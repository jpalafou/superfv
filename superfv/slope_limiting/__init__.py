from functools import lru_cache
from itertools import product
from types import ModuleType
from typing import List, Literal, Tuple, cast

from superfv.axes import DIM_TO_AXIS
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, insert_slice, merge_slices


def gather_neighbor_slices(
    active_dims: Tuple[Literal["x", "y", "z"], ...], include_corners: bool
) -> List[Tuple[slice, ...]]:
    """
    Returns a list of slice objects for gathering neighbors in up to 3 dimensions with
    the center slice as the first element.

    Args:
        active_dims (Tuple[Literal["x", "y", "z"], ...]): Active dimensions for
        interpolation.
        include_corners (bool): Whether to include corner neighbors.

    Returns:
        List[Tuple[slice, ...]]: List of slice objects for gathering neighbors.
    """
    return _gather_neighbor_slices(active_dims, include_corners)


@lru_cache(maxsize=None)
def _gather_neighbor_slices(
    active_dims: Tuple[Literal["x", "y", "z"], ...], include_corners: bool
) -> List[Tuple[slice, ...]]:
    ndim = len(active_dims)
    axes = tuple(DIM_TO_AXIS[dim] for dim in active_dims)

    # gather all slices excluding the center
    all_slices: List[Tuple[slice, ...]] = []
    if include_corners:
        for offset in product([-1, 0, 1], repeat=ndim):
            if offset == (0,) * ndim:
                continue
            all_slices.append(
                merge_slices(
                    *[
                        crop(i, (1 + shift, -1 + shift), ndim=4)
                        for i, shift in zip(axes, offset)
                    ]
                )
            )
    else:
        for ax in axes:
            for shift in [-1, 1]:
                all_slices.append(
                    merge_slices(
                        *[
                            crop(
                                i,
                                (1 + shift, -1 + shift) if ax == i else (1, -1),
                                ndim=4,
                            )
                            for i in axes
                        ]
                    )
                )

    # insert inner slices in beginning
    inner_slice = merge_slices(*[crop(i, (1, -1), ndim=4) for i in axes])
    all_slices.insert(0, inner_slice)

    return all_slices


def compute_dmp(
    xp: ModuleType,
    arr: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    *,
    out: ArrayLike,
    include_corners: bool,
) -> Tuple[slice, ...]:
    """
    Compute the minimum and maximum values of each point and its neighbors along
    specified dimensions.

    Args:
        xp: `np` namespace.
        arr: Array. First axis is assumed to be variable axis. Has shape
            (nvars, nx, ny, nz).
        active_dims: Dimensions to check. Must be a subset of ("x", "y", "z").
        out: Output array to store the results. Should have shape
            (nvars, nx, ny, nz, >=2). The DMP min will be stored in the last axis at
            index 0, and the DMP max at index 1.
        include_corners: Whether to include corners.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    all_slices = gather_neighbor_slices(active_dims, include_corners)
    inner_slice = all_slices[0]

    if hasattr(xp, "cuda"):
        M = xp.empty_like(arr)
        m = xp.empty_like(arr)
        compute_dmp_kernel_helper(arr, M, m, include_corners)
        out[insert_slice(inner_slice, 4, 1)] = M[inner_slice]
        out[insert_slice(inner_slice, 4, 0)] = m[inner_slice]
        modified = cast(Tuple[slice, ...], insert_slice(inner_slice, 4, slice(None, 2)))
        return modified

    # stack views of neighbors
    all_views = [arr[slc] for slc in all_slices]
    stacked = xp.stack(all_views, axis=0)

    # compute min an max
    out[insert_slice(inner_slice, 4, 0)] = xp.min(stacked, axis=0)
    out[insert_slice(inner_slice, 4, 1)] = xp.max(stacked, axis=0)

    # return inner slice
    modified = cast(Tuple[slice, ...], insert_slice(inner_slice, 4, slice(None, 2)))
    return modified


def compute_vis(
    xp: ModuleType, dmp: ArrayLike, rtol: float, atol: float, *, out: ArrayLike
):
    """
    Compute a boolean array indicating where the local DMP spread is significant and
    should be visualized. A cell is flagged True where:

    |dmp_max - dmp_min| > atol + rtol * max(|dmp_max|, |dmp_min|)

    Args:
        xp: `np` namespace.
        dmp: Array containing the discrete maximum principle values. Has shape
            (nvars, nx, ny, nz, >=2), where the last axis contains the min and max
            values.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        out: Output boolean array to store the results. Should have shape
            (nvars, nx, ny, nz).
    """
    m = dmp[..., 0]
    M = dmp[..., 1]
    out[...] = M - m > atol + rtol * xp.maximum(xp.abs(m), xp.abs(M))


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    dmp_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void dmp_kernel(
            const double* __restrict__ w,   // (nvars,nx,ny,nz) contiguous C-order
            double* __restrict__ M,      // same shape
            double* __restrict__ m,      // same shape
            const int nvars,
            const int nx,
            const int ny,
            const int nz,
            const int include_corners
        ){
            // Flattened index over nvars*nx*ny*nz
            const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
            const long long stride = (long long)blockDim.x * gridDim.x;

            const long long n = (long long)nvars * nx * ny * nz;
            const bool usingx = nx > 1;
            const bool usingy = ny > 1;
            const bool usingz = nz > 1;
            const int ndim = (int)usingx + (int)usingy + (int)usingz;

            for (long long i = tid; i < n; i += stride) {

                long long t = i;
                int iz = (int)(t % nz); t /= nz;
                int iy = (int)(t % ny); t /= ny;
                int ix = (int)(t % nx); t /= nx;
                int iv = (int)t;

                if (usingx && !(ix > 0 && ix < nx - 1)) continue;
                if (usingy && !(iy > 0 && iy < ny - 1)) continue;
                if (usingz && !(iz > 0 && iz < nz - 1)) continue;

                M[i] = w[i];
                m[i] = w[i];

                for (int offx = -1; offx <= 1; offx++) {
                    if (!usingx && offx != 0) continue;

                    for (int offy = -1; offy <= 1; offy++) {
                        if (!usingy && offy != 0) continue;

                        for (int offz = -1; offz <= 1; offz++) {
                            if (!usingz && offz != 0) continue;

                            if (offx == 0 && offy == 0 && offz == 0) continue;

                            if (include_corners == 0 && ndim > 1) {
                                int offsum = 0;
                                if (usingx) offsum += (offx != 0);
                                if (usingy) offsum += (offy != 0);
                                if (usingz) offsum += (offz != 0);
                                if (offsum > 1) continue;
                            }

                            int jx = ix + offx;
                            int jy = iy + offy;
                            int jz = iz + offz;
                            const long long j =
                                (((long long)iv * nx + jx) * ny + jy) * nz + jz;

                            M[i] = (w[j] > M[i]) ? w[j] : M[i];
                            m[i] = (w[j] < m[i]) ? w[j] : m[i];
                        }
                    }
                }
            }
        }
        """,
        "dmp_kernel",
    )

    def compute_dmp_kernel_helper(
        w: ArrayLike,
        M: ArrayLike,
        m: ArrayLike,
        include_corners: bool,
    ):
        nvars, nx, ny, nz = w.shape

        n = w.size
        threads = 256
        blocks = min(65535, (n + threads - 1) // threads)

        dmp_kernel(
            (blocks,), (threads,), (w, M, m, nvars, nx, ny, nz, int(include_corners))
        )
