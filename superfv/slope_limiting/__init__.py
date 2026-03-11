from functools import lru_cache
from itertools import product
from typing import List, Literal, Tuple

import numpy as np

from superfv.axes import DIM_TO_AXIS
from superfv.cuda_params import DEFAULT_THREADS_PER_BLOCK
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayLike
from superfv.tools.slicing import crop, merge_slices


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
    u: ArrayLike,
    M: ArrayLike,
    m: ArrayLike,
    active_dims: Tuple[Literal["x", "y", "z"], ...],
    include_corners: bool = True,
) -> Tuple[slice, ...]:
    """
    Compute the maximum and minimum values of each point and its neighbors along
    specified dimensions, writing the results to `M` and `m`, respectively.

    Args:
        u: Input array with shape (nvars, nx, ny, nz).
        M: Array to store the maximum values, must have the same shape as `u`.
        m: Array to store the minimum values, must have the same shape as `u`.
        active_dims: Tuple of active dimensions for interpolation, e.g., ("x", "y").
        include_corners: Whether to include corner neighbors in the computation.

    Returns:
        Slice objects indicating the modified regions in the output array.
    """
    all_slices = gather_neighbor_slices(active_dims, include_corners)
    inner_slice = all_slices[0]

    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        compute_dmp_kernel_helper(u, M, m, include_corners)
    else:
        # stack views of neighbors
        all_views = [u[slc] for slc in all_slices]
        stacked = np.stack(all_views, axis=0)

        # compute min an max
        M[inner_slice] = np.max(stacked, axis=0)
        m[inner_slice] = np.min(stacked, axis=0)

    return inner_slice


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    dmp_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void dmp_kernel(
            const double* __restrict__ u,   // (nvars,nx,ny,nz) contiguous C-order
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

                M[i] = u[i];
                m[i] = u[i];

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

                            M[i] = (u[j] > M[i]) ? u[j] : M[i];
                            m[i] = (u[j] < m[i]) ? u[j] : m[i];
                        }
                    }
                }
            }
        }
        """,
        "dmp_kernel",
    )

    def compute_dmp_kernel_helper(
        u: cp.ndarray,
        M: cp.ndarray,
        m: cp.ndarray,
        include_corners: bool,
    ):
        nvars, nx, ny, nz = u.shape

        if not u.flags.c_contiguous:
            raise ValueError("Input array must be C-contiguous")
        if not M.flags.c_contiguous:
            raise ValueError("Output array M must be C-contiguous")
        if not m.flags.c_contiguous:
            raise ValueError("Output array m must be C-contiguous")
        if u.dtype != cp.float64 or M.dtype != cp.float64 or m.dtype != cp.float64:
            raise ValueError("Input array must be of type float64")
        if u.ndim != 4:
            raise ValueError("Input array must have 4 dimensions (nvars, nx, ny, nz)")
        if M.shape != u.shape or m.shape != u.shape:
            raise ValueError(
                "Output arrays M and m must have the same shape as input array u"
            )

        threads_per_block = DEFAULT_THREADS_PER_BLOCK
        blocks_per_grid = (u.size + threads_per_block - 1) // threads_per_block

        dmp_kernel(
            (blocks_per_grid,),
            (threads_per_block,),
            (u, M, m, nvars, nx, ny, nz, int(include_corners)),
        )
