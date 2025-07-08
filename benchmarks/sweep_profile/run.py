import cupy as xp
from line_profiler import profile

from superfv.stencil import get_symmetric_slices
from superfv.tools.norms import linf_norm


@profile
def inplace_multistencil_sweep(
    xp,
    arr: xp.ndarray,
    stencils: xp.ndarray,
    axis: int,
    out: xp.ndarray,
):
    arr_ndims = arr.ndim
    nstencils, stencil_len = stencils.shape

    arr_slices = get_symmetric_slices(arr_ndims, stencil_len, axis)
    modified = arr_slices[stencil_len // 2]

    # Zero the central region across all stencils
    out_modified = modified + (slice(None, nstencils),)
    out[out_modified] = 0.0

    # Prepare the arrays for accumulation
    out_view = out[out_modified]
    w_stack = stencils.T.reshape((stencil_len,) + (1,) * arr_ndims + (nstencils,))
    for i, s in enumerate(arr_slices):
        w = w_stack[i]
        arr_sliced = arr[s][..., xp.newaxis]
        xp.add(out_view, xp.multiply(arr_sliced, w), out=out_view)

    return out_modified


@profile
def inplace_multistencil_sweep_with_einsum(
    xp, arr: xp.ndarray, stencils: xp.ndarray, axis: int, out: xp.ndarray
):
    arr_ndims = arr.ndim
    nstencils, stencil_len = stencils.shape

    arr_slices = get_symmetric_slices(arr_ndims, stencil_len, axis)
    modified = arr_slices[stencil_len // 2]
    out_modified = modified + (slice(None, nstencils),)

    # Stack all relevant slices: shape -> (stencil_len, ...arr.shape)
    arr_stack = xp.stack(
        [arr[s] for s in arr_slices], axis=0
    )  # (stencil_len, *arr.shape)

    # Broadcast stencils: (stencil_len, nstencils)
    # Result shape: (*arr.shape, nstencils)
    result = xp.einsum("i...,in->...n", arr_stack, stencils.T)

    # Write to output in-place
    out[out_modified] = result

    return out_modified


@profile
def inplace_multistencil_sweep_with_sync(
    xp,
    arr: xp.ndarray,
    stencils: xp.ndarray,
    axis: int,
    out: xp.ndarray,
):
    arr_ndims = arr.ndim
    nstencils, stencil_len = stencils.shape

    arr_slices = get_symmetric_slices(arr_ndims, stencil_len, axis)
    modified = arr_slices[stencil_len // 2]

    # Zero the central region across all stencils
    out_modified = modified + (slice(None, nstencils),)
    out[out_modified] = 0.0

    # Prepare the arrays for accumulation
    out_view = out[out_modified]
    w_stack = stencils.T.reshape((stencil_len,) + (1,) * arr_ndims + (nstencils,))
    for i, s in enumerate(arr_slices):
        w = w_stack[i]
        arr_sliced = arr[s][..., xp.newaxis]
        xp.add(out_view, xp.multiply(arr_sliced, w), out=out_view)

    xp.cuda.Device().synchronize()

    return out_modified


N = 256
ntrials = 5

u = xp.ones([5, N, N, 1])
out1 = xp.zeros([5, N, N, 1, 2])
out2 = xp.zeros([5, N, N, 1, 2])
out3 = xp.zeros([5, N, N, 1, 2])

stencil1 = xp.array([0.1, 0.2, 0.4, 0.2, 0.1])
stencil2 = xp.array([0.05, 0.1, 0.7, 0.1, 0.05])
stencils = xp.array([stencil1, stencil2])


def test_output():
    modified1 = inplace_multistencil_sweep(xp, u, stencils, 1, out1)
    modified2 = inplace_multistencil_sweep_with_einsum(xp, u, stencils, 1, out2)
    modified3 = inplace_multistencil_sweep(xp, u, stencils, 1, out3, sync=True)

    print(linf_norm(out1[modified1] - out2[modified2]))
    print(linf_norm(out1[modified1] - out3[modified3]))


@profile
def compare():
    for _ in range(ntrials):
        inplace_multistencil_sweep(xp, u, stencils, 1, out1)
        inplace_multistencil_sweep(xp, out1[..., 0], stencils, 2, out1)
        result1 = xp.min(out1)

    for _ in range(ntrials):
        inplace_multistencil_sweep_with_einsum(xp, u, stencils, 1, out2)
        inplace_multistencil_sweep_with_einsum(xp, out2[..., 0], stencils, 2, out2)
        result2 = xp.min(out2)

    for _ in range(ntrials):
        inplace_multistencil_sweep_with_sync(xp, u, stencils, 1, out3)
        inplace_multistencil_sweep_with_sync(xp, out3[..., 0], stencils, 2, out3)
        result3 = xp.min(out3)

    return result1, result2, result3


compare()
