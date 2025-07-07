import cupy as cp
import numpy as np

from superfv.stencil import inplace_multistencil_sweep, inplace_stencil_sweep

N = 64
stencil1 = cp.array([0.0, 1.0, 0.0])
stencil2 = cp.array([0.1, 0.8, 0.1])
multistencil = cp.array([stencil1, stencil2])

# allocate arrays
arr = cp.random.rand(5, N, N, N)
out1 = cp.empty((5, N, N, N, 2))
out2 = cp.empty((5, N, N, N, 2))

# perform serial stencil sweeps
for _ in range(10):
    modified1 = inplace_stencil_sweep(cp, arr, stencil1, axis=1, out=out1[..., 0])
    _ = inplace_stencil_sweep(cp, arr, stencil2, axis=1, out=out1[..., 1])

# perform multistencil sweep
for _ in range(10):
    modified2 = inplace_multistencil_sweep(
        cp, arr, multistencil, axis=1, out=out2[..., :2]
    )

assert np.array_equal(out1[modified1], out2[modified2]), "Outputs do not match!"
