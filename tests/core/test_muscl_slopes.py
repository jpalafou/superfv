import numpy as np
import pytest

from superfv.mesh import UniformFVMesh
from superfv.slope_limiting.muscl import (
    compute_limited_slopes,
    compute_PP2D_slopes,
    musclConfig,
)
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayManager, xp

if CUPY_AVAILABLE:
    from superfv.slope_limiting.muscl import (
        MUSCL_slopes_kernel_helper,
        PP2D_slopes_kernel_helper,
    )


@pytest.mark.parametrize("face_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen", "PP2D"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
@pytest.mark.parametrize("mode", ["uniform", "ramp"])
@pytest.mark.parametrize("cupy", [False, True])
def test_field(
    face_dim: str,
    active_dims: str,
    limiter: str,
    SED: bool,
    check_uniformity: bool,
    mode: str,
    cupy: bool,
):
    """
    Test limited slope computation on a uniform field.
    """
    if cupy and not CUPY_AVAILABLE:
        pytest.skip("CuPy is not available")
    if face_dim not in active_dims:
        pytest.skip("face_dim must be in active_dims")
    if limiter == "PP2D" and (len(active_dims) != 2 or face_dim != active_dims[0]):
        pytest.skip("PP2D limiter is only applicable in 2D")

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        physical_admissibility_detection=False,
        limiter=limiter,
    )

    N = 16
    array_manager = ArrayManager()
    if cupy:
        array_manager.transfer_to("gpu")
    mesh = UniformFVMesh(
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        slab_depth=0,
        array_manager=array_manager,
    )

    u = xp.empty((1, *mesh.shape)) if cupy else np.empty((1, *mesh.shape))
    alpha = xp.empty((1, *mesh.shape, 1)) if cupy else np.empty((1, *mesh.shape, 1))
    buffer = xp.empty((1, *mesh.shape, 20)) if cupy else np.empty((1, *mesh.shape, 20))
    dux = xp.empty((1, *mesh.shape, 1)) if cupy else np.empty((1, *mesh.shape, 1))
    duy = xp.empty((1, *mesh.shape, 1)) if cupy else np.empty((1, *mesh.shape, 1))

    # fill u with data
    if mode == "uniform":
        u[0] = 1.0
    elif mode == "ramp":

        def ramp(X, Y, Z):
            return 2 * X + 3 * Y + 5 * Z

        u[0] = ramp(*mesh.get_cell_centers())
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # compute limited slopes
    if limiter == "PP2D":
        if cupy:
            modified = PP2D_slopes_kernel_helper(
                u,
                alpha[..., 0],
                dux[..., 0],
                duy[..., 0],
                active_dims[0],
                active_dims[1],
                1e-20,
                SED,
            )
        else:
            modified = compute_PP2D_slopes(
                u,
                active_dims,
                Sx=dux,
                Sy=duy,
                buffer=buffer,
                eps=1e-20,
                config=config,
                alpha=alpha,
            )

        dux[...] = dux / (1 / N)
        duy[...] = duy / (1 / N)

    else:
        if cupy:
            modified = MUSCL_slopes_kernel_helper(
                u, alpha[..., 0], dux[..., 0], face_dim, limiter, SED
            )
        else:
            modified = compute_limited_slopes(
                u,
                face_dim,
                out=duy,
                alpha=alpha,
                buffer=buffer,
                config=config,
            )
        dux[...] = dux / (1 / N)

    # check that the slopes are correct
    if mode == "uniform":
        np.all(dux[modified] == 0.0)
        if limiter == "PP2D":
            np.all(duy[modified] == 0.0)
    elif mode == "ramp":
        if face_dim == "x":
            np.all(dux[modified] == 2)
        elif face_dim == "y":
            np.all(dux[modified] == 3)
        elif face_dim == "z":
            np.all(dux[modified] == 5)

        if limiter == "PP2D":
            if active_dims[1] == "x":
                np.all(duy[modified] == 2)
            elif active_dims[1] == "y":
                np.all(duy[modified] == 3)
            elif active_dims[1] == "z":
                np.all(duy[modified] == 5)
