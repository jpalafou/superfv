import numpy as np
import pytest

from superfv.mesh import UniformFVMesh
from superfv.slope_limiting.muscl import compute_MUSCL_slopes, compute_PP2D_slopes
from superfv.tools.device_management import CUPY_AVAILABLE, ArrayManager, xp


@pytest.mark.parametrize("face_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen", "PP2D"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
@pytest.mark.parametrize("mode", ["uniform", "ramp"])
def test_field(
    face_dim: str,
    active_dims: str,
    limiter: str,
    SED: bool,
    check_uniformity: bool,
    mode: str,
):
    """
    Test limited slope computation on a uniform field.
    """
    if face_dim not in active_dims:
        pytest.skip("face_dim must be in active_dims")
    if limiter == "PP2D" and (len(active_dims) != 2 or face_dim != active_dims[0]):
        pytest.skip("PP2D limiter is only applicable in 2D")

    N = 16
    array_manager = ArrayManager()
    if CUPY_AVAILABLE:
        array_manager.transfer_to("gpu")
    mesh = UniformFVMesh(
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        slab_depth=0,
        array_manager=array_manager,
    )

    u = xp.empty((1, *mesh.shape))
    alpha = xp.empty((1, *mesh.shape))
    dux = xp.empty((1, *mesh.shape))
    duy = xp.empty((1, *mesh.shape))

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
        modified = compute_PP2D_slopes(u, alpha, dux, duy, active_dims, SED=SED)

        dux[...] = dux / (1 / N)
        duy[...] = duy / (1 / N)

    else:
        modified = compute_MUSCL_slopes(u, alpha, dux, face_dim, limiter, SED=SED)
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
