import numpy as np
import pytest

from superfv.mesh import UniformFVMesh
from superfv.slope_limiting.muscl import compute_limited_slopes, musclConfig


@pytest.mark.parametrize("face_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_uniform_field(
    face_dim: str, active_dims: str, limiter: str, SED: bool, check_uniformity: bool
):
    """
    Test limited slope computation on a uniform field.
    """
    if face_dim not in active_dims:
        pytest.skip("face_dim must be in active_dims")
    if SED and limiter != "moncen":
        pytest.skip("SED is only applicable with certain limiters")

    active_dims_tuple = tuple(active_dims)

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        physical_admissibility_detection=False,
    )

    N = 16
    mesh = UniformFVMesh(
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        slab_depth=0,
    )

    u = np.empty((1, *mesh.shape))
    alpha = np.empty((1, *mesh.shape, 1))
    buffer = np.empty((1, *mesh.shape, 20))
    du = np.empty((1, *mesh.shape, 1))

    # fill u with a linear ramp
    def ramp(X, Y, Z):
        return 2 * X + 3 * Y + 5 * Z

    u[0] = ramp(mesh.X, mesh.Y, mesh.Z)

    # compute limited slopes
    modified = compute_limited_slopes(
        np,
        u,
        face_dim,
        active_dims_tuple,
        out=du,
        alpha=alpha,
        buffer=buffer,
        config=config,
    )
    du[...] = du / (1 / N)

    # check that the slopes are correct
    np.all(du[modified] == 0.0)


@pytest.mark.parametrize("face_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_linear_ramp(
    face_dim: str, active_dims: str, limiter: str, SED: bool, check_uniformity: bool
):
    """
    Test limited slope computation on a linear ramp.
    """
    if face_dim not in active_dims:
        pytest.skip("face_dim must be in active_dims")
    if SED and limiter != "moncen":
        pytest.skip("SED is only applicable with certain limiters")

    active_dims_tuple = tuple(active_dims)

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        physical_admissibility_detection=False,
    )

    N = 16
    mesh = UniformFVMesh(
        nx=N if "x" in active_dims else 1,
        ny=N if "y" in active_dims else 1,
        nz=N if "z" in active_dims else 1,
        slab_depth=0,
    )

    u = np.empty((1, *mesh.shape))
    alpha = np.empty((1, *mesh.shape, 1))
    buffer = np.empty((1, *mesh.shape, 20))
    du = np.empty((1, *mesh.shape, 1))

    # fill u with a linear ramp
    def ramp(X, Y, Z):
        return 2 * X + 3 * Y + 5 * Z

    u[0] = ramp(mesh.X, mesh.Y, mesh.Z)

    # compute limited slopes
    modified = compute_limited_slopes(
        np,
        u,
        face_dim,
        active_dims_tuple,
        out=du,
        alpha=alpha,
        buffer=buffer,
        config=config,
    )
    du[...] = du / (1 / N)

    # check that the slopes are correct
    if face_dim == "x":
        assert np.all(du[modified] == 2)
    elif face_dim == "y":
        assert np.all(du[modified] == 3)
    elif face_dim == "z":
        assert np.all(du[modified] == 5)
