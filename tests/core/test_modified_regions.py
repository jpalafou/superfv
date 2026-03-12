import warnings
from types import ModuleType

import numpy as np
import pytest

from superfv.axes import DIM_TO_AXIS
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.MOOD import (
    blend_troubled_cells,
    detect_NAD_violations,
    map_cells_values_to_face_values,
)
from superfv.slope_limiting.muscl import compute_MUSCL_slopes, compute_PP2D_slopes
from superfv.slope_limiting.shock_detection import detect_shocks
from superfv.slope_limiting.smooth_extrema_detection import compute_alpha
from superfv.slope_limiting.zhang_and_shu import ZhangShuConfig, compute_theta
from superfv.stencils import transverse_integration
from superfv.stencils.conservative_interpolation import (
    cell_center,
    gauss_legendre_nodes,
    left_right,
)
from superfv.sweep import stencil_sweep
from superfv.tools.device_management import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def configure_xp():
    if CUPY_AVAILABLE:
        warnings.warn("Running tests with CuPy", RuntimeWarning)
        return cp
    else:
        return np


def sample_data(dims: str, nout: int = 1, N: int = 32, *, xp: ModuleType) -> tuple:
    xyz_shape = (
        N if "x" in dims else 1,
        N if "y" in dims else 1,
        N if "z" in dims else 1,
    )
    u = xp.ones((5, *xyz_shape))
    buffer = xp.full((5, *xyz_shape, 20), xp.nan)
    out = xp.full((5, *xyz_shape, nout), xp.nan)
    return u, buffer, out


@pytest.mark.parametrize("dims", ["x", "y", "z"])
def test_blend_troubled_cells(dims: str):
    xp = configure_xp()

    troubles0, buffer, out = sample_data("xyz", nout=1, xp=xp)
    troubles1 = out[..., 0]
    modified = blend_troubled_cells(
        xp, troubles0, tuple(dims), out=troubles1, buffer=buffer
    )

    assert not xp.any(xp.isnan(troubles1[modified]))
    troubles1[modified] = xp.nan
    assert xp.all(xp.isnan(troubles1))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_compute_alpha(dims: str, check_uniformity: bool):
    xp = configure_xp()

    u, _, _ = sample_data(dims, nout=1, xp=xp)
    alpha = xp.full_like(u, xp.nan)

    modified = compute_alpha(u, alpha, tuple(dims), check_uniformity)

    assert not xp.any(xp.isnan(alpha[modified]))
    # skip all-nan check


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims: str, include_corners: bool):
    xp = configure_xp()

    u, _, _ = sample_data(dims, nout=1, xp=xp)
    M = xp.full_like(u, xp.nan)
    m = xp.full_like(u, xp.nan)

    modified = compute_dmp(u, M, m, tuple(dims), include_corners)

    assert not xp.any(xp.isnan(M[modified]))
    assert not xp.any(xp.isnan(m[modified]))
    M[modified] = xp.nan
    m[modified] = xp.nan
    assert xp.all(xp.isnan(M))
    assert xp.all(xp.isnan(m))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", [0, 1, 2])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_MUSCL_slopes(dims: str, limiter: int, SED: bool):
    xp = configure_xp()

    face_dim = dims[0]
    u, _, _ = sample_data(dims, nout=2, xp=xp)
    alpha = xp.zeros_like(u)
    slopes = xp.full_like(u, xp.nan)

    modified = compute_MUSCL_slopes(
        u,
        alpha,
        slopes,
        face_dim,
        {0: None, 1: "minmod", 2: "minmod"}[limiter],
        SED=SED,
    )

    assert not xp.any(xp.isnan(slopes[modified]))
    slopes[modified] = xp.nan
    assert xp.all(xp.isnan(slopes))


@pytest.mark.parametrize("dims", ["xy", "xz", "yz"])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_PP2D_slopes(dims: str, SED: bool):
    xp = configure_xp()

    u, _, _ = sample_data(dims, nout=3, xp=xp)
    alpha = xp.zeros_like(u)
    Sx = xp.full_like(u, xp.nan)
    Sy = xp.full_like(u, xp.nan)

    modified = compute_PP2D_slopes(
        u,
        alpha,
        Sx,
        Sy,
        tuple(dims),
        SED=SED,
    )

    assert not xp.any(xp.isnan(Sx[modified]))
    assert not xp.any(xp.isnan(Sy[modified]))
    Sx[modified] = xp.nan
    Sy[modified] = xp.nan
    assert xp.all(xp.isnan(Sx))
    assert xp.all(xp.isnan(Sy))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("check_uniformity", [False, True])
@pytest.mark.parametrize("PAD", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_theta(
    dims: str, check_uniformity: bool, PAD: bool, include_corners: bool
):
    pytest.skip("Skip for now")

    u, buffer, out = sample_data(dims, nout=1, xp=np)
    _, _, nodes = sample_data(dims, nout=1, xp=np)

    M = np.ones_like(u)
    m = np.ones_like(u)
    Mj = np.ones_like(u)
    mj = np.ones_like(u)

    config = ZhangShuConfig(
        shock_detection=False,
        smooth_extrema_detection=False,
        check_uniformity=check_uniformity,
        physical_admissibility_detection=PAD,
        include_corners=include_corners,
        PAD_atol=0,
        PAD_bounds=np.array([[0.0, 0.1] * 5]),
    )

    modified = compute_theta(
        u,
        nodes,
        nodes if "x" in dims else None,
        nodes if "y" in dims else None,
        nodes if "z" in dims else None,
        out=out,
        M=M,
        m=m,
        Mj=Mj,
        mj=mj,
        buffer=buffer,
        config=config,
    )

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize(
    "NAD_config",
    [dict(), dict(rtol=1e-2, atol=1e-15), dict(delta=False)],
)
@pytest.mark.parametrize("include_corners", [False, True])
def test_detect_NAD_violations(
    dims: str,
    NAD_config: dict,
    include_corners: bool,
):
    uold, buffer, _ = sample_data(dims, nout=1, xp=np)
    unew = np.ones_like(uold)
    M = np.ones_like(uold)
    m = np.ones_like(uold)
    out = np.full_like(uold, np.nan)

    modified = detect_NAD_violations(
        uold,
        unew,
        tuple(dims),
        include_corners=include_corners,
        out=out,
        M=M,
        m=m,
        buffer=buffer,
        **NAD_config,
    )

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_detect_shocks(dims: str):
    xp = configure_xp()
    u, _, _ = sample_data(dims, nout=1, xp=xp)
    eta = xp.full_like(u, xp.nan)
    has_shock = xp.full_like(eta[:1], -1, dtype=xp.int32)

    modified = detect_shocks(u, u, eta, has_shock, tuple(dims), 0.025)

    assert not xp.any(xp.isnan(eta[modified]))
    # skip all-nan check

    assert not xp.any(has_shock[modified] == -1)
    has_shock[modified] = -1
    assert xp.all(has_shock == -1)


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_map_cells_values_to_face_values(dims: str):
    xp = configure_xp()

    face_dim = dims[0]
    axis = DIM_TO_AXIS[face_dim]
    u, _, _ = sample_data(dims, nout=1, N=33, xp=xp)
    out = xp.full_like(u, xp.nan)
    if "x" in dims:
        u = u[:, :-1, :, :]
        if face_dim != "x":
            out = out[:, :-1, :, :]
    if "y" in dims:
        u = u[:, :, :-1, :]
        if face_dim != "y":
            out = out[:, :, :-1, :]
    if "z" in dims:
        u = u[:, :, :, :-1]
        if face_dim != "z":
            out = out[:, :, :, :-1]

    modified = map_cells_values_to_face_values(xp, u, axis, out=out)

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))


@pytest.mark.parametrize("interp_dim", ["x", "y", "z"])
@pytest.mark.parametrize("active_dims", ["x", "y", "z", "xy", "yz", "xz", "xyz"])
@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize(
    "stencil", ["ci:center", "ci:left_right", "ci:gauss_legendre", "transverse"]
)
@pytest.mark.parametrize("ninterps", [1, 2])
def test_stencil_sweep(interp_dim, active_dims, p, stencil, ninterps):
    if interp_dim not in active_dims:
        pytest.skip("Interpolation dimension must be among active dimensions")

    xp = configure_xp()

    match stencil:
        case "ci:center":
            weights = cell_center(p)
        case "ci:left_right":
            weights = left_right(p)
        case "ci:gauss_legendre":
            weights = gauss_legendre_nodes(p)
        case "transverse":
            weights = transverse_integration(p)
        case _:
            raise ValueError(f"Unknown stencil: {stencil}")
    nouterps, _ = weights.shape
    weights = xp.asarray(weights)

    _, _, u = sample_data(active_dims, nout=ninterps, xp=xp)
    u[...] = 1.0
    _, _, out = sample_data(active_dims, nout=ninterps * nouterps, xp=xp)

    modified = stencil_sweep(u, weights, out, interp_dim)

    assert not xp.any(xp.isnan(u))

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))
