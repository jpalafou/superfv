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
from superfv.slope_limiting.muscl import (
    compute_MUSCL_slopes,
    compute_PP2D_slopes,
    musclConfig,
)
from superfv.slope_limiting.shock_detection import detect_shocks
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
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

    from superfv.slope_limiting.smooth_extrema_detection import (
        compute_alpha_kernel_helper,
    )


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
def test_compute_alpha_kernel_helper(dims: str, check_uniformity: bool):
    xp = configure_xp()

    if not hasattr(xp, "cuda"):
        pytest.skip("compute_alpha_kernel_helper is only implemented for CuPy")

    u, _, alpha = sample_data(dims, nout=1, xp=xp)
    modified = compute_alpha_kernel_helper(
        u,
        alpha[..., 0],
        1e-16,
        check_uniformity,
        1e-16,
    )

    assert not xp.any(xp.isnan(alpha[modified]))
    alpha[modified] = xp.nan
    assert xp.all(xp.isnan(alpha))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims: str, include_corners: bool):
    xp = configure_xp()

    u, _, M = sample_data(dims, nout=1, xp=xp)
    _, _, m = sample_data(dims, nout=1, xp=xp)
    modified = compute_dmp(u, M[..., 0], m[..., 0], tuple(dims), include_corners)

    assert not xp.any(xp.isnan(M[modified]))
    assert not xp.any(xp.isnan(m[modified]))
    M[modified] = xp.nan
    m[modified] = xp.nan
    assert xp.all(xp.isnan(M))
    assert xp.all(xp.isnan(m))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_MUSCL_slopes(dims: str, SED: bool):
    xp = configure_xp()

    face_dim = dims[0]
    u, _, _ = sample_data(dims, nout=2, xp=xp)
    slopes = xp.empty_like(u) * xp.nan
    alpha = xp.empty_like(u) * xp.nan if SED else None

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=False,
        limiter="minmod",
        physical_admissibility_detection=False,
    )

    modified = compute_MUSCL_slopes(
        u,
        alpha,
        slopes,
        face_dim,
        config,
    )

    assert not xp.any(xp.isnan(slopes[modified]))
    # skipping all nan check since the stencils will modify some ghost cells

    if SED:
        assert xp.all(xp.isnan(alpha))


@pytest.mark.parametrize("dims", ["xy", "xz", "yz"])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_PP2D_slopes(dims: str, SED: bool):
    xp = configure_xp()

    u, _, _ = sample_data(dims, nout=3, xp=xp)
    Sx = xp.empty_like(u) * xp.nan
    Sy = xp.empty_like(u) * xp.nan
    alpha = xp.empty_like(u) * xp.nan if SED else None

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=False,
        limiter="PP2D",
        physical_admissibility_detection=False,
    )

    modified = compute_PP2D_slopes(
        u,
        alpha,
        Sx,
        Sy,
        tuple(dims),
        config,
    )

    assert not xp.any(xp.isnan(Sx[modified]))
    assert not xp.any(xp.isnan(Sy[modified]))
    # skipping all nan check since the stencils will modify some ghost cells

    if SED:
        assert xp.all(xp.isnan(alpha))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("check_uniformity", [False, True])
@pytest.mark.parametrize("PAD", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_theta(
    dims: str, check_uniformity: bool, PAD: bool, include_corners: bool
):
    u, buffer, out = sample_data(dims, nout=1, xp=np)
    nodes, _, _ = sample_data(dims, nout=1, xp=np)
    nodes = nodes[..., np.newaxis]

    M = np.empty(u.shape)
    m = np.empty(u.shape)
    Mj = np.empty(u.shape)
    mj = np.empty(u.shape)

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
    [dict(), dict(rtol=1e-2, atol=1e-14), dict(delta=False)],
)
@pytest.mark.parametrize("include_corners", [False, True])
def test_detect_NAD_violations(
    dims: str,
    NAD_config: dict,
    include_corners: bool,
):
    uold, buffer, out1 = sample_data(dims, nout=1, xp=np)
    unew, _, _ = sample_data(dims, nout=2, xp=np)

    M = np.empty(uold.shape)
    m = np.empty(uold.shape)
    out = out1[..., 0]

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
    u, _, eta = sample_data(dims, nout=3, xp=xp)
    has_shock = xp.full_like(eta[:1, ..., 0], -1, dtype=xp.int32)

    modified = detect_shocks(u, u, eta, has_shock, tuple(dims), 0.025)

    for i, dim in enumerate(["x", "y", "z"]):
        if dim in dims:
            assert not xp.any(xp.isnan(eta[..., i][modified]))
            # skipping all nan check since the stencils will modify some ghost cells

    assert not xp.any(has_shock[modified] == -1)
    has_shock[modified] = -1
    assert xp.all(has_shock == -1)


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_map_cells_values_to_face_values(dims: str):
    xp = configure_xp()

    face_dim = dims[0]
    axis = DIM_TO_AXIS[face_dim]
    u, _, out = sample_data(dims, nout=1, N=33, xp=xp)
    out = out[..., 0]
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


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_smooth_extrema_detection(dims: str, check_uniformity: bool):
    u, buffer, out = sample_data(dims, nout=1, xp=np)
    modified = smooth_extrema_detector(
        u, tuple(dims), check_uniformity, out=out, buffer=buffer
    )

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


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
