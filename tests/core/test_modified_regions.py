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
    compute_limited_slopes,
    compute_PP2D_slopes,
    musclConfig,
)
from superfv.slope_limiting.shock_detection import compute_shock_detector
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.slope_limiting.zhang_and_shu import ZhangShuConfig, compute_theta
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
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims: str, include_corners: bool):
    xp = configure_xp()

    u, _, out = sample_data(dims, nout=2, xp=xp)
    modified = compute_dmp(xp, u, tuple(dims), out=out, include_corners=include_corners)

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_compute_limited_slopes(
    dims: str, limiter: str, SED: bool, check_uniformity: bool
):
    xp = configure_xp()

    face_dim = dims[0]
    u, buffer, temp = sample_data(dims, nout=2, xp=xp)
    out = temp[..., :1]
    alpha = temp[..., 1:2]

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        limiter=limiter,
        physical_admissibility_detection=False,
    )

    modified = compute_limited_slopes(
        xp,
        u,
        face_dim,
        tuple(dims),
        out=out,
        buffer=buffer,
        alpha=alpha,
        config=config,
    )

    assert not xp.any(xp.isnan(out[modified]))
    # skipping all nan check since the stencils will leave some ghost cells non-nan

    if SED:
        assert not xp.any(xp.isnan(alpha[modified]))
        alpha[modified] = xp.nan
        assert xp.all(xp.isnan(alpha))


@pytest.mark.parametrize("dims", ["xy", "xz", "yz"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
def test_compute_PP2D_slopes(dims: str, SED: bool, check_uniformity: bool):
    xp = configure_xp()

    u, buffer, temp = sample_data(dims, nout=3, xp=xp)
    Sx = temp[..., :1]
    Sy = temp[..., 1:2]
    alpha = temp[..., 2:3]

    config = musclConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        limiter="PP2D",
        physical_admissibility_detection=False,
    )

    modified = compute_PP2D_slopes(
        xp, u, tuple(dims), Sx=Sx, Sy=Sy, buffer=buffer, alpha=alpha, config=config
    )

    assert not xp.any(xp.isnan(Sx[modified]))
    assert not xp.any(xp.isnan(Sy[modified]))
    # skipping all nan check since the 2D stencils will leave some ghost cells non-nan

    if SED:
        assert not xp.any(xp.isnan(alpha[modified]))
        alpha[modified] = xp.nan
        assert xp.all(xp.isnan(alpha))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_compute_shock_detector(dims: str):
    xp = configure_xp()

    u, buffer, temp = sample_data(dims, nout=2, xp=xp)
    out = temp[:1, ..., 0]
    eta = temp[..., 1]

    modified = compute_shock_detector(
        xp, u, u, tuple(dims), 0.025, out=out, eta=eta, buffer=buffer
    )

    assert not xp.any(xp.isnan(out[modified]))
    assert not xp.any(xp.isnan(eta[modified]))
    out[modified] = xp.nan
    eta[modified] = xp.nan
    assert xp.all(xp.isnan(out))
    assert xp.all(xp.isnan(eta))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("check_uniformity", [False, True])
@pytest.mark.parametrize("PAD", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_theta(
    dims: str, SED: bool, check_uniformity: bool, PAD: bool, include_corners: bool
):
    xp = configure_xp()

    u, mega_buffer, out = sample_data(dims, nout=1, xp=xp)
    nodes, _, _ = sample_data(dims, nout=1, xp=xp)
    nodes = nodes[..., xp.newaxis]

    dmp = mega_buffer[..., :2]
    alpha = mega_buffer[..., 2:3]
    buffer = mega_buffer[..., 3:]

    config = ZhangShuConfig(
        shock_detection=False,
        smooth_extrema_detection=SED,
        check_uniformity=check_uniformity,
        physical_admissibility_detection=PAD,
        include_corners=include_corners,
        PAD_atol=0,
        PAD_bounds=xp.array([[0.0, 0.1] * 5]),
    )

    modified = compute_theta(
        xp,
        u,
        nodes,
        nodes if "x" in dims else None,
        nodes if "y" in dims else None,
        nodes if "z" in dims else None,
        out=out,
        dmp=dmp,
        alpha=alpha,
        buffer=buffer,
        config=config,
    )

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("absolute_dmp", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_detect_NAD_violations(dims: str, absolute_dmp: bool, include_corners: bool):
    xp = configure_xp()

    uold, buffer, out = sample_data(dims, nout=1, xp=xp)
    unew, _, _ = sample_data(dims, nout=1, xp=xp)
    out = out[..., 0]
    modified = detect_NAD_violations(
        xp,
        uold,
        unew,
        tuple(dims),
        out=out,
        dmp=buffer,
        absolute_dmp=absolute_dmp,
        include_corners=include_corners,
    )

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))


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
    xp = configure_xp()

    u, buffer, out = sample_data(dims, nout=1, xp=xp)
    modified = smooth_extrema_detector(
        xp, u, tuple(dims), check_uniformity, out=out, buffer=buffer
    )

    assert not xp.any(xp.isnan(out[modified]))
    out[modified] = xp.nan
    assert xp.all(xp.isnan(out))
