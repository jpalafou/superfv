import numpy as np
import pytest

from superfv.fv import DIM_TO_AXIS
from superfv.slope_limiting import compute_dmp
from superfv.slope_limiting.MOOD import (
    blend_troubled_cells,
    detect_NAD_violations,
    map_cells_values_to_face_values,
)
from superfv.slope_limiting.muscl import compute_limited_slopes, compute_PP2D_slopes
from superfv.slope_limiting.smooth_extrema_detection import smooth_extrema_detector
from superfv.slope_limiting.zhang_and_shu import ZhangShuConfig, compute_theta


def sample_data(dims: str, nout: int = 1, N: int = 32) -> tuple:
    xyz_shape = (
        N if "x" in dims else 1,
        N if "y" in dims else 1,
        N if "z" in dims else 1,
    )
    u = np.ones((5, *xyz_shape))
    buffer = np.full((5, *xyz_shape, 20), np.nan)
    out = np.full((5, *xyz_shape, nout), np.nan)
    return u, buffer, out


@pytest.mark.parametrize("dims", ["x", "y", "z"])
def test_blend_troubled_cells(dims: str):
    troubles0, buffer, out = sample_data("xyz", nout=1)
    troubles1 = out[..., 0]
    modified = blend_troubled_cells(
        np, troubles0, tuple(dims), out=troubles1, buffer=buffer
    )

    assert not np.any(np.isnan(troubles1[modified]))
    troubles1[modified] = np.nan
    assert np.all(np.isnan(troubles1))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_dmp(dims: str, include_corners: bool):
    u, _, out = sample_data(dims, nout=2)
    modified = compute_dmp(np, u, tuple(dims), out=out, include_corners=include_corners)

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("absolute_dmp", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_detect_NAD_violations(dims: str, absolute_dmp: bool, include_corners: bool):
    uold, buffer, out = sample_data(dims, nout=1)
    unew, _, _ = sample_data(dims, nout=1)
    out = out[..., 0]
    modified = detect_NAD_violations(
        np,
        uold,
        unew,
        tuple(dims),
        out=out,
        dmp=buffer,
        absolute_dmp=absolute_dmp,
        include_corners=include_corners,
    )

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("limiter", ["minmod", "moncen"])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_limited_slopes(dims: str, limiter: str, SED: bool):
    face_dim = dims[0]
    u, buffer, temp = sample_data(dims, nout=2)
    out = temp[..., :1]
    alpha = temp[..., 1:2]
    modified = compute_limited_slopes(
        np,
        u,
        face_dim,
        tuple(dims),
        out=out,
        buffer=buffer,
        limiter=limiter,
        SED=SED,
        alpha=alpha,
    )

    assert not np.any(np.isnan(out[modified]))
    # skipping all nan check since the stencils will leave some ghost cells non-nan

    if SED:
        assert not np.any(np.isnan(alpha[modified]))
        alpha[modified] = np.nan
        assert np.all(np.isnan(alpha))


@pytest.mark.parametrize("dims", ["xy", "xz", "yz"])
@pytest.mark.parametrize("SED", [False, True])
def test_compute_PP2D_slopes(dims: str, SED: bool):
    u, buffer, temp = sample_data(dims, nout=3)
    Sx = temp[..., :1]
    Sy = temp[..., 1:2]
    alpha = temp[..., 2:3]
    modified = compute_PP2D_slopes(
        np, u, tuple(dims), Sx=Sx, Sy=Sy, buffer=buffer, SED=SED, alpha=alpha
    )

    assert not np.any(np.isnan(Sx[modified]))
    assert not np.any(np.isnan(Sy[modified]))
    # skipping all nan check since the 2D stencils will leave some ghost cells non-nan

    if SED:
        assert not np.any(np.isnan(alpha[modified]))
        alpha[modified] = np.nan
        assert np.all(np.isnan(alpha))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
@pytest.mark.parametrize("SED", [False, True])
@pytest.mark.parametrize("include_corners", [False, True])
def test_compute_theta(dims: str, SED: bool, include_corners: bool):
    u, mega_buffer, out = sample_data(dims, nout=1)
    nodes, _, _ = sample_data(dims, nout=1)
    nodes = nodes[..., np.newaxis]

    dmp = mega_buffer[..., :2]
    alpha = mega_buffer[..., 2:3]
    buffer = mega_buffer[..., 3:]

    config = ZhangShuConfig(SED, False, 0, include_corners)

    modified = compute_theta(
        np,
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

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_map_cells_values_to_face_values(dims: str):
    face_dim = dims[0]
    axis = DIM_TO_AXIS[face_dim]
    u, _, out = sample_data(dims, nout=1, N=33)
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
    modified = map_cells_values_to_face_values(np, u, axis, out=out)

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("dims", ["x", "y", "z", "xy", "xz", "yz", "xyz"])
def test_smooth_extrema_detection(dims: str):
    u, buffer, out = sample_data(dims, nout=1)
    modified = smooth_extrema_detector(np, u, tuple(dims), out=out, buffer=buffer)

    assert not np.any(np.isnan(out[modified]))
    out[modified] = np.nan
    assert np.all(np.isnan(out))
