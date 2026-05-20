from functools import lru_cache
from typing import Literal

import numpy as np

from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


def _prim_to_cs_np(
    w: np.ndarray,
    cs: np.ndarray,
    idx: VariableIndexMap,
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
):
    rho = w[idx("rho")]
    P = w[idx("P")]
    if isothermal:
        cs[...] = iso_cs
    else:
        np.sqrt(np.maximum(gamma * P / rho, 0.0), out=cs)


@lru_cache(maxsize=None)
def _make_prim_to_cs_kernel():
    in_params = "float64 rho, float64 P, float64 gamma, bool isothermal, float64 iso_cs"
    out_params = "float64 cs"

    body = """
    if (isothermal) {
        cs = iso_cs;
    } else {
        cs = sqrt(fmax(gamma * P / rho, 0.0));
    }
    """

    return cp.ElementwiseKernel(
        in_params=in_params,
        out_params=out_params,
        operation=body,
        name="prim_to_cs",
        no_return=True,
    )


def prim_to_cs(
    w: ArrayLike,
    cs: ArrayLike,
    idx: VariableIndexMap,
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
):
    """
    Compute the speed of sound from primitive variables and write it to `cs`
    with support for both NumPy and CuPy arrays.
    """
    if cs.shape != w.shape[1:]:
        raise ValueError("Shape of cs must match the shape of a single variable in w")
    if CUPY_AVAILABLE and isinstance(w, cp.ndarray):
        kernel = _make_prim_to_cs_kernel()
        kernel(
            w[idx("rho")],
            w[idx("P")],
            gamma,
            isothermal,
            iso_cs,
            cs,
        )
    else:
        _prim_to_cs_np(w, cs, idx, gamma, isothermal, iso_cs)


def _prim_to_cons_np(w: np.ndarray, u: np.ndarray, idx: VariableIndexMap, gamma: float):
    rho = w[idx("rho")]
    vx = w[idx("vx")]
    vy = w[idx("vy")]
    vz = w[idx("vz")]
    P = w[idx("P")]

    KE = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    u[idx("rho")] = rho
    u[idx("mx")] = rho * vx
    u[idx("my")] = rho * vy
    u[idx("mz")] = rho * vz
    u[idx("E")] = KE + P / (gamma - 1)

    if "passives" in idx:
        u[idx("passives")] = rho * w[idx("passives")]


@lru_cache(maxsize=None)
def _make_prim_to_cons_elementwise_kernel(npassives: int):
    in_params = "float64 rho, float64 vx, float64 vy, float64 vz, float64 P, float64 gamma"
    out_params = "float64 rho_out, float64 mx, float64 my, float64 mz, float64 E"

    body = """
    rho_out = rho;
    mx = rho * vx;
    my = rho * vy;
    mz = rho * vz;
    double K = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    E = K + P / (gamma - 1.0);
    """

    for i in range(npassives):
        in_params += f", float64 pass{i}"
        out_params += f", float64 upass{i}"
        body += f"\nupass{i} = rho * pass{i};"

    return cp.ElementwiseKernel(
        in_params=in_params,
        out_params=out_params,
        operation=body,
        name=f"prim_to_cons_npass_{npassives}",
        no_return=True,
    )


def prim_to_cons(
    w: ArrayLike,
    u: ArrayLike,
    idx: VariableIndexMap,
    gamma: float,
):
    """
    Compute the conservative variables from primitive variables and write them to `u`
    with support for both NumPy and CuPy arrays.
    """
    if CUPY_AVAILABLE and isinstance(w, cp.ndarray):
        passives = idx.group_var_map.get("passives", [])
        kernel = _make_prim_to_cons_elementwise_kernel(len(passives))
        kernel(
            w[idx("rho")],
            w[idx("vx")],
            w[idx("vy")],
            w[idx("vz")],
            w[idx("P")],
            gamma,
            *(w[idx(v)] for v in passives),
            u[idx("rho")],
            u[idx("mx")],
            u[idx("my")],
            u[idx("mz")],
            u[idx("E")],
            *(u[idx(v)] for v in passives),
        )
    else:
        _prim_to_cons_np(w, u, idx, gamma)


def _cons_to_prim_np(
    u: np.ndarray,
    w: np.ndarray,
    idx: VariableIndexMap,
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
):
    rho = u[idx("rho")]
    mx = u[idx("mx")]
    my = u[idx("my")]
    mz = u[idx("mz")]
    E = u[idx("E")]

    vx = mx / rho
    vy = my / rho
    vz = mz / rho
    KE = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    w[idx("rho")] = rho
    w[idx("vx")] = vx
    w[idx("vy")] = vy
    w[idx("vz")] = vz
    w[idx("P")] = rho * iso_cs**2 if isothermal else (gamma - 1) * (E - KE)

    if "passives" in idx:
        w[idx("passives")] = u[idx("passives")] / rho

    return w


@lru_cache(maxsize=None)
def _make_cons_to_prim_elementwise_kernel(npassives: int):
    in_params = (
        "float64 rho, float64 mx, float64 my, float64 mz, float64 E, "
        "float64 gamma, bool isothermal, float64 iso_cs"
    )
    out_params = "float64 rho_out, float64 vx, float64 vy, float64 vz, float64 P"

    body = """
    rho_out = rho;
    vx = mx / rho;
    vy = my / rho;
    vz = mz / rho;
    double K = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    P = isothermal ? rho * iso_cs * iso_cs : (gamma - 1.0) * (E - K);
    """

    for i in range(npassives):
        in_params += f", float64 upass{i}"
        out_params += f", float64 pass{i}"
        body += f"\npass{i} = upass{i} / rho;"

    return cp.ElementwiseKernel(
        in_params=in_params,
        out_params=out_params,
        operation=body,
        name=f"cons_to_prim_npass_{npassives}",
        no_return=True,
    )


def cons_to_prim(
    u: ArrayLike,
    w: ArrayLike,
    idx: VariableIndexMap,
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
):
    """
    Compute the primitive variables from conservative variables and write them to `w`
    with support for both NumPy and CuPy arrays.
    """
    if CUPY_AVAILABLE and isinstance(u, cp.ndarray):
        passives = idx.group_var_map.get("passives", [])
        kernel = _make_cons_to_prim_elementwise_kernel(len(passives))
        kernel(
            u[idx("rho")],
            u[idx("mx")],
            u[idx("my")],
            u[idx("mz")],
            u[idx("E")],
            gamma,
            isothermal,
            iso_cs,
            w[idx("rho")],
            w[idx("vx")],
            w[idx("vy")],
            w[idx("vz")],
            w[idx("P")],
            *(w[idx(v)] for v in passives),
        )
        return w
    else:
        return _cons_to_prim_np(u, w, idx, gamma, isothermal, iso_cs)


def prim_to_flux(
    w: np.ndarray,
    f: np.ndarray,
    idx: VariableIndexMap,
    dim: Literal["x", "y", "z"],
    gamma: float,
):
    """
    Compute the fluxes from primitive variables and write them to `f`.
    """
    d1 = dim
    d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    rho = w[idx("rho")]
    v1 = w[idx("v" + d1)]
    v2 = w[idx("v" + d2)]
    v3 = w[idx("v" + d3)]
    P = w[idx("P")]

    KE = 0.5 * rho * (v1**2 + v2**2 + v3**2)

    f[idx("rho")] = rho * v1
    f[idx("m" + d1)] = rho * v1**2 + P
    f[idx("m" + d2)] = rho * v1 * v2
    f[idx("m" + d3)] = rho * v1 * v3
    f[idx("E")] = (KE + P / (gamma - 1) + P) * v1

    if "passives" in idx:
        f[idx("passives")] = rho * v1 * w[idx("passives")]
