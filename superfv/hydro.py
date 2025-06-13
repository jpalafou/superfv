from typing import Any

import numpy as np

from superfv.tools.array_management import ArrayLike, VariableIndexMap


def conservatives_from_primitives(
    hydro: Any, idx: VariableIndexMap, w: ArrayLike, gamma: float
):
    """
    Compute the conservative variables from the primitive variables.

    Args:
        hydro: Hydro namespace.
        idx: VariableIndexMap object with indices for hydro variables.
        w: Array of primitive variables. Has shape (nvars, nx, ny, nz, ...).
        gamma (float): Adiabatic index.

    Returns:
        u: Array with conservative variables. Has shape (nvars, nx, ny, nz, ...).
    """
    u = np.empty_like(w)
    HAS_PASSIVES = "user_defined_passives" in idx.group_var_map
    (
        u[idx("rho")],
        u[idx("mx")],
        u[idx("my")],
        u[idx("mz")],
        u[idx("E")],
        passives,
    ) = hydro.conservatives_from_primitives(
        rho=w[idx("rho")],
        v1=w[idx("vx")],
        v2=w[idx("vy")],
        v3=w[idx("vz")],
        P=w[idx("P")],
        gamma=gamma,
        passives=w[idx("user_defined_passives")] if HAS_PASSIVES else None,
    )
    if HAS_PASSIVES:
        u[idx("user_defined_passives")] = passives
    return u


def primitives_from_conservatives(
    hydro: Any, idx: VariableIndexMap, u: ArrayLike, gamma: float
):
    """
    Compute the primitive variables from the conservative variables.

    Args:
        hydro: Hydro namespace.
        idx: VariableIndexMap object with indices for hydro variables.
        u: Array of conservative variables. Has shape (nvars, nx, ny, nz, ...).
        gamma: Adiabatic index.

    Returns:
        w (ArrayLike): Array with primitive variables. Has shape (nvars, nx, ny, nz, ...).
    """
    w = np.empty_like(u)
    HAS_PASSIVES = "user_defined_passives" in idx.group_var_map
    (
        w[idx("rho")],
        w[idx("mx")],
        w[idx("my")],
        w[idx("mz")],
        w[idx("E")],
        passives,
    ) = hydro.primitives_from_conservatives(
        rho=u[idx("rho")],
        m1=u[idx("vx")],
        m2=u[idx("vy")],
        m3=u[idx("vz")],
        E=u[idx("P")],
        gamma=gamma,
        conserved_passives=u[idx("user_defined_passives")] if HAS_PASSIVES else None,
    )
    if HAS_PASSIVES:
        w[idx("user_defined_passives")] = passives
    return w
