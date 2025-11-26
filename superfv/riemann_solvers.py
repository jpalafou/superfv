from types import ModuleType
from typing import Literal, Protocol, runtime_checkable

from .hydro import fluxes, prim_to_cons, sound_speed
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.slicing import VariableIndexMap
from .tools.stability import avoid0


@runtime_checkable
class AdvectionRiemannSolver(Protocol):
    def __call__(
        self,
        xp: ModuleType,
        idx: VariableIndexMap,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
    ) -> ArrayLike: ...


@runtime_checkable
class HydroRiemannSolver(Protocol):
    def __call__(
        self,
        xp: ModuleType,
        idx: VariableIndexMap,
        wl: ArrayLike,
        wr: ArrayLike,
        dim: Literal["x", "y", "z"],
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ) -> ArrayLike: ...


def advection_upwind(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
) -> ArrayLike:
    """
    Compute the upwind advection flux for a given dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension for the advection flux ("x", "y", or "z").

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    vl = wl[idx("v" + dim)]
    vr = wr[idx("v" + dim)]
    v = xp.where(xp.abs(vl) > xp.abs(vr), vl, vr)

    F = xp.empty_like(wl)

    F[idx("rho")] = v * xp.where(v > 0, wl[idx("rho")], wr[idx("rho")])
    F[idx("vx")] = 0.0
    F[idx("vy")] = 0.0
    F[idx("vz")] = 0.0
    if "passives" in idx.group_var_map:
        F[idx("passives")] = v * xp.where(
            v > 0, wl[idx("passives")], wr[idx("passives")]
        )

    return F


def llf(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
) -> ArrayLike:
    """
    Compute the LLF flux for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        gamma: Adiabatic index.
        isothermal: Whether to use isothermal sound speed.
        iso_cs: Isothermal sound speed (if isothermal is True).

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    ul = prim_to_cons(xp, idx, wl, gamma)
    ur = prim_to_cons(xp, idx, wr, gamma)

    csl = iso_cs if isothermal else sound_speed(xp, idx, wl, gamma)
    csr = iso_cs if isothermal else sound_speed(xp, idx, wr, gamma)
    sl = csl + xp.abs(wl[idx("v" + dim)])
    sr = csr + xp.abs(wr[idx("v" + dim)])
    smax = xp.maximum(sl, sr)

    Fl = fluxes(xp, idx, wl, dim, gamma)
    Fr = fluxes(xp, idx, wr, dim, gamma)

    F = 0.5 * (Fl + Fr - smax * (ur - ul))

    return F


def hllc(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
    isothermal: bool = False,
    iso_cs: float = 1.0,
) -> ArrayLike:
    """
    Compute the HLLC flux for the Euler equations in the specified dimension.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        idx: VariableIndexMap object with indices for hydro variables.
        wl: Array of the left state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        wr: Array of the right state as primitive variables. Has shape
            (nvars, nx, ny, nz, ...).
        dim: Dimension in which to compute the flux. Can be "x", "y", or "z".
        gamma: Adiabatic index.
        isothermal: Whether to use isothermal sound speed.
        iso_cs: Isothermal sound speed (if isothermal is True).

    Returns:
        F: Flux array. Has shape (nvars, nx, ny, nz, ...).
    """
    d1 = dim
    d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

    ul = prim_to_cons(xp, idx, wl, gamma)
    ur = prim_to_cons(xp, idx, wr, gamma)

    rhol = wl[idx("rho")]
    v1l = wl[idx("v" + d1)]
    v2l = wl[idx("v" + d2)]
    v3l = wl[idx("v" + d3)]
    Pl = wl[idx("P")]
    El = ul[idx("E")]
    rhor = wr[idx("rho")]
    v1r = wr[idx("v" + d1)]
    v2r = wr[idx("v" + d2)]
    v3r = wr[idx("v" + d3)]
    Pr = wr[idx("P")]
    Er = ur[idx("E")]

    cl = iso_cs if isothermal else sound_speed(xp, idx, wl, gamma)
    cr = iso_cs if isothermal else sound_speed(xp, idx, wr, gamma)
    cmax = xp.maximum(cl, cr)

    sl = xp.minimum(v1l, v1r) - cmax
    sr = xp.maximum(v1l, v1r) + cmax

    rcl = rhol * (v1l - sl)
    rcr = rhor * (sr - v1r)

    vP_star_denom = avoid0(xp, rcl + rcr)
    vstar = (rcr * v1r + rcl * v1l + (Pl - Pr)) / vP_star_denom
    Pstar = (rcr * Pl + rcl * Pr + rcl * rcr * (v1l - v1r)) / vP_star_denom

    rhoE_starl_denoml = avoid0(xp, sl - vstar)
    rhoE_starr_denomr = avoid0(xp, sr - vstar)
    rhostarl = rhol * (sl - v1l) / rhoE_starl_denoml
    rhostarr = rhor * (sr - v1r) / rhoE_starr_denomr
    Estarl = ((sl - v1l) * El - Pl * v1l + Pstar * vstar) / rhoE_starl_denoml
    Estarr = ((sr - v1r) * Er - Pr * v1r + Pstar * vstar) / rhoE_starr_denomr

    rhogdv = hllc_operator(xp, sl, sr, vstar, rhol, rhor, rhostarl, rhostarr)
    v1gdv = hllc_operator(xp, sl, sr, vstar, v1l, v1r, vstar, vstar)
    Pgdv = hllc_operator(xp, sl, sr, vstar, Pl, Pr, Pstar, Pstar)
    Egdv = hllc_operator(xp, sl, sr, vstar, El, Er, Estarl, Estarr)

    F = xp.empty_like(wl)
    F[idx("rho")] = rhogdv * v1gdv
    F[idx("m" + d1)] = F[idx("rho")] * v1gdv + Pgdv
    F[idx("E")] = v1gdv * (Egdv + Pgdv)
    F[idx("m" + d2)] = F[idx("rho")] * xp.where(vstar > 0, v2l, v2r)
    F[idx("m" + d3)] = F[idx("rho")] * xp.where(vstar > 0, v3l, v3r)
    if "passives" in idx.group_var_map:
        F[idx("passives")] = F[idx("rho")] * xp.where(
            vstar > 0, wl[idx("passives")], wr[idx("passives")]
        )

    return F


def hllc_operator(
    xp: ModuleType,
    sl: ArrayLike,
    sr: ArrayLike,
    vstar: ArrayLike,
    ql: ArrayLike,
    qr: ArrayLike,
    qstarl: ArrayLike,
    qstarr: ArrayLike,
) -> ArrayLike:
    """
    Helper function for HLLC flux calculation.

    Args:
        xp: ModuleType for the array operations (e.g., numpy or cupy).
        sl: Left wave speed.
        sr: Right wave speed.
        vstar: Star state velocity.
        ql: Left state variable.
        qr: Right state variable.
        qstarl: Left star state variable.
        qstarr: Right star state variable.

    Returns:
        Array with the HLLC operator applied.
    """
    return xp.where(
        sl > 0, ql, xp.where(vstar > 0, qstarl, xp.where(sr > 0, qstarr, qr))
    )


def hllct(
    xp: ModuleType,
    idx: VariableIndexMap,
    wl: ArrayLike,
    wr: ArrayLike,
    dim: Literal["x", "y", "z"],
    gamma: float,
) -> ArrayLike:
    """
    Compute Romain's HLLC flux for the Euler equations in the specified dimension.

    Args:
        xp: numpy/cupy-like module for array ops.
        idx: VariableIndexMap providing indices for 'rho', 'v*', 'P', 'E', etc.
        wl: Left primitive state [rho, v*, ..., P, passives?].
        wr: Right primitive state.
        dim: Flux direction ("x", "y", or "z").
        gamma: Adiabatic index.

    Returns:
        F: Flux array with same shape as wl (nvars, ...).
    """
    d1 = dim

    flux = 0.0 * wl
    uleft = prim_to_cons(xp, idx, wl, gamma)
    uright = prim_to_cons(xp, idx, wr, gamma)
    # left state
    dl = wl[idx("rho")]
    vl = wl[idx("v" + d1)]
    pl = wl[idx("P")]
    el = uleft[idx("E")]
    # right state
    dr = wr[idx("rho")]
    vr = wr[idx("v" + d1)]
    pr = wr[idx("P")]
    er = uright[idx("E")]
    # sound speed
    cl = xp.sqrt(gamma * pl / dl)
    cr = xp.sqrt(gamma * pr / dr)
    # waves speed
    sl = xp.minimum(vl, vr) - xp.maximum(cl, cr)
    sr = xp.maximum(vl, vr) + xp.maximum(cl, cr)
    dcl = dl * (vl - sl)
    dcr = dr * (sr - vr)
    # star state velocity and pressure
    vstar = (dcl * vl + dcr * vr + pl - pr) / (dcl + dcr)
    pstar = (dcl * pr + dcr * pl + dcl * dcr * (vl - vr)) / (dcl + dcr)
    # left and right star states
    dstarl = dl * (sl - vl) / (sl - vstar)
    dstarr = dr * (sr - vr) / (sr - vstar)
    estarl = ((sl - vl) * el - pl * vl + pstar * vstar) / (sl - vstar)
    estarr = ((sr - vr) * er - pr * vr + pstar * vstar) / (sr - vstar)
    # sample godunov state
    dg = xp.where(sl > 0, dl, xp.where(vstar > 0, dstarl, xp.where(sr > 0, dstarr, dr)))
    vg = xp.where(sl > 0, vl, xp.where(vstar > 0, vstar, xp.where(sr > 0, vstar, vr)))
    pg = xp.where(sl > 0, pl, xp.where(vstar > 0, pstar, xp.where(sr > 0, pstar, pr)))
    eg = xp.where(sl > 0, el, xp.where(vstar > 0, estarl, xp.where(sr > 0, estarr, er)))
    # compute godunov flux
    flux[idx("rho")] = dg * vg
    flux[idx("m" + d1)] = dg * vg * vg + pg
    flux[idx("E")] = (eg + pg) * vg
    return flux


if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore

    def make_hllc_elementwise_kernel(npassives: int):
        in_params = (
            "float64 rhol, float64 rhor, float64 v1l, float64 v1r, "
            "float64 v2l, float64 v2r, float64 v3l, float64 v3r, "
            "float64 Pl, float64 Pr, float64 cl, float64 cr, float64 gamma, int8 dim"
        )
        out_params = "float64 Frho, float64 Fm1, float64 Fm2, float64 Fm3, float64 FE"
        body = """
        double vl;
        double vr;
        if (dim == 1) {
            vl = v1l;
            vr = v1r;
        } else if (dim == 2) {
            vl = v2l;
            vr = v2r;
        } else {
            vl = v3l;
            vr = v3r;
        }

        double Kl = 0.5 * rhol * (v1l * v1l + v2l * v2l + v3l * v3l);
        double Kr = 0.5 * rhor * (v1r * v1r + v2r * v2r + v3r * v3r);

        double El = Pl / (gamma - 1.0) + Kl;
        double Er = Pr / (gamma - 1.0) + Kr;

        double cmax = fmax(cl, cr);

        double sl = fmin(vl, vr) - cmax;
        double sr = fmax(vl, vr) + cmax;

        double rcl = rhol * (vl - sl);
        double rcr = rhor * (sr - vr);
        double vP_star_denom = rcl + rcr;
        double vP_star_denom_safe = (
            fabs(vP_star_denom) > 1e-15
                ? vP_star_denom
                : (vP_star_denom > 0 ? 1e-15 : -1e-15)
        );
        double vstar = (rcr * vr + rcl * vl + (Pl - Pr)) / vP_star_denom_safe;
        double Pstar = (rcr * Pl + rcl * Pr + rcl * rcr * (vl - vr)) / vP_star_denom_safe;

        double rhoE_star_denoml = sl - vstar;
        double rhoE_star_denomr = sr - vstar;
        double rhoE_star_denoml_safe = (
            fabs(rhoE_star_denoml) > 1e-15
                ? rhoE_star_denoml
                : (rhoE_star_denoml > 0 ? 1e-15 : -1e-15)
        );
        double rhoE_star_denomr_safe = (
            fabs(rhoE_star_denomr) > 1e-15
                ? rhoE_star_denomr
                : (rhoE_star_denomr > 0 ? 1e-15 : -1e-15)
        );
        double rhostarl = rhol * (sl - vl) / rhoE_star_denoml_safe;
        double rhostarr = rhor * (sr - vr) / rhoE_star_denomr_safe;
        double Estarl = ((sl - vl) * El - Pl * vl + Pstar * vstar) / rhoE_star_denoml_safe;
        double Estarr = ((sr - vr) * Er - Pr * vr + Pstar * vstar) / rhoE_star_denomr_safe;

        double rhogdv;
        double vgdv;
        double Pgdv;
        double Egdv;

        if (sl > 0) {
            rhogdv = rhol;
            vgdv = vl;
            Pgdv = Pl;
            Egdv = El;
        } else if (vstar > 0) {
            rhogdv = rhostarl;
            vgdv = vstar;
            Pgdv = Pstar;
            Egdv = Estarl;
        } else if (sr > 0) {
            rhogdv = rhostarr;
            vgdv = vstar;
            Pgdv = Pstar;
            Egdv = Estarr;
        } else {
            rhogdv = rhor;
            vgdv = vr;
            Pgdv = Pr;
            Egdv = Er;
        }

        Frho = rhogdv * vgdv;
        if (dim == 1) {
            Fm1 = Frho * vgdv + Pgdv;
            Fm2 = Frho * (vstar > 0 ? v2l : v2r);
            Fm3 = Frho * (vstar > 0 ? v3l : v3r);
        } else if (dim == 2) {
            Fm1 = Frho * (vstar > 0 ? v1l : v1r);
            Fm2 = Frho * vgdv + Pgdv;
            Fm3 = Frho * (vstar > 0 ? v3l : v3r);
        } else {
            Fm1 = Frho * (vstar > 0 ? v1l : v1r);
            Fm2 = Frho * (vstar > 0 ? v2l : v2r);
            Fm3 = Frho * vgdv + Pgdv;
        }
        FE = vgdv * (Egdv + Pgdv);
        """

        for i in range(npassives):
            in_params += f", float64 passl{i}, float64 passr{i}"
            out_params += f", float64 Fpass{i}"
            body += f"\nFpass{i} = Frho * (vstar > 0 ? passl{i} : passr{i});"

        return cp.ElementwiseKernel(
            in_params=in_params,
            out_params=out_params,
            operation=body,
            name=f"hllc_npass_{npassives}",
            no_return=True,
        )
