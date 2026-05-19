from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from typing import Literal

import numpy as np

from .hydro import compute_fluxes, prim_to_cons, sound_speed
from .tools.device_management import CUPY_AVAILABLE, ArrayLike
from .tools.stability import avoid0
from .tools.variable_index_map import VariableIndexMap

if CUPY_AVAILABLE:
    import cupy as cp  # type: ignore


class RiemannSolver(Enum):
    UPWIND = 0
    LLF = 1
    HLLC = 2
    HLLC_TEYSSIER = 3


class RiemmannSolverBase(ABC):
    def __init__(self, npassives: int):
        self.npassives = npassives

    @abstractmethod
    def numpy_func(
        self,
        wl: np.ndarray,
        wr: np.ndarray,
        fluxes: np.ndarray,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        pass

    @abstractmethod
    def cuda_elementwise_kernel_body(self, npassives: int) -> str:
        pass

    def cuda_elementwise_kernel_in_params(self, npassives: int) -> str:
        in_params = (
            "float64 rhol, float64 rhor, float64 v1l, float64 v1r, float64 v2l, float64 v2r, "
            "float64 v3l, float64 v3r, float64 Pl, float64 Pr, "
            "float64 gamma, bool isothermal, float64 iso_cs, int32 dim"
        )
        for i in range(npassives):
            in_params += f", float64 passl{i}, float64 passr{i}"
        return in_params

    def cuda_elementwise_kernel_out_params(self, npassives: int) -> str:
        out_params = "float64 Frho, float64 Fm1, float64 Fm2, float64 Fm3, float64 FE"
        for i in range(npassives):
            out_params += f", float64 Fpass{i}"
        return out_params

    @cached_property
    def cuda_kernel(self):
        return cp.ElementwiseKernel(
            in_params=self.cuda_elementwise_kernel_in_params(self.npassives),
            out_params=self.cuda_elementwise_kernel_out_params(self.npassives),
            operation=self.cuda_elementwise_kernel_body(self.npassives),
            name=f"{self.__class__.__name__}_npass_{self.npassives}",
            no_return=True,
        )

    def __call__(
        self,
        wl: ArrayLike,
        wr: ArrayLike,
        fluxes: ArrayLike,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        fluxes[...] = 0.0  # Initialize zero fluxes

        if CUPY_AVAILABLE and isinstance(wl, cp.ndarray):
            self.cuda_kernel(
                wl[idx("rho")],
                wr[idx("rho")],
                wl[idx("vx")],
                wr[idx("vx")],
                wl[idx("vy")],
                wr[idx("vy")],
                wl[idx("vz")],
                wr[idx("vz")],
                wl[idx("P")],
                wr[idx("P")],
                gamma,
                isothermal,
                iso_cs,
                {"x": 1, "y": 2, "z": 3}[dim],
                *[
                    x
                    for v in idx.group_var_map.get("passives", [])
                    for x in (wl[idx(v)], wr[idx(v)])
                ],
                fluxes[idx("rho")],
                fluxes[idx("mx")],
                fluxes[idx("my")],
                fluxes[idx("mz")],
                fluxes[idx("E")],
                *[fluxes[idx(v)] for v in idx.group_var_map.get("passives", [])],
            )
        else:
            self.numpy_func(wl, wr, fluxes, dim, idx, gamma, isothermal, iso_cs)


class UpwindRiemannSolver(RiemmannSolverBase):
    def numpy_func(
        self,
        wl: np.ndarray,
        wr: np.ndarray,
        fluxes: np.ndarray,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        vl = wl[idx("v" + dim)]
        vr = wr[idx("v" + dim)]
        v = np.where(np.abs(vl) > np.abs(vr), vl, vr)

        fluxes[idx("rho")] = v * np.where(v > 0, wl[idx("rho")], wr[idx("rho")])
        if "passives" in idx.group_var_map:
            fluxes[idx("passives")] = fluxes[idx("rho")] * np.where(
                v > 0, wl[idx("passives")], wr[idx("passives")]
            )

    def cuda_elementwise_kernel_body(self, npassives: int) -> str:
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

        double v = fabs(vl) > fabs(vr) ? vl : vr;

        Frho = v * (v > 0 ? rhol : rhor);
        """
        for i in range(npassives):
            body += f"\nFpass{i} = Frho * (v > 0 ? passl{i} : passr{i});"
        return body


class LLF_RiemannSolver(RiemmannSolverBase):
    def numpy_func(
        self,
        wl: np.ndarray,
        wr: np.ndarray,
        fluxes: np.ndarray,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        ul = prim_to_cons(idx, wl, gamma)
        ur = prim_to_cons(idx, wr, gamma)

        csl = iso_cs if isothermal else np.sqrt(gamma * wl[idx("P")] / wl[idx("rho")])
        csr = iso_cs if isothermal else np.sqrt(gamma * wr[idx("P")] / wr[idx("rho")])
        sl = csl + np.abs(wl[idx("v" + dim)])
        sr = csr + np.abs(wr[idx("v" + dim)])
        smax = np.maximum(sl, sr)

        Fl = compute_fluxes(idx, wl, dim, gamma)
        Fr = compute_fluxes(idx, wr, dim, gamma)

        fluxes[...] = 0.5 * (Fl + Fr - smax * (ur - ul))

    def cuda_elementwise_kernel_body(self, npassives: int) -> str:
        body = """
        // Compute maximum wave speed
        double vl, vr;
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
        double csl = isothermal ? iso_cs : sqrt(gamma * Pl / rhol);
        double csr = isothermal ? iso_cs : sqrt(gamma * Pr / rhor);
        double sl = csl + fabs(vl);
        double sr = csr + fabs(vr);
        double smax = fmax(sl, sr);

        // Compute left and right conservative variables
        double m1l = rhol * v1l;
        double m1r = rhor * v1r;
        double m2l = rhol * v2l;
        double m2r = rhor * v2r;
        double m3l = rhol * v3l;
        double m3r = rhor * v3r;
        double KEl = 0.5 * rhol * (v1l * v1l + v2l * v2l + v3l * v3l);
        double KEr = 0.5 * rhor * (v1r * v1r + v2r * v2r + v3r * v3r);
        double El = Pl / (gamma - 1.0) + KEl;
        double Er = Pr / (gamma - 1.0) + KEr;

        // Compute left and right fluxes
        double Flrho = rhol * vl;
        double Frrho = rhor * vr;
        Fm1l = Flrho * v1l;
        Fm1r = Frrho * v1r;
        Fm2l = Flrho * v2l;
        Fm2r = Frrho * v2r;
        Fm3l = Flrho * v3l;
        Fm3r = Frrho * v3r;
        if (dim == 1) {
            Fm1l += Pl;
            Fm1r += Pr;
        } else if (dim == 2) {
            Fm2l += Pl;
            Fm2r += Pr;
        } else {
            Fm3l += Pl;
            Fm3r += Pr;
        }
        FEl = vl * (El + Pl);
        FEr = vr * (Er + Pr);

        // Compute LLF flux
        Frho = 0.5 * (Flrho + Frrho - smax * (rhor - rhol));
        Fm1 = 0.5 * (Fm1l + Fm1r - smax * (m1r - m1l));
        Fm2 = 0.5 * (Fm2l + Fm2r - smax * (m2r - m2l));
        Fm3 = 0.5 * (Fm3l + Fm3r - smax * (m3r - m3l));
        FE = 0.5 * (FEl + FEr - smax * (Er - El));
        """
        for i in range(npassives):
            body += f"""
            double conspassl{i} = rhol * passl{i};
            double conspassr{i} = rhor * passr{i};
            double Fpassl = conspassl{i} * vl;
            double Fpassr = conspassr{i} * vr;
            Fpass{i} = 0.5 * (Fpassl + Fpassr - smax * (conspassr{i} - conspassl{i}));
            """
        return body


def hllc_operator(
    sl: np.ndarray,
    sr: np.ndarray,
    vstar: np.ndarray,
    ql: np.ndarray,
    qr: np.ndarray,
    qstarl: np.ndarray,
    qstarr: np.ndarray,
) -> np.ndarray:
    """
    Helper function for HLLC flux calculation.

    Args:
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
    return np.where(sl > 0, ql, np.where(vstar > 0, qstarl, np.where(sr > 0, qstarr, qr)))


class HLLC_RiemannSolver(RiemmannSolverBase):
    def numpy_func(
        self,
        wl: np.ndarray,
        wr: np.ndarray,
        fluxes: np.ndarray,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        d1 = dim
        d2, d3 = {"x": ("y", "z"), "y": ("x", "z"), "z": ("x", "y")}[dim]

        ul = prim_to_cons(idx, wl, gamma)
        ur = prim_to_cons(idx, wr, gamma)

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

        cl = iso_cs if isothermal else sound_speed(idx, wl, gamma)
        cr = iso_cs if isothermal else sound_speed(idx, wr, gamma)
        cmax = np.maximum(cl, cr)

        sl = np.minimum(v1l, v1r) - cmax
        sr = np.maximum(v1l, v1r) + cmax

        rcl = rhol * (v1l - sl)
        rcr = rhor * (sr - v1r)

        vP_star_denom = avoid0(rcl + rcr)
        vstar = (rcr * v1r + rcl * v1l + (Pl - Pr)) / vP_star_denom
        Pstar = (rcr * Pl + rcl * Pr + rcl * rcr * (v1l - v1r)) / vP_star_denom

        rhoE_starl_denoml = avoid0(sl - vstar)
        rhoE_starr_denomr = avoid0(sr - vstar)
        rhostarl = rhol * (sl - v1l) / rhoE_starl_denoml
        rhostarr = rhor * (sr - v1r) / rhoE_starr_denomr
        Estarl = ((sl - v1l) * El - Pl * v1l + Pstar * vstar) / rhoE_starl_denoml
        Estarr = ((sr - v1r) * Er - Pr * v1r + Pstar * vstar) / rhoE_starr_denomr

        rhogdv = hllc_operator(sl, sr, vstar, rhol, rhor, rhostarl, rhostarr)
        v1gdv = hllc_operator(sl, sr, vstar, v1l, v1r, vstar, vstar)
        Pgdv = hllc_operator(sl, sr, vstar, Pl, Pr, Pstar, Pstar)
        Egdv = hllc_operator(sl, sr, vstar, El, Er, Estarl, Estarr)

        fluxes[idx("rho")] = rhogdv * v1gdv
        fluxes[idx("m" + d1)] = fluxes[idx("rho")] * v1gdv + Pgdv
        fluxes[idx("E")] = v1gdv * (Egdv + Pgdv)
        fluxes[idx("m" + d2)] = fluxes[idx("rho")] * np.where(vstar > 0, v2l, v2r)
        fluxes[idx("m" + d3)] = fluxes[idx("rho")] * np.where(vstar > 0, v3l, v3r)
        if "passives" in idx.group_var_map:
            fluxes[idx("passives")] = fluxes[idx("rho")] * np.where(
                vstar > 0, wl[idx("passives")], wr[idx("passives")]
            )

    def cuda_elementwise_kernel_body(self, npassives: int):
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

        double cl = isothermal ? iso_cs : sqrt(gamma * Pl / rhol);
        double cr = isothermal ? iso_cs : sqrt(gamma * Pr / rhor);
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
            body += f"\nFpass{i} = Frho * (vstar > 0 ? passl{i} : passr{i});"
        return body


class HLLC_Teyssier_RiemannSolver(RiemmannSolverBase):
    def numpy_func(
        self,
        wl: np.ndarray,
        wr: np.ndarray,
        fluxes: np.ndarray,
        dim: Literal["x", "y", "z"],
        idx: VariableIndexMap,
        gamma: float,
        isothermal: bool = False,
        iso_cs: float = 1.0,
    ):
        if sum(wl.shape[i] > 1 for i in [1, 2, 3]) > 1:
            raise NotImplementedError("No support for 1D problems.")
        if "passives" in idx.group_var_map:
            raise NotImplementedError("No support for passive scalars.")
        uleft = prim_to_cons(idx, wl, gamma)
        uright = prim_to_cons(idx, wr, gamma)
        # left state
        dl = wl[idx("rho")]
        vl = wl[idx("v" + dim)]
        pl = wl[idx("P")]
        el = uleft[idx("E")]
        # right state
        dr = wr[idx("rho")]
        vr = wr[idx("v" + dim)]
        pr = wr[idx("P")]
        er = uright[idx("E")]
        # sound speed
        cl = iso_cs if isothermal else np.sqrt(gamma * pl / dl)
        cr = iso_cs if isothermal else np.sqrt(gamma * pr / dr)
        # waves speed
        sl = np.minimum(vl, vr) - np.maximum(cl, cr)
        sr = np.maximum(vl, vr) + np.maximum(cl, cr)
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
        dg = np.where(sl > 0, dl, np.where(vstar > 0, dstarl, np.where(sr > 0, dstarr, dr)))
        vg = np.where(sl > 0, vl, np.where(vstar > 0, vstar, np.where(sr > 0, vstar, vr)))
        pg = np.where(sl > 0, pl, np.where(vstar > 0, pstar, np.where(sr > 0, pstar, pr)))
        eg = np.where(sl > 0, el, np.where(vstar > 0, estarl, np.where(sr > 0, estarr, er)))
        # compute godunov flux
        fluxes[idx("rho")] = dg * vg
        fluxes[idx("m" + dim)] = dg * vg * vg + pg
        fluxes[idx("E")] = (eg + pg) * vg

    def cuda_elementwise_kernel_body(self, npassives: int) -> str:
        raise NotImplementedError("HLLC Teyssier Riemann solver is not implemented yet.")
