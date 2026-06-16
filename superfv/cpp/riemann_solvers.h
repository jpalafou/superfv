#pragma once

#include <algorithm>
#include <cmath>

#include "hydro.h"

void hllc_flux(
    const Primitives& left,
    const Primitives& right,
    Conservatives& flux,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    double csl;
    double csr;
    prim_to_cs(left, csl, gamma, isothermal, iso_cs);
    prim_to_cs(right, csr, gamma, isothermal, iso_cs);

    double rhol = left.rho;
    double v1l = left.vx;
    double v2l = left.vy;
    double v3l = left.vz;
    double Pl = left.P;

    double rhor = right.rho;
    double v1r = right.vx;
    double v2r = right.vy;
    double v3r = right.vz;
    double Pr = right.P;

    double Kl = 0.5 * rhol * (v1l * v1l + v2l * v2l + v3l * v3l);
    double Kr = 0.5 * rhor * (v1r * v1r + v2r * v2r + v3r * v3r);

    double El = Pl / (gamma - 1.0) + Kl;
    double Er = Pr / (gamma - 1.0) + Kr;

    double cmax = std::max(csl, csr);

    double sl = std::min(v1l, v1r) - cmax;
    double sr = std::max(v1l, v1r) + cmax;

    double rcl = rhol * (v1l - sl);
    double rcr = rhor * (sr - v1r);
    double vP_star_denom = rcl + rcr;
    double vP_star_denom_safe = (
        std::fabs(vP_star_denom) > 1e-15
            ? vP_star_denom
            : (vP_star_denom >= 0 ? 1e-15 : -1e-15)
    );
    double vstar = (rcr * v1r + rcl * v1l + (Pl - Pr)) / vP_star_denom_safe;
    double Pstar = (rcr * Pl + rcl * Pr + rcl * rcr * (v1l - v1r)) / vP_star_denom_safe;

    double rhoE_star_denoml = sl - vstar;
    double rhoE_star_denomr = sr - vstar;
    double rhoE_star_denoml_safe = (
        std::fabs(rhoE_star_denoml) > 1e-15
            ? rhoE_star_denoml
            : (rhoE_star_denoml >= 0 ? 1e-15 : -1e-15)
    );
    double rhoE_star_denomr_safe = (
        std::fabs(rhoE_star_denomr) > 1e-15
            ? rhoE_star_denomr
            : (rhoE_star_denomr >= 0 ? 1e-15 : -1e-15)
    );
    double rhostarl = rhol * (sl - v1l) / rhoE_star_denoml_safe;
    double rhostarr = rhor * (sr - v1r) / rhoE_star_denomr_safe;
    double Estarl = ((sl - v1l) * El - Pl * v1l + Pstar * vstar) / rhoE_star_denoml_safe;
    double Estarr = ((sr - v1r) * Er - Pr * v1r + Pstar * vstar) / rhoE_star_denomr_safe;

    double rhogdv;
    double vgdv;
    double Pgdv;
    double Egdv;

    if (sl > 0) {
        rhogdv = rhol;
        vgdv = v1l;
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
        vgdv = v1r;
        Pgdv = Pr;
        Egdv = Er;
    }

    flux.rho = rhogdv * vgdv;
    flux.mx = flux.rho * vgdv + Pgdv;
    flux.my = flux.rho * (vstar > 0 ? v2l : v2r);
    flux.mz = flux.rho * (vstar > 0 ? v3l : v3r);
    flux.E = vgdv * (Egdv + Pgdv);

    for (int i = 0; i < flux.npassive; ++i) {
        flux.passive[i] = flux.rho * (vstar > 0 ? left.passive[i] : right.passive[i]);
    }
}
