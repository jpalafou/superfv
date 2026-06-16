#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

struct Primitives {
    double rho;
    double vx;
    double vy;
    double vz;
    double P;
    double* passive;
    int npassive;
};

struct Conservatives {
    double rho;
    double mx;
    double my;
    double mz;
    double E;
    double* passive;
    int npassive;
};

void cons_to_prim(
    const Conservatives& cons,
    Primitives& prim,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    prim.rho = cons.rho;
    prim.vx = cons.mx / cons.rho;
    prim.vy = cons.my / cons.rho;
    prim.vz = cons.mz / cons.rho;
    if (isothermal) {
        prim.P = iso_cs * iso_cs * cons.rho;
    } else {
        double kinetic_energy = 0.5 * (prim.vx * prim.vx + prim.vy * prim.vy + prim.vz * prim.vz);
        double internal_energy = cons.E / cons.rho - kinetic_energy;
        prim.P = (gamma - 1.0) * internal_energy * cons.rho;
    }
    for (int i = 0; i < cons.npassive; ++i) {
        prim.passive[i] = cons.passive[i] / cons.rho;
    }
}

void prim_to_cons(
    const Primitives& prim,
    Conservatives& cons,
    const double gamma
) {
    cons.rho = prim.rho;
    cons.mx = prim.vx * prim.rho;
    cons.my = prim.vy * prim.rho;
    cons.mz = prim.vz * prim.rho;

    double internal_energy = prim.P / ((gamma - 1.0) * prim.rho);
    cons.E = internal_energy * prim.rho + 0.5 * (prim.vx * prim.vx + prim.vy * prim.vy + prim.vz * prim.vz) * prim.rho;

    for (int i = 0; i < cons.npassive; ++i) {
        cons.passive[i] = prim.passive[i] * cons.rho;
    }
}

void prim_to_cs(
    const Primitives& prim,
    double& cs,
    const double gamma,
    const bool isothermal,
    const double iso_cs
) {
    if (isothermal) {
        cs = iso_cs;
    } else {
        cs = std::sqrt(std::max(gamma * prim.P / prim.rho, 0.0));
    }
}
