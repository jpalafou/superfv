from functools import partial
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from spd.finite_volume.muscl import Slope_limiter as spd_muscl_limiter
from spd.sdfb_simulator import SDFB_Simulator

from superfv import HydroSolver, ics

N = 192
T = 1.95

output = Path("sd-compare/rayleigh-taylor-instability/results/muscl_hancock.png")
output.unlink(missing_ok=True)  # remove the output file if it exists


def run_fv():
    def gravity(idx, u, *, xp):
        gx = 0.0
        gy = 1.0
        out = xp.zeros_like(u)
        out[idx("mx")] = u[idx("rho")] * gx
        out[idx("my")] = u[idx("rho")] * gy
        out[idx("E")] = u[idx("mx")] * gx + u[idx("my")] * gy
        return out

    sim = HydroSolver(
        ic=partial(ics.rayleigh_taylor, gamma=5 / 3, P0=1.0),
        gamma=5 / 3,
        source=gravity,
        xlims=(0, 0.25),
        ylims=(0, 1),
        nx=N // 4,
        ny=N,
        bcy=("reflective", "reflective"),
        p=1,
        nu=1e-3,
        CFL=0.8,
        use_MUSCL=True,
        use_SED=False,
        MUSCL_limiter="moncen",
        cupy=True,
    )
    sim.run(T, "muscl_hancock")

    return sim


def run_sd():
    def rayleigh_taylor(xy: np.ndarray, case: int, P0, gamma):
        x = xy[0]
        y = xy[1]

        yc = 0.5
        rho1 = 2.0
        rho2 = 1.0

        if case == 0:
            # density
            return np.where(y > yc, rho2, rho1)
        elif case == 1:
            # vx
            return np.zeros(x.shape)
        elif case == 2:
            # vy
            dv = np.sqrt(gamma * (P0 + rho1 * yc - P0 + 1) / rho1)
            return -0.025 * dv * np.cos(8 * np.pi * x)
        elif case == 4:
            # Pressure
            return np.where(y > yc, P0 + rho2 * y + (rho1 - rho2) * yc, P0 + rho1 * y)
        else:
            return np.zeros(x.shape)

    sim = SDFB_Simulator(
        N=(N // 4, N),
        xlim=(0.0, 0.25),
        ylim=(0.0, 1.0),
        BC=(
            ("periodic", "periodic"),  # x
            ("reflective", "reflective"),  # y
        ),
        init_fct=partial(rayleigh_taylor, P0=1.0, gamma=5 / 3),
        gamma=5 / 3,
        potential=True,
        scheme="FV",
        p=1,
        viscosity=True,
        nu=1e-3,
        time_integrator="rk1",
        cfl_coeff=0.8,
        riemann_solver_sd="hllc",  # SD high-order flux
        riemann_solver_fv="hllc",  # MUSCL fallback flux
        superfv_viscosity=True,
        use_cupy=True,
    )

    # add a constant gravitational acceleration in the negative y-direction
    sim.dm.grad_phi = cp.zeros((sim.ndim,) + sim.dm.U_cv.shape[1:])
    sim.dm.grad_phi[sim.dims["y"]] = -1.0

    # select the slope limiter for the MUSCL-Hancock scheme
    sim.ho_scheme.slope_limiter = spd_muscl_limiter("moncen")

    sim.perform_time_evolution(T)

    return sim


def compare(sim_fv, sim_sd):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, constrained_layout=True)

    for ax in axs:
        ax.set_aspect("equal")

    x_sd = sim_sd.regular_faces()[0]
    y_sd = sim_sd.regular_faces()[1]
    z_sd = sim_sd.dm.W_cv[0]  # W_cv_initial[2]

    x_fv, y_fv, _ = sim_fv.mesh.faces
    z_fv = sim_fv.snapshot_history[-1].w[0, :, :, 0].T  # sim_fv.snapshot_history[0].w[2, :, :, 0].T

    vmin = np.minimum(z_fv, z_sd).min().item()
    vmax = np.maximum(z_fv, z_sd).max().item()

    axs[0].set_title("SuperFV")
    _ = axs[0].pcolormesh(cp.asnumpy(x_fv), cp.asnumpy(y_fv), z_fv, vmin=vmin, vmax=vmax)

    axs[1].set_title("SPD")
    m1 = axs[1].pcolormesh(x_sd, y_sd, z_sd, vmin=vmin, vmax=vmax)
    fig.colorbar(m1, ax=axs[1], pad=0.15)

    axs[2].set_title(r"$|\text{SPD} - \text{SuperFV}|$")
    m2 = axs[2].pcolormesh(x_sd, y_sd, np.abs(z_sd - z_fv), cmap="RdPu")
    fig.colorbar(m2, ax=axs[2], pad=0.15)

    fig.savefig(
        "sd-compare/rayleigh-taylor-instability/results/muscl_hancock.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    sim_fv = run_fv()
    sim_sd = run_sd()
    compare(sim_fv, sim_sd)
