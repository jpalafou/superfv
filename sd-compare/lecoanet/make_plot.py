import cupy as cp
import matplotlib.pyplot as plt
from script import (
    params_to_dedalus_filename,
    project_dedalus_to_x_y_dye,
    run_MUSCL_Hancock_sim,
    run_spd_sim,
    run_superfv_sim,
    spd_to_uniform_cell_averaged_dye,
    superfv_to_uniform_cell_averaged_dye,
)

Re_base10 = 5
Nref = 4096
density_jump = 2
target_times = [2, 4, 6]
nout = 1
if not 1 <= nout <= len(target_times):
    raise ValueError(f"nout must be between 1 and {len(target_times)}.")
t_plot = target_times[nout - 1]

NDOF = 2048

dedalus = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_plot)

musclhancock = run_MUSCL_Hancock_sim(
    name="",
    NDOF=NDOF,
    Re_base10=Re_base10,
    Nref=Nref,
    density_jump=density_jump,
    target_times=target_times,
)

superfv4 = run_superfv_sim(
    name="",
    p=3,
    NDOF=NDOF,
    Re_base10=Re_base10,
    Nref=Nref,
    density_jump=density_jump,
    target_times=target_times,
    rtol=1e-5,
)

superfv8 = run_superfv_sim(
    name="",
    p=7,
    NDOF=NDOF,
    Re_base10=Re_base10,
    Nref=Nref,
    density_jump=density_jump,
    target_times=target_times,
    rtol=1e-5,
)

spd4 = run_spd_sim(
    name="",
    p=3,
    NDOF=NDOF,
    Re_base10=Re_base10,
    Nref=Nref,
    density_jump=density_jump,
    target_times=target_times,
    tolerance=1e-5,
)

spd8 = run_spd_sim(
    name="",
    p=7,
    NDOF=NDOF,
    Re_base10=Re_base10,
    Nref=Nref,
    density_jump=density_jump,
    target_times=target_times,
    tolerance=1e-5,
)


def plot_dedalus(ax, filename):
    ax.set_aspect("equal")
    ax.set_ylim(0, 1.0)

    x, y, c = project_dedalus_to_x_y_dye(dedalus)
    return ax.pcolormesh(x, y, c.T, shading="nearest")


def plot_fv(ax, sim):
    ax.set_aspect("equal")
    ax.set_ylim(0, 1.0)

    x_fv, y_fv, _ = sim.mesh.faces
    z_fv = superfv_to_uniform_cell_averaged_dye(sim, nout).T
    return ax.pcolormesh(cp.asnumpy(x_fv), cp.asnumpy(y_fv), z_fv)


def plot_sd(ax, sim):
    ax.set_aspect("equal")
    ax.set_ylim(0, 1.0)

    x_sd = sim.regular_faces()[0]
    y_sd = sim.regular_faces()[1]
    z_sd = spd_to_uniform_cell_averaged_dye(sim, nout)
    return ax.pcolormesh(cp.asnumpy(x_sd), cp.asnumpy(y_sd), cp.asnumpy(z_sd))


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))

    axs[0, 0].set_title(f"dedalus (N={Nref})")
    plot_dedalus(axs[0, 0], dedalus)

    axs[1, 0].set_title(f"MUSCL-Hancock (N={NDOF})")
    plot_fv(axs[1, 0], musclhancock)

    axs[0, 1].set_title(f"FV4 (N={NDOF})")
    plot_fv(axs[0, 1], superfv4)

    axs[0, 2].set_title(f"FV8 (N={NDOF})")
    plot_fv(axs[0, 2], superfv8)

    axs[1, 1].set_title(f"SD4 (N={NDOF})")
    plot_sd(axs[1, 1], spd4)

    axs[1, 2].set_title(f"SD8 (N={NDOF})")
    plot_sd(axs[1, 2], spd8)

    fig.savefig(f"sd-compare/lecoanet/t={t_plot}.png", dpi=300, bbox_inches="tight")
