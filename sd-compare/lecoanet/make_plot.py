import cupy as cp
import matplotlib.pyplot as plt
from script import (
    params_to_dedalus_filename,
    project_dedalus_to_x_y_dye,
    run_spd_sim,
    run_superfv_sim,
    spd_to_uniform_cell_averaged_dye,
    superfv_to_uniform_cell_averaged_dye,
)

base_path = "/scratch/gpfs/jp7427/FVvsSD/lecoanet-large"

Re_base10 = 5
Nref = 4096
density_jump = 2
target_times = [2, 4, 6]
t_plot = 6
if t_plot not in target_times:
    raise ValueError(f"t_plot must be one of {target_times}.")
nout = target_times.index(t_plot) + 1

NDOF = 2048


def plot_dedalus(ax, filename):
    ax.set_aspect("equal")
    ax.set_ylim(0, 1.0)

    x, y, c = project_dedalus_to_x_y_dye(filename)
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
    dedalus_filename = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_plot)

    fv4_nolim = run_superfv_sim(
        name="unlimited",
        p=3,
        NDOF=NDOF,
        Re_base10=Re_base10,
        Nref=Nref,
        density_jump=density_jump,
        target_times=target_times,
        limiting=False,
        read_only=True,
    )

    fv4_mm = run_superfv_sim(
        name="",
        p=3,
        NDOF=NDOF,
        Re_base10=Re_base10,
        Nref=Nref,
        density_jump=density_jump,
        target_times=target_times,
        rtol=1e-5,
        read_only=True,
    )

    sd4_nolim = run_spd_sim(
        name="unlimited",
        p=3,
        NDOF=NDOF,
        Re_base10=Re_base10,
        Nref=Nref,
        density_jump=density_jump,
        target_times=target_times,
        limiting=False,
        read_only=True,
    )

    sd4_mm = run_spd_sim(
        name="",
        p=3,
        NDOF=NDOF,
        Re_base10=Re_base10,
        Nref=Nref,
        density_jump=density_jump,
        target_times=target_times,
        tolerance=1e-5,
        read_only=True,
    )

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))

    axs[0, 0].set_title(f"dedalus (N={Nref})")
    plot_dedalus(axs[0, 0], dedalus_filename)

    axs[0, 1].set_title(f"FV4 (N=2048, 7.4 hrs)")
    plot_fv(axs[0, 1], fv4_nolim)

    axs[0, 2].set_title(f"SD4 (N=2048, 9.4 hrs)")
    plot_sd(axs[0, 2], sd4_nolim)

    axs[1, 1].set_title(f"FV4-MM (N=2048, 14.1 hrs)")
    plot_fv(axs[1, 1], fv4_mm)

    axs[1, 2].set_title(f"SD4-MM (N=2048, 66.8 hrs)")
    plot_sd(axs[1, 2], sd4_mm)

    fig.savefig(
        f"/scratch/gpfs/jp7427/FVvsSD/lecoanet-large/t={t_plot}.png", dpi=300, bbox_inches="tight"
    )
