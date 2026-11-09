import shutil
import warnings
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import spd.initial_conditions as spd_ics
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput
from superfv import ics as superfv_ics
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.tools.slicing import crop

gamma = 1.4

base_output_path = Path("/scratch/gpfs/jp7427/FVvsSD/dmr")


def dirichlet_x0(idx, x, y, z, t, xp):
    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = 8.0
    out[idx("vx")] = 8.25 * np.cos(np.pi / 6)
    out[idx("vy")] = -4.125
    out[idx("P")] = 116.5
    return out


def dirichlet_y1(idx, x, y, z, t, xp):
    shock_x = 10 * t / xp.sin(np.pi / 3) + 1 / 6 + y / xp.tan(np.pi / 3)
    behind_shock = x < shock_x

    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = xp.where(behind_shock, 8.0, gamma)
    out[idx("vx")] = xp.where(behind_shock, 8.25 * np.cos(np.pi / 6), 0.0)
    out[idx("vy")] = xp.where(behind_shock, -4.125, 0.0)
    out[idx("P")] = xp.where(behind_shock, 116.5, 1.0)
    return out


def patch_bc(u, context):
    x = cp.asnumpy(context.mesh.Centers[0][:, 0, 0])
    i = np.max(np.where(x < 1 / 6)[0]).item() + context.nghost

    apply_free_bc(u[crop(1, (None, i), ndim=4)], context)
    apply_reflective_bc(u[crop(1, (i, None), ndim=4)], context)


def fv_dmr_params(nx):
    return dict(
        ic=superfv_ics.double_mach_reflection,
        gamma=gamma,
        xlims=(0, 4),
        ylims=(0, 1),
        nx=nx,
        ny=nx // 4,
        bcx=("dirichlet", "free"),
        bcy=("patch", "dirichlet"),
        bcx_callable_lower=dirichlet_x0,
        bcy_callable_lower=patch_bc,
        bcy_callable_upper=dirichlet_y1,
    )


def run_muscl_hancock_simulation(t, nx, sed=True):
    params = f"{t=}_{nx=}_{sed=}"
    path = base_output_path / ("muscl_hancock_" + params)

    try:
        sim = HydroSolverOutput(path)
    except Exception:
        if path.exists():
            print(f"Removing bad superfv output at '{path}'")
            shutil.rmtree(path)
        print(f"Running superfv simulation with output path '{path}'")
    else:
        print(f"Loaded superfv output from '{path}'")
        save_density_plot(sim, path)
        return sim

    sim = HydroSolver(
        **fv_dmr_params(nx),
        p=1,
        CFL=0.8,
        use_MUSCL=True,
        use_SED=sed,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t, "muscl_hancock")
    save_density_plot(sim, path)

    return sim


def run_fv_simulation(t, nx, p, rtol, vx_vy_nad=True, sed=True, nad_cfl=True):
    params = f"{t=}_{nx=}_{p=}_{rtol=}_{vx_vy_nad=}_{sed=}_{nad_cfl=}"
    path = base_output_path / ("fv_" + params)

    try:
        sim = HydroSolverOutput(path)
    except Exception:
        if path.exists():
            print(f"Removing bad superfv output at '{path}'")
            shutil.rmtree(path)
        print(f"Running superfv simulation with output path '{path}'")
    else:
        print(f"Loaded superfv output from '{path}'")
        save_density_plot(sim, path)
        return sim

    sim = HydroSolver(
        **fv_dmr_params(nx),
        p=p,
        CFL=0.8,
        use_MOOD=True,
        positivity_guard=False,
        omit_vars=None if vx_vy_nad else ["vx", "vy"],
        use_SED=sed,
        rtol=rtol,
        scale_by_cfl=nad_cfl,
        MUSCL_limiter="moncen",
        cupy=True,
        output_path=path,
    )
    sim.run(t)
    save_density_plot(sim, path)

    return sim


def run_sd_simulation(t, ndofx, p, rtol, vx_vy_nad=True, sed=True, nad_cfl=True):
    params = f"{t=}_{ndofx=}_{p=}_{rtol=}_{vx_vy_nad=}_{sed=}_{nad_cfl=}"
    path = base_output_path / ("sd_" + params)

    nelemsx = ndofx // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(nelemsx, nelemsx // 4),
        xlim=(0.0, 4.0),
        ylim=(0.0, 1.0),
        gamma=gamma,
        min_rho=1e-12,
        min_P=1e-12,
        BC=(
            ("doublemach", "doublemach"),
            ("doublemach", "doublemach"),
        ),
        init_fct=spd_ics.double_mach_reflection(gamma=gamma),
        cfl_coeff={3: 0.8, 7: 0.5}[p],
        use_cupy=True,
        time_integrator="rk3",
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
        potential=False,
        NAD="",
        tolerance=rtol,
        NAD_cfl=nad_cfl,
        limiting_variables=[0, 1, 2, 4] if vx_vy_nad else [0, 4],
        PAD=True,
        SED=sed,
        blending=False,
        folder=str(path),
    )

    try:
        sim.load_output()
    except Exception:
        if path.exists():
            print(f"Removing bad spd output at '{path}'")
            shutil.rmtree(path)
        print(f"Running spd simulation with output path '{path}'")
    else:
        print(f"Loaded spd output from '{path}'")
        save_density_plot(sim, path)
        return sim

    sim.output()
    sim.perform_time_evolution(t)
    sim.output()
    save_density_plot(sim, path)
    return sim


def plot_fv_density(ax, sim, i=-1, **kwargs):
    x = sim.mesh.faces[0]
    y = sim.mesh.faces[1]
    rho = sim.snapshot_history[i].u[sim.params.variable_index_map("rho"), :, :, 0]

    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    if isinstance(y, cp.ndarray):
        y = cp.asnumpy(y)

    ax.pcolormesh(x, y, rho.T, shading="flat", **kwargs)
    ax.set_aspect("equal")


def get_sd_density_array(sim, uniform=False):
    rho = sim.dm.W_cv[0:1]
    if uniform:
        x, y = sim.regular_faces()
        if sim.p > 1:
            # regular_mesh expects point values rather than cell averages.
            rho = sim.regular_mesh(sim.compute_sp_from_cv(rho))
    else:
        ngh = sim.Nghc
        x = sim.faces["x"][ngh:-ngh]
        y = sim.faces["y"][ngh:-ngh]
    rho = sim.transpose_to_fv(rho)[0]
    return rho, x, y


def plot_sd_density(ax, sim, uniform=False, **kwargs):
    rho, x, y = get_sd_density_array(sim, uniform=uniform)

    x, y, rho = (sim.dm.asnumpy(a) for a in (x, y, rho))
    ax.pcolormesh(x, y, rho, shading="flat", **kwargs)
    ax.set_aspect("equal")


def save_density_plot(sim, snapshot_dir):
    if isinstance(sim, SPD_Simulator) and sim.rank != 0:
        return

    fig, ax = plt.subplots(figsize=(8, 2.5))
    if isinstance(sim, (HydroSolver, HydroSolverOutput)):
        plot_fv_density(ax, sim, cmap="afmhot")
        t = sim.snapshot_history[-1].t
    else:
        plot_sd_density(ax, sim, uniform=True, cmap="afmhot")
        t = sim.time
    ax.set_title(rf"$\rho$ at $t={t:.3g}$")
    fig.tight_layout()
    fig.savefig(Path(snapshot_dir) / "rho.png", dpi=150)
    plt.close(fig)


def apply_moving_average(t, data, n_avg):
    kernel = np.ones(n_avg) / n_avg
    data = np.convolve(data, kernel, mode="valid")
    t = t[n_avg - 1 :]
    return t, data


def plot_fv_fallback_history(ax, sim, n_avg=None, **kwargs):
    ntotal = sim.mesh.nx * sim.mesh.ny
    t = np.array([s.t_sim for s in sim.step_history.steps])
    fallback_rates = np.array(
        [
            sum(ss.fallback_counts[1] for ss in s.substeps) / 3 / ntotal
            for s in sim.step_history.steps
        ]
    )
    assert len(t) == len(fallback_rates)

    if n_avg is not None:
        t, fallback_rates = apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)


def plot_sd_fallback_history(ax, sim, n_avg=None, **kwargs):
    warnings.warn("plot_sd_fallback_history plots over an average time step size.")

    ntotal = sim.domain_size * sim.comms.size
    fallback_counts = np.asarray(sim.trouble_counts, dtype=float)
    assert fallback_counts.size % 3 == 0
    fallback_counts = fallback_counts.reshape(-1, 3).mean(axis=1)
    fallback_rates = fallback_counts / ntotal

    n_steps = len(fallback_counts)
    dt_avg = sim.time / n_steps
    t = dt_avg * np.arange(1, n_steps + 1)

    if n_avg is not None:
        t, fallback_rates = apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)


def get_fv_rho_maxmin(sim):
    rho = sim.snapshot_history[-1].u[sim.params.variable_index_map("rho"), :, :, 0]
    return (rho.max(), rho.min())


def get_sd_rho_maxmin(sim, uniform=False):
    rho, _, _ = get_sd_density_array(sim, uniform=uniform)
    return (rho.max(), rho.min())


def get_simulation_rho_maxmin(sims, uniform_sd=False):
    maxmins = [
        (
            get_fv_rho_maxmin(sim)
            if isinstance(sim, HydroSolver) or isinstance(sim, HydroSolverOutput)
            else get_sd_rho_maxmin(sim, uniform_sd)
        )
        for sim in sims
    ]
    rhomax = max(x[0] for x in maxmins)
    rhomin = min(x[1] for x in maxmins)
    return rhomax, rhomin


if __name__ == "__main__":
    t = 0.2
    N = 192
    nad_cfl = True
    configs = [(p, rtol) for p in [3, 7] for rtol in [1e-5, 1e-3, 1e-1]]

    # run all the simulations
    _ = run_muscl_hancock_simulation(t=t, nx=N)
    for p, rtol in configs:
        _ = run_fv_simulation(t=t, nx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)
        _ = run_sd_simulation(t=t, ndofx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)

    # plot them

    def plot_simulation(ax, solver, p, rtol, uniform_sd=False, **kwargs):
        if solver == "fv":
            sim = run_fv_simulation(t=t, nx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)
            plot_fv_density(ax, sim, **kwargs)
        else:
            sim = run_sd_simulation(t=t, ndofx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)
            plot_sd_density(ax, sim, uniform_sd, **kwargs)

    def plot_fallback_history(ax, solver, p, rtol, n_avg=None, **kwargs):
        if solver == "fv":
            sim = run_fv_simulation(t=t, nx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)
            plot_fv_fallback_history(ax, sim, n_avg, **kwargs)
        else:
            sim = run_sd_simulation(t=t, ndofx=N, p=p, rtol=rtol, nad_cfl=nad_cfl)
            plot_sd_fallback_history(ax, sim, n_avg, **kwargs)

    print("Done!")
