import shutil

import cupy as cp
import numpy as np
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput


def _get_path(solver, output_folder, t, **solver_config):
    if solver not in ("muscl_hancock", "fv", "sd"):
        raise ValueError(f"Invalid {solver=}.")
    return output_folder / "_".join(
        [solver, f"{t=}"] + [f"{key}={val}" for key, val in solver_config.items()]
    )


def generate_muscl_runner(output_folder, **ic_config):
    """
    Returns function that runs or reads superfv MUSCL-Hancock simulation for a given:
        - output_folder
        - **ic_config
    """

    def f(t, **solver_config):
        path = _get_path("muscl_hancock", output_folder, t, **solver_config)

        try:
            sim = HydroSolverOutput(path)
        except Exception:
            if path.exists():
                print(f"Removing bad superfv output at '{path}'")
                shutil.rmtree(path)
            print(f"Running superfv simulation with output path '{path}'")
        else:
            print(f"Loaded superfv output from '{path}'")
            return sim

        sim = HydroSolver(p=1, use_MUSCL=True, **ic_config, **solver_config, output_path=path)
        sim.run(t, "muscl_hancock")
        print("Finished MUSCL-Hancock simulation!\n")
        return sim

    return f


def generate_fv_runner(output_folder, **ic_config):
    """
    Returns function that runs or reads superfv simulation for a given:
        - output_folder
        - **ic_config
    """

    def f(t, **solver_config):
        path = _get_path("fv", output_folder, t, **solver_config)

        try:
            sim = HydroSolverOutput(path)
        except Exception:
            if path.exists():
                print(f"Removing bad superfv output at '{path}'")
                shutil.rmtree(path)
            print(f"Running superfv simulation with output path '{path}'")
        else:
            print(f"Loaded superfv output from '{path}'")
            return sim

        sim = HydroSolver(use_MUSCL=False, **ic_config, **solver_config, output_path=path)
        sim.run(t, "ssprk3")
        print("Finished superfv simulation!\n")
        return sim

    return f


def generate_sd_runner(output_folder, **ic_config):
    """
    Returns function that runs or reads spd simulation for a given:
        - output_folder
        - **ic_config
    """

    def f(t, **solver_config):
        path = _get_path("sd", output_folder, t, **solver_config)

        sim = SPD_Simulator(time_integrator="rk3", **ic_config, **solver_config, folder=str(path))

        try:
            sim.load_output()
        except Exception:
            if path.exists():
                print(f"Removing bad spd output at '{path}'")
                shutil.rmtree(path)
            print(f"Running spd simulation with output path '{path}'")
        else:
            print(f"Loaded spd output from '{path}'")
            return sim

        try:
            sim.output()
            sim.perform_time_evolution(t)
            sim.output()
        except BaseException:
            if path.exists():
                print(f"Removing failed spd output at '{path}'")
                shutil.rmtree(path)
            raise
        print("Finished spd simulation!\n")
        return sim

    return f


def plot_fv_2d_density_map(ax, sim, i=-1, **kwargs):
    x = sim.mesh.faces[0]
    y = sim.mesh.faces[1]
    rho = sim.snapshot_history[i].u[sim.params.variable_index_map("rho"), :, :, 0]

    if isinstance(x, cp.ndarray):
        x = cp.asnumpy(x)
    if isinstance(y, cp.ndarray):
        y = cp.asnumpy(y)

    ax.pcolormesh(x, y, rho.T, shading="flat", **kwargs)
    ax.set_aspect("equal")


def _get_sd_density_array(sim, uniform):
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


def plot_sd_2d_density_map(ax, sim, uniform=True, **kwargs):
    rho, x, y = _get_sd_density_array(sim, uniform)

    x, y, rho = (sim.dm.asnumpy(a) for a in (x, y, rho))
    ax.pcolormesh(x, y, rho, shading="flat", **kwargs)
    ax.set_aspect("equal")


def get_fv_rho_maxmin(sim):
    rho = sim.snapshot_history[-1].u[sim.params.variable_index_map("rho"), :, :, 0]
    return (rho.max(), rho.min())


def get_sd_rho_maxmin(sim, uniform=False):
    rho, _, _ = _get_sd_density_array(sim, uniform=uniform)
    return (rho.max(), rho.min())


def update_maxmin(sim, rhomax=-np.inf, rhomin=np.inf, *, uniform_sd=False):
    if isinstance(sim, (HydroSolver, HydroSolverOutput)):
        newrhomax, newrhomin = get_fv_rho_maxmin(sim)
    else:
        newrhomax, newrhomin = get_sd_rho_maxmin(sim, uniform=uniform_sd)
    if newrhomin < rhomin:
        rhomin = newrhomin
    if newrhomax > rhomax:
        rhomax = newrhomax
    return rhomax, rhomin


def _apply_moving_average(t, data, n_avg):
    if not 1 <= n_avg <= len(data):
        raise ValueError("n_avg must be between 1 and the number of history samples.")
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
        t, fallback_rates = _apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)


def plot_sd_fallback_history(ax, sim, n_avg=None, **kwargs):
    ntotal = sim.domain_size * sim.comms.size
    fallback_counts = np.asarray(sim.trouble_counts, dtype=float)
    assert fallback_counts.size % 3 == 0
    fallback_counts = fallback_counts.reshape(-1, 3).mean(axis=1)
    fallback_rates = fallback_counts / ntotal

    t = np.asarray(sim.step_times)
    assert len(t) == len(fallback_rates)

    if n_avg is not None:
        t, fallback_rates = _apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)
