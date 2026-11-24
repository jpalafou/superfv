import shutil
import warnings
from dataclasses import replace

import cupy as cp
import numpy as np
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput
from superfv.finite_volume_driver import get_interior_view
from superfv.hydro_driver import (
    compute_hydro_dt,
    update_hydro_fluxes,
    update_hydro_workspace,
)
from superfv.slope_limiting.mood import mood_loop
from superfv.tools.step_history import SubstepSummary


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


def generate_fv_runner(output_folder, *, time_integrator="ssprk3", **ic_config):
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
        sim.run(t, time_integrator)
        print("Finished superfv simulation!\n")
        return sim

    return f


def generate_sd_runner(output_folder, *, time_integrator="rk3", **ic_config):
    """
    Returns function that runs or reads spd simulation for a given:
        - output_folder
        - **ic_config
    """

    def f(t, **solver_config):
        path = _get_path("sd", output_folder, t, **solver_config)

        sim = SPD_Simulator(
            time_integrator=time_integrator, **ic_config, **solver_config, folder=str(path)
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


def plot_fv_2d_trouble_map(ax, sim, i=-1, **kwargs):
    """Recompute binary trouble flags from one forward-Euler candidate at snapshot i."""
    params, mesh = sim.params, sim.mesh
    scheme = params.fv_scheme
    if not scheme.mood_params.use_MOOD:
        raise ValueError("Trouble maps require a simulation configured with MOOD.")
    if mesh.active_dims != ("x", "y"):
        raise ValueError("Trouble maps require a 2D xy simulation.")

    if scheme.lazy_primitive_mode != "full":
        raise NotImplementedError("Trouble maps require full lazy primitives.")
    # Detect only the initial candidate, regardless of the run's revision limit.
    scheme = replace(
        scheme, mood_params=replace(scheme.mood_params, max_revs=1, count_cell_fallbacks=False)
    )

    snapshot = sim.snapshot_history[i]
    xp = cp if mesh.cupy else np
    u = xp.asarray(snapshot.u)
    idx = params.variable_index_map
    dt = compute_hydro_dt(u, idx, mesh, params.hydro)
    interior = get_interior_view(mesh.active_dims, mesh.nghost)
    shape = (idx.nvars, *mesh._shape_)
    uold, wold, unew, wnew = (xp.empty(shape) for _ in range(4))
    alpha = xp.empty(shape)
    troubles = xp.zeros((1, *mesh._shape_), dtype=xp.int32)
    cascade = xp.zeros_like(troubles)
    revisable = xp.zeros((1, *mesh.shape), dtype=xp.int32)
    F = xp.empty((idx.nvars, mesh.nx + 1, mesh._shape_[1], mesh._shape_[2]))
    G = xp.empty((idx.nvars, mesh._shape_[0], mesh.ny + 1, mesh._shape_[2]))
    F_fallback = xp.empty((2, *F.shape))
    G_fallback = xp.empty((2, *G.shape))
    empty = xp.empty(0)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        update_hydro_workspace(
            u,
            uold,
            wold,
            empty,
            empty,
            snapshot.t,
            idx,
            mesh,
            scheme,
            params.bc,
            params.hydro,
        )
        update_hydro_fluxes(
            uold,
            wold,
            empty,
            F,
            G,
            empty,
            empty,
            empty,
            alpha,
            idx,
            mesh.active_dims,
            mesh,
            scheme,
            params.hydro,
        )
        source = xp.zeros_like(u) if params.source is None else params.source(idx, u, xp=xp)
        mood_loop(
            uold,
            wold,
            empty,
            unew,
            wnew,
            alpha,
            empty,
            empty,
            troubles,
            revisable,
            cascade,
            F,
            G,
            empty,
            F_fallback,
            G_fallback,
            empty,
            source,
            idx,
            snapshot.t,
            dt,
            mesh,
            scheme,
            params.bc,
            SubstepSummary(
                substep=0,
                t_wall=np.nan,
                n_MOOD_revisions=0,
                n_troubles_hist=[],
                fallback_counts={},
            ),
            params.hydro,
        )

    x, y = mesh.faces[:2]
    flags = troubles[interior][0, :, :, 0].T
    if xp is cp:
        x, y, flags = (cp.asnumpy(a) for a in (x, y, flags))
    ax.pcolormesh(x, y, flags, shading="flat", **kwargs)
    ax.set_aspect("equal")


def plot_sd_2d_trouble_map(ax, sim, uniform=False, **kwargs):
    """Recompute forward-Euler flags in place; call last when plotting a snapshot.

    Leaves sim in FV layout and appends a diagnostic trouble count.
    """
    if uniform:
        raise NotImplementedError("Uniform remapping of trouble flags is not implemented.")
    if sim.ndim != 2:
        raise ValueError("Trouble maps require a 2D simulation.")
    if sim.ader:
        raise NotImplementedError("Trouble maps currently require an RK simulation.")
    if not hasattr(sim.lo_scheme.dm, "troubles") or sim.lo_scheme.godunov:
        raise ValueError("Trouble maps require an SD simulation with trouble detection.")

    warnings.warn(
        "plot_sd_2d_trouble_map mutates sim: it leaves sim in FV layout and appends "
        "a diagnostic trouble count. Call it last when plotting a snapshot.",
        stacklevel=2,
    )
    sim.convert_solution(W=True, call_timer=False)
    sim.compute_dt()
    ho, lo = sim.ho_scheme, sim.lo_scheme
    u = ho.get_solution()
    if sim.WB:
        u = u - ho.dm.U_eq_sp
    ho.solve_faces(u, ader=False)
    ho.switch_to_finite_volume(U_sp=u)
    lo.working_arrays()
    lo.store_high_order_fluxes(0, ader=False)
    lo.W_cv[...] = ho.compute_primitives_cv(lo.U_cv, call_timer=False)
    lo.apply_fluxes(sim.dt)
    lo.detect_troubles(sim.dt)

    ngh = sim.Nghc
    x, y = (lo.faces[dim][ngh:-ngh] for dim in ("x", "y"))
    flags = lo.dm.troubles[0]
    x, y, flags = (lo.dm.asnumpy(a) for a in (x, y, flags))
    ax.pcolormesh(x, y, flags, shading="flat", **kwargs)
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
    steps = sim.step_history.steps[1:]  # Skip the initial condition.
    ntotal = sim.mesh.nx * sim.mesh.ny * sim.mesh.nz
    t = np.array([s.t_sim for s in steps])
    fallback_rates = np.empty(len(steps))
    for i, s in enumerate(steps):
        if s.n_dt_revisions:
            raise NotImplementedError("FV limiting fractions do not support timestep revisions.")
        if len(s.substeps) != 3:
            raise NotImplementedError("FV limiting fractions currently require SSPRK3.")
        if any(ss.n_MOOD_revisions > 1 for ss in s.substeps):
            raise NotImplementedError(
                "FV limiting fractions do not support multiple MOOD revisions."
            )
        n1, n2, n3 = (ss.n_troubles_hist[0] for ss in s.substeps)
        fallback_rates[i] = (n1 + n2 + 4 * n3) / (6 * ntotal)

    if n_avg is not None:
        t, fallback_rates = _apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)


def plot_sd_fallback_history(ax, sim, n_avg=None, **kwargs):
    fallback_rates = np.asarray(sim.limiting_fractions)
    t = np.asarray(sim.step_times)
    assert len(t) == len(fallback_rates)

    if n_avg is not None:
        t, fallback_rates = _apply_moving_average(t, fallback_rates, n_avg)

    ax.plot(t, fallback_rates, **kwargs)
