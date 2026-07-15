from functools import partial
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import spd.initial_conditions as ic
from spd.sdfb_simulator import SPD_Simulator

from superfv import HydroSolver, HydroSolverOutput, ics
from superfv.hydro_solver import TimeIntegrator

base_directory = Path("/scratch/gpfs/jp7427/FVvsSD/lecoanet/")
dataset_directory = Path("/scratch/gpfs/jp7427/FVvsSD/Lecoanet_dataset/")

Re_base10 = 5
Nref = 4096
density_jump = 2
t_sim_approx = 6

gamma = 5.0 / 3.0
NDOF = 2048
p = 7  # only used for FV and SD simulations
which = "fv"  # "mh", "fv", or "sd"


def nu_from_Re(Re: float) -> float:
    return 2.0 / Re


def params_to_dedalus_filename(
    Re_base10: int, Nref: int, density_jump: int, t_sim_approx: int
) -> Path:
    return dataset_directory / f"{Re_base10}_{Nref}_{density_jump}_{t_sim_approx}.h5"


def project_dedalus_to_t_exact(filename: Path) -> float:
    with h5py.File(filename, "r") as f:
        out = float(f["scales"]["sim_time"][0])
    print(f"t_exact from '{filename}' is {out}")
    return out


def project_dedalus_to_x_y_dye(filename: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(filename, "r") as f:
        x = np.asarray(f["scales"]["x"])
        y = np.asarray(f["scales"]["z"])
        c = np.asarray(f["tasks"]["c"]).squeeze()
    return x, y, c


def _grid_values_to_uniform_cell_averages(
    values: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    nxf, nyf = values.shape
    coeff = np.fft.fft2(values) / (nxf * nyf)

    kx = np.fft.fftfreq(nxf) * nxf
    ky = np.fft.fftfreq(nyf) * nyf

    # Average each Fourier mode over a target FV cell and shift to cell centers.
    coeff *= (
        np.sinc(kx / nx)[:, None]
        * np.sinc(ky / ny)[None, :]
        * np.exp(1j * np.pi * kx / nx)[:, None]
        * np.exp(1j * np.pi * ky / ny)[None, :]
    )

    coarse_coeff = np.zeros((nx, ny), dtype=np.complex128)
    np.add.at(
        coarse_coeff,
        (
            np.mod(kx.astype(int), nx)[:, None],
            np.mod(ky.astype(int), ny)[None, :],
        ),
        coeff,
    )

    return np.fft.ifft2(coarse_coeff * (nx * ny)).real


def run_MUSCL_Hancock_sim(name, NDOF, Re_base10, Nref, density_jump, t_sim_approx, **kwargs):
    dedalus = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_sim_approx)
    path = base_directory / f"MUSCL_Hancock_{name}_{NDOF=}_{dedalus.stem}"
    t_exact = project_dedalus_to_t_exact(dedalus)
    nu = nu_from_Re(10**Re_base10)

    try:
        out = HydroSolverOutput(path)
        print(f"Loaded output from '{path}'")
        return out
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        if path.exists():
            print(f"Path '{path}' exists. Returning early.")
            return None

    sim = HydroSolver(
        ic=partial(ics.lecoanet_kelvin_helmholtz, density_jump=density_jump - 1.0),
        passive_ics={"dye": ics.lecoanet_kelvin_helmholtz_dye},
        gamma=gamma,
        nu=nu,
        Chi=nu,
        nu_dye=nu,
        rho_min=1e-10,
        P_min=1e-10,
        nx=NDOF,
        ny=2 * NDOF,
        xlims=(0.0, 1.0),
        ylims=(0.0, 2.0),
        p=1,
        use_MUSCL=True,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(t_exact, time_integrator=TimeIntegrator.MUSCL_HANCOCK)
    return sim


def run_superfv_sim(name, p, NDOF, Re_base10, Nref, density_jump, t_sim_approx, **kwargs):
    dedalus = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_sim_approx)
    path = base_directory / f"FV_{name}_{p=}_{NDOF=}_{dedalus.stem}"
    t_exact = project_dedalus_to_t_exact(dedalus)
    nu = nu_from_Re(10**Re_base10)

    try:
        out = HydroSolverOutput(path)
        print(f"Loaded output from '{path}'")
        return out
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        if path.exists():
            print(f"Path '{path}' exists. Returning early.")
            return None

    sim = HydroSolver(
        ic=partial(ics.lecoanet_kelvin_helmholtz, density_jump=density_jump - 1.0),
        passive_ics={"dye": ics.lecoanet_kelvin_helmholtz_dye},
        gamma=gamma,
        nu=nu,
        Chi=nu,
        nu_dye=nu,
        rho_min=1e-10,
        P_min=1e-10,
        nx=NDOF,
        ny=2 * NDOF,
        xlims=(0.0, 1.0),
        ylims=(0.0, 2.0),
        p=p,
        use_MOOD=True,
        detect_closing_troubles=False,
        cupy=True,
        output_path=path,
        **kwargs,
    )
    sim.run(
        t_exact,
    )
    return sim


def run_spd_sim(name, p, NDOF, Re_base10, Nref, density_jump, t_sim_approx, **kwargs):
    dedalus = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_sim_approx)
    path = base_directory / f"SD_{name}_{p=}_{NDOF=}_{dedalus.stem}"
    t_exact = project_dedalus_to_t_exact(dedalus)
    nu = nu_from_Re(10**Re_base10)
    Nelements = NDOF // (p + 1)

    sim = SPD_Simulator(
        p=p,
        N=(Nelements, 2 * Nelements),
        xlim=(0.0, 1.0),
        ylim=(0.0, 2.0),
        gamma=gamma,
        viscosity=True,
        thdiffusion=True,
        nu=nu,
        chi=nu,
        BC=(("periodic", "periodic"), ("periodic", "periodic")),
        init_fct=ic.KH_instability(density_jump=density_jump - 1.0),
        passives=["dye"],
        cfl_coeff={3: 0.4, 7: 0.2}[p],
        use_cupy=True,
        time_integrator="rk3",
        scheme="SDFB",
        fallback="MUSCL",
        slope_limiter="moncen",
        potential=True,
        limiting_variables=[0, 1, 2, 4],
        PAD=True,
        SED=True,
        blending=False,
        riemann_solver_sd="hllc",
        riemann_solver_fv="hllc",
        folder=str(path),
        **kwargs,
    )

    try:
        sim.load_output()
        print(f"Loaded output from '{path}'")
        return sim
    except Exception as e:
        print(f"Failed to load output from '{path}' with: {e}")
        if path.exists():
            print(f"Path '{path}' exists. Returning early.")
            return None

    sim.output()
    sim.perform_time_evolution(t_exact)
    sim.output()

    return sim


def project_dedalus_to_uniform_cell_averaged_dye(
    filename: Path,
    nx: int,
    ny: int,
) -> np.ndarray:
    with h5py.File(filename, "r") as f:
        x = np.asarray(f["scales"]["x"]).squeeze()
        y = np.asarray(f["scales"]["z"]).squeeze()
        c = np.asarray(f["tasks"]["c"])

    c = c[0]
    if c.shape != (len(x), len(y)):
        raise ValueError(f"Expected c.shape == {(len(x), len(y))}, got {c.shape}.")

    return _grid_values_to_uniform_cell_averages(c, nx, ny)


def superfv_to_uniform_cell_averaged_dye(sim):
    idx = sim.params.variable_index_map
    return sim.snapshot_history[-1].w[idx("dye")].squeeze()


def spd_to_uniform_cell_averaged_dye(sim):
    W_sp = sim.ho_scheme.compute_sp_from_cv(sim.dm.W_cv)
    W_fv = sim.ho_scheme.compute_cv_from_sp_fv(W_sp)
    return W_fv[sim._p_ + 1].T


if __name__ == "__main__":
    reference_file = params_to_dedalus_filename(Re_base10, Nref, density_jump, t_sim_approx)
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_file}")

    if which == "mh":
        run_MUSCL_Hancock_sim(
            name="",
            NDOF=NDOF,
            Re_base10=Re_base10,
            Nref=Nref,
            density_jump=density_jump,
            t_sim_approx=t_sim_approx,
        )
    elif which == "fv":
        run_superfv_sim(
            name="",
            p=p,
            NDOF=NDOF,
            Re_base10=Re_base10,
            Nref=Nref,
            density_jump=density_jump,
            t_sim_approx=t_sim_approx,
            rtol=1e-5,
        )
    elif which == "sd":
        run_spd_sim(
            name="",
            p=p,
            NDOF=NDOF,
            Re_base10=Re_base10,
            Nref=Nref,
            density_jump=density_jump,
            t_sim_approx=t_sim_approx,
            tolerance=1e-5,
        )
