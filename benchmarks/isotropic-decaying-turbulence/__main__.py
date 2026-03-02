import argparse
import os
from functools import partial
from itertools import product

import numpy as np

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import decaying_isotropic_turbulence
from superfv.tools.device_management import CUPY_AVAILABLE

parser = argparse.ArgumentParser(prog="isoturb")
parser.add_argument("--N", type=int, required=True)
args = parser.parse_args()

N = args.N
cupy = N >= 128
overwrite = False

if not cupy and CUPY_AVAILABLE:
    raise RuntimeError("Requested CPU run but CuPy is available.")
if cupy and not CUPY_AVAILABLE:
    raise RuntimeError("Requested GPU run but CuPy is not available.")

base_path = f"/scratch/gpfs/jp7427/out/isotropic-decaying-turbulence/{N}x{N}/"
if cupy:
    base_path += "cupy/"

# Loop parameters
M_max_values = [0.01, 0.1, 1, 10, 20, 30, 40, 50]

seeds = range(1, 31)

common = dict(PAD={"rho": (0, None)}, SED=False)
musclhancock = dict(p=1, MUSCL=True, **common)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    limiting_vars=("rho", "vx", "vy"),
    **common,
)
aposteriori1 = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori2 = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori)
aposteriori3 = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori)

no_v = dict(limiting_vars=("rho",))
aposteriori_no_v = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    **no_v,
    **common,
)
aposteriori1_no_v = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori_no_v)
aposteriori2_no_v = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori_no_v)
aposteriori3_no_v = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori_no_v)

configs = {
    "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/3revs/no_delta/rtol_1e-5": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori3
    ),
    "MM7/3revs/no_delta/rtol_1e-5": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori3
    ),
    "MM3/3revs/no_delta/rtol_1e-3": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-3, **aposteriori3
    ),
    "MM7/3revs/no_delta/rtol_1e-3": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-3, **aposteriori3
    ),
    "MM3/3revs/no_delta/rtol_1e-2": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-2, **aposteriori3
    ),
    "MM7/3revs/no_delta/rtol_1e-2": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-2, **aposteriori3
    ),
    "MM3/3revs/no_delta/rtol_1e-1": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-1, **aposteriori3
    ),
    "MM7/3revs/no_delta/rtol_1e-1": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-1, **aposteriori3
    ),
    "MM3/2revs/no_delta/rtol_1e-3": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-3, **aposteriori2
    ),
    "MM7/2revs/no_delta/rtol_1e-3": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-3, **aposteriori2
    ),
    "MM3/2revs/no_delta/rtol_1e-2": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-2, **aposteriori2
    ),
    "MM7/2revs/no_delta/rtol_1e-2": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-2, **aposteriori2
    ),
    "MM3/1rev/no_delta/rtol_1e-5": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-5": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
}

no_fail_set = set()  # {"p0", "MUSCL-Hancock", "ZS3", "ZS7"}


def compute_velocity_rms(sim):
    idx = sim.variable_index_map
    xp = sim.xp

    u = sim.arrays["u"]
    w = xp.empty_like(u)
    sim.conservatives_to_primitives(u, w)

    v = xp.sqrt(xp.mean(xp.sum(xp.square(w[idx("v")]), axis=0))).item()

    return v


def compute_turbulence_crossing_time(sim):
    mesh = sim.mesh

    Lx = mesh.xlim[1] - mesh.xlim[0]
    Ly = mesh.ylim[1] - mesh.ylim[0]
    Lz = mesh.zlim[1] - mesh.zlim[0]
    L = max(Lx, Ly, Lz)

    sigma = compute_velocity_rms(sim)

    return L / sigma


def compute_reference_dt(sim):
    mesh = sim.mesh

    h = min(mesh.hx, mesh.hy, mesh.hz)
    sigma = compute_velocity_rms(sim)

    return h / (3 * sigma)


# Run simulations
for (name, config), M_max, seed in product(configs.items(), M_max_values, seeds):
    if M_max < 1 and seed > 1:
        # only need one seed for low Mach numbers
        continue

    sim_path = f"{base_path}{name}/M_max_{M_max}/seed_{seed:02d}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"- - - Starting simulation: {name}, seed={seed}, M_max={M_max} - - -")
        print(f"\tRunning config with name '{name}' and writing to path '{sim_path}'.")

        # dummy sim used purely for computing T and dt_ref
        dummy_sim = EulerSolver(
            ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=-5 / 3),
            isothermal=True,
            nx=N,
            ny=N,
            **config,
        )

        t_cross = compute_turbulence_crossing_time(dummy_sim)
        dt_ref = compute_reference_dt(dummy_sim)
        max_steps = 10 * int(t_cross / dt_ref) if M_max > 1 else None
        print(
            f"\tt_cross = {t_cross:.4f}, dt_ref = {dt_ref:.2e}, max_steps = {max_steps}"
        )

        sim = EulerSolver(
            ic=partial(decaying_isotropic_turbulence, seed=seed, M=M_max, slope=-5 / 3),
            isothermal=True,
            nx=N,
            ny=N,
            cupy=cupy,
            **config,
        )

        try:
            sim.run(
                [t.item() for t in np.linspace(0, t_cross, 4)[1:]],
                allow_overshoot=True,
                q_max=2,
                muscl_hancock=config.get("MUSCL", False),
                log_freq=100,
                max_steps=max_steps,
                path=sim_path,
                overwrite=True,
            )
            sim.write_timings()

            if sim.t < t_cross:
                raise RuntimeError(
                    f"\tSimulation ended at t={sim.t:.4f} before target t={t_cross:.4f}."
                )

            print("\tSuccess!\n")

            # clean up error file if it exists
            if os.path.exists(f"{sim_path}error.txt"):
                os.remove(f"{sim_path}error.txt")

        except RuntimeError as e:
            print(f"  Failed: {e}")
            with open(f"{sim_path}error.txt", "w") as f:
                f.write(str(e))

            continue
