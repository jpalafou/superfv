import os

from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import kelvin_helmholtz_2d

# loop parameters
base_path = "/scratch/gpfs/jp7427/out/kelvin-helmholtz/"
overwrite = False

common = dict(PAD={"rho": (0, None), "P": (0, None)})
musclhancock = dict(p=1, MUSCL=True, **common)
apriori = dict(ZS=True, lazy_primitives="adaptive", **common)
aposteriori = dict(
    MOOD=True,
    face_fallback=False,
    lazy_primitives="full",
    MUSCL_limiter="PP2D",
    **common,
)
aposteriori1 = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori2 = dict(cascade="muscl1", max_MOOD_iters=2, **aposteriori)
aposteriori3 = dict(cascade="muscl1", max_MOOD_iters=3, **aposteriori)

no_v = dict(limiting_vars=("rho", "P"))

configs = {
    "MUSCL-Hancock": dict(MUSCL_limiter="PP2D", **musclhancock),
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/no_delta/rtol_1e-5": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-5": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-5, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_1e-4": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-4, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-4": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-4, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_1e-3": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-3, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-3": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-3, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_5e-3": dict(
        p=3, NAD_delta=False, NAD_rtol=5e-3, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_5e-3": dict(
        p=7, NAD_delta=False, NAD_rtol=5e-3, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_1e-2": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-2, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-2": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-2, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_5e-2": dict(
        p=3, NAD_delta=False, NAD_rtol=5e-2, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_5e-2": dict(
        p=7, NAD_delta=False, NAD_rtol=5e-2, **aposteriori1
    ),
    "MM3/1rev/no_delta/rtol_1e-1": dict(
        p=3, NAD_delta=False, NAD_rtol=1e-1, **aposteriori1
    ),
    "MM7/1rev/no_delta/rtol_1e-1": dict(
        p=7, NAD_delta=False, NAD_rtol=1e-1, **aposteriori1
    ),
}

# simulation parameters
N = 2048
T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
gamma = 1.4

# run simulations
for name, config in configs.items():
    sim_path = f"{base_path}{name}/"

    try:
        if overwrite:
            raise FileNotFoundError

        if os.path.exists(f"{sim_path}error.txt"):
            print(f"Error exists for config={name}, skipping...")
            continue

        sim = OutputLoader(sim_path)
    except FileNotFoundError:
        print(f"Running simulation config={name}...")

        sim = EulerSolver(
            ic=kelvin_helmholtz_2d,
            gamma=gamma,
            nx=N,
            ny=N,
            cupy=True,
            **config,
        )

        try:
            sim.run(
                T,
                allow_overshoot=True,
                q_max=2,
                muscl_hancock=name == "MUSCL-Hancock",
                time_degree=2 if name == "MUSCL3" else None,
                log_freq=1000,
                path=sim_path,
                overwrite=True,
            )
            sim.write_timings()

            # clean up error file if it exists
            if os.path.exists(f"{sim_path}error.txt"):
                os.remove(f"{sim_path}error.txt")

        except RuntimeError as e:
            print(f"  Failed: {e}")
            with open(f"{sim_path}error.txt", "w") as f:
                f.write(str(e))

            continue
