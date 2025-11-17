from superfv import EulerSolver
from superfv.initial_conditions import kelvin_helmholtz_2d

# loop parameters
base_path = "/scratch/gpfs/jp7427/out/kelvin-helmholtz/"

PAD = {"rho": (0, None), "P": (0, None)}
apriori = dict(ZS=True, lazy_primitives="adaptive", PAD=PAD)
aposteriori = dict(MOOD=True, lazy_primitives="adaptive", PAD=PAD)

configs = {
    "ZS3": dict(p=3, GL=True, **apriori),
    "p0": dict(p=0),
    "MUSCL-Hancock": dict(p=1, MUSCL=True, MUSCL_limiter="PP2D"),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3": dict(p=3, **aposteriori),
    "MM7": dict(p=7, **aposteriori),
    "MM3(1e-1)": dict(p=3, NAD_rtol=1e-1, **aposteriori),
    "MM7(1e-1)": dict(p=7, NAD_rtol=1e-1, **aposteriori),
    "MM3(1)": dict(p=3, NAD_rtol=1.0, **aposteriori),
    "MM7(1)": dict(p=7, NAD_rtol=1.0, **aposteriori),
}

# simulation parameters
N = 2048
T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
gamma = 1.4

# run simulations
for name, config in configs.items():
    print(f"Running {name}...")

    sim_path = base_path + name

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
            muscl_hancock=config.get("MUSCL", False),
            log_freq=1000,
            path=sim_path,
        )
        sim.write_timings()
    except Exception as e:
        print(f"\nFailed: {e}\n")
        continue
