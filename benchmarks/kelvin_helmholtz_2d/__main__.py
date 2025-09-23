from superfv import EulerSolver, OutputLoader
from superfv.initial_conditions import kelvin_helmholtz_2d

N = 2048
T = 0.8
gamma = 1.4

path = "/scratch/gpfs/jp7427/out/kelvin-helmholtz_N=2048/"
overwrite = []

configs = {
    "MUSCL-Hancock": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        p=1,
        MUSCL=True,
        MUSCL_limiter="moncen",
        SED=True,
    ),
    "ZS3": dict(
        GL=True,
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=3,
        ZS=True,
        include_corners=True,
        adaptive_dt=False,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "MM3": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=3,
        MOOD=True,
        cascade="muscl",
        MUSCL_limiter="moncen",
        max_MOOD_iters=1,
        limiting_vars="actives",
        NAD=True,
        include_corners=True,
        NAD_rtol=1e-2,
        NAD_atol=1e-2,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
}

sims = {}
for name, config in configs.items():
    if overwrite != "all":
        if name not in overwrite:
            try:
                sim = OutputLoader(path + name)
                sims[name] = sim
                continue
            except FileNotFoundError:
                pass

    print(f"Running {name}...")
    sim = EulerSolver(
        ic=kelvin_helmholtz_2d,
        gamma=gamma,
        nx=N,
        ny=N,
        cupy=True,
        **config,
    )

    try:
        if config.get("MUSCL", False):
            sim.musclhancock(
                T, allow_overshoot=True, path=path + name, overwrite=True, log_freq=20
            )
        else:
            sim.run(
                T,
                q_max=2,
                allow_overshoot=True,
                path=path + name,
                overwrite=True,
                log_freq=20,
            )
        sim.write_timings()
        sims[name] = sim
    except Exception as e:
        print(f"Failed: {e}")
        continue
