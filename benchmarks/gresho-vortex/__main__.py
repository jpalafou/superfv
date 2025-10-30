from functools import partial
from itertools import product

from superfv import EulerSolver
from superfv.initial_conditions import gresho_vortex

N = 96
T = [0.2, 0.4, 0.6, 0.8, 1.0]
gamma = 5 / 3
v0 = 5.0
M_max_values = [1e-1, 1e-2, 1e-3]
output_base_dir = "out/gresho_vortex"

configs = {
    "MUSCL-Hancock": dict(
        riemann_solver="hllc",
        p=1,
        MUSCL=True,
        MUSCL_limiter="PP2D",
        flux_recipe=2,
    ),
    "ZS3": dict(
        riemann_solver="hllc",
        p=3,
        flux_recipe=2,
        lazy_primitives=True,
        ZS=True,
        GL=True,
        include_corners=True,
        PAD={"rho": (0, None), "P": (0, None)},
    ),
    "MM3": dict(
        riemann_solver="hllc",
        p=3,
        flux_recipe=2,
        lazy_primitives=True,
        MOOD=True,
        limiting_vars="actives",
        cascade="muscl",
        MUSCL_limiter="PP2D",
        max_MOOD_iters=1,
        NAD=True,
        NAD_rtol=1e-2,
        NAD_atol=1e-8,
        include_corners=True,
        PAD={"rho": (0, None), "P": (0, None)},
    ),
}

for (name, config), M_max in product(configs.items(), M_max_values):
    print(f"Running {name} with M_max = {M_max}")
    sim = EulerSolver(
        ic=partial(gresho_vortex, gamma=gamma, M_max=M_max, v0=v0),
        gamma=gamma,
        nx=N,
        ny=N,
        **config,
    )

    try:
        sim.run(
            T,
            q_max=2,
            muscl_hancock=config.get("MUSCL", False),
            allow_overshoot=True,
            log_freq=20,
            path=output_base_dir + "/" + name + f"-M_max={M_max}",
        )
    except RuntimeError as e:
        print(f"Simulation '{name}' failed: {e}")
