import numpy as np

from superfv import EulerSolver, OutputLoader
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.initial_conditions import double_mach_reflection
from superfv.tools.slicing import crop

Nx = 3200
T = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
gamma = 1.4

path = "/scratch/gpfs/jp7427/out/double-mach-reflection/"
overwrite = []

configs = {
    "MUSCL-Hancock": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        p=1,
        MUSCL=True,
        MUSCL_limiter="PP2D",
        SED=True,
    ),
    "ZS7": dict(
        GL=True,
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=7,
        ZS=True,
        include_corners=True,
        adaptive_dt=False,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "ZS7-T": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=7,
        ZS=True,
        include_corners=True,
        adaptive_dt=False,
        SED=True,
    ),
    "MM7(0.1)": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=7,
        MOOD=True,
        cascade="muscl",
        MUSCL_limiter="PP2D",
        max_MOOD_iters=1,
        limiting_vars="actives",
        NAD=True,
        include_corners=True,
        NAD_rtol=1e-1,
        NAD_atol=1e-8,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
    "MM7(0.01)": dict(
        riemann_solver="hllc",
        flux_recipe=2,
        lazy_primitives=True,
        p=7,
        MOOD=True,
        cascade="muscl",
        MUSCL_limiter="PP2D",
        max_MOOD_iters=1,
        limiting_vars="actives",
        NAD=True,
        include_corners=True,
        NAD_rtol=1e-2,
        NAD_atol=1e-8,
        PAD={"rho": (0, None), "P": (0, None)},
        SED=True,
    ),
}


def dirichlet_x0(idx, x, y, z, t, xp):
    out = xp.zeros((len(idx.idxs), *x.shape))
    out[idx("rho")] = 8.0
    out[idx("vx")] = 7.145
    out[idx("vy")] = -4.125
    out[idx("P")] = 116.5
    return out


def dirichlet_y1(idx, x, y, z, t, xp):
    theta = np.pi / 3
    dx = (10 * t / xp.sin(theta)) + (1 / 6) + (y / xp.tan(theta))

    rho = xp.where(x < dx, 8.0, gamma)
    vx = xp.where(x < dx, 7.145, 0.0)
    vy = xp.where(x < dx, -4.125, 0.0)
    P = xp.where(x < dx, 116.5, 1.0)

    out = xp.zeros((len(idx.idxs), *x.shape))

    out[idx("rho")] = rho
    out[idx("vx")] = vx
    out[idx("vy")] = vy
    out[idx("P")] = P

    return out


def patch_bc(_u_, context):
    xp = context.xp

    slab_thickness = context.slab_thickness
    mesh = context.mesh

    X, _, _ = mesh.get_cell_centers()
    x = X[:, 0, 0]
    idx = xp.max(xp.where(x < 1 / 6)[0]).item() + slab_thickness

    section1 = crop(1, (None, idx), ndim=4)
    section2 = crop(1, (idx, None), ndim=4)

    apply_free_bc(_u_[section1], context)
    apply_reflective_bc(_u_[section2], context)


for name, config in configs.items():
    if overwrite != "all" and name not in overwrite:
        try:
            sim = OutputLoader(path + name)
            continue
        except FileNotFoundError:
            pass

    print(f"Running {name}...")
    sim = EulerSolver(
        gamma=gamma,
        ic=double_mach_reflection,
        bcx=("dirichlet", "free"),
        bcy=("patch", "dirichlet"),
        bcx_callable=(dirichlet_x0, None),
        bcy_callable=(patch_bc, dirichlet_y1),
        xlim=(0, 4),
        nx=Nx,
        ny=Nx // 4,
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
    except Exception as e:
        print(f"Failed: {e}")
        continue
