import numpy as np

from superfv import EulerSolver
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.initial_conditions import double_mach_reflection
from superfv.tools.slicing import crop

# loop parameters
base_path = "/scratch/gpfs/jp7427/out/double-mach-reflection/"

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
Nx = 3200
T = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
gamma = 1.4


# define boundary condition callables
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


# run simulations
for name, config in configs.items():
    print(f"Running {name}...")

    sim_path = base_path + name

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
