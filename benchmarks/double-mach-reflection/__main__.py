import os

import matplotlib.pyplot as plt
import numpy as np

from superfv import plot_2d_slice
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.initial_conditions import double_mach_reflection
from superfv.tools.run_helper import run_multiple_simulations
from superfv.tools.slicing import crop

# define boundary conditions
gamma = 1.4


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


# simulation parameters
Nx = 3200
init_params = dict(
    ic=double_mach_reflection,
    gamma=gamma,
    xlim=(0, 4),
    nx=Nx,
    ny=Nx // 4,
    bcx=("dirichlet", "free"),
    bcy=("patch", "dirichlet"),
    bcx_callable=(dirichlet_x0, None),
    bcy_callable=(patch_bc, dirichlet_y1),
    PAD={"rho": (0, None), "P": (0, None)},
    cupy=True,
)
run_params = dict(T=np.linspace(0, 0.2, 11).tolist(), allow_overshoot=True)

# loop parameters
musclhancock = dict(p=1, MUSCL=True, MUSCL_limiter="PP2D")
apriori = dict(ZS=True, lazy_primitives="adaptive")
aposteriori = dict(MOOD=True, lazy_primitives="full", MUSCL_limiter="PP2D")
aposteriori_1rev = dict(cascade="muscl", max_MOOD_iters=1, **aposteriori)
aposteriori_2revs = dict(cascade="muscl0", max_MOOD_iters=2, **aposteriori)
aposteriori_3revs = dict(cascade="muscl0", max_MOOD_iters=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "ZS3": dict(p=3, GL=True, **apriori),
    "ZS7": dict(p=7, GL=True, **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/rtol_1e-1": dict(p=3, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-1": dict(p=7, NAD_rtol=1e-1, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-3": dict(p=3, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, NAD_rtol=1e-3, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, NAD_rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, NAD_rtol=1e-5, **aposteriori_1rev),
}


def makeplot(name, sim):
    plot_path = f"out/double-mach-reflection-plots/{name}.png"
    dir_name = os.path.dirname(plot_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 3)

    plot_2d_slice(
        sim,
        ax,
        "rho",
        cmap="grey",
        colorbar=False,
        levels=15,
        linewidths=0.25,
    )
    fig.savefig(plot_path, dpi=300)


run_multiple_simulations(
    {name: (init_params | config, run_params) for name, config in configs.items()},
    "/scratch/gpfs/jp7427/out/double-mach-reflection/",
    overwrite=False,
    postprocess=makeplot,
)
