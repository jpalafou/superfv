import matplotlib.pyplot as plt
import numpy as np

from superfv import EulerSolver, plot_2d_slice
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.initial_conditions import double_mach_reflection
from superfv.tools.slicing import crop

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
    dx = (10 * t / np.sin(theta)) + (1 / 6) + (y / np.tan(theta))

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
    idx = xp.max(xp.where(x < 1 / 6)[0]) + slab_thickness

    section1 = crop(1, (None, idx), ndim=4)
    section2 = crop(1, (idx, None), ndim=4)

    apply_free_bc(_u_[section1], context)
    apply_reflective_bc(_u_[section2], context)


Nx = 256

sim = EulerSolver(
    ic=double_mach_reflection,
    bcx=("dirichlet", "free"),
    bcy=("patch", "dirichlet"),
    bcx_callable=(dirichlet_x0, None),
    bcy_callable=(patch_bc, dirichlet_y1),
    xlim=(0, 4),
    nx=Nx,
    ny=Nx // 4,
    p=0,
)

sim.run(0.2, log_freq=10)


fig, ax = plt.subplots()
ax.set_xlim(0, 3)

plot_2d_slice(sim, ax, "rho", cell_averaged=True, cmap="GnBu_r", colorbar=True)

fig.savefig("benchmarks/double-mach-reflection/plot.png", dpi=300)
