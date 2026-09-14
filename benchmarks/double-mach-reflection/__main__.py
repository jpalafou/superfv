import os

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from superfv import run_multiple_simulations
from superfv.boundary_conditions import apply_free_bc, apply_reflective_bc
from superfv.initial_conditions import double_mach_reflection
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
    nghost = context.nghost
    mesh = context.mesh

    x = cp.asnumpy(mesh.Centers[0][:, 0, 0])
    idx = np.max(np.where(x < 1 / 6)[0]).item() + nghost

    section1 = crop(1, (None, idx), ndim=4)
    section2 = crop(1, (idx, None), ndim=4)

    apply_free_bc(_u_[section1], context)
    apply_reflective_bc(_u_[section2], context)


# simulation parameters
Nx = 3200
init_params = dict(
    ic=double_mach_reflection,
    gamma=gamma,
    xlims=(0, 4),
    nx=Nx,
    ny=Nx // 4,
    bcx=("dirichlet", "free"),
    bcy=("patch", "dirichlet"),
    bcx_callable_lower=dirichlet_x0,
    bcy_callable_lower=patch_bc,
    bcy_callable_upper=dirichlet_y1,
    PAD_bounds={"rho": (0, None), "P": (0, None)},
    cupy=True,
)
run_params = dict(t=np.linspace(0, 0.2, 11)[1:].tolist(), allow_overshoot=True)

# loop parameters
musclhancock = dict(p=1, use_MUSCL=True, MUSCL_limiter="pp2d")
apriori = dict(use_ZS=True, lazy_primitive_mode="adaptive")
aposteriori = dict(use_MOOD=True, lazy_primitive_mode="full", MUSCL_limiter="pp2d")
aposteriori_1rev = dict(fallback_cascade="muscl", max_revs=1, **aposteriori)
aposteriori_2revs = dict(fallback_cascade="muscl0", max_revs=2, **aposteriori)
aposteriori_3revs = dict(fallback_cascade="muscl0", max_revs=3, **aposteriori)

configs = {
    "MUSCL-Hancock": musclhancock,
    "MUSCL-RK3": musclhancock | dict(CFL=0.5),
    "MUSCL-RK3-minmod": musclhancock | dict(MUSCL_limiter="minmod"),
    "ZS3": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "ZS3t": dict(p=3, adaptive_dt=False, **apriori),
    "ZS7t": dict(p=7, adaptive_dt=False, **apriori),
    "MM3/1rev/rtol_1e-3": dict(p=3, rtol=1e-3, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-3": dict(p=7, rtol=1e-3, **aposteriori_1rev),
    "MM3/1rev/rtol_1e-5": dict(p=3, rtol=1e-5, **aposteriori_1rev),
    "MM7/1rev/rtol_1e-5": dict(p=7, rtol=1e-5, **aposteriori_1rev),
    "ZS3-RK4": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7-RK4": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "MM3-RK4/1rev/rtol_1e-1": dict(p=3, rtol=1e-1, **aposteriori_1rev),
    "MM7-RK4/1rev/rtol_1e-1": dict(p=7, rtol=1e-1, **aposteriori_1rev),
}


def makeplot(name, sim):
    plot_path = f"out/double-mach-reflection-plots/{name}.pdf"
    dir_name = os.path.dirname(plot_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 3)

    nlevels = 15
    idx = sim.params.variable_index_map
    rho_min = sim.snapshot_history[-1].u[idx("rho")].min().item()
    rho_max = sim.snapshot_history[-1].u[idx("rho")].max().item()
    levels = np.linspace(rho_min, rho_max, nlevels)
    ax.set_title(rf"$\rho \in [{rho_min:.2f}, {rho_max:.2f}]$ contoured with {nlevels} levels")

    x_centers, y_centers = map(cp.asnumpy, sim.mesh.centers[:2])

    ax.contour(
        x_centers,
        y_centers,
        sim.snapshot_history[-1].w[idx("rho"), :, :, 0].T,
        levels=levels,
        cmap="grey",
        linewidths=0.25,
    )
    fig.savefig(plot_path, bbox_inches="tight")


run_multiple_simulations(
    {
        name: (
            init_params | config,
            dict(
                time_integrator=(
                    "rk4"
                    if "RK4" in name
                    else (
                        "ssprk3"
                        if "RK3" in name
                        else (
                            "muscl_hancock"
                            if config.get("use_MUSCL", False)
                            else "match_p_up_to_ssprk3"
                        )
                    )
                ),
                **run_params,
            ),
        )
        for name, config in configs.items()
    },
    "/scratch/gpfs/jp7427/out/double-mach-reflection/",
    overwrite=False,
    postprocess=makeplot,
)
