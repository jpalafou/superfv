from functools import partial
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from superfv import AdvectionSolver, ics
from superfv.hydro_solver_output import HydroSolverOutput

output = Path("/scratch/gpfs/jp7427/siam-nnp26/slotted_disk/")

times = [float(i + 1) * 2 * np.pi for i in range(10)]

config = dict(
    ic=partial(ics.slotted_disk, rho_min=0.0, rho_max=1.0),
    nx=256,
    ny=256,
    bcx=("ic", "ic"),
    bcy=("ic", "ic"),
    PAD_bounds={"rho": (0.0, 1.0)},
    cupy=True,
)


def print_rho_min_max(sim):
    rho_min = min(sim.step_history.get_history("rho_min"))
    rho_max = max(sim.step_history.get_history("rho_max"))
    print(f"rho_min: {rho_min}, rho_max: {rho_max}")


def plot_result(sim):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)

    x = cp.asnumpy(sim.mesh.faces[0])
    y = cp.asnumpy(sim.mesh.faces[1])
    z = sim.snapshot_history[-1].u[sim.params.variable_index_map("rho"), :, :, 0].T
    ax.pcolormesh(x, y, z, shading="flat", cmap="jet")
    fig.savefig(sim.params.output_path / Path("final_rho.png"), dpi=300)


def run_simulation(folder_name, **kwargs):
    output_path = output / Path(folder_name)

    try:
        sim = HydroSolverOutput(output_path)
    except FileNotFoundError:
        sim = AdvectionSolver(**kwargs, **config)
        sim.run(times)
    print_rho_min_max(sim)
    plot_result(sim)


run_simulation(
    "a_priori_8th_order_transverse",
    p=7,
    use_ZS=True,
    flux_quadrature="transverse",
)


run_simulation(
    "a_priori_8th_order_gauss_legendre",
    p=7,
    use_ZS=True,
    flux_quadrature="gauss_legendre",
    adaptive_dt=True,
)

run_simulation(
    "a_posteriori_8th_order_1rev",
    p=7,
    use_MOOD=True,
    max_revs=1,
    fallback_cascade="muscl",
)

run_simulation(
    "a_posteriori_8th_order_3rev",
    p=7,
    use_MOOD=True,
    max_revs=3,
    fallback_cascade="muscl0",
)
