from functools import partial

import numpy as np

from superfv import HydroSolver, HydroSolverOutput, ics

path = "/scratch/gpfs/jp7427/superfv/reference-solutions/pp2d"

ref_sim = None
try:
    ref_sim = HydroSolverOutput(path)
except Exception as e:
    print(f"Failed to load reference solution from {path}: {e}")

sim = HydroSolver(
    ic=partial(ics.square, vx=2, vy=1),
    nx=64,
    ny=64,
    p=1,
    use_MUSCL=True,
    MUSCL_limiter="pp2d",
    cupy=True,
    output_path=path if ref_sim is None else None,
    overwrite=True,
)
sim.take_n_steps(100, time_integrator="muscl_hancock")

if ref_sim is not None:
    print("Measuring error between simulation and solution saved to disk...")
    idx = sim.idx
    E_error = sim.snapshot_history[-1].u[idx("E")] - ref_sim.snapshot_history[-1].u[idx("E")]
    print(f"Energy error: {np.max(np.abs(E_error)).item():.6e}")
