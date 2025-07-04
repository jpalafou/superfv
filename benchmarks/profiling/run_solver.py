import pickle

import matplotlib.pyplot as plt
import numpy as np

import superfv.visualization as vis

with open("out/profiling_solver.pkl", "rb") as f:
    sim = pickle.load(f)

sim.euler(n=5, progress_bar=False)

n_cells_updated_per_s = (
    np.array(sim.minisnapshots["n_updates"])[1:]
    / np.array(sim.minisnapshots["run_time"])[1:]
)

for i in range(len(n_cells_updated_per_s)):
    print(f"Step {i+1}: {n_cells_updated_per_s[i]:.2e} cells updated per second")

fig, ax = plt.subplots()
vis.plot_1d_slice(sim, ax, "rho", x=None)
fig.savefig("out/profiling_solver.png", dpi=300)
