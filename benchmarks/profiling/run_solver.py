import pickle

import matplotlib.pyplot as plt

import superfv.visualization as vis

with open("out/profiling_solver.pkl", "rb") as f:
    sim = pickle.load(f)

sim.euler(n=5, progress_bar=False)


n_updates = sim.minisnapshots["n_updates"]
run_times = sim.minisnapshots["run_time"]
for i in range(1, sim.step_count + 1):
    print(
        f"Update {i}: {n_updates[i]} cells update in {1000*run_times[i]:.2f} ms, "
        f"resulting in {n_updates[i]/run_times[i]:.2e} cells/second."
    )

fig, ax = plt.subplots()
vis.plot_1d_slice(sim, ax, "rho", x=None)
fig.savefig("out/profiling_solver.png", dpi=300)
