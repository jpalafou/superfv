import pickle

with open("out/profiling_solver.pkl", "rb") as f:
    sim = pickle.load(f)

sim.euler(n=5, progress_bar=False, no_snapshots=True)
