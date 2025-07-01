import superfv.initial_conditions as ic
from superfv import AdvectionSolver

N = 1024
p = 3

sim = AdvectionSolver(
    ic=lambda idx, x, y, z: ic.square(idx, x, y, z, vx=2, vy=1),
    nx=N,
    ny=N,
    p=p,
    interpolation_scheme="transverse",
    cupy=True,
    log_every_step=False,
)

if __name__ == "__main__":
    sim.run(n=5, no_snapshots=True)
