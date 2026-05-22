from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import pandas

from superfv import HydroSolver, TimeIntegrator, ics, run_multiple_simulations
from superfv.tools.norms import linf_norm

base_path = "/Users/jonathan/Desktop/out"
overwrite = True

N_list = [16, 32, 64]
schemes = {"FV1": dict(p=0), "FV2": dict(p=1), "FV3": dict(p=2), "FV4": dict(p=3)}
init_params = dict(ic=partial(ics.sinus, vx=2.0, vy=1.0))
run_params = dict(t=1.0, time_integrator=TimeIntegrator.MATCH_P_UP_TO_RK4)


def get_CFL(N: int, p: int) -> float:
    cfl = 0.8
    qmax = 3
    if p > qmax:
        cfl *= (1 / N) ** ((p - qmax) / (qmax + 1))
    return cfl


data = []


def update_data_and_plot_error(name: str, sim: HydroSolver):
    N = sim.params.mesh.nx
    rho0 = sim.snapshot_history[0].u[sim.idx("rho")]
    rho1 = sim.snapshot_history[1].u[sim.idx("rho")]
    err = linf_norm(rho1 - rho0)

    scheme_name = name.split("/N=")[0]
    data.append(dict(scheme=scheme_name, N=N, err=err))
    df = pandas.DataFrame(data)

    # plot err over N for each scheme and write to image file
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$N$")
    ax.set_ylabel("err")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    for scheme, scheme_df in df.groupby("scheme"):
        scheme_df.plot(x="N", y="err", kind="line", ax=ax, label=scheme)
    ax.legend()
    fig.savefig(f"{base_path}/convergence.png")


run_multiple_simulations(
    configs={
        f"{name}/{N=}": (
            dict(scheme_params, **init_params, nx=N, ny=N, CFL=get_CFL(N, scheme_params["p"])),
            run_params,
        )
        for (name, scheme_params), N in product(schemes.items(), N_list)
    },
    base_path=base_path,
    overwrite=overwrite,
    postprocess=update_data_and_plot_error,
)
