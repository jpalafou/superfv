import argparse
from functools import partial
from itertools import product

from superfv import ics, run_multiple_simulations

parser = argparse.ArgumentParser()
parser.add_argument("--cupy", action="store_true", help="Use CuPy for GPU acceleration")
cupy = parser.parse_args().cupy

run_params = dict(n=11, snapshot_mode="none")
init_params = dict(
    ic=partial(ics.sinus, rho_min=1, rho_max=2, vx=2, vy=1, P=1),
    gamma=1.4,
    skip_trouble_counts=True,
    cupy=cupy,
    profile=False,
)

# loop parameters
if cupy:
    N_values = [32, 64, 128, 256, 512, 1024, 2048, 3072]
else:
    N_values = [32, 64, 128, 256, 512]

musclhancock = dict(p=1, use_MUSCL=True, MUSCL_limiter="pp2d")
apriori = dict(
    use_ZS=True,
    lazy_primitive_mode="adaptive",
    adaptive_dt=False,
)
aposteriori = dict(
    use_MOOD=True,
    lazy_primitive_mode="full",
    MUSCL_limiter="pp2d",
)
aposteriori_1rev = dict(
    fallback_cascade="muscl",
    max_revs=1,
    **aposteriori,
)
aposteriori_2revs = dict(
    fallback_cascade="muscl0",
    max_revs=2,
    **aposteriori,
)
aposteriori_3revs = dict(
    fallback_cascade="muscl0",
    max_revs=3,
    **aposteriori,
)

configs = {
    "p0": dict(p=0),
    "p1": dict(p=1),
    "p3": dict(p=3),
    "p7": dict(p=7),
    "p3_GL": dict(p=3, flux_quadrature="gauss_legendre"),
    "p7_GL": dict(p=7, flux_quadrature="gauss_legendre"),
    "MUSCL-Hancock": musclhancock,
    "ZS3": dict(p=3, flux_quadrature="gauss_legendre", **apriori),
    "ZS7": dict(p=7, flux_quadrature="gauss_legendre", **apriori),
    "ZS3t": dict(p=3, **apriori),
    "ZS7t": dict(p=7, **apriori),
    "MM3_1rev_rtol_0": dict(p=3, rtol=0, **aposteriori_1rev),
    "MM7_1rev_rtol_0": dict(p=7, rtol=0, **aposteriori_1rev),
    "MM3_2revs_rtol_0": dict(p=3, rtol=0, **aposteriori_2revs),
    "MM7_2revs_rtol_0": dict(p=7, rtol=0, **aposteriori_2revs),
    "MM3_3revs_rtol_0": dict(p=3, rtol=0, **aposteriori_3revs),
    "MM7_3revs_rtol_0": dict(p=7, rtol=0, **aposteriori_3revs),
}

run_multiple_simulations(
    {
        f"{name}/N_{N}": (
            dict(nx=N, ny=N, **config, **init_params),
            run_params | (dict(time_integrator="muscl_hancock") if name == "MUSCL-Hancock" else {}),
        )
        for (name, config), N in product(configs.items(), N_values)
    },
    "/scratch/gpfs/jp7427/superfv/sinus-2d-profiling/" + ("cupy/" if cupy else ""),
    overwrite=True,
)
