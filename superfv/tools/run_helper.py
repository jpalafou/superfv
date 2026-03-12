import os
from typing import Any, Dict, Literal

from superfv import AdvectionSolver, EulerSolver, OutputLoader


def run_multiple_simulations(
    system: Literal["advection", "euler"],
    configs: Dict[str, Dict[str, Any]],
    base_path: str,
    mode: Literal["n", "T"] = "n",
    *,
    n: int = 10,
    T: float = 1.0,
    q_max: int = 2,
    allow_mh: bool = True,
    snapshot_mode: Literal["target", "none", "every"] = "target",
    overwrite: bool = False,
):
    """
    Helper function for running many simulations and saving their outputs to the same
    `base_path`.

    Args:
        system: Determines FiniteVolumeSolver subclass to init:
            "advection" for AdvectionSolver or "euler" for EulerSolver.
        configs: Dictionary of configurations for the FiniteVolumeSolver init:
            `sim = FiniteVolumeSolver(**config) for name, config in configs`
        base_path: Root directory for storing snapshots. The snapshot is stored in
            'base_path/name/', where name is the key of the config in `configs`.
        mode: "n_steps" for fixed number of steps, "T" for fixed physical time.
        n: Number of steps to run for each simulation if `mode="n"
        T: Physical time to run for each simulation if `mode="T"`.
        q_max: `FiniteVolumeSolver.run` argument.
        allow_mh: Whether to allow MUSCL-Hancock. If True, `muscl_hancock` in
            `FiniteVolumeSolver.run` is set to true only for configs who contain
            `MUSCL=True`.
        snapshot_mode: `FiniteVolumeSolver.run` argument.
        overwrite: Whether to overwrite existing directories under 'base_path/name/'.
    """
    for name, config in configs.items():
        sim_path = os.path.join(base_path, name)
        error_path = os.path.join(sim_path, "error.txt")

        try:
            if overwrite:
                raise FileNotFoundError

            if os.path.exists(error_path):
                print(f"Error exists for {name} with the following contents:")
                with open(error_path, "r") as f:
                    print(f.read())
                print("\nSkipping...\n")
                continue

            sim = OutputLoader(sim_path)

            print(f"Output exists for {name}, skipping...")

            continue

        except FileNotFoundError:
            print(f"Running simulation with configuration `{name}`:")
            for argument, value in config.items():
                print(f"\t{argument}: {value}")
            print("...")

            if system == "advection":
                sim = AdvectionSolver(**config)
            elif system == "euler":
                sim = EulerSolver(**config)
            else:
                raise ValueError(f"Unknown system: {system}")

            try:
                sim.run(
                    **({"n": dict(n=n), "T": dict(T=T)}[mode]),
                    q_max=q_max,
                    muscl_hancock=config.get("MUSCL", False) if allow_mh else False,
                    path=sim_path,
                    snapshot_mode=snapshot_mode,
                    overwrite=True,
                )
                sim.write_timings()

                print(f"Successfully completed {name}!")

                # clean up error file if it exists
                if os.path.exists(error_path):
                    os.remove(error_path)

            except RuntimeError as e:
                print(f"  Failed: {e}")
                with open(error_path, "w") as f:
                    f.write(str(e))

                continue
