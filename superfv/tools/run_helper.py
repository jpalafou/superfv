import os
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from superfv import AdvectionSolver, EulerSolver, OutputLoader


def run_multiple_simulations(
    configs: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]],
    base_path: str,
    system: Literal["advection", "euler"] = "euler",
    overwrite: bool = False,
    skip_errors: bool = True,
    postprocess: Optional[
        Callable[[str, Union[OutputLoader, AdvectionSolver, EulerSolver]], None]
    ] = None,
):
    """
    Helper function for running many simulations and saving their outputs to the same
    `base_path`.

    Args:
        configs: A dictionary mapping simulation names to tuples of the form
            `(init_params, run_params)`, where `init_params` is a dictionary of
            parameters to pass to the solver initialization and `run_params` is
            a dictionary of parameters to pass to the `run` method.
        base_path: The directory to save all simulation outputs to. Each simulation
            will be saved to a subdirectory of `base_path` with the name of the
            simulation provieed in `configs`.
        system: The system to simulate, either "advection" or "euler".
        overwrite: If True, overwrite existing outputs. If False, skip simulations
            that already have outputs. If an error file exists for a simulation, the
            behavior depends on `skip_errors`.
        skip_errors: If True, skip simulations that have an error file. If False,
            re-run simulations that have an error file.
        postprocess: An optional function that takes the name of a simulation and
            either the loaded output (if it exists) or the simulation object (if it
            was just run) and performs some postprocessing.
    """
    for name, (init_params, run_params) in configs.items():
        if "overwrite" in run_params:
            raise ValueError(
                "Cannot specify `overwrite` in run_params. Use the `overwrite` "
                "argument of `run_multiple_simulations` instead."
            )
        if "path" in run_params:
            raise ValueError(
                "Cannot specify `path` in run_params. The path for each simulation "
                "is determined by the `base_path` and the simulation name in "
                "`configs`."
            )

        sim_path = os.path.join(base_path, name)
        error_path = os.path.join(sim_path, "error.txt")

        try:
            if overwrite:
                raise FileNotFoundError

            if os.path.exists(error_path):
                print(f"Error exists for {name} with the following contents:")
                with open(error_path, "r") as f:
                    print(f.read())
                if skip_errors:
                    print("\nSkipping...\n")
                    continue
                else:
                    print("\nRe-running...\n")
                    raise FileNotFoundError

            output = OutputLoader(sim_path)

            if postprocess is not None:
                postprocess(name, output)

            print(f"Output exists for {name}, skipping...")

            continue

        except FileNotFoundError:
            sim: Union[AdvectionSolver, EulerSolver]

            print(f"Running simulation with config `{name}`:")
            print("\t__init__ parameters:")
            for argument, value in init_params.items():
                print(f"\t\t{argument}: {value}")
            print("\trun parameters:")
            for argument, value in run_params.items():
                print(f"\t\t{argument}: {value}")
            print("...")

            if system == "advection":
                sim = AdvectionSolver(**init_params)
            elif system == "euler":
                sim = EulerSolver(**init_params)
            else:
                raise ValueError(f"Unknown system: {system}")

            try:
                sim.run(**run_params, path=sim_path, overwrite=True)
                sim.write_timings()

                if postprocess is not None:
                    postprocess(name, sim)

                print(f"Successfully completed {name}!\n\n")

                # clean up error file if it exists
                if os.path.exists(error_path):
                    os.remove(error_path)

            except RuntimeError as e:
                print(f"Failed: {e}\n\n")
                with open(error_path, "w") as f:
                    f.write(str(e))

                continue
