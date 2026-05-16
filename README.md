# SuperFV

[![Build](https://github.com/jpalafou/superfv/actions/workflows/ci.yml/badge.svg)](https://github.com/jpalafou/superfv/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Super high-order Uniform-grid solver for hyPERbolic conservation laws using the Finite-Volume method**.

## Overview

SuperFV is a high-order finite volume solver for the Euler equations of hydrodynamics in 1D, 2D, and 3D.

## Features

- Up-to-eighth-order accuracy in space and fourth-order accuracy in time
- Options for *a priori* slope limiting (MUSCL, Zhang-Shu)
- Options for *a posteriori* slope limiting (MOOD)
- Support for passive scalars
- Supports GPU acceleration using custom CUDA kernels and CuPy

## Installation

SuperFV can be installed using pip:

```bash
git clone git@github.com:jpalafou/superfv.git
cd superfv
pip install .
```

## Initialization

Initialize a `HydroSolver` object with the desired parameters:

```python
from superfv import HydroSolver, ics, BC
sim = HydroSolver(ic=ics.square, nx=64, bcx=(BC.PERIODIC, BC.PERIODIC), ...)
```

Documentation for `HydroSolver.__init__` includes a complete list of arguments and their purpose.

## Running simulations

The simulation can be evolved for a fixed number of steps:

```python
sim.take_n_steps(n=10, ...)
```

Or, until a target time is reached:

```python
sim.run(t=1.0, ...)
```

## Post-processing

`sim.step_history` includes useful information about each step like conserved energy and timing. Snapshots are similarly recorded in `sim.snapshot_history`. To write snapshots to disk, provide an output path:

```python
sim = HydroSolver(ic=ics.square, nx=64, output_path="myfolder/outputs")
```

Otherwise, they are stored in memory. You can supply multiple target times that will trigger snapshots:

```python
from superfv import SnapshotMode
sim.run(t=[0.2, 0.4, 0.6, 0.8, 1.0], snapshot_mode=SnapshotMode.TARGET,  allow_overshoot=False)
```

Snapshots after the initial snapshot can be turned off entirely with `SnapshotMode.NONE`, and, if memory permits, a snapshot can be taken at every step with `SnapshotMode.EVERY`.

## Example

- ['notebooks/Example.ipynb'](notebooks/Example.ipynb)
