# SuperFV

[![Build](https://github.com/jpalafou/superfv/actions/workflows/ci.yml/badge.svg)](https://github.com/jpalafou/superfv/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Super high-order Uniform-grid solver for hyPERbolic conservation laws using the Finite-Volume method**.

## Overview

SuperFV is a computational solver for hyperbolic conservation laws based on the finite-volume method with arbitrarily high-order accuracy in its spatial discretization. The solver operates on a uniform mesh, supporting up to three dimensions.

## Features

- High-order accuracy in space in 1D, 2D, or 3D
- Support for passive scalars


## Installation

SuperFV can be installed using pip:

```bash
git clone git@github.com:jpalafou/superfv.git
cd superfv
pip install .
```
