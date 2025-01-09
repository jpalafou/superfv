from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np

from .tools.array_management import ArrayLike, ArraySlicer


class InitialCondition(ABC):
    """
    Initial conditions base class.
    """

    @abstractmethod
    def base_ic(
        self, x: ArrayLike, y: ArrayLike, z: ArrayLike, *args, **kwargs
    ) -> ArrayLike:
        """
        Returns initial condition array with flexible arguments.

        Args:
            x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
            y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
            z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ArrayLike: Initial condition array, has shape (nvars, nx, ny, nz).
        """
        pass

    def __init__(self, *args, **kwargs):
        """
        Initializes the initial conditions object with arguments that are passed to the
        base_ic method.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs

    def __call__(
        self, array_slicer: ArraySlicer, dims: str
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
        """
        Returns a callable object that can be used to generate the initial condition.

        Args:
            array_slicer (ArraySlicer): Array slicer object. Defines the variables used
                in the initial condition.
            dims (str): Dimensions of the simulation. Can be "x", "y", "z", or any
                combination.
        """
        if not all(dim in "xyz" for dim in dims):
            raise ValueError(
                f"Invalid dimension string: {dims}. Must be a subset of 'xyz'."
            )

        self.array_slicer = array_slicer
        self.dims = dims
        return lambda x, y, z: self.base_ic(x, y, z, *self.args, **self.kwargs)


class Sinus(InitialCondition):
    def base_ic(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        vx: float = 0,
        vy: float = 0,
        vz: float = 0,
        bounds: Tuple[float, float] = (0, 1),
    ) -> ArrayLike:
        """
        Returns initial condition array for the sinusoidal initial condition that is
        periodic on the interval [0, 1] in each dimension.

        Args:
            x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
            y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
            z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
            vx (float): x-component of the velocity.
            vy (float): y-component of the velocity.
            vz (float): z-component of the velocity.
            bounds (Tuple[float, float]): Bounds of the sinusoidal function.
        """
        _slc = self.array_slicer
        dims = self.dims

        # Validate variables in ArraySlicer
        if _slc.vars == {"u", "vx", "vy", "vz"}:
            # advection case
            out = np.zeros((4, *x.shape))
            r = int("x" in dims) * x + int("y" in dims) * y + int("z" in dims) * z
            out[_slc("u")] = (bounds[1] - bounds[0]) * np.sin(2 * np.pi * r) + bounds[0]
            out[_slc("vx")] = vx
            out[_slc("vy")] = vy
            out[_slc("vz")] = vz
        else:
            raise NotImplementedError(
                f"Initial condition not implemented for variables: {_slc.vars}. "
                "Supported variables: {'u', 'vx', 'vy', 'vz'}."
            )
        return out


class Square(InitialCondition):
    def base_ic(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        vx: float = 0,
        vy: float = 0,
        vz: float = 0,
        bounds: Tuple[float, float] = (0, 1),
    ) -> ArrayLike:
        """
        Returns initial condition array for the square initial condition that is
        periodic on the interval [0, 1] in each dimension.

        Args:
            x (ArrayLike): x-coordinates, has shape (nx, ny, nz).
            y (ArrayLike): y-coordinates, has shape (nx, ny, nz).
            z (ArrayLike): z-coordinates, has shape (nx, ny, nz).
            vx (float): x-component of the velocity.
            vy (float): y-component of the velocity.
            vz (float): z-component of the velocity.
            bounds (Tuple[float, float]): Bounds of the square function.
        """
        _slc = self.array_slicer
        dims = self.dims

        # Validate variables in ArraySlicer
        if _slc.vars == {"u", "vx", "vy", "vz"}:
            # advection case
            out = np.zeros((4, *x.shape))
            r = np.ones_like(x).astype(bool)
            if "x" in dims:
                r &= (x >= 0.25) & (x <= 0.75)
            if "y" in dims:
                r &= (y >= 0.25) & (y <= 0.75)
            if "z" in dims:
                r &= (z >= 0.25) & (z <= 0.75)
            r = r.astype(float)
            out[_slc("u")] = (bounds[1] - bounds[0]) * r + bounds[0]
            out[_slc("vx")] = vx
            out[_slc("vy")] = vy
            out[_slc("vz")] = vz
        else:
            raise NotImplementedError(
                f"Initial condition not implemented for variables: {_slc.vars}. "
                "Supported variables: {'u', 'vx', 'vy', 'vz'}."
            )
        return out
