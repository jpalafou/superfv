from typing import Union

import numpy as np
import rationalpy as rp

class Stencil:
    x: np.ndarray
    w: Union[np.ndarray, rp.RationalArray]
    rational: bool
    size: int

    def rescope(
        self, x: np.ndarray = None, h: int = 1, inplace: bool = True
    ) -> Union[None, "Stencil"]: ...
    def asnumpy(
        self, mode: str = "numerator", trim_leading_and_trailing_zeros: bool = True
    ) -> np.ndarray: ...
