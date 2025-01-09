from typing import Union

from stencilpal.stencil import Stencil

def conservative_interpolation_stencil(
    p: int, x: Union[int, float, str]
) -> Stencil: ...
def uniform_quadrature(p: int) -> Stencil: ...
