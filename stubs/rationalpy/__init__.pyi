from typing import Any, List, Optional, Tuple, Union

import numpy as np

class RationalArray:
    numerator: np.ndarray
    denominator: np.ndarray

    def __add__(self, other: Any) -> "RationalArray": ...
    def __mul__(self, other: Any) -> "RationalArray": ...
    def __getitem__(self, index: Any) -> Any: ...

def rational_array(
    numerator: Union[int, List[int], Tuple[int, ...], np.ndarray],
    denominator: Optional[Union[int, List[int], Tuple[int, ...], np.ndarray]] = None,
) -> "RationalArray": ...
