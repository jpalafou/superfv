from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Union

import numpy as np


@dataclass
class VariableIndexMap:
    var_idx_map: Dict[str, int]
    group_var_map: Dict[str, List[str]]

    @property
    def var_names(self) -> Set[str]:
        return set(self.var_idx_map.keys())

    @property
    def group_names(self) -> Set[str]:
        return set(self.group_var_map.keys())

    @property
    def all_names(self) -> Set[str]:
        return self.var_names | self.group_names

    @property
    def idxs(self) -> List[int]:
        return sorted(set(self.var_idx_map.values()))

    @property
    def nvars(self) -> int:
        return len(self.idxs)

    def __post_init__(self):
        # check that no group names are also variable names
        if self.var_names & self.group_names:
            raise KeyError("Variables and groups cannot share names.")

        # check that no group contains a non-existent variable
        for group in self.group_var_map.keys():
            gen = self._retrieve_vars_from_group(group)
            _ = list(gen)

        # clear cache
        object.__setattr__(self, "_cache", {})

    def add_var(self, name: str, idx: int):
        if name in self.all_names:
            raise KeyError(f"Name '{name}' already exists.")
        self.var_idx_map[name] = idx
        self.__post_init__()

    def add_var_to_group(self, var_name: str, group_name: str):
        if group_name not in self.group_var_map:
            if group_name in self.var_idx_map:
                raise KeyError(f"Name '{group_name}' already exists.")
            self.group_var_map[group_name] = []
        self.group_var_map[group_name].append(var_name)
        self.__post_init__()

    def __call__(
        self, name: str, keepdims: bool = False
    ) -> Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]:
        cache_key = (name, keepdims)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute(name, keepdims)
        self._cache[cache_key] = result
        return result

    def _compute(
        self, name: str, keepdims: bool
    ) -> Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]:
        # Single variable
        if name in self.var_idx_map:
            idx = self.var_idx_map[name]
            return slice(idx, idx + 1) if keepdims else idx

        # Group of variables
        if name in self.group_var_map:
            if not self.group_var_map[name]:
                raise ValueError(f"Group '{name}' has no members.")
            idxs = sorted(set(self.var_idx_map[v] for v in self._retrieve_vars_from_group(name)))
            # Return a slice if contiguous, otherwise an index array
            if idxs == list(range(idxs[0], idxs[-1] + 1)):
                return slice(idxs[0], idxs[-1] + 1)
            return np.array(idxs)

        raise KeyError(f"Name '{name}' not found.")

    def _retrieve_vars_from_group(
        self, group_name: str, _visiting: Optional[set] = None
    ) -> Iterator[str]:
        if _visiting is None:
            _visiting = set()

        if group_name in _visiting:
            raise ValueError("Circular group reference detected.")

        for member in self.group_var_map[group_name]:
            if member in self.var_idx_map:
                yield member
            elif member in self.group_var_map:
                yield from self._retrieve_vars_from_group(member, _visiting | {group_name})
            else:
                raise KeyError(f"Member '{member}' not found as variable or group.")

    def __contains__(self, name: str) -> bool:
        return name in self.all_names
