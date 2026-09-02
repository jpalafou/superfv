from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

Index = Union[int, slice, np.ndarray[Any, np.dtype[np.int_]]]


class VariableIndexMap:
    def __init__(
        self,
        var_idx_map: Optional[Dict[str, int]] = None,
        group_var_map: Optional[Dict[str, List[str]]] = None,
    ):
        self.var_idx_map: Dict[str, int] = {}
        self.group_var_map: Dict[str, List[str]] = {}
        self._cache: Dict[Tuple[str, bool], Index] = {}

        for var, idx in (var_idx_map or {}).items():
            self.var_idx_map[var] = idx

        for group, members in (group_var_map or {}).items():
            if not members:
                raise ValueError(f"Group '{group}' has no members.")
            self.group_var_map[group] = members

        self._validate_groups()

    def _validate_groups(self):
        for group_name in self.group_var_map:
            if group_name in self.var_idx_map:
                raise KeyError(f"Name '{group_name}' already exists as a variable.")
            for member in self.group_var_map[group_name]:
                if member not in self.var_idx_map and member not in self.group_var_map:
                    raise KeyError(f"Member '{member}' does not exist.")
        self._check_for_cycles()

    def _check_for_cycles(self):
        for group in self.group_var_map:
            self.flattened_var_names(group)

    def flattened_var_names(self, group_name: str) -> List[str]:
        if group_name in self.group_var_map:
            return list(self._iter_group_vars(group_name, set()))
        raise KeyError(f"Group name '{group_name}' not found.")

    def _iter_group_vars(self, group_name: str, visiting: Set[str]) -> Iterator[str]:
        if group_name not in self.group_var_map:
            raise KeyError(f"Group name '{group_name}' not found.")
        if group_name in visiting:
            raise ValueError("Circular group reference detected.")
        for member in self.group_var_map[group_name]:
            if member in self.var_idx_map:
                yield member
            else:
                yield from self._iter_group_vars(member, visiting | {group_name})

    def _reset_cache(self):
        self._cache.clear()

    @property
    def idxs(self) -> List[int]:
        return sorted(set(self.var_idx_map.values()))

    @property
    def nvars(self) -> int:
        return len(self.idxs)

    @property
    def is_contiguous(self) -> bool:
        if min(self.idxs, default=0) != 0:
            return False
        if self.idxs != list(range(self.nvars)):
            return False
        return True

    def add_var(self, name: str, idx: int):
        if name in self.var_idx_map:
            if idx == self.var_idx_map[name]:
                return
            else:
                raise KeyError(f"Variable '{name}' already exists with a different index.")
        if name in self.group_var_map:
            raise KeyError(f"Name '{name}' already exists as a group.")
        self.var_idx_map[name] = idx

        self._reset_cache()

    def add_member_to_group(self, member_name: str, group_name: str):
        self._add_member_to_group(member_name, group_name, recurse=True)

    def _add_member_to_group(self, member_name: str, group_name: str, recurse: bool = True):
        # test for circular references on a copy
        if recurse:
            test_copy = self.copy()
            test_copy._add_member_to_group(member_name, group_name, recurse=False)

        if group_name in self.group_var_map:
            if member_name in self.group_var_map[group_name]:
                return  # Redundant addition, do nothing
            self.group_var_map[group_name].append(member_name)
        else:
            self.group_var_map[group_name] = [member_name]

        self._validate_groups()
        self._reset_cache()

    def __call__(self, name: str, keepdims: bool = False) -> Index:
        cache_key = (name, keepdims)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute(name, keepdims)
        self._cache[cache_key] = result
        return result

    def _compute(self, name: str, keepdims: bool) -> Index:
        # Single variable
        if name in self.var_idx_map:
            idx = self.var_idx_map[name]
            return slice(idx, idx + 1) if keepdims else idx

        # Group of variables
        if name in self.group_var_map:
            if not self.group_var_map[name]:
                raise ValueError(f"Group '{name}' has no members.")
            idxs = sorted({self.var_idx_map[v] for v in self.flattened_var_names(name)})
            # Return a slice if contiguous, otherwise an index array
            if idxs == list(range(idxs[0], idxs[-1] + 1)):
                return slice(idxs[0], idxs[-1] + 1)
            return np.array(idxs, dtype=np.int_)

        raise KeyError(f"Name '{name}' not found.")

    def is_in_group(self, member_name: str, group_name: str) -> bool:
        return member_name in self.flattened_var_names(group_name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, VariableIndexMap):
            raise NotImplementedError(f"Comparison of VariableIndexMap with {type(other)}")
        return self.var_idx_map == other.var_idx_map and self.group_var_map == other.group_var_map

    def copy(self) -> "VariableIndexMap":
        new_map = VariableIndexMap()
        new_map.var_idx_map = self.var_idx_map.copy()
        new_map.group_var_map = {k: v.copy() for k, v in self.group_var_map.items()}
        return new_map
