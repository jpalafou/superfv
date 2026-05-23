# superfv/tools/yaml_helper.py
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial
from pathlib import PurePath
from typing import IO, Any, List, Tuple, Union, cast

import numpy as np
import yaml
from yaml.nodes import Node, SequenceNode

from superfv.tools.variable_index_map import VariableIndexMap

TUP_TAG = "!tuple"


class TupleDumper(yaml.SafeDumper):
    # don't write anchors/aliases like &id001/*id001
    def ignore_aliases(self, data: Any) -> bool:  # type: ignore[override]
        return True


class TupleLoader(yaml.SafeLoader):
    pass


# --- Representers with precise types the stubs expect ---


def _repr_flow_list(dumper: TupleDumper, value: List[Any]) -> Node:
    # lists inline: [a, b, c]
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


def _repr_tuple(dumper: TupleDumper, value: Tuple[Any, ...]) -> Node:
    # tuples inline with custom tag: !tuple [a, b]
    return dumper.represent_sequence(TUP_TAG, list(value), flow_style=True)


def _serialize_function(data: Any) -> str:
    return f"{data.__module__}.{data.__qualname__}"


def _serialize_partial(data: partial) -> str:
    func_str = _serialize_function(data.func)
    args_str = ", ".join(repr(arg) for arg in data.args)
    kwargs = data.keywords or {}
    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    all_args = ", ".join(filter(None, [func_str, args_str, kwargs_str]))
    return f"partial({all_args})"


def _prepare_for_yaml(data: Any) -> Any:
    if is_dataclass(data) and not isinstance(data, type):
        if isinstance(data, VariableIndexMap):
            return {
                "var_idx_map": _prepare_for_yaml(data.var_idx_map),
                "group_var_map": _prepare_for_yaml(data.group_var_map),
            }
        return {
            field.name: _prepare_for_yaml(getattr(data, field.name))
            for field in fields(data)
            if not field.name.startswith("_")
        }
    if isinstance(data, dict):
        return {_prepare_for_yaml(key): _prepare_for_yaml(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_prepare_for_yaml(item) for item in data]
    if isinstance(data, tuple):
        return tuple(_prepare_for_yaml(item) for item in data)
    if isinstance(data, Enum):
        return data.name
    if isinstance(data, partial):
        return _serialize_partial(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.generic):
        return data.item()
    if isinstance(data, PurePath):
        return str(data)
    if callable(data) and hasattr(data, "__module__") and hasattr(data, "__qualname__"):
        return _serialize_function(data)
    return data


TupleDumper.add_representer(list, _repr_flow_list)
TupleDumper.add_representer(tuple, _repr_tuple)

# --- Constructor for !tuple with precise node type ---


def _construct_tuple(loader: TupleLoader, node: SequenceNode) -> Tuple[Any, ...]:
    seq = loader.construct_sequence(node)
    return tuple(seq)


TupleLoader.add_constructor(TUP_TAG, _construct_tuple)

# --- Public API ---


def yaml_dump(obj: Any) -> str:
    # dicts stay block style; sequences are forced flow by representers
    return yaml.dump(
        _prepare_for_yaml(obj),
        Dumper=TupleDumper,
        sort_keys=False,
        default_flow_style=False,
    )


def yaml_load(src: Union[str, IO[str]]) -> Any:
    # Accept either a string or an open text file
    if hasattr(src, "read"):
        text = cast(IO[str], src).read()
    else:
        text = cast(str, src)
    return yaml.load(text, Loader=TupleLoader)
