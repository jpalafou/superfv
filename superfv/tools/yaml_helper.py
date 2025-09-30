# superfv/tools/yaml_helper.py
from __future__ import annotations

from typing import IO, Any, List, Tuple, Union, cast

import yaml
from yaml.nodes import Node, SequenceNode

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
    return yaml.dump(obj, Dumper=TupleDumper, sort_keys=False, default_flow_style=None)


def yaml_load(src: Union[str, IO[str]]) -> Any:
    # Accept either a string or an open text file
    if hasattr(src, "read"):
        text = cast(IO[str], src).read()
    else:
        text = cast(str, src)
    return yaml.load(text, Loader=TupleLoader)
