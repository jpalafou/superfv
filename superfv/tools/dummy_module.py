from dataclasses import dataclass


@dataclass
class DummyModule:
    def __getattr__(self, name):
        raise AttributeError(
            "DummyModule has no attributes and is meant to be used as a placeholder"
            " for pickling."
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass
