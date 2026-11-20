"""Load generators only when requested; data dependencies stay out of training."""
from importlib import import_module
from types import ModuleType

REGISTRY: dict[str, ModuleType] = {}
_NAMES = {"chart", "music", "chem", "statics", "circuit", "graph", "graph_pair",
          "materials", "thermo", "pk_lab", "bio", "maps"}


def get_generator(name: str) -> ModuleType:
    if name not in _NAMES:
        raise ValueError(f"unknown notebook RL generator: {name}")
    if name not in REGISTRY:
        module = import_module(f"{__name__}.{name}")
        if not all(hasattr(module, key) for key in (
            "GENERATOR_VERSION", "FAMILY", "OBSERVATION_INSTRUCTION", "generate", "self_check"
        )):
            raise RuntimeError(f"incomplete generator contract: {name}")
        REGISTRY[name] = module
    return REGISTRY[name]
