from typing import Any, Dict, Type

from torch import nn


class BaseClassifier(nn.Module):
    """Base class for pluggable classifiers."""

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseClassifier":
        raise NotImplementedError


_MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {}


def register_model(name: str):
    def decorator(cls: Type[BaseClassifier]) -> Type[BaseClassifier]:
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator
