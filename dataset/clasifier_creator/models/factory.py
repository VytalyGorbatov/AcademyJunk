import importlib
import logging
from typing import Any, Dict

from torch import nn

from .base import BaseClassifier, _MODEL_REGISTRY

logger = logging.getLogger(__name__)


def build_model(config: Dict[str, Any]) -> BaseClassifier:
    model_cfg = config.get("model", {})
    model_type = str(model_cfg.get("type", "cnn")).lower()
    class_path = model_cfg.get("class_path")

    if class_path:
        return _build_from_class_path(class_path, config)

    model_cls = _MODEL_REGISTRY.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model type '{model_type}'.")

    logger.info("Building model type '%s'", model_type)
    return model_cls.from_config(config)


def _build_from_class_path(class_path: str, config: Dict[str, Any]) -> BaseClassifier:
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ValueError("class_path must be in 'module.ClassName' format.")

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    if not issubclass(cls, nn.Module):
        raise TypeError("Custom classifier must inherit from torch.nn.Module.")

    if hasattr(cls, "from_config"):
        logger.info("Building model from class_path '%s' via from_config", class_path)
        return cls.from_config(config)

    init_args = config.get("model", {}).get("init_args", {})
    logger.info("Building model from class_path '%s' with init_args", class_path)
    return cls(**init_args)
