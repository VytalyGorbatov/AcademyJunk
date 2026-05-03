"""NIDS ML training package."""

# Backward-compatible re-exports so old `from nids_ml.X import ...` still works.
from .data import DatasetBuilder, TwoWayDatasetBuilder  # noqa: F401
from .pipelines import ClassifierPipeline, TwoWayPipeline  # noqa: F401
from .training import Trainer, TwoWayTrainer  # noqa: F401
