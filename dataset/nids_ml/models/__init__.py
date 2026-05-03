"""Model registry and built-in classifiers."""

# Base must be imported first so the registry exists before models register themselves.
from .base import BaseClassifier, _MODEL_REGISTRY, register_model  # noqa: F401
from .cnn import Conv1DClassifier  # noqa: F401
from .lstm import LSTMClassifier  # noqa: F401
from .byte_cnn import ByteCNNClassifier  # noqa: F401
from .tcn import ByteTCNClassifier  # noqa: F401
from .tcn_2way import ByteTCNBackbone, ByteTCN2WayClassifier, Heads  # noqa: F401
from .factory import build_model  # noqa: F401
