from typing import Any, Dict, List

import torch
from torch import nn

from .base import BaseClassifier, register_model


@register_model("cnn")
class Conv1DClassifier(BaseClassifier):
    """1D Convolutional Neural Network for sequence classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
        input_length: int,
    ) -> None:
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes)

        layers: List[nn.Module] = []
        prev_c = in_channels
        length = input_length

        for out_c, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(prev_c, out_c, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_c = out_c
            length = max(length // 2, 1)

        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(prev_c * length, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Conv1DClassifier":
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 1)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            conv_channels=model_cfg.get("conv_channels", [16, 32, 64]),
            kernel_sizes=model_cfg.get("kernel_sizes", [7, 5, 3]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            input_length=int(data_cfg.get("fixed_len", 1024)),
        )
