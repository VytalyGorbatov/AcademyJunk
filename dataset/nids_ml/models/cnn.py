from typing import Any, Dict, List

import torch
from torch import nn

from .base import BaseClassifier, register_model


@register_model("cnn")
class Conv1DClassifier(BaseClassifier):
    """1D CNN for byte-sequence classification with learned embedding."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have same length")

        self.embed = nn.Embedding(vocab_size, embed_dim)

        layers: List[nn.Module] = []
        prev_c = embed_dim
        for out_c, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(prev_c, out_c, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_c = out_c

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(prev_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] long byte-ID tensor."""
        x = self.embed(x)            # [B, L, E]
        x = x.transpose(1, 2)        # [B, E, L]
        x = self.features(x)         # [B, C_last, L']
        x = self.global_pool(x).squeeze(-1)  # [B, C_last]
        return self.classifier(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Conv1DClassifier":
        model_cfg = config.get("model", {})
        return cls(
            vocab_size=int(model_cfg.get("vocab_size", 257)),
            embed_dim=int(model_cfg.get("embed_dim", 16)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            conv_channels=model_cfg.get("conv_channels", [16, 32, 64]),
            kernel_sizes=model_cfg.get("kernel_sizes", [7, 5, 3]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
