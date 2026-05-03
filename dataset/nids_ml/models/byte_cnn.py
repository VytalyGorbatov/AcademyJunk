from typing import Any, Dict, List

import torch
from torch import nn

from .base import BaseClassifier, register_model


@register_model("byte_cnn")
class ByteCNNClassifier(BaseClassifier):
    """CNN-based classifier for raw byte sequences."""

    output_mode = "binary"

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
        mlp_dims: List[int],
    ) -> None:
        super().__init__()
        if len(conv_channels) != len(kernel_sizes):
            raise ValueError("conv_channels and kernel_sizes must have same length")
        if len(conv_channels) != 3:
            raise ValueError("ByteCNNClassifier expects exactly 3 conv blocks")
        if len(mlp_dims) != 2:
            raise ValueError("ByteCNNClassifier expects 2 hidden MLP dimensions")

        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)

        layers: List[nn.Module] = []
        in_channels = embed_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(conv_channels[-1], mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] long byte-ID tensor."""
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x.view(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ByteCNNClassifier":
        model_cfg = config.get("model", {})
        return cls(
            vocab_size=int(model_cfg.get("vocab_size", 257)),
            embed_dim=int(model_cfg.get("embed_dim", 16)),
            conv_channels=model_cfg.get("conv_channels", [64, 128, 256]),
            kernel_sizes=model_cfg.get("kernel_sizes", [7, 5, 3]),
            dropout=float(model_cfg.get("dropout", 0.2)),
            mlp_dims=model_cfg.get("mlp_dims", [256, 64]),
        )
