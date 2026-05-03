from typing import Any, Dict

import torch
from torch import nn

from .base import BaseClassifier, register_model


@register_model("lstm")
class LSTMClassifier(BaseClassifier):
    """LSTM-based classifier for sequence classification."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
        vocab_size: int = 256,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L] long byte-ID tensor."""
        x = self.embed(x)

        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            final_h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_h = h_n[-1]

        return self.fc(final_h)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LSTMClassifier":
        model_cfg = config.get("model", {})
        return cls(
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            vocab_size=int(model_cfg.get("vocab_size", 257)),
            embed_dim=int(model_cfg.get("embed_dim", 32)),
        )
