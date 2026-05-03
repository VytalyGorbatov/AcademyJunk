from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Config
# =========================

@dataclass
class ByteTCNFusionNetConfig:
    # Byte / sequence
    vocab_size: int = 257          # 256 bytes + PAD
    pad_idx: int = 256
    byte_embed_dim: int = 32

    # TCN
    tcn_channels: int = 96
    tcn_kernel_size: int = 5
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    tcn_dropout: float = 0.10
    norm_groups: int = 8

    # Prefix pooling
    prefix_lengths: Tuple[int, ...] = (128, 256, 512)

    # Encoded vector sizes
    buffer_proj_dim: int = 192
    telemetry_hidden_dim: int = 128
    fusion_hidden_dim: int = 256

    # Telemetry
    numeric_dim: int = 0
    categorical_cardinalities: Optional[Dict[str, int]] = None
    categorical_embed_dim: int = 8

    # Model behavior
    share_byte_encoder: bool = False
    use_film: bool = True

    # Heads
    num_classes: Optional[int] = None   # optional multi-class auxiliary head


# =========================
# Utility functions
# =========================

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [B, C, L] or compatible
    mask: [B, 1, L] or broadcastable boolean mask (True = valid)
    """
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return (x * mask_f).sum(dim=dim) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    mask=False positions are ignored.
    """
    neg_inf = torch.finfo(x.dtype).min
    x_masked = x.masked_fill(~mask, neg_inf)
    out = x_masked.max(dim=dim).values
    # In case every element is masked, replace -inf with zero.
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def make_prefix_mask(base_mask: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    base_mask: [B, L] bool
    returns: [B, L] bool masked to first prefix_len positions
    """
    bsz, seq_len = base_mask.shape
    device = base_mask.device
    prefix_positions = torch.arange(seq_len, device=device).unsqueeze(0) < prefix_len
    return base_mask & prefix_positions


# =========================
# Building blocks
# =========================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.dropout(x)
        return x


class ResTCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = min(norm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = DepthwiseSeparableConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout=dropout,
        )

        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = DepthwiseSeparableConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=1,
            dropout=dropout,
        )

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)

        return residual + out


class PrefixPooling(nn.Module):
    """
    Produces:
    [global_avg, global_max, prefix1_avg, prefix1_max, ..., prefixN_avg, prefixN_max]
    Each item has shape [B, C]
    Final output shape: [B, C * 2 * (1 + len(prefix_lengths))]
    """
    def __init__(self, prefix_lengths: Sequence[int]) -> None:
        super().__init__()
        self.prefix_lengths = tuple(prefix_lengths)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        valid_mask: [B, L] bool, True = valid token
        """
        if x.ndim != 3:
            raise ValueError(f"x must be [B, C, L], got {tuple(x.shape)}")
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [B, L], got {tuple(valid_mask.shape)}")

        mask_3d = valid_mask.unsqueeze(1)  # [B, 1, L]

        pooled: List[torch.Tensor] = []

        # Global pools
        pooled.append(masked_mean(x, mask_3d, dim=2))
        pooled.append(masked_max(x, mask_3d, dim=2))

        # Prefix pools
        seq_len = x.size(-1)
        for k in self.prefix_lengths:
            k_eff = min(k, seq_len)
            p_mask = make_prefix_mask(valid_mask, k_eff).unsqueeze(1)  # [B,1,L]
            pooled.append(masked_mean(x, p_mask, dim=2))
            pooled.append(masked_max(x, p_mask, dim=2))

        return torch.cat(pooled, dim=1)


class ByteEncoder(nn.Module):
    """
    Byte encoder:
    bytes -> embedding -> stem conv -> ResTCN stack -> prefix/global pooling -> projection
    """
    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.byte_embed_dim,
            padding_idx=cfg.pad_idx,
        )

        self.stem = nn.Conv1d(
            in_channels=cfg.byte_embed_dim,
            out_channels=cfg.tcn_channels,
            kernel_size=1,
            bias=False,
        )

        self.blocks = nn.ModuleList([
            ResTCNBlock(
                channels=cfg.tcn_channels,
                kernel_size=cfg.tcn_kernel_size,
                dilation=d,
                dropout=cfg.tcn_dropout,
                norm_groups=cfg.norm_groups,
            )
            for d in cfg.tcn_dilations
        ])

        self.pool = PrefixPooling(cfg.prefix_lengths)

        pooled_multiplier = 2 * (1 + len(cfg.prefix_lengths))
        pooled_dim = cfg.tcn_channels * pooled_multiplier

        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, cfg.buffer_proj_dim),
            nn.SiLU(),
            nn.Dropout(cfg.tcn_dropout),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        byte_ids: [B, L] long
        valid_mask: [B, L] bool, optional. If None -> byte_ids != pad_idx
        returns: [B, buffer_proj_dim]
        """
        if byte_ids.ndim != 2:
            raise ValueError(f"byte_ids must be [B, L], got {tuple(byte_ids.shape)}")
        if byte_ids.dtype != torch.long:
            raise TypeError(f"byte_ids must be torch.long, got {byte_ids.dtype}")

        if valid_mask is None:
            valid_mask = byte_ids.ne(self.cfg.pad_idx)
        else:
            if valid_mask.shape != byte_ids.shape:
                raise ValueError("valid_mask shape must match byte_ids shape")
            valid_mask = valid_mask.bool()

        # [B, L, E]
        x = self.embedding(byte_ids)

        # [B, E, L]
        x = x.transpose(1, 2)
        x = self.stem(x)

        for block in self.blocks:
            x = block(x)

        pooled = self.pool(x, valid_mask)   # [B, pooled_dim]
        return self.proj(pooled)


class TelemetryEncoder(nn.Module):
    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        card = cfg.categorical_cardinalities or {}
        self.cat_names = list(card.keys())

        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=cardinality, embedding_dim=cfg.categorical_embed_dim)
            for name, cardinality in card.items()
        })

        cat_total_dim = len(self.cat_names) * cfg.categorical_embed_dim
        num_dim = cfg.numeric_dim
        in_dim = cat_total_dim + num_dim

        if in_dim == 0:
            self.encoder = None
            self.out_dim = 0
        else:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, cfg.telemetry_hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.10),
                nn.Linear(cfg.telemetry_hidden_dim, cfg.telemetry_hidden_dim),
                nn.SiLU(),
            )
            self.out_dim = cfg.telemetry_hidden_dim

    def forward(
        self,
        cat_features: Optional[Dict[str, torch.Tensor]] = None,
        num_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        cat_features: dict[name] -> [B] long
        num_features: [B, numeric_dim] float
        """
        if self.encoder is None:
            # Return empty tensor with proper batch size inference
            if num_features is not None:
                bsz = num_features.size(0)
                device = num_features.device
            elif cat_features:
                first = next(iter(cat_features.values()))
                bsz = first.size(0)
                device = first.device
            else:
                raise ValueError("TelemetryEncoder received no features and has zero configured input dims")
            return torch.zeros(bsz, 0, device=device)

        parts: List[torch.Tensor] = []

        if self.cat_names:
            if cat_features is None:
                raise ValueError(f"Expected categorical features: {self.cat_names}")
            for name in self.cat_names:
                if name not in cat_features:
                    raise KeyError(f"Missing categorical feature '{name}'")
                x = cat_features[name]
                if x.ndim != 1:
                    raise ValueError(f"Categorical feature '{name}' must have shape [B], got {tuple(x.shape)}")
                parts.append(self.cat_embeddings[name](x.long()))

        if self.cfg.numeric_dim > 0:
            if num_features is None:
                raise ValueError(f"Expected numeric features with dim={self.cfg.numeric_dim}")
            if num_features.ndim != 2 or num_features.size(1) != self.cfg.numeric_dim:
                raise ValueError(
                    f"num_features must be [B, {self.cfg.numeric_dim}], got {tuple(num_features.shape)}"
                )
            parts.append(num_features)

        x = torch.cat(parts, dim=1)
        return self.encoder(x)


class FiLMGating(nn.Module):
    """
    Telemetry-conditioned affine modulation for buffer vectors:
    out = x * (1 + gamma) + beta
    """
    def __init__(self, cond_dim: int, target_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, target_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.size(1) == 0:
            return x
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


# =========================
# Main model
# =========================

class ByteTCNFusionNet(nn.Module):
    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.header_encoder = ByteEncoder(cfg)
        self.body_encoder = self.header_encoder if cfg.share_byte_encoder else ByteEncoder(cfg)

        self.telemetry_encoder = TelemetryEncoder(cfg)

        if cfg.use_film and self.telemetry_encoder.out_dim > 0:
            self.header_film = FiLMGating(self.telemetry_encoder.out_dim, cfg.buffer_proj_dim)
            self.body_film = FiLMGating(self.telemetry_encoder.out_dim, cfg.buffer_proj_dim)
        else:
            self.header_film = None
            self.body_film = None

        fusion_in_dim = cfg.buffer_proj_dim * 2 + self.telemetry_encoder.out_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, cfg.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
        )

        self.risk_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.snort_alert_head = nn.Linear(cfg.fusion_hidden_dim, 1)

        self.class_head = None
        if cfg.num_classes is not None and cfg.num_classes > 1:
            self.class_head = nn.Linear(cfg.fusion_hidden_dim, cfg.num_classes)

    def forward(
        self,
        header_bytes: torch.Tensor,
        body_bytes: torch.Tensor,
        *,
        header_mask: Optional[torch.Tensor] = None,
        body_mask: Optional[torch.Tensor] = None,
        cat_features: Optional[Dict[str, torch.Tensor]] = None,
        num_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        header_bytes: [B, Lh] long
        body_bytes:   [B, Lb] long

        Returns dict:
            risk_logit: [B]
            risk_prob: [B]
            snort_alert_logit: [B]
            snort_alert_prob: [B]
            fused: [B, H]
            header_vec: [B, D]
            body_vec: [B, D]
            telemetry_vec: [B, T]
            class_logits: [B, C]  (optional)
        """
        header_vec = self.header_encoder(header_bytes, valid_mask=header_mask)
        body_vec = self.body_encoder(body_bytes, valid_mask=body_mask)

        telemetry_vec = self.telemetry_encoder(
            cat_features=cat_features,
            num_features=num_features,
        )

        if self.header_film is not None:
            header_vec = self.header_film(header_vec, telemetry_vec)
        if self.body_film is not None:
            body_vec = self.body_film(body_vec, telemetry_vec)

        fused_in = torch.cat([header_vec, body_vec, telemetry_vec], dim=1)
        fused = self.fusion(fused_in)

        risk_logit = self.risk_head(fused).squeeze(-1)
        snort_alert_logit = self.snort_alert_head(fused).squeeze(-1)

        out = {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "snort_alert_logit": snort_alert_logit,
            "snort_alert_prob": torch.sigmoid(snort_alert_logit),
            "fused": fused,
            "header_vec": header_vec,
            "body_vec": body_vec,
            "telemetry_vec": telemetry_vec,
        }

        if self.class_head is not None:
            out["class_logits"] = self.class_head(fused)

        return out


# =========================
# Example loss
# =========================

def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    *,
    risk_target: torch.Tensor,
    snort_alert_target: Optional[torch.Tensor] = None,
    class_target: Optional[torch.Tensor] = None,
    risk_weight: float = 1.0,
    snort_weight: float = 0.25,
    class_weight: float = 0.25,
    pos_weight_risk: Optional[torch.Tensor] = None,
    pos_weight_snort: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    risk_target: [B] float {0,1}
    snort_alert_target: [B] float {0,1}
    class_target: [B] long
    """
    total = torch.tensor(0.0, device=outputs["risk_logit"].device)
    metrics: Dict[str, float] = {}

    risk_loss = F.binary_cross_entropy_with_logits(
        outputs["risk_logit"],
        risk_target.float(),
        pos_weight=pos_weight_risk,
    )
    total = total + risk_weight * risk_loss
    metrics["risk_loss"] = float(risk_loss.detach())

    if snort_alert_target is not None:
        snort_loss = F.binary_cross_entropy_with_logits(
            outputs["snort_alert_logit"],
            snort_alert_target.float(),
            pos_weight=pos_weight_snort,
        )
        total = total + snort_weight * snort_loss
        metrics["snort_alert_loss"] = float(snort_loss.detach())

    if class_target is not None and "class_logits" in outputs:
        class_loss = F.cross_entropy(outputs["class_logits"], class_target.long())
        total = total + class_weight * class_loss
        metrics["class_loss"] = float(class_loss.detach())

    metrics["total_loss"] = float(total.detach())
    return total, metrics


# =========================
# Minimal usage example
# =========================

if __name__ == "__main__":
    torch.manual_seed(7)

    cfg = ByteTCNFusionNetConfig(
        vocab_size=257,
        pad_idx=256,
        byte_embed_dim=32,
        tcn_channels=96,
        tcn_kernel_size=5,
        tcn_dilations=(1, 2, 4, 8),
        prefix_lengths=(128, 256, 512),
        buffer_proj_dim=192,
        telemetry_hidden_dim=128,
        fusion_hidden_dim=256,
        numeric_dim=6,
        categorical_cardinalities={
            "proto": 8,
            "dir": 4,
            "eth_type": 16,
            "pkt_gen": 32,
        },
        categorical_embed_dim=8,
        share_byte_encoder=False,
        use_film=True,
        num_classes=5,
    )

    model = ByteTCNFusionNet(cfg)

    B = 4
    Lh = 640
    Lb = 768

    # Random demo tensors
    header_bytes = torch.randint(0, 256, (B, Lh), dtype=torch.long)
    body_bytes = torch.randint(0, 256, (B, Lb), dtype=torch.long)

    # Simulate padding in tails
    header_bytes[:, -50:] = cfg.pad_idx
    body_bytes[:, -120:] = cfg.pad_idx

    cat_features = {
        "proto": torch.randint(0, 8, (B,), dtype=torch.long),
        "dir": torch.randint(0, 4, (B,), dtype=torch.long),
        "eth_type": torch.randint(0, 16, (B,), dtype=torch.long),
        "pkt_gen": torch.randint(0, 32, (B,), dtype=torch.long),
    }
    num_features = torch.randn(B, cfg.numeric_dim)

    outputs = model(
        header_bytes=header_bytes,
        body_bytes=body_bytes,
        cat_features=cat_features,
        num_features=num_features,
    )

    risk_target = torch.randint(0, 2, (B,), dtype=torch.float32)
    snort_target = torch.randint(0, 2, (B,), dtype=torch.float32)
    class_target = torch.randint(0, cfg.num_classes, (B,), dtype=torch.long)

    loss, metrics = compute_multitask_loss(
        outputs,
        risk_target=risk_target,
        snort_alert_target=snort_target,
        class_target=class_target,
        risk_weight=1.0,
        snort_weight=0.2,
        class_weight=0.2,
    )

    print("risk_logit shape:", outputs["risk_logit"].shape)
    print("class_logits shape:", outputs["class_logits"].shape)
    print("loss:", float(loss))
    print("metrics:", metrics)