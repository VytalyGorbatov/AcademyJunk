import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Constants & utils
# =========================

PAD_IDX = 256             # special pad token (not a real byte)
VOCAB_SIZE = 257          # 0..255 bytes + PAD_IDX
SEP_BYTE = 0x1E           # record separator (decimal 30) between header and body

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# =========================
# Robust decoding of buffers
# =========================

def decode_buffers_field(x: Any) -> List[int]:
    """
    Robustly decode JSON field 'buffers' into list of ints in [0,255] plus possibly SEP.
    Supported:
      - list[int] already
      - list[float] in [0,1] (legacy normalized) -> round(x*255)
      - str (latin-1) -> bytes via encode('latin1')
      - dict with {"b64": "..."} or {"base64": "..."} (optional)
    """
    if isinstance(x, list):
        if len(x) == 0:
            return []
        if isinstance(x[0], int):
            return [int(v) & 0xFF for v in x]
        if isinstance(x[0], float):
            # NOTE: ambiguous padding if your padding used 0.0; prefer raw bytes in production
            return [int(round(float(v) * 255.0)) & 0xFF for v in x]
        raise TypeError(f"Unsupported list element type: {type(x[0])}")

    if isinstance(x, str):
        # Interpret as latin-1 transport of bytes
        b = x.encode("latin1", errors="ignore")
        return list(b)

    if isinstance(x, dict):
        import base64
        b64 = x.get("b64") or x.get("base64")
        if b64 is None:
            raise TypeError("Unsupported dict format for buffers")
        b = base64.b64decode(b64)
        return list(b)

    raise TypeError(f"Unsupported buffers field type: {type(x)}")


def pad_or_truncate(ids: List[int], fixed_len: int, pad_idx: int) -> List[int]:
    if len(ids) >= fixed_len:
        return ids[:fixed_len]
    return ids + [pad_idx] * (fixed_len - len(ids))


def split_header_body(
    ids: List[int],
    fixed_len: int,
    sep_byte: int = SEP_BYTE,
    pad_idx: int = PAD_IDX,
    fallback_header_len: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Split stream into header/body by separator byte.
    If separator missing, fallback to fixed header_len or equal split.
    Output sequences are padded/truncated to (header_len, body_len) summing to fixed_len.
    """
    if fallback_header_len is None:
        fallback_header_len = fixed_len // 2

    header_len = fallback_header_len
    body_len = fixed_len - header_len

    # normalize to fixed_len in stream space, then split
    ids_fixed = pad_or_truncate(ids, fixed_len, pad_idx)

    try:
        sep_pos = ids_fixed.index(sep_byte)
        header_raw = ids_fixed[:sep_pos]
        body_raw = ids_fixed[sep_pos + 1:]
    except ValueError:
        header_raw = ids_fixed[:header_len]
        body_raw = ids_fixed[header_len:]

    header_ids = pad_or_truncate(header_raw, header_len, pad_idx)
    body_ids = pad_or_truncate(body_raw, body_len, pad_idx)
    return header_ids, body_ids


# =========================
# Category mapping (optional)
# =========================

class CategoryMapper:
    def __init__(self) -> None:
        self.maps: Dict[str, Dict[Any, int]] = {}

    def fit(self, records: List[Dict[str, Any]], cat_fields: List[str]) -> None:
        for f in cat_fields:
            uniq = set()
            for r in records:
                if f in r:
                    uniq.add(r[f])
            # reserve 0 for "UNK"
            mapping = {"__UNK__": 0}
            for v in sorted(list(uniq), key=lambda x: str(x))[:100000]:
                mapping[v] = len(mapping)
            self.maps[f] = mapping

    def transform_one(self, r: Dict[str, Any], f: str) -> int:
        m = self.maps.get(f, {"__UNK__": 0})
        return int(m.get(r.get(f, "__UNK__"), 0))

    def cardinality(self, f: str) -> int:
        return len(self.maps.get(f, {"__UNK__": 0}))


# =========================
# Rule family derivation (example)
# =========================

def derive_rule_family(r: Dict[str, Any]) -> int:
    """
    Example: map SID to coarse family = sid // 1000.
    If missing, return -1.
    Adjust to your metadata (classtype, gid, etc.).
    """
    sid = r.get("sid")
    if sid is None:
        return -1
    try:
        sid_i = int(sid)
        return sid_i // 1000
    except Exception:
        return -1


# =========================
# Dataset
# =========================

@dataclass
class DataConfig:
    fixed_len: int = 1024
    fallback_header_len: int = 512
    cat_fields: Tuple[str, ...] = ("proto", "dir", "eth_type", "pkt_gen")
    num_fields: Tuple[str, ...] = ("pkt_len", "payload_len", "ttl", "sport", "dport", "tcp_flags")
    sep_byte: int = SEP_BYTE


class SipJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        cfg: DataConfig,
        cat_mapper: Optional[CategoryMapper] = None,
        for_training: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cat_mapper = cat_mapper
        self.for_training = for_training

        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]

        ids = decode_buffers_field(r["buffers"])
        header_ids, body_ids = split_header_body(
            ids,
            fixed_len=self.cfg.fixed_len,
            sep_byte=self.cfg.sep_byte,
            pad_idx=PAD_IDX,
            fallback_header_len=self.cfg.fallback_header_len,
        )

        alerted = int(r.get("alerted", 0))
        is_attack = int(r.get("is_attack", 0))

        # Telemetry
        cat = {}
        if self.cat_mapper is not None:
            for f in self.cfg.cat_fields:
                cat[f] = self.cat_mapper.transform_one(r, f)
        else:
            for f in self.cfg.cat_fields:
                cat[f] = int(r.get(f, 0))

        num = []
        for f in self.cfg.num_fields:
            num.append(float(r.get(f, 0.0)))

        rule_family = int(r.get("rule_family", derive_rule_family(r)))

        return {
            "header_ids": header_ids,
            "body_ids": body_ids,
            "alerted": alerted,
            "is_attack": is_attack,
            "rule_family": rule_family,
            "cat": cat,
            "num": num,
        }


def collate_fn(batch: List[Dict[str, Any]], cfg: DataConfig) -> Dict[str, torch.Tensor]:
    header = torch.tensor([b["header_ids"] for b in batch], dtype=torch.long)
    body = torch.tensor([b["body_ids"] for b in batch], dtype=torch.long)

    header_mask = header.ne(PAD_IDX)
    body_mask = body.ne(PAD_IDX)

    alerted = torch.tensor([b["alerted"] for b in batch], dtype=torch.float32)
    is_attack = torch.tensor([b["is_attack"] for b in batch], dtype=torch.float32)

    rule_family = torch.tensor([b["rule_family"] for b in batch], dtype=torch.long)

    # categorical fields -> dict of [B] tensors
    cat = {}
    for f in cfg.cat_fields:
        cat[f] = torch.tensor([b["cat"][f] for b in batch], dtype=torch.long)

    num = torch.tensor([b["num"] for b in batch], dtype=torch.float32)

    return {
        "header_ids": header,
        "body_ids": body,
        "header_mask": header_mask,
        "body_mask": body_mask,
        "alerted": alerted,
        "is_attack": is_attack,
        "rule_family": rule_family,
        "num": num,
        **{f"cat_{k}": v for k, v in cat.items()},
    }


# =========================
# ByteTCN backbone (minimal, derived from the earlier design)
# =========================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size, padding=padding,
                            dilation=dilation, groups=channels, bias=False)
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pw(self.dw(x)))


class ResTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.gn2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = nn.SiLU()
        self.c1 = DepthwiseSeparableConv1d(channels, kernel, dilation, dropout)
        self.c2 = DepthwiseSeparableConv1d(channels, kernel, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.c1(self.act(self.gn1(x)))
        x = self.c2(self.act(self.gn2(x)))
        return r + x


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.to(x.dtype)
    denom = m.sum(dim=dim).clamp_min(1.0)
    return (x * m).sum(dim=dim) / denom

def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    neg_inf = torch.finfo(x.dtype).min
    x2 = x.masked_fill(~mask, neg_inf)
    out = x2.max(dim=dim).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

class PrefixPooling(nn.Module):
    def __init__(self, prefix_lengths: Tuple[int, ...]) -> None:
        super().__init__()
        self.prefix_lengths = prefix_lengths

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: [B,C,L], valid_mask: [B,L]
        B, C, L = x.shape
        m = valid_mask.unsqueeze(1)  # [B,1,L]
        feats = []
        feats.append(masked_mean(x, m, dim=2))
        feats.append(masked_max(x, m, dim=2))

        pos = torch.arange(L, device=x.device).unsqueeze(0)  # [1,L]
        for k in self.prefix_lengths:
            kk = min(k, L)
            pm = (pos < kk) & valid_mask  # [B,L]
            pm = pm.unsqueeze(1)
            feats.append(masked_mean(x, pm, dim=2))
            feats.append(masked_max(x, pm, dim=2))
        return torch.cat(feats, dim=1)

class ByteEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        channels: int = 96,
        kernel: int = 5,
        dilations: Tuple[int, ...] = (1,2,4,8),
        dropout: float = 0.1,
        prefix_lengths: Tuple[int, ...] = (64,128,256),
        proj_dim: int = 192,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.stem = nn.Conv1d(embed_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([ResTCNBlock(channels, kernel, d, dropout) for d in dilations])
        self.pool = PrefixPooling(prefix_lengths)

        pooled_mul = 2 * (1 + len(prefix_lengths))
        pooled_dim = channels * pooled_mul
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)                  # [B,L,E]
        x = x.transpose(1,2)               # [B,E,L]
        x = self.stem(x)                   # [B,C,L]
        for b in self.blocks:
            x = b(x)
        pooled = self.pool(x, mask)        # [B, pooled_dim]
        return self.proj(pooled)           # [B, proj_dim]


class TelemetryEncoder(nn.Module):
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int, cat_emb_dim: int = 8, out_dim: int = 128):
        super().__init__()
        self.cat_names = list(cat_cardinalities.keys())
        self.embs = nn.ModuleDict({
            k: nn.Embedding(v, cat_emb_dim) for k, v in cat_cardinalities.items()
        })
        in_dim = len(self.cat_names) * cat_emb_dim + num_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
        )
        self.out_dim = out_dim

    def forward(self, cat: Dict[str, torch.Tensor], num: torch.Tensor) -> torch.Tensor:
        parts = []
        for k in self.cat_names:
            parts.append(self.embs[k](cat[k]))
        parts.append(num)
        x = torch.cat(parts, dim=1)
        return self.mlp(x)

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, target_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * target_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        g, b = self.lin(c).chunk(2, dim=1)
        return x * (1.0 + g) + b


class ByteTCNBackbone(nn.Module):
    """
    Returns a representation vector z for each sample.
    """
    def __init__(self, cat_cardinalities: Dict[str, int], num_dim: int) -> None:
        super().__init__()
        self.enc_h = ByteEncoder()
        self.enc_b = ByteEncoder()
        self.tel = TelemetryEncoder(cat_cardinalities, num_dim)

        self.film_h = FiLM(self.tel.out_dim, 192)
        self.film_b = FiLM(self.tel.out_dim, 192)

        fusion_in = 192 + 192 + self.tel.out_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.out_dim = 256

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cat = {k.replace("cat_", ""): v for k, v in batch.items() if k.startswith("cat_")}
        zt = self.tel(cat, batch["num"])
        zh = self.enc_h(batch["header_ids"], batch["header_mask"])
        zb = self.enc_b(batch["body_ids"], batch["body_mask"])
        zh = self.film_h(zh, zt)
        zb = self.film_b(zb, zt)
        z = torch.cat([zh, zb, zt], dim=1)
        return self.fusion(z)


# =========================
# Heads and losses (PU + auxiliary + contrastive)
# =========================

class Heads(nn.Module):
    def __init__(self, in_dim: int, num_rule_families: int) -> None:
        super().__init__()
        self.risk = nn.Linear(in_dim, 1)            # main PU risk head
        self.alerted = nn.Linear(in_dim, 1)         # auxiliary IDS teacher head
        self.rule = nn.Linear(in_dim, num_rule_families) if num_rule_families > 1 else None

        # projection head for contrastive SSL
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 128)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {
            "risk_logit": self.risk(z).squeeze(-1),
            "alerted_logit": self.alerted(z).squeeze(-1),
            "proj": F.normalize(self.proj(z), dim=1),
        }
        if self.rule is not None:
            out["rule_logits"] = self.rule(z)
        return out


def logistic_loss(logit: torch.Tensor, y_pm1: torch.Tensor) -> torch.Tensor:
    # y in {+1,-1}; loss = log(1+exp(-y * f(x))) = softplus(-y*f)
    return F.softplus(-y_pm1 * logit)

def nnpu_loss(
    logit_p: torch.Tensor,
    logit_u: torch.Tensor,
    pi_p: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    nnPU risk estimator (Kiryo et al., Eq. 6):
      R = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
    Using logistic surrogate:
      R_p^+ = E_P[softplus(-f)]
      R_p^- = E_P[softplus(+f)]
      R_u^- = E_U[softplus(+f)]
    """
    # positives P
    rp_pos = logistic_loss(logit_p, torch.ones_like(logit_p)).mean()
    rp_neg = logistic_loss(logit_p, -torch.ones_like(logit_p)).mean()

    # unlabeled U treated as y=-1 for the negative-part expectation
    ru_neg = logistic_loss(logit_u, -torch.ones_like(logit_u)).mean()

    pos_risk = pi_p * rp_pos
    neg_risk = ru_neg - pi_p * rp_neg
    total = pos_risk + torch.clamp(neg_risk, min=0.0)

    stats = {
        "pos_risk": float(pos_risk.detach()),
        "neg_risk_raw": float(neg_risk.detach()),
        "nnpu_risk": float(total.detach()),
    }
    return total, stats


def contrastive_nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent loss (SimCLR-style) over a batch.
    z1,z2: [B,D] normalized.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = (z @ z.t()) / temperature  # [2B,2B]
    sim = sim - torch.eye(2*B, device=z.device) * 1e9  # mask self similarity

    # positives: i <-> i+B
    targets = torch.arange(2*B, device=z.device)
    targets = (targets + B) % (2*B)

    loss = F.cross_entropy(sim, targets)
    return loss


# =========================
# Byte augmentations for SSL
# =========================

def augment_ids(ids: torch.Tensor, mask: torch.Tensor, p_drop: float = 0.05, p_span: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lightweight, protocol-tolerant augmentations for bytes:
      - random token dropout (replace with PAD)
      - random span masking
    """
    B, L = ids.shape
    out = ids.clone()
    out_mask = mask.clone()

    # token dropout
    drop = (torch.rand(B, L, device=ids.device) < p_drop) & out_mask
    out[drop] = PAD_IDX
    out_mask = out.ne(PAD_IDX)

    # span masking
    if p_span > 0:
        for b in range(B):
            if torch.rand(1).item() < p_span:
                # choose a span within valid region
                valid_positions = torch.where(out_mask[b])[0]
                if valid_positions.numel() < 8:
                    continue
                start = valid_positions[0].item()
                end = valid_positions[-1].item()
                span_len = int(min(32, max(8, (end - start) * 0.1)))
                s = random.randint(start, max(start, end - span_len))
                out[b, s:s+span_len] = PAD_IDX
        out_mask = out.ne(PAD_IDX)

    return out, out_mask


# =========================
# Metrics and threshold tuning on PR curve
# =========================

@torch.no_grad()
def pr_curve_best_f1(scores: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    Compute PR curve and best F1 threshold.
    scores: [N] in [0,1]
    y_true: [N] in {0,1}
    """
    scores = scores.detach().cpu()
    y_true = y_true.detach().cpu()

    # sort by descending score
    idx = torch.argsort(scores, descending=True)
    s = scores[idx]
    y = y_true[idx]

    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    fn = tp[-1] - tp

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)

    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    best_i = int(torch.argmax(f1).item())
    best_thr = float(s[best_i].item())
    best_f1 = float(f1[best_i].item())

    # PR-AUC (approx trapezoid in recall space)
    # ensure recall is non-decreasing
    r = recall
    p = precision
    pr_auc = float(torch.trapz(p, r).item())

    return {
        "best_threshold": best_thr,
        "best_f1": best_f1,
        "pr_auc": pr_auc,
        "precision_at_best": float(precision[best_i].item()),
        "recall_at_best": float(recall[best_i].item()),
    }


@torch.no_grad()
def eval_on_loader(
    backbone: nn.Module,
    heads: Heads,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    heads.eval()
    all_scores = []
    all_y = []

    for batch in loader:
        batch = to_device(batch, device)
        z = backbone(batch)
        out = heads(z)
        prob = torch.sigmoid(out["risk_logit"])
        all_scores.append(prob)
        all_y.append(batch["is_attack"])

    scores = torch.cat(all_scores, dim=0)
    y = torch.cat(all_y, dim=0)
    pr_stats = pr_curve_best_f1(scores, y)
    return pr_stats


# =========================
# Training loops
# =========================

@dataclass
class TrainConfig:
    seed: int = 7
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs_pretrain: int = 5
    max_epochs_pu: int = 10
    patience: int = 3
    clip_grad: float = 1.0

    # Loss weights
    w_ssl: float = 1.0
    w_alert: float = 0.2
    w_rule: float = 0.2
    w_pu: float = 1.0

    # PU prior (inside the filtered population, e.g., FP group)
    pi_p: float = 0.10

    # Contrastive temperature
    temp: float = 0.2


class EarlyStopper:
    def __init__(self, patience: int = 3, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad > self.patience


def train(
    train_p_path: str,
    train_u_path: str,
    val_path: str,
    cfg_data: DataConfig,
    cfg_train: TrainConfig,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg_train.seed)

    # Load records to build category mapping
    tmp_records = []
    for p in [train_p_path, train_u_path]:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tmp_records.append(json.loads(line))
    mapper = CategoryMapper()
    mapper.fit(tmp_records, list(cfg_data.cat_fields))

    cat_card = {f: mapper.cardinality(f) for f in cfg_data.cat_fields}
    num_dim = len(cfg_data.num_fields)

    ds_p = SipJsonlDataset(train_p_path, cfg_data, mapper, for_training=True)  # alerted=1
    ds_u = SipJsonlDataset(train_u_path, cfg_data, mapper, for_training=True)  # alerted=0 (unlabeled)
    ds_val = SipJsonlDataset(val_path, cfg_data, mapper, for_training=False)

    # derive number of rule families for head sizing
    fams = set()
    for r in ds_p.records:
        fams.add(int(r.get("rule_family", derive_rule_family(r))))
    fams.discard(-1)
    num_rule_families = max(fams) + 1 if fams else 1

    loader_p = DataLoader(ds_p, batch_size=cfg_train.batch_size, shuffle=True,
                          num_workers=cfg_train.num_workers, collate_fn=lambda b: collate_fn(b, cfg_data))
    loader_u = DataLoader(ds_u, batch_size=cfg_train.batch_size, shuffle=True,
                          num_workers=cfg_train.num_workers, collate_fn=lambda b: collate_fn(b, cfg_data))
    loader_val = DataLoader(ds_val, batch_size=cfg_train.batch_size, shuffle=False,
                            num_workers=cfg_train.num_workers, collate_fn=lambda b: collate_fn(b, cfg_data))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = ByteTCNBackbone(cat_card, num_dim).to(device)
    heads = Heads(backbone.out_dim, num_rule_families).to(device)

    optim = torch.optim.AdamW(list(backbone.parameters()) + list(heads.parameters()),
                              lr=cfg_train.lr, weight_decay=cfg_train.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=1, min_lr=1e-6)

    # -------------------------
    # Stage 1: Offline pretraining
    # -------------------------
    print("== Stage 1: pretraining ==")
    for epoch in range(cfg_train.max_epochs_pretrain):
        backbone.train(); heads.train()
        total_loss = 0.0
        n = 0

        # use unlabeled loader for SSL pretraining; can also mix with positives
        for batch in loader_u:
            batch = to_device(batch, device)
            # create two augmented views for contrastive
            h1, hm1 = augment_ids(batch["header_ids"], batch["header_mask"])
            b1, bm1 = augment_ids(batch["body_ids"], batch["body_mask"])
            view1 = dict(batch)
            view1["header_ids"], view1["header_mask"] = h1, hm1
            view1["body_ids"], view1["body_mask"] = b1, bm1

            h2, hm2 = augment_ids(batch["header_ids"], batch["header_mask"])
            b2, bm2 = augment_ids(batch["body_ids"], batch["body_mask"])
            view2 = dict(batch)
            view2["header_ids"], view2["header_mask"] = h2, hm2
            view2["body_ids"], view2["body_mask"] = b2, bm2

            z1 = backbone(view1)
            z2 = backbone(view2)
            out1 = heads(z1)
            out2 = heads(z2)

            loss_ssl = contrastive_nt_xent(out1["proj"], out2["proj"], temperature=cfg_train.temp)

            # IDS-structured auxiliary tasks (optional during pretraining)
            loss_alert = F.binary_cross_entropy_with_logits(out1["alerted_logit"], batch["alerted"])
            loss_rule = torch.tensor(0.0, device=device)
            if "rule_logits" in out1:
                mask_rule = (batch["alerted"] > 0.5) & (batch["rule_family"] >= 0)
                if mask_rule.any():
                    loss_rule = F.cross_entropy(out1["rule_logits"][mask_rule], batch["rule_family"][mask_rule])

            loss = cfg_train.w_ssl * loss_ssl + cfg_train.w_alert * loss_alert + cfg_train.w_rule * loss_rule

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(heads.parameters()), cfg_train.clip_grad)
            optim.step()

            total_loss += float(loss.detach())
            n += 1

        # Validate representation via risk head PR-AUC on val (research-only)
        val_stats = eval_on_loader(backbone, heads, loader_val, device)
        sched.step(val_stats["pr_auc"])
        print(f"[pretrain epoch {epoch}] loss={total_loss/max(1,n):.4f} val_pr_auc={val_stats['pr_auc']:.4f} bestF1={val_stats['best_f1']:.4f}")

        # checkpoint each epoch
        torch.save({
            "backbone": backbone.state_dict(),
            "heads": heads.state_dict(),
            "cat_card": cat_card,
            "num_rule_families": num_rule_families,
            "cfg_data": cfg_data.__dict__,
            "cfg_train": cfg_train.__dict__,
        }, os.path.join(out_dir, f"pretrain_epoch{epoch}.pt"))

    # -------------------------
    # Stage 2: Main nnPU multitask training
    # -------------------------
    print("== Stage 2: nnPU training ==")
    stopper = EarlyStopper(patience=cfg_train.patience, mode="max")
    best_path = os.path.join(out_dir, "best.pt")
    best_pr_auc = -1.0

    for epoch in range(cfg_train.max_epochs_pu):
        backbone.train(); heads.train()
        total = 0.0
        n = 0

        iter_u = iter(loader_u)
        for batch_p in loader_p:
            try:
                batch_u = next(iter_u)
            except StopIteration:
                iter_u = iter(loader_u)
                batch_u = next(iter_u)

            batch_p = to_device(batch_p, device)
            batch_u = to_device(batch_u, device)

            z_p = backbone(batch_p)
            z_u = backbone(batch_u)

            out_p = heads(z_p)
            out_u = heads(z_u)

            # Main PU risk (Kiryo nnPU)
            loss_pu, pu_stats = nnpu_loss(out_p["risk_logit"], out_u["risk_logit"], pi_p=cfg_train.pi_p)

            # Auxiliary: predict alerted (teacher semantics)
            loss_alert = (
                F.binary_cross_entropy_with_logits(out_p["alerted_logit"], batch_p["alerted"]) +
                F.binary_cross_entropy_with_logits(out_u["alerted_logit"], batch_u["alerted"])
            ) / 2.0

            # Auxiliary: rule family (only for alerted positives, typically in P)
            loss_rule = torch.tensor(0.0, device=device)
            if "rule_logits" in out_p:
                mask_rule = (batch_p["alerted"] > 0.5) & (batch_p["rule_family"] >= 0)
                if mask_rule.any():
                    loss_rule = F.cross_entropy(out_p["rule_logits"][mask_rule], batch_p["rule_family"][mask_rule])

            # Optional SSL regularizer: contrastive on unlabeled batch
            # (keeps representations stable as PU head learns)
            h1, hm1 = augment_ids(batch_u["header_ids"], batch_u["header_mask"])
            b1, bm1 = augment_ids(batch_u["body_ids"], batch_u["body_mask"])
            view1 = dict(batch_u); view1["header_ids"], view1["header_mask"] = h1, hm1; view1["body_ids"], view1["body_mask"] = b1, bm1
            h2, hm2 = augment_ids(batch_u["header_ids"], batch_u["header_mask"])
            b2, bm2 = augment_ids(batch_u["body_ids"], batch_u["body_mask"])
            view2 = dict(batch_u); view2["header_ids"], view2["header_mask"] = h2, hm2; view2["body_ids"], view2["body_mask"] = b2, bm2
            z1 = backbone(view1); z2 = backbone(view2)
            ssl = contrastive_nt_xent(heads(z1)["proj"], heads(z2)["proj"], temperature=cfg_train.temp)

            loss = cfg_train.w_pu * loss_pu + cfg_train.w_alert * loss_alert + cfg_train.w_rule * loss_rule + 0.1 * cfg_train.w_ssl * ssl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(heads.parameters()), cfg_train.clip_grad)
            optim.step()

            total += float(loss.detach())
            n += 1

        val_stats = eval_on_loader(backbone, heads, loader_val, device)
        pr_auc = val_stats["pr_auc"]
        sched.step(pr_auc)

        print(
            f"[nnPU epoch {epoch}] loss={total/max(1,n):.4f} "
            f"val_pr_auc={pr_auc:.4f} bestF1={val_stats['best_f1']:.4f} "
            f"thr={val_stats['best_threshold']:.3f}"
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save({
                "backbone": backbone.state_dict(),
                "heads": heads.state_dict(),
                "best_val": val_stats,
                "cat_card": cat_card,
                "num_rule_families": num_rule_families,
            }, best_path)

        if stopper.step(pr_auc):
            print("Early stopping triggered.")
            break

    print(f"Best checkpoint saved at: {best_path} (val_pr_auc={best_pr_auc:.4f})")