"""
2-Stage trainer for the 2-Way ByteTCN.

Stage 1: Contrastive SSL pretraining with auxiliary IDS-structured tasks.
Stage 2: nnPU multitask training with optional SSL regulariser.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.common import augment_ids, to_device
from .base import BaseTrainer, EarlyStopper
from .losses import contrastive_nt_xent, nnpu_loss
from .metrics import pr_curve_best_f1
from ..models.tcn_2way import ByteTCN2WayClassifier, ByteTCNBackbone, Heads

logger = logging.getLogger(__name__)


# ─── Evaluation helpers ──────────────────────────────


@torch.no_grad()
def eval_on_loader(
    backbone: ByteTCNBackbone,
    heads: Heads,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    backbone.eval()
    heads.eval()
    all_scores: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []

    for batch in loader:
        batch = to_device(batch, device)
        z = backbone(batch)
        out = heads(z)
        all_scores.append(torch.sigmoid(out["risk_logit"]))
        all_y.append(batch["is_attack"])

    if not all_scores:
        return {
            "best_threshold": 0.5, "best_f1": 0.0, "pr_auc": 0.0,
            "precision_at_best": 0.0, "recall_at_best": 0.0,
        }

    return pr_curve_best_f1(torch.cat(all_scores), torch.cat(all_y))


# ─── Trainer ─────────────────────────────────────────


class TwoWayTrainer(BaseTrainer):
    """Manages the 2-stage pretrain → nnPU training loop."""

    def __init__(
        self,
        model: ByteTCN2WayClassifier,
        device: torch.device,
        training_cfg: Dict[str, Any],
    ) -> None:
        super().__init__(device, training_cfg)

        self.model = model.to(device)
        self.backbone = model.backbone
        self.heads = model.heads

        self.lr = float(training_cfg.get("lr", training_cfg.get("learning_rate", 3e-4)))
        self.weight_decay = float(training_cfg.get("weight_decay", 1e-4))
        self.max_epochs_pretrain = int(training_cfg.get("max_epochs_pretrain", 5))
        self.max_epochs_pu = int(training_cfg.get("max_epochs_pu", 10))

        self.w_ssl = float(training_cfg.get("w_ssl", 1.0))
        self.w_alert = float(training_cfg.get("w_alert", 0.2))
        self.w_pu = float(training_cfg.get("w_pu", 1.0))
        self.pi_p = float(training_cfg.get("pi_p", training_cfg.get("pu_prior", 0.10)))
        self.temp = float(training_cfg.get("temp", 0.2))

        self.all_params = list(self.model.parameters())
        self.sched_cfg = {"factor": 0.5, "patience": 3, "min_lr": 1e-6}

    # ── helpers ─────────────────────────────────────

    def _make_augmented_view(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return a copy of *batch* with augmented header/body ids."""
        h, hm = augment_ids(batch["header_ids"], batch["header_mask"])
        b, bm = augment_ids(batch["body_ids"], batch["body_mask"])
        view = dict(batch)
        view["header_ids"], view["header_mask"] = h, hm
        view["body_ids"], view["body_mask"] = b, bm
        return view

    def load_pretrained(self, path: Path) -> None:
        """Load backbone + heads weights from a pretrain checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.heads.load_state_dict(ckpt["heads"])
        logger.info("Loaded pretrained checkpoint from %s", path)

    def _reset_optimizer(self, use_adamw: bool = True) -> None:
        """Create a fresh optimizer + scheduler at full LR."""
        self.optim = self._init_optimizer(
            self.all_params, self.lr, self.weight_decay, use_adamw=use_adamw,
        )
        self.sched = self._init_scheduler(
            self.optim, self.sched_cfg, mode="max",
        )

    def _do_optim_step(self, loss: torch.Tensor) -> None:
        self._optim_step(loss, self.optim, self.all_params)

    # ── Stage 1: contrastive pretraining ────────────

    def pretrain(
        self,
        loader_u: DataLoader,
        loader_val: DataLoader,
        out_dir: Path,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, float]:
        logger.info(
            "== Stage 1: contrastive pretraining (%d epochs) ==",
            self.max_epochs_pretrain,
        )
        self._reset_optimizer()
        best_val: Dict[str, float] = {}

        for epoch in range(self.max_epochs_pretrain):
            if stop_flag and stop_flag():
                logger.info("Stop signal; exiting pretrain.")
                break

            self.backbone.train()
            self.heads.train()
            total_loss = 0.0
            n = 0

            for batch in loader_u:
                if stop_flag and stop_flag():
                    break
                batch = to_device(batch, self.device)

                view1 = self._make_augmented_view(batch)
                view2 = self._make_augmented_view(batch)

                z1 = self.backbone(view1)
                z2 = self.backbone(view2)
                out1 = self.heads(z1)
                out2 = self.heads(z2)

                loss_ssl = contrastive_nt_xent(
                    out1["proj"], out2["proj"], temperature=self.temp,
                )
                loss_alert = F.binary_cross_entropy_with_logits(
                    out1["alerted_logit"], batch["alerted"],
                )
                loss = self.w_ssl * loss_ssl + self.w_alert * loss_alert

                self._do_optim_step(loss)
                total_loss += float(loss.detach())
                n += 1

            val_stats = eval_on_loader(
                self.backbone, self.heads, loader_val, self.device,
            )
            # No scheduler step during pretraining — pr_auc is not the
            # pretraining objective; decaying LR based on it is harmful.
            best_val = val_stats
            lr = self.optim.param_groups[0]["lr"]
            logger.info(
                "[pretrain epoch %d] loss=%.4f val_pr_auc=%.4f bestF1=%.4f lr=%.2e",
                epoch, total_loss / max(1, n),
                val_stats["pr_auc"], val_stats["best_f1"], lr,
            )

            torch.save(
                {"backbone": self.backbone.state_dict(),
                 "heads": self.heads.state_dict()},
                out_dir / f"pretrain_epoch{epoch}.pt",
            )

        return best_val

    # ── Stage 2: nnPU multitask training ────────────

    def train_pu(
        self,
        loader_p: DataLoader,
        loader_u: DataLoader,
        loader_val: DataLoader,
        out_dir: Path,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, float], Path]:
        logger.info(
            "== Stage 2: nnPU training (%d epochs, π=%.3f) ==",
            self.max_epochs_pu, self.pi_p,
        )
        self._reset_optimizer()
        stopper = EarlyStopper(patience=self.patience, mode="max")
        best_path = out_dir / "model_best.pt"
        best_pr_auc = -1.0
        best_val: Dict[str, float] = {}

        for epoch in range(self.max_epochs_pu):
            if stop_flag and stop_flag():
                logger.info("Stop signal; exiting nnPU training.")
                break

            self.backbone.train()
            self.heads.train()
            total = 0.0
            n = 0

            iter_u = iter(loader_u)
            for batch_p in loader_p:
                if stop_flag and stop_flag():
                    break
                try:
                    batch_u = next(iter_u)
                except StopIteration:
                    iter_u = iter(loader_u)
                    batch_u = next(iter_u)

                batch_p = to_device(batch_p, self.device)
                batch_u = to_device(batch_u, self.device)

                z_p = self.backbone(batch_p)
                z_u = self.backbone(batch_u)
                out_p = self.heads(z_p)
                out_u = self.heads(z_u)

                # Main PU risk
                loss_pu, _ = nnpu_loss(
                    out_p["risk_logit"], out_u["risk_logit"], pi_p=self.pi_p,
                )

                # Auxiliary: predict alerted
                loss_alert = (
                    F.binary_cross_entropy_with_logits(
                        out_p["alerted_logit"], batch_p["alerted"],
                    )
                    + F.binary_cross_entropy_with_logits(
                        out_u["alerted_logit"], batch_u["alerted"],
                    )
                ) / 2.0

                # SSL regulariser on unlabeled
                view1 = self._make_augmented_view(batch_u)
                view2 = self._make_augmented_view(batch_u)
                z1 = self.backbone(view1)
                z2 = self.backbone(view2)
                ssl = contrastive_nt_xent(
                    self.heads(z1)["proj"], self.heads(z2)["proj"],
                    temperature=self.temp,
                )

                loss = (
                    self.w_pu * loss_pu
                    + self.w_alert * loss_alert
                    + 0.1 * self.w_ssl * ssl
                )

                self._do_optim_step(loss)
                total += float(loss.detach())
                n += 1

            val_stats = eval_on_loader(
                self.backbone, self.heads, loader_val, self.device,
            )
            pr_auc = val_stats["pr_auc"]
            self._step_scheduler(self.sched, pr_auc)
            lr = self.optim.param_groups[0]["lr"]
            logger.info(
                "[nnPU epoch %d] loss=%.4f val_pr_auc=%.4f bestF1=%.4f "
                "thr=%.3f lr=%.2e",
                epoch, total / max(1, n), pr_auc, val_stats["best_f1"],
                val_stats["best_threshold"], lr,
            )

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_val = val_stats
                torch.save(
                    {"backbone": self.backbone.state_dict(),
                     "heads": self.heads.state_dict(),
                     "best_val": val_stats},
                    best_path,
                )

            if stopper.step(pr_auc):
                logger.info("Early stopping triggered.")
                break

        logger.info("Best checkpoint: val_pr_auc=%.4f", best_pr_auc)
        torch.save(
            {"backbone": self.backbone.state_dict(),
             "heads": self.heads.state_dict()},
            out_dir / "model_last.pt",
        )
        return best_val, best_path

    # ── Abstract interface implementation ───────────

    def train(
        self,
        loaders: Dict[str, DataLoader],
        out_dir: Path,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, float], Path]:
        """Full 2-stage training: pretrain then nnPU."""
        self.pretrain(loaders["train_u"], loaders["val"], out_dir, stop_flag=stop_flag)
        return self.train_pu(
            loaders["train_p"], loaders["train_u"], loaders["val"],
            out_dir, stop_flag=stop_flag,
        )

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        return eval_on_loader(self.backbone, self.heads, loader, self.device)
