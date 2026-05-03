import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import BaseTrainer, EarlyStopper
from .metrics import MetricUtils
from ..local_types import Metrics
from .losses import PULoss
from ..data.common import PAD_IDX

logger = logging.getLogger(__name__)


def _is_dict_model(model: nn.Module) -> bool:
    """Return True if the model's forward() expects a Dict[str, Tensor] batch."""
    return hasattr(model, "tcn")  # ByteTCNClassifier wraps ByteTCNFusionNet


class Trainer(BaseTrainer):
    """Single-stage supervised / nnPU training loop."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        class_weights: Optional[torch.Tensor] = None,
        output_mode: str = "multiclass",
        threshold: float = 0.5,
        patience: int = 0,
        lr_scheduler_cfg: Optional[Dict[str, Any]] = None,
        pu_prior: Optional[float] = None,
    ) -> None:
        # Build a training_cfg dict for the base class
        training_cfg: Dict[str, Any] = {
            "clip_grad": 1.0,
            "patience": patience,
        }
        super().__init__(device, training_cfg)

        self.model = model.to(device)
        self.output_mode = output_mode
        self.threshold = threshold
        self.pu_mode = pu_prior is not None
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if self.pu_mode and self.output_mode == "binary":
            logger.info(
                "Using nnPU loss with class prior π=%.3f "
                "(Positive-Unlabeled learning mode)",
                pu_prior,
            )
            self.criterion = PULoss(prior=pu_prior, nnpu=True)
        elif self.output_mode == "binary":
            pos_weight = None
            if class_weights is not None:
                if class_weights.numel() >= 2:
                    denom = float(max(class_weights[0].item(), 1e-12))
                    pos_weight = class_weights[1] / denom
                else:
                    pos_weight = class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = self._init_optimizer(
            self.model.parameters(), learning_rate, weight_decay,
        )
        self.scheduler = self._init_scheduler(
            self.optimizer, lr_scheduler_cfg, mode="max",
        )

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[float, Metrics, bool]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        all_losses: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        stopped_early = False
        dict_mode = _is_dict_model(self.model)

        with torch.set_grad_enabled(train):
            for batch in loader:
                if stop_flag and stop_flag():
                    stopped_early = True
                    break

                # ── unpack batch ────────────────────────
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    yb = batch.get("alerted", batch.get("is_attack"))

                    if dict_mode:
                        logits = self.model(batch)
                    else:
                        ids = torch.cat(
                            [batch["header_ids"], batch["body_ids"]], dim=1,
                        )
                        logits = self.model(ids)
                else:
                    # Legacy tuple batch (backward compat)
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        xb, yb = batch[0], batch[1]
                    else:
                        xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)

                # ── loss + backward ─────────────────────
                if self.output_mode == "binary":
                    logits = logits.view(-1)
                    loss = self.criterion(logits, yb.float())
                else:
                    loss = self.criterion(logits, yb.long())

                if train:
                    self._optim_step(loss, self.optimizer, self.model.parameters())

                all_losses.append(loss.item())
                if self.output_mode == "binary":
                    probs = torch.sigmoid(logits)
                    preds = (probs >= self.threshold).long()
                    all_true.extend(yb.long().tolist())
                    all_pred.extend(preds.tolist())
                else:
                    preds = torch.argmax(logits, dim=1)
                    all_true.extend(yb.tolist())
                    all_pred.extend(preds.tolist())

        avg_loss = float(np.mean(all_losses)) if all_losses else 0.0

        if not all_true:
            return avg_loss, {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "loss": avg_loss,
            }, stopped_early

        metrics = MetricUtils.compute_binary_metrics(
            np.array(all_true), np.array(all_pred)
        )
        metrics["loss"] = avg_loss
        return avg_loss, metrics, stopped_early

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        best_metric_name: str,
        out_dir: Path,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, List[Metrics]], Metrics, Optional[Dict[str, Any]]]:
        best_metric = -float("inf")
        best_state: Optional[Dict[str, Any]] = None
        last_state: Optional[Dict[str, Any]] = None
        history: Dict[str, List[Metrics]] = {"train": [], "val": []}
        stopper = EarlyStopper(patience=self.patience, mode="max") if self.patience > 0 else None

        logger.info("Starting training for %s epochs on %s...", epochs, self.device)

        for epoch in range(1, epochs + 1):
            if stop_flag and stop_flag():
                logger.info("Stop signal detected before epoch %s; exiting training.", epoch)
                break

            train_loss, train_metrics, stopped_train = self._run_epoch(
                train_loader, train=True, stop_flag=stop_flag
            )
            if stopped_train:
                logger.info("Stop signal detected during train epoch %s; exiting training.", epoch)
                break

            _, val_metrics, stopped_val = self._run_epoch(
                val_loader, train=False, stop_flag=stop_flag
            )
            if stopped_val:
                logger.info("Stop signal detected during val epoch %s; exiting training.", epoch)
                break

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            last_state = {
                "model_state_dict": copy.deepcopy(self.model.state_dict()),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }

            metric_val = float(val_metrics.get(best_metric_name, 0.0))

            self._step_scheduler(self.scheduler, metric_val)

            if metric_val > best_metric:
                best_metric = metric_val
                best_state = {
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }

            current_lr = self.optimizer.param_groups[0]["lr"]
            bad_epochs = stopper.bad if stopper else 0
            logger.info(
                "Epoch %s/%s | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f "
                "| Val F1: %.4f | LR: %.2e | No-improve: %s/%s",
                epoch,
                epochs,
                train_loss,
                val_metrics.get("loss", 0.0),
                val_metrics.get("accuracy", 0.0),
                val_metrics.get("f1", 0.0),
                current_lr,
                bad_epochs,
                self.patience if self.patience > 0 else "off",
            )

            if stopper and stopper.step(metric_val):
                logger.info(
                    "Early stopping triggered: no improvement for %s epochs.",
                    self.patience,
                )
                break

        if best_state:
            torch.save(best_state, out_dir / "model_best.pt")
            logger.info("Best model saved with %s=%.4f", best_metric_name, best_metric)
            return history, best_state["val_metrics"], last_state

        return history, {}, last_state

    def evaluate(self, loader: DataLoader) -> Metrics:
        _, metrics, _ = self._run_epoch(loader, train=False)
        return metrics
