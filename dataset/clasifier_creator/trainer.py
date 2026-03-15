import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import MetricUtils
from .types import Metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Manages the training loop and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        class_weights: Optional[torch.Tensor] = None,
        output_mode: str = "multiclass",
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.output_mode = output_mode
        self.threshold = threshold
        if class_weights is not None:
            class_weights = class_weights.to(device)

        if self.output_mode == "binary":
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.max_grad_norm: float = 1.0

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

        with torch.set_grad_enabled(train):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    xb, yb = batch[0], batch[1]
                else:
                    xb, yb = batch
                if stop_flag and stop_flag():
                    stopped_early = True
                    break
                xb = xb.to(self.device).unsqueeze(1)
                yb = yb.to(self.device)

                logits = self.model(xb)
                if self.output_mode == "binary":
                    logits = logits.view(-1)
                    loss = self.criterion(logits, yb.float())
                else:
                    loss = self.criterion(logits, yb)

                if train:
                    self.optimizer.zero_grad()
                    if torch.isfinite(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=self.max_grad_norm,
                        )
                        self.optimizer.step()
                    else:
                        logger.warning("Non-finite loss detected; skipping optimizer step.")

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
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = {
                    "model_state_dict": copy.deepcopy(self.model.state_dict()),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }

            logger.info(
                "Epoch %s/%s | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f",
                epoch,
                epochs,
                train_loss,
                val_metrics.get("loss", 0.0),
                val_metrics.get("accuracy", 0.0),
                val_metrics.get("f1", 0.0),
            )

        if best_state:
            torch.save(best_state, out_dir / "model_best.pt")
            logger.info("Best model saved with %s=%.4f", best_metric_name, best_metric)
            return history, best_state["val_metrics"], last_state

        return history, {}, last_state

    def evaluate(self, loader: DataLoader) -> Metrics:
        _, metrics, _ = self._run_epoch(loader, train=False)
        return metrics
