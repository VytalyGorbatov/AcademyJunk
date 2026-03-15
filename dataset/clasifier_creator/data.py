import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .types import TensorPair, TensorTriple
from .utils import DataUtils

logger = logging.getLogger(__name__)


class BufferDataset(Dataset):
    """Simple Tensor Dataset wrapper."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        raw_buffers: List[str] | None = None,
    ) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Size mismatch: features {features.shape[0]} != labels {labels.shape[0]}"
            )
        if raw_buffers is not None and features.shape[0] != len(raw_buffers):
            raise ValueError(
                f"Size mismatch: features {features.shape[0]} != raw_buffers {len(raw_buffers)}"
            )
        self.features = features
        self.labels = labels
        self.raw_buffers = raw_buffers

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        if self.raw_buffers is None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], self.labels[idx], self.raw_buffers[idx]


class DatasetBuilder:
    """Handles loading JSON datasets, processing features, and splitting data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.buffer_field = self.data_cfg.get("buffer_field", "buffer")
        self.label_field = self.data_cfg.get("label_field", "alerted")
        self.split_by = self.data_cfg.get("split_by", self.label_field)
        self.fixed_len = int(self.data_cfg.get("fixed_len", 1024))
        self.seed = int(config.get("seed", 42))

    def build_datasets(self) -> Dict[str, BufferDataset]:
        """Builds train, val, and test datasets based on configuration."""
        benign_paths_cfg = self.config.get("benign_paths", {})
        attack_paths_cfg = self.config.get("attack_paths", {})
        attack_percent = float(self.config.get("attack_percent", 0.5))

        sampling_cfg = self.config.get("sampling", {})
        with_replacement = bool(sampling_cfg.get("with_replacement", False))

        split_seed_offsets = {"train": 11, "val": 22, "test": 33}
        datasets: Dict[str, BufferDataset] = {}

        for split in ["train", "val", "test"]:
            benign_paths = DataUtils.ensure_path_list(benign_paths_cfg.get(split))
            attack_paths = DataUtils.ensure_path_list(attack_paths_cfg.get(split))

            x_attack, y_train_attack, y_split_attack, b_attack = self._load_group(
                attack_paths, default_label=1, is_test_split=(split == "test")
            )
            x_benign, y_train_benign, y_split_benign, b_benign = self._load_group(
                benign_paths, default_label=0, is_test_split=(split == "test")
            )

            x_all = torch.cat([x_attack, x_benign], dim=0)
            y_train_all = torch.cat([y_train_attack, y_train_benign], dim=0)
            y_split_all = torch.cat([y_split_attack, y_split_benign], dim=0)

            all_buffers = b_attack + b_benign

            if x_all.numel() == 0:
                logger.warning("Split '%s' is empty.", split)
                datasets[split] = BufferDataset(x_all, y_train_all, all_buffers)
                continue

            if split == "test":
                datasets[split] = BufferDataset(x_all, y_train_all, all_buffers)
                self._log_label_stats(split, y_train_all, "train_label")
                self._log_label_stats(split, y_split_all, "split_label")
                continue

            # Partition by the configured split_by field so that
            # attack_percent controls the pos/neg ratio in the split.
            if self.split_by == self.label_field:
                split_labels = y_train_all
            else:
                split_labels = y_split_all
            pos_mask = split_labels == 1
            neg_mask = split_labels == 0

            pos_indices = pos_mask.nonzero(as_tuple=False).view(-1).tolist()
            neg_indices = neg_mask.nonzero(as_tuple=False).view(-1).tolist()
            pos_buffers = [all_buffers[i] for i in pos_indices]
            neg_buffers = [all_buffers[i] for i in neg_indices]

            pos_data = (x_all[pos_mask], y_train_all[pos_mask], pos_buffers)
            neg_data = (x_all[neg_mask], y_train_all[neg_mask], neg_buffers)

            current_seed = self.seed + split_seed_offsets.get(split, 0)
            (mixed_x, mixed_y, mixed_b), split_counts = self._mix_split(
                pos_data,
                neg_data,
                attack_percent,
                with_replacement,
                current_seed,
            )

            datasets[split] = BufferDataset(mixed_x, mixed_y, mixed_b)
            self._log_label_stats(split, mixed_y, "train_label")
            if split_counts is not None:
                split_pos, split_neg = split_counts
                self._log_counts(split, split_pos, split_neg, "split_label")

        return datasets

    def _load_group(
        self, paths: List[Path], default_label: int, is_test_split: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Loads a group of files (e.g., all benign files) and concatenates them."""
        features_list: List[torch.Tensor] = []
        train_labels_list: List[torch.Tensor] = []
        split_labels_list: List[torch.Tensor] = []

        buffers_list: List[str] = []

        for p in paths:
            x, y_t, y_s, b = self._load_json_file(p, default_label, is_test_split)
            if x.numel() > 0:
                features_list.append(x)
                train_labels_list.append(y_t)
                split_labels_list.append(y_s)
                buffers_list.extend(b)

        if not features_list:
            return (
                torch.empty(0, self.fixed_len, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
                [],
            )

        return (
            torch.cat(features_list, dim=0),
            torch.cat(train_labels_list, dim=0),
            torch.cat(split_labels_list, dim=0),
            buffers_list,
        )

    def _load_json_file(
        self, path: Path, default_label: int, is_test_split: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Parses a single JSON file into tensors."""
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return self._empty_quad()

        dataset = obj.get("dataset", [])
        if not isinstance(dataset, list):
            logger.error("File %s invalid: 'dataset' is not a list", path)
            return self._empty_quad()

        features: List[np.ndarray] = []
        train_labels: List[int] = []
        split_labels: List[int] = []
        buffers: List[str] = []

        for rec in dataset:
            if not isinstance(rec, dict):
                continue

            buf = rec.get(self.buffer_field)
            if not isinstance(buf, str):
                continue

            is_attack_val = 1 if int(rec.get("is_attack", default_label)) == 1 else 0
            s_label = is_attack_val

            if is_test_split:
                t_label = is_attack_val
            else:
                t_label = 1 if int(rec.get(self.label_field, default_label)) == 1 else 0

            arr = DataUtils.buffer_to_fixed_array(buf, self.fixed_len)
            features.append(arr)
            train_labels.append(t_label)
            split_labels.append(s_label)
            buffers.append(buf)

        if not features:
            return self._empty_quad()

        return (
            torch.tensor(np.stack(features, axis=0), dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
            torch.tensor(split_labels, dtype=torch.long),
            buffers,
        )

    def _empty_quad(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        return (
            torch.empty(0, self.fixed_len, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [],
        )

    def _mix_split(
        self,
        pos_data: Tuple[torch.Tensor, torch.Tensor, List[str]],
        neg_data: Tuple[torch.Tensor, torch.Tensor, List[str]],
        attack_percent: float,
        with_replacement: bool,
        seed: int,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, List[str]], Tuple[int, int]]:
        """Resamples positive and negative data to match start percentages."""
        pos_x, pos_y, pos_b = pos_data
        neg_x, neg_y, neg_b = neg_data
        n_pos, n_neg = pos_x.shape[0], neg_x.shape[0]

        n_pos_sample, n_neg_sample = self._compute_sample_counts(
            n_pos, n_neg, attack_percent, with_replacement
        )

        if n_pos_sample == 0 and n_neg_sample == 0:
            return (
                torch.empty(0, self.fixed_len, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                [],
            ), (0, 0)

        rng = np.random.RandomState(seed)

        sampled_pos_x, sampled_pos_y, sampled_pos_b = self._sample_indices(
            pos_x, pos_y, pos_b, n_pos_sample, with_replacement, rng
        )
        sampled_neg_x, sampled_neg_y, sampled_neg_b = self._sample_indices(
            neg_x, neg_y, neg_b, n_neg_sample, with_replacement, rng
        )

        all_x = torch.cat([sampled_pos_x, sampled_neg_x], dim=0)
        all_y = torch.cat([sampled_pos_y, sampled_neg_y], dim=0)
        all_b = sampled_pos_b + sampled_neg_b

        if all_x.numel() == 0:
            return (all_x, all_y, all_b), (0, 0)

        perm = torch.from_numpy(rng.permutation(all_x.shape[0])).long()
        perm_list = perm.tolist()
        all_b = [all_b[i] for i in perm_list]
        return (all_x[perm], all_y[perm], all_b), (n_pos_sample, n_neg_sample)

    def _sample_indices(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        buffers: List[str],
        n_sample: int,
        replace: bool,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        n_available = x.shape[0]
        if n_available == 0 or n_sample == 0:
            return (
                torch.empty(0, self.fixed_len, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                [],
            )

        if replace or n_sample > n_available:
            indices = rng.randint(0, n_available, size=n_sample)
        else:
            indices = rng.choice(n_available, size=n_sample, replace=False)

        if isinstance(indices, np.ndarray):
            idx_list = indices.tolist()
        else:
            idx_list = list(indices)
        return x[indices], y[indices], [buffers[i] for i in idx_list]

    def _compute_sample_counts(
        self, n_pos: int, n_neg: int, attack_percent: float, with_replacement: bool
    ) -> Tuple[int, int]:
        p = float(attack_percent)
        if p <= 0.0:
            return 0, n_neg
        if p >= 1.0:
            return n_pos, 0

        if with_replacement:
            total = n_pos + n_neg
            target_pos = int(round(p * total))
            target_neg = total - target_pos
        else:
            max_pos = n_pos / p if p > 0 else float("inf")
            max_neg = n_neg / (1.0 - p) if p < 1.0 else float("inf")
            total = int(min(max_pos, max_neg))

            target_pos = min(int(round(p * total)), n_pos)
            target_neg = min(total - target_pos, n_neg)

        return max(target_pos, 0), max(target_neg, 0)

    def _log_label_stats(self, name: str, labels: torch.Tensor, tag: str) -> None:
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        self._log_counts(name, pos, neg, tag)

    def _log_counts(self, name: str, pos: int, neg: int, tag: str) -> None:
        total = pos + neg
        logger.info(
            "Split %s (%s): Total=%s, Pos=%s, Neg=%s", name, tag, total, pos, neg
        )
