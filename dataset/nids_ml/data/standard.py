"""Data loading for the standard training pipeline.

Uses byte-ID representation (integers 0-255, PAD=256) with header/body split,
matching the 2-way pipeline's data format.  Preserves attack_percent sampling
logic for controlling the positive/negative ratio in train/val splits.
"""
from __future__ import annotations

import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .common import (
    DataConfig2Way,
    PAD_IDX,
    SEP_BYTE,
    decode_buffers_field,
    split_header_body,
    twoway_collate_fn,
)
from ..utils import DataUtils

logger = logging.getLogger(__name__)


# ─── Dataset ──────────────────────────────────────────


class RecordDataset(Dataset):
    """Dataset backed by pre-loaded record dicts with byte-ID processing."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        cfg: DataConfig2Way,
    ) -> None:
        super().__init__()
        self.records = records
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        ids = decode_buffers_field(r[self.cfg.buffer_field])
        header_ids, body_ids = split_header_body(
            ids, self.cfg.fixed_len, self.cfg.sep_byte, self.cfg.fallback_header_len,
        )
        return {
            "header_ids": header_ids,
            "body_ids": body_ids,
            "alerted": int(r.get("alerted", 0)),
            "is_attack": int(r.get("is_attack", 0)),
        }


# ─── Builder ──────────────────────────────────────────


class DatasetBuilder:
    """Loads JSON datasets, applies attack_percent sampling, returns RecordDatasets.

    Produces datasets whose items are dicts with byte-ID sequences (header/body).
    Use ``build_loaders()`` for ready-to-iterate DataLoaders with the correct
    collate function, or ``build_datasets()`` if you need raw Dataset objects.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})

        # sep_byte: prefer data section, fall back to model.split_byte for compat
        sep_byte = int(data_cfg.get(
            "sep_byte",
            config.get("model", {}).get("split_byte", SEP_BYTE),
        ))
        fixed_len = int(data_cfg.get("fixed_len", 1024))

        self.cfg = DataConfig2Way(
            fixed_len=fixed_len,
            fallback_header_len=int(data_cfg.get("fallback_header_len", fixed_len // 2)),
            buffer_field=data_cfg.get("buffer_field", "buffers"),
            sep_byte=sep_byte,
        )
        self.label_field = data_cfg.get("label_field", "alerted")
        self.split_by = data_cfg.get("split_by", self.label_field)
        self.seed = int(config.get("seed", 42))

    # ── public API ──────────────────────────────────

    def build_datasets(self) -> Dict[str, RecordDataset]:
        """Build train, val, test datasets with attack_percent sampling."""
        benign_paths_cfg = self.config.get("benign_paths", {})
        attack_paths_cfg = self.config.get("attack_paths", {})
        attack_percent = float(self.config.get("attack_percent", 0.5))

        sampling_cfg = self.config.get("sampling", {})
        with_replacement = bool(sampling_cfg.get("with_replacement", False))

        split_seed_offsets = {"train": 11, "val": 22, "test": 33}
        datasets: Dict[str, RecordDataset] = {}

        for split in ["train", "val", "test"]:
            benign_paths = DataUtils.ensure_path_list(benign_paths_cfg.get(split))
            attack_paths = DataUtils.ensure_path_list(attack_paths_cfg.get(split))

            attack_records = self._load_group(
                attack_paths, default_label=1, is_test_split=(split == "test"),
            )
            benign_records = self._load_group(
                benign_paths, default_label=0, is_test_split=(split == "test"),
            )

            all_records = attack_records + benign_records

            if not all_records:
                logger.warning("Split '%s' is empty.", split)
                datasets[split] = RecordDataset([], self.cfg)
                continue

            if split == "test":
                datasets[split] = RecordDataset(all_records, self.cfg)
                self._log_label_stats(split, all_records, "_train_label", "train_label")
                self._log_label_stats(split, all_records, "_split_label", "split_label")
                continue

            # Partition by the configured split_by field so that
            # attack_percent controls the pos/neg ratio in the split.
            if self.split_by == self.label_field:
                label_key = "_train_label"
            else:
                label_key = "_split_label"

            pos_records = [r for r in all_records if r[label_key] == 1]
            neg_records = [r for r in all_records if r[label_key] == 0]

            current_seed = self.seed + split_seed_offsets.get(split, 0)
            mixed_records, split_counts = self._mix_split(
                pos_records, neg_records,
                attack_percent, with_replacement, current_seed,
            )

            datasets[split] = RecordDataset(mixed_records, self.cfg)
            self._log_label_stats(split, mixed_records, "_train_label", "train_label")
            if split_counts is not None:
                self._log_counts(split, *split_counts, "split_label")

        return datasets

    def build_loaders(self) -> Dict[str, DataLoader]:
        """Build DataLoaders with the standard collate function."""
        datasets = self.build_datasets()

        training_cfg = self.config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 128))
        num_workers = int(training_cfg.get("num_workers", 0))

        collate = partial(twoway_collate_fn, cfg=self.cfg)

        return {
            split: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=collate,
            )
            for split, ds in datasets.items()
        }

    # ── loading ─────────────────────────────────────

    def _load_group(
        self, paths: List[Path], default_label: int, is_test_split: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load and concatenate records from multiple JSON files."""
        records: List[Dict[str, Any]] = []
        for p in paths:
            records.extend(self._load_json_file(p, default_label, is_test_split))
        return records

    def _load_json_file(
        self, path: Path, default_label: int, is_test_split: bool = False,
    ) -> List[Dict[str, Any]]:
        """Parse one JSON file into a list of record dicts."""
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return []

        dataset = obj.get("dataset", [])
        if not isinstance(dataset, list):
            logger.error("File %s invalid: 'dataset' is not a list", path)
            return []

        buffer_field = self.cfg.buffer_field
        records: List[Dict[str, Any]] = []

        for rec in dataset:
            if not isinstance(rec, dict):
                continue
            if buffer_field not in rec:
                continue

            is_attack_val = 1 if int(rec.get("is_attack", default_label)) == 1 else 0
            split_label = is_attack_val

            if is_test_split:
                train_label = is_attack_val
            else:
                train_label = (
                    1 if int(rec.get(self.label_field, default_label)) == 1 else 0
                )

            rec["_train_label"] = train_label
            rec["_split_label"] = split_label
            records.append(rec)

        return records

    # ── sampling ────────────────────────────────────

    def _mix_split(
        self,
        pos_records: List[Dict[str, Any]],
        neg_records: List[Dict[str, Any]],
        attack_percent: float,
        with_replacement: bool,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
        """Resample positive/negative records to match attack_percent ratio."""
        n_pos, n_neg = len(pos_records), len(neg_records)

        n_pos_sample, n_neg_sample = self._compute_sample_counts(
            n_pos, n_neg, attack_percent, with_replacement,
        )

        if n_pos_sample == 0 and n_neg_sample == 0:
            return [], (0, 0)

        rng = np.random.RandomState(seed)

        sampled_pos = self._sample_records(pos_records, n_pos_sample, with_replacement, rng)
        sampled_neg = self._sample_records(neg_records, n_neg_sample, with_replacement, rng)

        combined = sampled_pos + sampled_neg
        if not combined:
            return [], (0, 0)

        perm = rng.permutation(len(combined)).tolist()
        combined = [combined[i] for i in perm]
        return combined, (n_pos_sample, n_neg_sample)

    @staticmethod
    def _sample_records(
        records: List[Dict[str, Any]],
        n_sample: int,
        replace: bool,
        rng: np.random.RandomState,
    ) -> List[Dict[str, Any]]:
        n_available = len(records)
        if n_available == 0 or n_sample == 0:
            return []

        if replace or n_sample > n_available:
            indices = rng.randint(0, n_available, size=n_sample)
        else:
            indices = rng.choice(n_available, size=n_sample, replace=False)

        return [records[i] for i in indices.tolist()]

    @staticmethod
    def _compute_sample_counts(
        n_pos: int, n_neg: int, attack_percent: float, with_replacement: bool,
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

    # ── logging ─────────────────────────────────────

    @staticmethod
    def _log_label_stats(
        name: str, records: List[Dict[str, Any]], key: str, tag: str,
    ) -> None:
        pos = sum(1 for r in records if r.get(key) == 1)
        neg = sum(1 for r in records if r.get(key) == 0)
        DatasetBuilder._log_counts(name, pos, neg, tag)

    @staticmethod
    def _log_counts(name: str, pos: int, neg: int, tag: str) -> None:
        total = pos + neg
        logger.info(
            "Split %s (%s): Total=%s, Pos=%s, Neg=%s", name, tag, total, pos, neg,
        )
