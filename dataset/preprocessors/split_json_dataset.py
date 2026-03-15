#!/usr/bin/env python3
"""
Split a JSON dataset (array of objects or values) into train / validation / test subsets.

Usage examples:

  python split_json_dataset.py dataset.json --splits 0.7 0.15 0.15 \
      --output-dir data_splits --seed 42

  python split_json_dataset.py dataset.json --train 0.8 --val 0.1 --test 0.1

If ratios don't sum exactly to 1.0 due to floating point, they will be normalized.
Output files will be named (by default):
  <base>_train.json
  <base>_val.json
  <base>_test.json
and written to the chosen output directory (default: same as input file).

Supports large (but memory‑fit) JSON arrays. For massive files which cannot fit into
memory, consider implementing a streaming reservoir sampling approach (not included here).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Any, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a JSON array dataset into train/val/test subsets.")
    p.add_argument("input", help="Path to input JSON file containing a top-level list (array).")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--splits", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                       help="Three floats summing (approximately) to 1.0 for train/val/test.")
    group2 = p.add_argument_group("Individual split ratios (alternative to --splits)")
    group2.add_argument("--train", type=float, help="Train ratio")
    group2.add_argument("--val", type=float, help="Validation ratio")
    group2.add_argument("--test", type=float, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--shuffle", action="store_true", default=True, help="Shuffle before splitting (default: True)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable shuffling")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to place output files (default: input dir)")
    p.add_argument("--base-name", type=str, default=None,
                   help="Base name for output files (default: input filename stem)")
    return p.parse_args()


def resolve_ratios(args: argparse.Namespace) -> Tuple[float, float, float]:
    if args.splits:
        train_r, val_r, test_r = args.splits
    else:
        if args.train is None or args.val is None or args.test is None:
            raise SystemExit("Error: Provide either --splits TRAIN VAL TEST or all of --train --val --test")
        train_r, val_r, test_r = args.train, args.val, args.test
    total = train_r + val_r + test_r
    if total <= 0:
        raise SystemExit("Error: Sum of ratios must be > 0")
    # Normalize
    train_r, val_r, test_r = (train_r / total, val_r / total, test_r / total)
    return train_r, val_r, test_r


def compute_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train_r, val_r, test_r = ratios
    # Floor for first two, remainder to last to guarantee total n
    train_n = math.floor(n * train_r)
    val_n = math.floor(n * val_r)
    test_n = n - train_n - val_n
    # Adjust if any split expected >0 but got 0 and we still have items to reassign
    def bump(count: int, ratio: float) -> int:
        return 1 if count == 0 and ratio > 0 else 0
    # Ensure at least one item in a non-zero ratio split if possible
    needed = bump(train_n, train_r) + bump(val_n, val_r) + bump(test_n, test_r)
    while needed > 0 and (train_n + val_n + test_n) < n:
        # Give an item to the split with ratio >0 and currently zero
        if train_n == 0 and train_r > 0:
            train_n += 1
        elif val_n == 0 and val_r > 0:
            val_n += 1
        elif test_n == 0 and test_r > 0:
            test_n += 1
        needed = bump(train_n, train_r) + bump(val_n, val_r) + bump(test_n, test_r)
    # If we've over-allocated correct by reducing from the largest
    overflow = (train_n + val_n + test_n) - n
    if overflow > 0:
        # Reduce from the split with largest fractional excess preference
        for _ in range(overflow):
            # Choose candidate with count>1 to avoid wiping a split to zero if its ratio >0 and dataset large enough
            candidates = [
                (train_n, 'train'),
                (val_n, 'val'),
                (test_n, 'test'),
            ]
            candidates.sort(reverse=True)
            for _, name in candidates:
                if name == 'train' and train_n > 1:
                    train_n -= 1
                    break
                if name == 'val' and val_n > 1:
                    val_n -= 1
                    break
                if name == 'test' and test_n > 1:
                    test_n -= 1
                    break
    return train_n, val_n, test_n


def main():
    args = parse_args()
    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        try:
            data: Any = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Failed to parse JSON: {e}")

    dataset = data['dataset']

    if not isinstance(dataset, list):
        raise SystemExit("Top-level JSON must be an array/list")

    n = len(dataset)
    if n == 0:
        raise SystemExit("Dataset is empty; nothing to split.")

    ratios = resolve_ratios(args)
    indices = list(range(n))
    if args.shuffle:
        random.shuffle(indices)

    train_n, val_n, test_n = compute_counts(n, ratios)

    train_idx = indices[:train_n]
    val_idx = indices[train_n:train_n + val_n]
    test_idx = indices[train_n + val_n:train_n + val_n + test_n]

    # Sanity check
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    def subset(idxs: List[int]) -> List[Any]:
        return [dataset[i] for i in idxs]

    output_dir = args.output_dir or input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base = args.base_name or input_path.stem

    out_map = {
        'train': subset(train_idx),
        'val': subset(val_idx),
        'test': subset(test_idx),
    }

    print("Summary:")
    print(f"  Total: {n}")
    print(f"  Train: {len(out_map['train'])} ({len(out_map['train'])/n:.2%})")
    print(f"  Val:   {len(out_map['val'])} ({len(out_map['val'])/n:.2%})")
    print(f"  Test:  {len(out_map['test'])} ({len(out_map['test'])/n:.2%})")

    for split_name, split_data in out_map.items():
        out_path = output_dir / f"{base}_{split_name}.json"

        out_data = data.copy()
        out_data['dataset'] = split_data

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False)
        print(f"Wrote {split_name}: {len(split_data)} -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
