#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_samples(json_path: Path) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Очікується JSON-масив записів (list).")

    required = {"raw_score", "calibrated_score", "is_attack", "alerted"}
    missing = [k for k in required if not all(k in row for row in data if isinstance(row, dict))]
    if missing:
        raise ValueError(f"У JSON відсутні обов'язкові поля: {missing}")

    return data


def extract_scores(samples: List[Dict[str, Any]], score_field: str, predicate) -> np.ndarray:
    values: List[float] = []
    for row in samples:
        if not isinstance(row, dict):
            continue
        if predicate(row):
            try:
                values.append(float(row[score_field]))
            except (TypeError, ValueError):
                continue
    return np.array(values, dtype=np.float64)


def plot_hist(ax, values: np.ndarray, bins: int, label: str, color: str, alpha: float = 0.45) -> None:
    if values.size == 0:
        return
    ax.hist(
        values,
        bins=bins,
        range=(0.0, 1.0),
        alpha=alpha,
        label=f"{label} (n={values.size})",
        color=color,
        edgecolor="black",
        linewidth=0.3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Побудова гістограм raw_score для attack/alert/FN/benign без alert."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/nids_ml/artifacts/test_samples.json"),
        help="Шлях до JSON зі зразками (default: dataset/nids_ml/artifacts/test_samples.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/nids_ml/artifacts/score_distributions.png"),
        help="Куди зберегти PNG (default: dataset/nids_ml/artifacts/score_distributions.png)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Кількість бінів гістограми (default: 50)",
    )
    args = parser.parse_args()

    samples = load_samples(args.input)

    def build_series(score_field: str) -> Dict[str, np.ndarray]:
        return {
            "is_attack_1": extract_scores(samples, score_field, lambda r: int(r["is_attack"]) == 1),
            "is_attack_0": extract_scores(samples, score_field, lambda r: int(r["is_attack"]) == 0),
            "alerted_1": extract_scores(samples, score_field, lambda r: int(r["alerted"]) == 1),
            "alerted_0": extract_scores(samples, score_field, lambda r: int(r["alerted"]) == 0),
            "false_negative": extract_scores(
                samples, score_field, lambda r: int(r["is_attack"]) == 1 and int(r["alerted"]) == 0
            ),
            "benign_no_alert": extract_scores(
                samples, score_field, lambda r: int(r["is_attack"]) == 0 and int(r["alerted"]) == 0
            ),
        }

    raw = build_series("raw_score")
    calibrated = build_series("calibrated_score")

    fig, axes = plt.subplots(3, 2, figsize=(18, 16), sharex=True)

    for col_idx, (series, field_name) in enumerate(((raw, "raw_score"), (calibrated, "calibrated_score"))):
        # score_distribution_raw
        plot_hist(axes[0, col_idx], series["is_attack_0"], args.bins, "is_attack=0", "#4C78A8", alpha=0.45)
        plot_hist(axes[0, col_idx], series["is_attack_1"], args.bins, "is_attack=1", "#F58518", alpha=0.45)
        plot_hist(axes[0, col_idx], series["alerted_0"], args.bins, "alerted=0", "#54A24B", alpha=0.22)
        plot_hist(axes[0, col_idx], series["alerted_1"], args.bins, "alerted=1", "#E45756", alpha=0.30)
        axes[0, col_idx].set_title(f"score_distribution_raw ({field_name})")
        axes[0, col_idx].set_ylabel("count")
        axes[0, col_idx].grid(alpha=0.2)
        axes[0, col_idx].legend(loc="upper center", ncol=2)

        # score_distribution_benign
        plot_hist(
            axes[1, col_idx],
            series["benign_no_alert"],
            args.bins,
            "is_attack=0, alerted=0",
            "#72B7B2",
            alpha=0.7,
        )
        axes[1, col_idx].set_title(f"score_distribution_benign ({field_name})")
        axes[1, col_idx].set_ylabel("count")
        axes[1, col_idx].grid(alpha=0.2)
        axes[1, col_idx].legend(loc="upper right")

        # score_distribution_false_negative
        plot_hist(
            axes[2, col_idx],
            series["false_negative"],
            args.bins,
            "is_attack=1, alerted=0",
            "#FF9DA6",
            alpha=0.7,
        )
        axes[2, col_idx].set_title(f"score_distribution_false_negative ({field_name})")
        axes[2, col_idx].set_xlabel(field_name)
        axes[2, col_idx].set_ylabel("count")
        axes[2, col_idx].grid(alpha=0.2)
        axes[2, col_idx].legend(loc="upper right")

    for ax in axes.ravel():
        ax.set_xlim(0.0, 1.0)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)

    print(f"Saved PNG: {args.output}")
    print(
        "Counts: "
        f"raw:is_attack=0={raw['is_attack_0'].size}, "
        f"raw:is_attack=1={raw['is_attack_1'].size}, "
        f"raw:alerted=0={raw['alerted_0'].size}, "
        f"raw:alerted=1={raw['alerted_1'].size}, "
        f"raw:benign_no_alert={raw['benign_no_alert'].size}, "
        f"raw:false_negative={raw['false_negative'].size}; "
        f"calibrated:is_attack=0={calibrated['is_attack_0'].size}, "
        f"calibrated:is_attack=1={calibrated['is_attack_1'].size}, "
        f"calibrated:alerted=0={calibrated['alerted_0'].size}, "
        f"calibrated:alerted=1={calibrated['alerted_1'].size}, "
        f"calibrated:benign_no_alert={calibrated['benign_no_alert'].size}, "
        f"calibrated:false_negative={calibrated['false_negative'].size}"
    )


if __name__ == "__main__":
    main()
