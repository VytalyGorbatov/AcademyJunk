#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Set

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "dataset" not in data or not isinstance(data["dataset"], list):
        raise ValueError("JSON має містити ключ 'dataset' зі списком записів.")
    df = pd.DataFrame(data["dataset"])
    return df


def load_and_merge_datasets(json_paths: Tuple[Path, Path]) -> pd.DataFrame:
    frames = [load_dataset(path) for path in json_paths]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return merged


def coerce_numeric(df: pd.DataFrame, ignore_cols: Set[str]) -> pd.DataFrame:
    df = df.copy()
    # Спробувати перетворити object-колонки на числові, що можливо.
    for col in df.columns:
        if col in ignore_cols:
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Залишити лише числові колонки.
    num_df = df.select_dtypes(include=[np.number])
    return num_df


def drop_low_information(num_df: pd.DataFrame, min_variance: Optional[float]) -> pd.DataFrame:
    num_df = num_df.copy()
    # Прибрати константні колонки.
    nunique = num_df.nunique(dropna=True)
    num_df = num_df.loc[:, nunique > 1]

    # Додатково — фільтр за дисперсією, якщо задано.
    if min_variance is not None:
        variances = num_df.var(ddof=0, numeric_only=True)
        keep = variances[variances > min_variance].index
        num_df = num_df[keep]

    # Прибрати колонки, де всі значення NaN після перетворень.
    num_df = num_df.dropna(axis=1, how="all")
    return num_df


def compute_correlation(num_df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Використовує pairwise complete obs (pandas робить це за замовчуванням).
    corr = num_df.corr(method=method)
    return corr


def plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    mask_upper: bool,
    figsize: Tuple[float, float],
    output: Optional[Path],
) -> None:
    if corr.empty or corr.shape[0] < 2:
        raise ValueError("Замало числових фіч для побудови кореляційної матриці.")

    # Маска верхнього трикутника (не включає діагональ).
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Якщо фіч небагато, додаємо підписи; інакше — без анотацій.
    annot = corr.shape[0] <= 25

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        corr,
        mask=mask,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=annot,
        fmt=".2f" if annot else "",
    )
    ax.set_title(title)
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Збережено: {output.resolve()}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Побудова хітмапи кореляцій з JSON-файлу (ключ 'dataset')."
    )
    parser.add_argument(
        "json_paths",
        nargs=2,
        type=Path,
        help="Два шляхи до JSON-файлів з даними.",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Метод кореляції (за замовчуванням: pearson).",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=["buffers", "buffer_names"],
        help="Стовпці, які потрібно ігнорувати.",
    )
    parser.add_argument(
        "--min-variance",
        type=float,
        default=None,
        help="Мінімальна дисперсія для відбору фіч (необов’язково).",
    )
    parser.add_argument(
        "--mask-upper",
        action="store_true",
        help="Маскувати верхній трикутник матриці.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10.0, 8.0],
        help="Розмір фігури, наприклад: --figsize 12 10.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Feature Correlation Heatmap",
        help="Заголовок графіку.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Шлях для збереження (PNG/SVG/PDF). Якщо не задано — показати вікно.",
    )

    args = parser.parse_args()

    df = load_and_merge_datasets(tuple(args.json_paths))
    ignore_cols = set(args.drop_cols)

    # Видалити явно ігноровані колонки, якщо є.
    cols_to_drop = [c for c in ignore_cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Перетворити до числових де можливо і відфільтрувати лише числові.
    num_df = coerce_numeric(df, ignore_cols=set())

    # Прибрати константні та (за бажанням) низькодисперсні фічі.
    num_df = drop_low_information(num_df, min_variance=args.min_variance)

    if num_df.shape[1] < 2:
        raise ValueError(
            "Після фільтрації лишилося менше двох числових фіч. "
            "Перевірте дані або параметри фільтрації."
        )

    corr = compute_correlation(num_df, method=args.method)

    plot_heatmap(
        corr=corr,
        title=args.title,
        mask_upper=args.mask_upper,
        figsize=tuple(args.figsize),
        output=args.output,
    )


if __name__ == "__main__":
    main()