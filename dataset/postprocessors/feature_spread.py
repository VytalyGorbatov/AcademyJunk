#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "dataset" not in data or not isinstance(data["dataset"], list):
        raise ValueError("JSON має містити ключ 'dataset' зі списком записів.")
    return pd.DataFrame(data["dataset"])


def coerce_numeric(df: pd.DataFrame, ignore_cols: Set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in ignore_cols:
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.select_dtypes(include=[np.number])


def compute_spread(
    num_df: pd.DataFrame,
    percentiles: List[float],
    normalize: str = "max",
) -> pd.DataFrame:
    # Перевірка перцентилів.
    percentiles = sorted(set(float(p) for p in percentiles))
    for p in percentiles:
        if p < 0 or p > 100:
            raise ValueError(f"Недопустимий перцентиль: {p}. Дозволено [0, 100].")

    # Прибрати повністю NaN та константні колонки.
    num_df = num_df.dropna(axis=1, how="all")
    nunique = num_df.nunique(dropna=True)
    num_df = num_df.loc[:, nunique > 1]
    if num_df.shape[1] == 0:
        raise ValueError("Немає придатних числових фіч після фільтрації.")

    rows: List[Dict[str, Any]] = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        if s.empty:
            continue

        if normalize == "max":
            denom = float(s.max())
        elif normalize == "max_abs":
            denom = float(np.max(np.abs(s.values)))
        elif normalize == "none":
            denom = None
        else:
            raise ValueError("normalize має бути 'max', 'max_abs' або 'none'.")

        perc_vals = np.percentile(s.values, percentiles)
        min_v = float(np.min(s.values))
        max_v = float(np.max(s.values))
        count = int(s.size)
        missing = int(num_df[col].size - s.size)

        row: Dict[str, Any] = {
            "feature": col,
            "count": count,
            "missing": missing,
            "min": min_v,
            "max": max_v,
        }

        # Перцентилі (сира шкала).
        for p, v in zip(percentiles, perc_vals):
            row[f"p{int(p)}"] = float(v)

        # Нормалізовані значення (якщо задано).
        if denom is not None and not np.isclose(denom, 0.0):
            for p, v in zip(percentiles, perc_vals):
                row[f"p{int(p)}_over_{normalize}"] = float(v) / denom
            # Узагальнення розбросу.
            if 95.0 in percentiles and 5.0 in percentiles:
                p95 = perc_vals[percentiles.index(95.0)]
                p5 = perc_vals[percentiles.index(5.0)]
                row[f"spread_p95_p5_over_{normalize}"] = float(p95 - p5) / denom
            if 75.0 in percentiles and 25.0 in percentiles:
                p75 = perc_vals[percentiles.index(75.0)]
                p25 = perc_vals[percentiles.index(25.0)]
                row[f"iqr_over_{normalize}"] = float(p75 - p25) / denom
        else:
            # Для 'none' нормалізаційні колонки не створюємо.
            pass

        rows.append(row)

    return pd.DataFrame(rows)


def ensure_required_percentiles(percentiles: List[float], metric: str) -> List[float]:
    req = set(percentiles)
    if metric.lower() in ("iqr", "p75_p25"):
        req.update({25.0, 75.0})
    elif metric.lower() in ("p95_p5",):
        req.update({5.0, 95.0})
    else:
        # Може бути pXX
        if metric.lower().startswith("p"):
            try:
                val = float(metric[1:])
                req.add(val)
            except ValueError:
                pass
    return sorted(req)


def select_metric_column(spread_df: pd.DataFrame, metric: str, normalize: str) -> Tuple[pd.DataFrame, str]:
    m = metric.lower()
    if normalize == "none":
        if m == "iqr" or m == "p75_p25":
            if not {"p75", "p25"}.issubset(spread_df.columns):
                raise ValueError("Для IQR потрібні перцентилі 25 і 75. Додайте їх через --percentiles.")
            spread_df = spread_df.copy()
            spread_df["value"] = spread_df["p75"] - spread_df["p25"]
            y_label = "IQR (p75 - p25)"
        elif m == "p95_p5":
            if not {"p95", "p5"}.issubset(spread_df.columns):
                raise ValueError("Для p95-p5 потрібні перцентилі 5 і 95. Додайте їх через --percentiles.")
            spread_df = spread_df.copy()
            spread_df["value"] = spread_df["p95"] - spread_df["p5"]
            y_label = "p95 - p5"
        elif m.startswith("p"):
            col = m  # напр., 'p95'
            if col not in spread_df.columns:
                raise ValueError(f"Колонку {col} не знайдено. Додайте відповідний перцентиль через --percentiles.")
            spread_df = spread_df.rename(columns={col: "value"})
            y_label = col
        else:
            raise ValueError("Невідомий metric для normalize=none. Використайте iqr, p95_p5 або pXX.")
    else:
        if m == "iqr" or m == "p75_p25":
            col = f"iqr_over_{normalize}"
        elif m == "p95_p5":
            col = f"spread_p95_p5_over_{normalize}"
        elif m.startswith("p"):
            col = f"{m}_over_{normalize}"  # напр., 'p95_over_max'
        else:
            raise ValueError("Невідомий metric. Використайте iqr, p95_p5 або pXX.")
        if col not in spread_df.columns:
            raise ValueError(f"Колонку {col} не знайдено. Перевірте --percentiles та --normalize.")
        spread_df = spread_df.rename(columns={col: "value"})
        y_label = col

    # Прибрати NaN у вибраному значенні.
    spread_df = spread_df.dropna(subset=["value"])
    return spread_df[["feature", "value"]], y_label


def plot_bars(
    plot_df: pd.DataFrame,
    y_label: str,
    title: str,
    output_image: Optional[Path],
    figsize: Tuple[float, float],
    top: Optional[int],
    ascending: bool,
    horizontal: bool,
    bar_color: str,
    annot: bool,
    fmt: str,
) -> None:
    if plot_df.empty:
        raise ValueError("Немає даних для побудови графіка (усі значення NaN або відсутні).")

    # Сортування і відбір топ-N.
    plot_df = plot_df.sort_values("value", ascending=ascending, kind="mergesort")
    if top is not None and top > 0:
        plot_df = plot_df.head(top)

    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)

    if horizontal:
        ax = sns.barplot(data=plot_df, y="feature", x="value", color=bar_color)
        ax.set_xlabel(y_label)
        ax.set_ylabel("Feature")
    else:
        ax = sns.barplot(data=plot_df, x="feature", y="value", color=bar_color)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Feature")
        # Повернути підписи, щоб влізли.
        plt.xticks(rotation=60, ha="right")

    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8)

    if annot:
        # Підписати значення на барах.
        for p in ax.patches:
            if horizontal:
                val = p.get_width()
                if np.isnan(val):
                    continue
                y = p.get_y() + p.get_height() / 2
                ax.text(val, y, fmt % val, va="center", ha="left", fontsize=8)
            else:
                val = p.get_height()
                if np.isnan(val):
                    continue
                x = p.get_x() + p.get_width() / 2
                ax.text(x, val, fmt % val, va="bottom", ha="center", fontsize=8, rotation=0)

    plt.tight_layout()
    if output_image:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_image, dpi=200, bbox_inches="tight")
        print(f"Збережено: {output_image.resolve()}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Обчислення розбросу фіч і побудова стовпчикового графіка."
    )
    parser.add_argument("json_path", type=Path, help="Шлях до JSON-файлу (ключ 'dataset').")
    parser.add_argument(
        "--drop-cols", nargs="*", default=["buffers", "buffer_names"],
        help="Стовпці, які потрібно ігнорувати.",
    )
    parser.add_argument(
        "--percentiles", nargs="+", type=float, default=[5, 25, 50, 75, 95],
        help="Список перцентилів (0–100), напр.: --percentiles 5 25 50 75 95.",
    )
    parser.add_argument(
        "--metric", type=str, default="p5",
        help="Що показувати: iqr | p95_p5 | pXX (напр., p95, p50).",
    )
    parser.add_argument(
        "--normalize", choices=["max", "max_abs", "none"], default="max",
        help="Нормалізація значень: 'max', 'max_abs' або 'none'.",
    )
    parser.add_argument(
        "--output-image", type=Path, default=None,
        help="Шлях для зображення (PNG/SVG/PDF). Якщо не задано — показати вікно.",
    )
    parser.add_argument(
        "--title", type=str, default="Feature Spread",
        help="Заголовок графіка.",
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[12.0, 8.0],
        help="Розмір фігури, напр.: --figsize 12 8.",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Показати лише топ-N фіч за метрикою (після сортування).",
    )
    parser.add_argument(
        "--ascending", action="store_true",
        help="Сортувати за зростанням (за замовчуванням — за спаданням).",
    )
    parser.add_argument(
        "--horizontal", action="store_true",
        help="Горизонтальні стовпці (зручно для великої кількості фіч).",
    )
    parser.add_argument(
        "--bar-color", type=str, default="#1f77b4",
        help="Колір стовпців (hex або назва кольору matplotlib).",
    )
    parser.add_argument(
        "--annot", action="store_true",
        help="Показувати підписи значень на стовпцях.",
    )
    parser.add_argument(
        "--float-format", type=str, default="%.4f",
        help="Формат чисел у підписах (напр., %.3f).",
    )

    args = parser.parse_args()

    # Завантаження і підготовка.
    df = load_dataset(args.json_path)
    drop_cols = [c for c in args.drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    num_df = coerce_numeric(df, ignore_cols=set())
    if num_df.shape[1] == 0:
        raise ValueError("Не знайдено числових фіч для аналізу.")

    # Переконатися, що для обраної метрики є потрібні перцентилі.
    percentiles_needed = ensure_required_percentiles(args.percentiles, args.metric)
    spread_df = compute_spread(num_df, percentiles=percentiles_needed, normalize=args.normalize)

    # Вибрати колонку метрики і підготувати дані до побудови.
    plot_df, y_label = select_metric_column(spread_df, metric=args.metric, normalize=args.normalize)

    # Побудова.
    plot_bars(
        plot_df=plot_df,
        y_label=y_label,
        title=args.title,
        output_image=args.output_image,
        figsize=tuple(args.figsize),
        top=args.top,
        ascending=args.ascending,
        horizontal=args.horizontal,
        bar_color=args.bar_color,
        annot=args.annot,
        fmt=args.float_format,
    )


if __name__ == "__main__":
    main()