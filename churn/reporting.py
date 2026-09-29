"""Noninteractive charts; no desktop window or global theme changes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix, precision_recall_curve

from churn.data import normalize_features


def save_eda(X: pd.DataFrame, y: pd.Series, output: Path) -> None:
    """Explore training rows only so held-out outcomes do not guide modeling."""
    clean = normalize_features(X)
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    bars = left.bar(["Stayed", "Churned"], counts, color=["#4C78A8", "#E45756"])
    left.bar_label(bars, padding=3)
    left.set(title="Training customer outcomes", ylabel="Customers")
    rates = y.groupby(clean["Contract"].fillna("Missing")).mean().sort_values(ascending=False)
    right.bar(rates.index, rates, color="#4C78A8")
    right.set(title="Training churn rate by contract", ylabel="Churn rate", ylim=(0, 1))
    right.yaxis.set_major_formatter(PercentFormatter(1))
    fig.savefig(output / "exploratory_analysis.png", dpi=160)


def save_evaluation(y: pd.Series, scores: np.ndarray, threshold: float,
                    name: str, output: Path) -> None:
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    left, right = fig.subplots(1, 2)
    matrix = confusion_matrix(y, scores >= threshold, labels=[0, 1])
    left.imshow(matrix, cmap="Blues")
    for (row, column), count in np.ndenumerate(matrix):
        left.text(column, row, str(count), ha="center", va="center",
                  color="white" if count > matrix.max() / 2 else "black")
    left.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stayed", "Churned"],
             yticklabels=["Stayed", "Churned"], xlabel="Predicted", ylabel="Actual",
             title=f"{name}: test set (cutoff {threshold:.3f})")
    precision, recall, _ = precision_recall_curve(y, scores)
    right.step(recall, precision, where="post", label=name, color="#4C78A8")
    right.axhline(float(y.mean()), linestyle="--", color="#E45756", label="Churn prevalence")
    right.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05),
              title="Selected model: test precision / recall")
    right.legend()
    fig.savefig(output / "test_evaluation.png", dpi=160)
