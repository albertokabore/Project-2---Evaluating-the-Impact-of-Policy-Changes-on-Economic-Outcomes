"""Reusable plotting helpers for the counterfactual analysis notebook/report."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"

COLOR_ACTUAL = "#1b4965"
COLOR_COUNTERFACTUAL = "#c1440e"
COLOR_CI = "#c1440e"
COLOR_TRAIN = "#5fa8d3"


def _save(fig: plt.Figure, filename: str | None) -> None:
    if filename:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")


def plot_series_with_intervention(
    df: pd.DataFrame,
    columns: list[str],
    intervention_date: pd.Timestamp,
    title: str,
    ylabel: str,
    filename: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for col in columns:
        ax.plot(df.index, df[col], label=col, linewidth=1.6)
    ax.axvline(intervention_date, color="black", linestyle="--", linewidth=1, label="Policy change (Mar 2022)")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_actual_vs_counterfactual(
    result,
    train_actual: pd.Series | None = None,
    title: str = "",
    ylabel: str = "",
    filename: str | None = None,
) -> plt.Figure:
    """Plot the observed series against the modeled counterfactual, with a 95% band."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if train_actual is not None:
        ax.plot(train_actual.index, train_actual.values, color=COLOR_TRAIN, linewidth=1.3, label="Pre-intervention (training data)")

    ax.plot(result.actual_post.index, result.actual_post.values, color=COLOR_ACTUAL, linewidth=1.8, label="Observed (actual)")
    ax.plot(result.counterfactual_mean.index, result.counterfactual_mean.values, color=COLOR_COUNTERFACTUAL, linewidth=1.8, linestyle="--", label="Counterfactual (no policy change)")
    ax.fill_between(
        result.counterfactual_mean.index,
        result.counterfactual_ci_lower.values,
        result.counterfactual_ci_upper.values,
        color=COLOR_CI,
        alpha=0.15,
        label="95% prediction interval",
    )
    ax.axvline(result.actual_post.index[0], color="black", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_effect(result, title: str = "", ylabel: str = "", filename: str | None = None) -> plt.Figure:
    """Two-panel plot: pointwise effect (bars) and cumulative effect (line)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    colors = ["#2a9d8f" if v <= 0 else "#e76f51" for v in result.pointwise_effect.values]
    axes[0].bar(result.pointwise_effect.index, result.pointwise_effect.values, width=20, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title(f"{title} -- Pointwise Effect (Actual - Counterfactual)")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.3)

    axes[1].plot(result.cumulative_effect.index, result.cumulative_effect.values, color=COLOR_COUNTERFACTUAL, linewidth=1.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].fill_between(result.cumulative_effect.index, 0, result.cumulative_effect.values, color=COLOR_COUNTERFACTUAL, alpha=0.15)
    axes[1].set_title("Cumulative Effect")
    axes[1].set_ylabel(f"Cumulative {ylabel}")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, filename)
    return fig
