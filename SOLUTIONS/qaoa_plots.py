"""Plotting helpers for QAOA optimization traces."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


_SERIES_COLORS = ["#0066CC", "#00356B", "#2D8C3C", "#C41230"]


def save_training_loss_plot(
    histories: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    output_path: str | Path,
    title: str = "QAOA Training Loss",
) -> Path:
    """Save the per-block loss history as a PNG."""
    if not histories:
        raise ValueError("Need at least one loss history to plot")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if len(histories) > 2:
        n_cols = 2
        n_rows = math.ceil(len(histories) / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 4 * n_rows),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        fig.patch.set_facecolor("white")
        axes_flat = list(axes.flat)

        for index, (label, objective_values, best_objective_values) in enumerate(histories):
            ax = axes_flat[index]
            ax.set_facecolor("#F3F4F6")
            if objective_values:
                color = _SERIES_COLORS[index % len(_SERIES_COLORS)]
                steps = range(1, len(objective_values) + 1)
                ax.plot(
                    steps,
                    objective_values,
                    color=color,
                    alpha=0.35,
                    linewidth=1.5,
                    label="trial loss",
                )
                ax.plot(
                    steps,
                    best_objective_values,
                    color=color,
                    linewidth=2.5,
                    label="best so far",
                )
            ax.set_title(label)
            ax.set_xlabel("Objective evaluation")
            ax.set_ylabel("Mean sampled QUBO energy")
            ax.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.8)
            ax.legend(frameon=False, loc="upper right")

        for ax in axes_flat[len(histories) :]:
            ax.axis("off")

        fig.suptitle(title)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F3F4F6")

        for index, (label, objective_values, best_objective_values) in enumerate(histories):
            if not objective_values:
                continue
            color = _SERIES_COLORS[index % len(_SERIES_COLORS)]
            steps = range(1, len(objective_values) + 1)
            ax.plot(
                steps,
                objective_values,
                color=color,
                alpha=0.35,
                linewidth=1.5,
                label=f"{label} trial loss",
            )
            ax.plot(
                steps,
                best_objective_values,
                color=color,
                linewidth=2.5,
                label=f"{label} best so far",
            )

        ax.set_title(title)
        ax.set_xlabel("Objective evaluation")
        ax.set_ylabel("Mean sampled QUBO energy")
        ax.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.8)
        ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination
