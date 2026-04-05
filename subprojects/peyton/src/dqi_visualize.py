"""Visualization helpers for DQI runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history: list[float], *, out_path: Path | None = None, title: str = "DQI convergence"):
    """Plot objective value across optimizer evaluations."""
    if not history:
        raise ValueError("history must not be empty")
    xs = np.arange(1, len(history) + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    ax.plot(xs, history, marker="o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Objective (lower is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, ax


def plot_bitstring_histogram(
    bitstring_counts: dict[str, int],
    *,
    top_k: int = 15,
    out_path: Path | None = None,
    title: str = "Sampled bitstrings",
):
    """Plot top-k measured bitstrings with normalized frequencies."""
    if not bitstring_counts:
        raise ValueError("bitstring_counts must not be empty")
    ordered = sorted(bitstring_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    labels = [k for k, _ in ordered]
    counts = np.array([v for _, v in ordered], dtype=float)
    probs = counts / float(np.sum(counts))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(labels)), 4.2), layout="constrained")
    ax.bar(labels, probs, color="#0B63CE")
    ax.set_xlabel("Bitstring")
    ax.set_ylabel("Relative frequency (top-k normalized)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, ax
