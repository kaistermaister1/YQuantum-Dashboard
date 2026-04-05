"""Visualization helpers for DQI runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(history: list[float], *, out_path: Path | None = None, title: str = "DQI convergence"):
    """Plot objective values; for one-shot DQI this is typically a single point."""
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


METHOD_ORDER = ("dqi", "random", "local_search", "bruteforce", "qaoa")

METHOD_STYLE: dict[str, dict[str, Any]] = {
    "dqi": {"color": "#0B63CE", "label": "DQI"},
    "random": {"color": "#78909C", "label": "Random sample"},
    "local_search": {"color": "#E65100", "label": "Multistart 1-opt"},
    "bruteforce": {"color": "#2E7D32", "label": "Exact (brute)"},
    "qaoa": {"color": "#6A1B9A", "label": "QAOA p=1"},
}


def plot_scaling_benchmark(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_keys: Sequence[str] = ("dqi", "random", "local_search"),
    out_path: Path | None = None,
    title_prefix: str = "DQI vs classical baselines",
    figsize: tuple[float, float] = (14.0, 4.6),
) -> tuple[Any, np.ndarray]:
    """Side-by-side panels: optimality gap, wall time, workflow cost.

    Each row mapping should include:

    - ``n``: problem size
    - ``methods``: nested map ``method -> {value, time_s, cost}``
    - ``reference_value`` (optional): exact optimum; if absent, ``min(method values)`` is used
    """
    if not rows:
        raise ValueError("rows must not be empty")
    ns = [int(r["n"]) for r in rows]
    x = np.arange(len(ns), dtype=float)
    n_methods = len(method_keys)
    width = 0.8 / max(1, n_methods)

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")

    # --- Panel 1: normalized optimality gap (minimization; 0 = on reference, >0 worse)
    # Uses (f - f*) / max(|f*|, eps) so sign of QUBO values does not invert the ranking.
    gaps: dict[str, list[float]] = {k: [] for k in method_keys}
    for r in rows:
        methods = r["methods"]
        ref = r.get("reference_value")
        if ref is None:
            ref = min(float(methods[k]["value"]) for k in method_keys if k in methods)
        ref = float(ref)
        denom = max(abs(ref), 1e-12)
        for k in method_keys:
            if k not in methods:
                gaps[k].append(float("nan"))
            else:
                gaps[k].append((float(methods[k]["value"]) - ref) / denom)

    for i, k in enumerate(method_keys):
        off = (i - (n_methods - 1) / 2.0) * width
        sty = METHOD_STYLE.get(k, {"color": "#455A64", "label": k})
        axes[0].bar(
            x + off,
            gaps[k],
            width=width * 0.92,
            label=sty["label"],
            color=sty["color"],
            edgecolor="#263238",
            linewidth=0.6,
        )
    axes[0].axhline(0.0, color="#1B5E20", linestyle="--", linewidth=1.0, alpha=0.75, label="Reference")
    axes[0].set_xticks(x, labels=[str(n) for n in ns])
    axes[0].set_xlabel("Instance size $n$")
    axes[0].set_ylabel(r"Optimality gap $(f - f^\ast)/\max(|f^\ast|,\epsilon)$")
    axes[0].set_title(f"{title_prefix} — solution quality")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.25)

    # --- Panel 2: wall time
    for i, k in enumerate(method_keys):
        off = (i - (n_methods - 1) / 2.0) * width
        sty = METHOD_STYLE.get(k, {"color": "#455A64", "label": k})
        times = [float(r["methods"][k]["time_s"]) if k in r["methods"] else float("nan") for r in rows]
        axes[1].bar(
            x + off,
            times,
            width=width * 0.92,
            label=sty["label"],
            color=sty["color"],
            edgecolor="#263238",
            linewidth=0.6,
        )
    axes[1].set_xticks(x, labels=[str(n) for n in ns])
    axes[1].set_xlabel("Instance size $n$")
    axes[1].set_ylabel("Wall time (s)")
    axes[1].set_title("Runtime (log scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)

    # --- Panel 3: workflow cost (reported energy evaluations / shots)
    for i, k in enumerate(method_keys):
        off = (i - (n_methods - 1) / 2.0) * width
        sty = METHOD_STYLE.get(k, {"color": "#455A64", "label": k})
        costs = [
            float(r["methods"][k]["cost"]) if k in r["methods"] else float("nan") for r in rows
        ]
        axes[2].bar(
            x + off,
            costs,
            width=width * 0.92,
            label=sty["label"],
            color=sty["color"],
            edgecolor="#263238",
            linewidth=0.6,
        )
    axes[2].set_xticks(x, labels=[str(n) for n in ns])
    axes[2].set_xlabel("Instance size $n$")
    axes[2].set_ylabel("Workflow cost (energy evals or shots)")
    axes[2].set_title("Reported workflow cost (log scale)")
    axes[2].set_yscale("log")
    axes[2].grid(True, axis="y", which="both", alpha=0.25)

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
    return fig, axes


class LiveBenchmarkComparison:
    """Horizontal bar chart that refreshes as benchmark methods finish (use with ``plt.ion()``)."""

    def __init__(
        self,
        *,
        title: str = "Best QUBO energy by method (lower is better)",
        figsize: tuple[float, float] = (8.2, 4.8),
    ) -> None:
        plt.ion()
        self._fig, self.ax = plt.subplots(figsize=figsize, layout="constrained")
        self._title = title

    def update(self, results: Mapping[str, Any]) -> None:
        """Redraw bars for all keys present in ``results`` (expects ``.best_value`` on each)."""
        self.ax.clear()
        keys = [k for k in METHOD_ORDER if k in results]
        if not keys:
            self.ax.set_title(self._title)
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.05)
            return
        vals = np.array([float(results[k].best_value) for k in keys], dtype=float)
        best_i = int(np.argmin(vals))
        face = ["#5C9BD5" if i != best_i else "#2E7D32" for i in range(len(keys))]
        bars = self.ax.barh(np.arange(len(keys)), vals, color=face, edgecolor="#263238", linewidth=0.9)
        bars[best_i].set_linewidth(2.0)
        bars[best_i].set_edgecolor("#1B5E20")
        self.ax.set_yticks(np.arange(len(keys)), labels=keys)
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Best objective found")
        self.ax.set_title(self._title)
        self.ax.grid(True, axis="x", alpha=0.28)
        lim = self.ax.get_xlim()
        span = lim[1] - lim[0] if lim[1] > lim[0] else 1.0
        pad = 0.02 * span
        for bar, v in zip(bars, vals):
            x = float(bar.get_width())
            y = float(bar.get_y() + bar.get_height() * 0.5)
            mid = lim[0] + 0.5 * span
            tx = x + pad if x >= mid else x - pad
            ha = "left" if x >= mid else "right"
            self.ax.text(tx, y, f"{v:.5g}", va="center", ha=ha, fontsize=9, clip_on=False)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.12)

    def savefig(self, path: Path, *, dpi: int = 150) -> None:
        self._fig.savefig(path, dpi=dpi)

    def show(self, *, block: bool = True) -> None:
        """Keep the window open (set ``block=False`` for non-blocking)."""
        plt.show(block=block)
