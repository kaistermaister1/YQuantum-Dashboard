#!/usr/bin/env python3
"""Rebuild hybrid DQI graph suite from existing hybrid artifacts/summaries."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _rows_from_official_summary(summary_path: Path) -> list[dict]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for r in data:
        methods = {}
        if "dqi" in r:
            methods["dqi"] = {
                "value": float(r["dqi"]["value"]),
                "time_s": float(r["dqi"]["time_s"]),
                "cost": float(r["dqi"]["cost"]),
            }
        if "random" in r:
            methods["random"] = {
                "value": float(r["random"]["value"]),
                "time_s": float(r["random"]["time_s"]),
                "cost": float(r["random"]["cost"]),
            }
        if "local_search" in r:
            methods["local_search"] = {
                "value": float(r["local_search"]["value"]),
                "time_s": float(r["local_search"]["time_s"]),
                "cost": float(r["local_search"]["cost"]),
            }
        rows.append(
            {
                "n": int(r["n"]),
                "n_vars": int(r.get("n_vars", r["n"])),
                "reference_value": r.get("reference_optimal"),
                "methods": methods,
            }
        )
    return rows


def _plot_hybrid_panels(rows: list[dict], out_dir: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    method_keys = ("dqi", "random", "local_search")

    # comparison 3-panel (same style as benchmark_official/test_plot)
    from src.dqi_visualize import plot_scaling_benchmark

    out_cmp = out_dir / "hybrid_dqi_benchmark_comparison.png"
    plot_scaling_benchmark(rows, method_keys=method_keys, out_path=out_cmp, title_prefix="Hybrid DQI vs classical baselines")

    # runtime
    out_rt = out_dir / "hybrid_dqi_benchmark_runtime.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    for key, color, label in (
        ("dqi", "#0B63CE", "Hybrid DQI runtime"),
        ("random", "#78909C", "Random runtime"),
        ("local_search", "#E65100", "Multistart 1-opt runtime"),
    ):
        ys = [float(r["methods"][key]["time_s"]) if key in r["methods"] else np.nan for r in rows]
        ax.plot(ns, ys, marker="o", linewidth=1.3, color=color, label=label)
    ax.set_xlabel("Instance size n")
    ax.set_ylabel("Runtime (s)")
    ax.set_yscale("log")
    ax.set_title("Hybrid benchmark runtime")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.savefig(out_rt, dpi=150)
    plt.close(fig)

    # workflow cost
    out_cost = out_dir / "hybrid_dqi_benchmark_workflow_cost.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    for key, color, label in (
        ("dqi", "#0B63CE", "Hybrid DQI cost"),
        ("random", "#78909C", "Random cost"),
        ("local_search", "#E65100", "Multistart 1-opt cost"),
    ):
        ys = [float(r["methods"][key]["cost"]) if key in r["methods"] else np.nan for r in rows]
        ax.plot(ns, ys, marker="o", linewidth=1.3, color=color, label=label)
    ax.set_xlabel("Instance size n")
    ax.set_ylabel("Workflow cost (energy evals or shots)")
    ax.set_yscale("log")
    ax.set_title("Hybrid benchmark workflow cost")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.savefig(out_cost, dpi=150)
    plt.close(fig)

    # gap-to-exact (where reference exists)
    out_gap = out_dir / "hybrid_dqi_benchmark_gap_to_exact.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    for key, color, label in (
        ("dqi", "#0B63CE", "Hybrid DQI normalized gap"),
        ("random", "#78909C", "Random normalized gap"),
        ("local_search", "#E65100", "Multistart 1-opt normalized gap"),
    ):
        ys = []
        for r in rows:
            ref = r.get("reference_value")
            if ref is None or key not in r["methods"]:
                ys.append(np.nan)
                continue
            denom = max(abs(float(ref)), 1e-12)
            ys.append((float(r["methods"][key]["value"]) - float(ref)) / denom)
        ax.plot(ns, ys, marker="o", linewidth=1.3, color=color, label=label)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#1B5E20")
    ax.set_xlabel("Instance size n")
    ax.set_ylabel("(f - f*) / max(|f*|, eps)")
    ax.set_title("Hybrid gap-to-exact (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_gap, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-root", type=Path, default=ROOT / "artifacts")
    args = ap.parse_args()

    art = args.artifacts_root
    off = art / "benchmark_official"
    hy = art / "benchmark_hybrid"
    hy.mkdir(parents=True, exist_ok=True)
    off.mkdir(parents=True, exist_ok=True)

    copied = []
    copied.append(
        _copy_if_exists(art / "dqi_convergence_10var.png", art / "hybrid_dqi_convergence_10var.png")
    )
    copied.append(
        _copy_if_exists(off / "dqi_official_sizes_comparison.png", off / "hybrid_dqi_official_sizes_comparison.png")
    )
    copied.append(
        _copy_if_exists(off / "test_plot.png", off / "hybrid_test_plot.png")
    )
    copied.append(
        _copy_if_exists(art / "dqi_histogram_10var.png", hy / "hybrid_dqi_histogram_energy.png")
    )
    copied.append(
        _copy_if_exists(art / "dqi_objective_10var.png", hy / "hybrid_dqi_objective_10var.png")
    )
    copied.append(
        _copy_if_exists(art / "dqi_benchmark_comparison.png", hy / "hybrid_dqi_optimizer_benchmark.png")
    )

    summary_path = off / "dqi_official_sizes_summary.json"
    if summary_path.is_file():
        rows = _rows_from_official_summary(summary_path)
        _plot_hybrid_panels(rows, hy)
        (hy / "hybrid_rebuilt_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("rebuilt hybrid graphs from existing artifacts.")
    print("copied_count:", int(sum(1 for x in copied if x)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

