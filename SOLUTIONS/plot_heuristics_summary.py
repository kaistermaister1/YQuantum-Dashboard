#!/usr/bin/env python3
"""Aggregate HEURISTICS run_summaries.csv into one compact performance figure with error bars.

Writes a single PNG (2×2 panels plus a settings table: seed(s), λ, n_total, run_id).
Error-bar conventions (see panel subtitles):
- Best profit / classical ratio and wall-clock time: sample standard deviation across
  independent runs in the same (optimizer, n, m, p) group. Omitted (zero height) when
  only one run exists.
- Feasible-sample fraction: pooled binomial standard error across all shots summed in
  the group, sqrt(p_hat * (1 - p_hat) / N_total), treating shots as i.i.d. Bernoulli.

Requires: numpy, matplotlib (same stack as other SOLUTIONS scripts).
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# Align with qaoa_plots.py / YQuantum design tokens
QAOA_BLUE = "#0066CC"
YALE_BLUE = "#00356B"
CLASSICAL_GREEN = "#2D8C3C"
TRAVELERS_RED = "#C41230"
SHELL_BG = "#F3F4F6"

_SERIES = [QAOA_BLUE, YALE_BLUE, CLASSICAL_GREEN, TRAVELERS_RED]

_ERR_KW = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#374151"}


def _f(row: dict[str, str], key: str) -> float | None:
    v = row.get(key, "").strip()
    if v == "":
        return None
    return float(v)


def _i(row: dict[str, str], key: str) -> int | None:
    v = row.get(key, "").strip()
    if v == "":
        return None
    return int(float(v))


def _s(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def format_lambda_display(raw: str, max_len: int = 96) -> str:
    """Pretty-print λ vector from CSV (list string); fall back to truncated raw."""
    raw = raw.strip()
    if not raw:
        return "—"
    try:
        vals = ast.literal_eval(raw)
        if isinstance(vals, (list, tuple)) and vals:
            inner = ", ".join(f"{float(v):.6g}" for v in vals)
            out = f"[{inner}]"
            return out if len(out) <= max_len else out[: max_len - 1] + "…"
    except (SyntaxError, ValueError, TypeError):
        pass
    return raw if len(raw) <= max_len else raw[: max_len - 1] + "…"


def collect_settings_from_rows(rs: list[dict[str, str]]) -> dict[str, Any]:
    """Distinct seeds, λ strings, and run_ids for rows in one aggregate group."""
    seeds: set[str] = set()
    lambdas_raw: set[str] = set()
    run_ids: list[str] = []
    for r in rs:
        rid = _s(r, "run_id")
        if rid:
            run_ids.append(rid)
        sd = _s(r, "seed")
        if sd != "":
            seeds.add(sd)
        lam = _s(r, "lambda")
        if lam:
            lambdas_raw.add(lam)
    seeds_sorted = sorted(seeds, key=lambda s: (len(s), s))
    seeds_display = ", ".join(seeds_sorted) if seeds_sorted else "—"
    if len(lambdas_raw) == 1:
        (only_lam,) = tuple(lambdas_raw)
        lambda_display = format_lambda_display(only_lam)
    elif not lambdas_raw:
        lambda_display = "—"
    else:
        parts = [format_lambda_display(l, max_len=72) for l in sorted(lambdas_raw)]
        lambda_display = " | ".join(parts)
    return {
        "seeds_display": seeds_display,
        "lambda_display": lambda_display,
        "run_ids": run_ids,
        "n_distinct_lambda": len(lambdas_raw),
    }


def load_qaoa_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("algorithm", "").strip() == "qaoa":
                rows.append(row)
    return rows


def group_key(row: dict[str, str]) -> tuple[str, int, int, int]:
    opt = row.get("optimizer", "").strip() or "unknown"
    n = _i(row, "N_local")
    m = _i(row, "M_blocks")
    p = _i(row, "p")
    if n is None or m is None or p is None:
        raise ValueError(f"Missing N_local/M_blocks/p in row {row.get('run_id')}")
    return (opt, n, m, p)


def aggregate_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        buckets[group_key(r)].append(r)

    out: list[dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[1], k[2], k[3], k[0])):
        opt, n, m, p = key
        rs = buckets[key]
        bp = np.array([_f(x, "best_profit") for x in rs], dtype=float)
        co = np.array([_f(x, "classical_opt_profit") for x in rs], dtype=float)
        rt = np.array([_f(x, "runtime_sec") for x in rs], dtype=float)
        tot = np.array([_i(x, "num_samples_total") or 0 for x in rs], dtype=float)
        feas = np.array([_i(x, "num_samples_feasible") or 0 for x in rs], dtype=float)
        gates = np.array([_i(x, "two_qubit_gate_count") or 0 for x in rs], dtype=float)

        ratio = bp / co
        n_runs = len(rs)
        ratio_mean = float(np.mean(ratio))
        ratio_std = float(np.std(ratio, ddof=1)) if n_runs > 1 else 0.0
        if math.isnan(ratio_std):
            ratio_std = 0.0
        rt_mean = float(np.mean(rt))
        rt_std = float(np.std(rt, ddof=1)) if n_runs > 1 else 0.0
        if math.isnan(rt_std):
            rt_std = 0.0

        sum_feas = float(np.sum(feas))
        sum_tot = float(np.sum(tot))
        if sum_tot > 0:
            p_hat = sum_feas / sum_tot
            feas_se = math.sqrt(max(p_hat * (1.0 - p_hat) / sum_tot, 0.0))
        else:
            p_hat = float("nan")
            feas_se = 0.0

        settings = collect_settings_from_rows(rs)
        n_totals = sorted({_i(x, "n_total") for x in rs if _i(x, "n_total") is not None})
        n_total_display = ", ".join(str(t) for t in n_totals) if n_totals else "—"

        out.append(
            {
                "optimizer": opt,
                "N_local": n,
                "M_blocks": m,
                "p": p,
                "n_runs": n_runs,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "rt_mean": rt_mean,
                "rt_std": rt_std,
                "feas_pooled": p_hat,
                "feas_se": feas_se,
                "sum_tot": sum_tot,
                "gates_mean": float(np.mean(gates)) if len(gates) else 0.0,
                "seeds_display": settings["seeds_display"],
                "lambda_display": settings["lambda_display"],
                "run_ids": settings["run_ids"],
                "n_total_display": n_total_display,
            }
        )
    return out


def _label(g: dict[str, Any]) -> str:
    return f"{g['optimizer'].upper()}\nn={g['N_local']} m={g['M_blocks']} p={g['p']}"


def _colors(n: int) -> list[str]:
    return [_SERIES[i % len(_SERIES)] for i in range(n)]


def _add_settings_table(fig: plt.Figure, groups: list[dict[str, Any]], gs_table: gridspec.GridSpec) -> None:
    ax_tbl = fig.add_subplot(gs_table)
    ax_tbl.axis("off")
    ax_tbl.set_title(
        "Run settings (from run_summaries.csv: seed, λ per block, n_total; run_id for traceability)",
        fontsize=9,
        loc="left",
        pad=6,
    )

    headers = ["Configuration", "runs", "n_total", "seed(s)", "λ (block penalties)", "run_id"]
    rows_data: list[list[str]] = []
    for g in groups:
        cfg = f"{g['optimizer'].upper()} n={g['N_local']} m={g['M_blocks']} p={g['p']}"
        rids = g.get("run_ids") or []
        if len(rids) == 1:
            rid_cell = rids[0]
        else:
            rid_cell = "\n".join(rids) if rids else "—"
        rows_data.append(
            [
                cfg,
                str(g["n_runs"]),
                str(g.get("n_total_display", "—")),
                str(g.get("seeds_display", "—")),
                str(g.get("lambda_display", "—")),
                rid_cell,
            ]
        )

    ncols = len(headers)
    table = ax_tbl.table(
        cellText=rows_data,
        colLabels=headers,
        loc="upper center",
        cellLoc="left",
        colColours=["#E5E7EB"] * ncols,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=6.5)
            cell.set_facecolor("#E5E7EB")
        else:
            cell.set_facecolor("#FAFAFA" if (row - 1) % 2 == 0 else "white")


def plot_combined_summary(groups: list[dict[str, Any]], out: Path) -> None:
    n_groups = len(groups)
    x = np.arange(n_groups)
    colors = _colors(n_groups)
    tick_labels = [_label(g) for g in groups]
    tick_fs = 7 if n_groups > 5 else 8

    table_row_weight = max(0.55, min(1.35, 0.22 * (n_groups + 1)))
    fig = plt.figure(figsize=(12, 9.5 + 0.35 * n_groups))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0, table_row_weight],
        hspace=0.38,
        wspace=0.28,
        top=0.93,
        bottom=0.06,
        left=0.07,
        right=0.98,
    )
    fig.suptitle("QAOA heuristics summary (from run_summaries.csv)", fontsize=12, y=0.97)

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[1, 0])
    axes[1, 1] = fig.add_subplot(gs[1, 1])

    # (0,0) Profit ratio
    ax = axes[0, 0]
    ax.set_facecolor(SHELL_BG)
    means = [g["ratio_mean"] for g in groups]
    errs = [g["ratio_std"] for g in groups]
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=3,
        color=colors,
        edgecolor="#1F2937",
        linewidth=0.5,
        error_kw=_ERR_KW,
    )
    ax.axhline(1.0, color=CLASSICAL_GREEN, linestyle="--", linewidth=1.2, label="ratio = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_ylabel("best / classical optimum")
    ax.set_title("Profit recovery (error bars: SD across runs in group)", fontsize=9)
    hi = max((m + (e or 0.0) for m, e in zip(means, errs, strict=True)), default=1.0)
    ax.set_ylim(0, max(1.05, hi * 1.12))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)

    # (0,1) Feasibility
    ax = axes[0, 1]
    ax.set_facecolor(SHELL_BG)
    means = [g["feas_pooled"] for g in groups]
    errs = [g["feas_se"] for g in groups]
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=3,
        color=colors,
        edgecolor="#1F2937",
        linewidth=0.5,
        error_kw=_ERR_KW,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_ylabel("feasible / total (pooled)")
    ax.set_title("Feasible shot rate (error bars: pooled binomial SE)", fontsize=9)
    top_f = max((m + e for m, e in zip(means, errs, strict=True) if not math.isnan(m)), default=0.2)
    ax.set_ylim(0, max(0.25, top_f * 1.15))
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)

    # (1,0) Runtime
    ax = axes[1, 0]
    ax.set_facecolor(SHELL_BG)
    means = [g["rt_mean"] / 60.0 for g in groups]
    errs = [g["rt_std"] / 60.0 for g in groups]
    ax.bar(
        x,
        means,
        yerr=errs,
        capsize=3,
        color=colors,
        edgecolor="#1F2937",
        linewidth=0.5,
        error_kw=_ERR_KW,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_ylabel("wall time (minutes)")
    ax.set_title("Runtime (error bars: SD across runs in group)", fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)

    # (1,1) Two-qubit gates
    ax = axes[1, 1]
    ax.set_facecolor(SHELL_BG)
    means = [g["gates_mean"] for g in groups]
    ax.bar(x, means, color=colors, edgecolor="#1F2937", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=tick_fs)
    ax.set_ylabel("two-qubit gates (template)")
    ax.set_title("Circuit template (no error bar)", fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=8)

    _add_settings_table(fig, groups, gs[2, :])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        type=Path,
        default=Path(__file__).resolve().parent / "HEURISTICS" / "run_summaries.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "HEURISTICS" / "plots",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="heuristics_summary_v1.png",
        help="Filename for the combined figure (written under --out-dir).",
    )
    args = parser.parse_args()

    rows = load_qaoa_rows(args.summaries)
    if not rows:
        raise SystemExit(f"No qaoa rows found in {args.summaries}")

    groups = aggregate_groups(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / args.output_name
    plot_combined_summary(groups, out_path)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
