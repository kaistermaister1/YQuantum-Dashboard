#!/usr/bin/env python3
"""Grouped bar chart: best_profit / classical optimum vs N_local for fixed (m, p, λ).

Uses Classical + QAOA (COBYLA / SPSA) rows from HEURISTICS/run_summaries.csv.
Horizontal reference line at ratio = 1. Requires matplotlib + numpy."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_algorithm_comparison_table import (
    DEFAULT_LAMBDA_BY_N,
    load_rows,
    parse_lambda_tuple,
    pick_classical,
    pick_qaoa,
    lambda_display,
)

REPO_SOLUTIONS = Path(__file__).resolve().parent

QAOA_COBYLA = "#0066CC"
QAOA_SPSA = "#00356B"
CLASSICAL_GREEN = "#2D8C3C"
SHELL_BG = "#F3F4F6"


def _f(row: dict[str, str], key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return float(v)


def ratio_vs_classical(q: dict[str, str] | None, classical: dict[str, str] | None) -> float | None:
    if q is None or classical is None:
        return None
    bq = _f(q, "best_profit")
    bc = _f(classical, "best_profit")
    if bq is None or bc is None or bc == 0:
        return None
    return bq / bc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summaries",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv",
    )
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--p", type=int, default=1)
    p.add_argument(
        "--n-list",
        type=str,
        default="5,7,10",
        help="Comma-separated N_local values (order preserved).",
    )
    p.add_argument(
        "--lambda",
        dest="lambda_str",
        default=None,
        help="λ list; default: DEFAULT_LAMBDA_BY_N[--n] for first n in --n-list, or fail.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "plots",
    )
    p.add_argument("--output-name", default="profit_ratio_vs_n_m3_p1.png")
    args = p.parse_args()

    ns = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    if not ns:
        raise SystemExit("Empty --n-list")

    if args.lambda_str:
        lam = parse_lambda_tuple(args.lambda_str)
        if lam is None:
            raise SystemExit("Could not parse --lambda")
        lambda_target = lam
    else:
        ref_n = ns[0]
        if ref_n not in DEFAULT_LAMBDA_BY_N:
            raise SystemExit(f"No default λ for n={ref_n}; pass --lambda explicitly.")
        lambda_target = DEFAULT_LAMBDA_BY_N[ref_n]

    rows = load_rows(args.summaries)
    cobyla_r: list[float | None] = []
    spsa_r: list[float | None] = []
    for n in ns:
        classical = pick_classical(rows, n=n, m=args.m)
        qc = pick_qaoa(
            rows, n=n, m=args.m, p=args.p, lambda_target=lambda_target, optimizer="cobyla"
        )
        qs = pick_qaoa(
            rows, n=n, m=args.m, p=args.p, lambda_target=lambda_target, optimizer="spsa"
        )
        cobyla_r.append(ratio_vs_classical(qc, classical))
        spsa_r.append(ratio_vs_classical(qs, classical))

    if all(v is None for v in cobyla_r + spsa_r):
        raise SystemExit("No ratios computed (missing classical or QAOA rows for this filter).")

    x = np.arange(len(ns))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SHELL_BG)

    def _vals(series: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
        y = np.array([0.0 if v is None else v for v in series], dtype=float)
        mask = np.array([v is not None for v in series], dtype=bool)
        return y, mask

    y_c, m_c = _vals(cobyla_r)
    y_s, m_s = _vals(spsa_r)
    ax.bar(
        x - width / 2,
        np.where(m_c, y_c, np.nan),
        width,
        label="QAOA COBYLA",
        color=QAOA_COBYLA,
        edgecolor="#1F2937",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        np.where(m_s, y_s, np.nan),
        width,
        label="QAOA SPSA",
        color=QAOA_SPSA,
        edgecolor="#1F2937",
        linewidth=0.5,
    )
    ax.axhline(1.0, color=CLASSICAL_GREEN, linestyle="--", linewidth=1.2, label="classical optimum")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N_local (coverages)")
    ax.set_ylabel("best_profit / classical optimum")
    ax.set_title(
        f"Profit recovery vs subsample size (m={args.m}, p={args.p}, λ = {lambda_display(lambda_target)})",
        fontsize=10,
    )
    chunks = [y_c[m_c], y_s[m_s]]
    chunks = [c for c in chunks if c.size > 0]
    hi = float(np.nanmax(np.concatenate(chunks))) if chunks else 1.0
    ax.set_ylim(0, max(1.05, hi * 1.12))
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.tick_params(axis="both", labelsize=9)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.output_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
