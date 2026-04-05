#!/usr/bin/env python3
"""Wall-clock runtime vs M_blocks (packages) with a *proportional shot budget*.

Selects local QAOA rows where ``num_samples_total = k * M_blocks`` for a fixed integer
``k`` (same workload per package block), so comparing wall time across ``m`` is not
confounded by holding total shots fixed while circuit width grows.

Plots measured points with a linear least-squares fit and R². Use ``--auto`` to pick
the (N_local, p, optimizer, k) group with the most distinct ``m`` values."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_algorithm_comparison_table import load_rows

REPO_SOLUTIONS = Path(__file__).resolve().parent

QAOA_COBYLA = "#0066CC"
QAOA_SPSA = "#00356B"
SHELL_BG = "#F3F4F6"

_RUN_TS = re.compile(r"_(\d{8}T\d{6}Z)$")


def _i(row: dict[str, str], key: str) -> int | None:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return int(float(v))


def _f(row: dict[str, str], key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return float(v)


def is_local_qaoa(row: dict[str, str]) -> bool:
    if (row.get("algorithm") or "").strip() != "qaoa":
        return False
    opt = (row.get("optimizer") or "").strip().lower()
    if opt not in {"cobyla", "spsa"}:
        return False
    return "execution_target=local" in (row.get("notes") or "")


def run_timestamp(run_id: str) -> str:
    m = _RUN_TS.search(run_id or "")
    return m.group(1) if m else ""


def row_key_shots_per_m(row: dict[str, str]) -> tuple[int, int, str, int, int] | None:
    """(N_local, p, optimizer_lower, k, M_blocks) if shots = k * m."""
    m = _i(row, "M_blocks")
    sh = _i(row, "num_samples_total")
    n = _i(row, "N_local")
    p = _i(row, "p")
    opt = (row.get("optimizer") or "").strip().lower()
    if m is None or sh is None or n is None or p is None or m <= 0 or sh <= 0:
        return None
    if sh % m != 0:
        return None
    k = sh // m
    return (n, p, opt, k, m)


def latest_per_m(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """Prefer lexicographically greatest run_id suffix (timestamp) per M_blocks."""
    by_m: dict[int, dict[str, str]] = {}
    for r in rows:
        mm = _i(r, "M_blocks")
        if mm is None:
            continue
        cur = by_m.get(mm)
        if cur is None:
            by_m[mm] = r
            continue
        t_new, t_old = run_timestamp(r["run_id"]), run_timestamp(cur["run_id"])
        if t_new > t_old:
            by_m[mm] = r
    return by_m


def find_best_auto(rows: list[dict[str, str]]) -> tuple[int, int, str, int, dict[int, dict[str, str]]]:
    q = [r for r in rows if is_local_qaoa(r)]
    from collections import defaultdict

    groups: dict[tuple[int, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for r in q:
        key = row_key_shots_per_m(r)
        if key is None:
            continue
        n, p, opt, k, m = key
        groups[(n, p, opt, k)].append(r)

    best_rank: tuple[int, int, int, int, int] | None = None
    best_payload: tuple[int, int, str, int, dict[int, dict[str, str]]] | None = None
    for (n, p, opt, k), rs in groups.items():
        by_m = latest_per_m(rs)
        sz = len(by_m)
        if sz < 3:
            continue
        # Tie-break: more m values, then higher p, then COBYLA before SPSA, then larger n, then k.
        rank = (sz, p, 1 if opt == "cobyla" else 0, n, k)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (n, p, opt, k, by_m)
    if best_payload is None:
        raise SystemExit(
            "No (N_local, p, optimizer, k) group with ≥3 distinct M_blocks where "
            "num_samples_total is divisible by M_blocks (constant shots per package). "
            "Add runs or pass explicit --n --p --optimizer --shots-per-m."
        )
    return best_payload


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return slope, intercept, R²."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--summaries",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_SOLUTIONS / "HEURISTICS" / "plots")
    ap.add_argument("--output-name", default=None)
    ap.add_argument("--auto", action="store_true", help="Pick largest proportional-budget series.")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--optimizer", type=str, default=None, choices=["cobyla", "spsa"])
    ap.add_argument(
        "--shots-per-m",
        type=int,
        default=None,
        help="Integer k such that num_samples_total = k * M_blocks (required if not --auto).",
    )
    ap.add_argument(
        "--second-optimizer",
        type=str,
        default=None,
        choices=["cobyla", "spsa"],
        help="Optional second series (same n, p, k) for comparison.",
    )
    args = ap.parse_args()

    rows = load_rows(args.summaries)

    if args.auto:
        n, p, opt, k, by_m = find_best_auto(rows)
    else:
        if args.n is None or args.p is None or args.optimizer is None or args.shots_per_m is None:
            raise SystemExit("Pass --auto or all of --n --p --optimizer --shots-per-m.")
        n, p, opt, k = args.n, args.p, args.optimizer.lower(), args.shots_per_m
        q = [
            r
            for r in rows
            if is_local_qaoa(r)
            and _i(r, "N_local") == n
            and _i(r, "p") == p
            and (r.get("optimizer") or "").strip().lower() == opt
            and _i(r, "num_samples_total") == k * (_i(r, "M_blocks") or -1)
        ]
        by_m = latest_per_m(q)
        if len(by_m) < 2:
            raise SystemExit(f"No matching rows for n={n} p={p} {opt} with shots = {k} * M_blocks.")

    ms = np.array(sorted(by_m.keys()), dtype=float)
    ts = np.array([_f(by_m[int(mi)], "runtime_sec") or 0.0 for mi in ms], dtype=float)
    evals = [_i(by_m[int(mi)], "num_objective_evals") for mi in ms]
    shots = [_i(by_m[int(mi)], "num_samples_total") for mi in ms]

    slope, icept, r2 = linear_fit_r2(ms, ts)
    x_line = np.linspace(ms.min(), ms.max(), 100)
    y_line = slope * x_line + icept

    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor=SHELL_BG)
    ax.set_facecolor("#FFFFFF")

    color = QAOA_COBYLA if opt == "cobyla" else QAOA_SPSA
    ax.scatter(ms, ts, s=64, color=color, zorder=3, label=f"QAOA ({opt.upper()}) measured")
    ax.plot(x_line, y_line, "--", color=color, linewidth=1.8, alpha=0.85, label="Linear fit")

    series_note = (
        f"$N_{{\\mathrm{{local}}}}={n}$, $p={p}$, {opt.upper()}, local\n"
        f"Fixed shots/package $k={k}$ "
        f"({{\\tt num\\_samples\\_total}} $= k \\cdot M_{{\\mathrm{{blocks}}}}$)\n"
        f"Objective evals at points: {evals}"
    )

    if args.second_optimizer and args.second_optimizer != opt:
        opt2 = args.second_optimizer
        q2 = [
            r
            for r in rows
            if is_local_qaoa(r)
            and _i(r, "N_local") == n
            and _i(r, "p") == p
            and (r.get("optimizer") or "").strip().lower() == opt2
            and _i(r, "num_samples_total") == k * (_i(r, "M_blocks") or -1)
        ]
        by_m2 = latest_per_m(q2)
        if len(by_m2) >= 2:
            ms2 = np.array(sorted(by_m2.keys()), dtype=float)
            ts2 = np.array([_f(by_m2[int(mi)], "runtime_sec") or 0.0 for mi in ms2], dtype=float)
            c2 = QAOA_SPSA if opt2 == "spsa" else QAOA_COBYLA
            s2, i2, r2_2 = linear_fit_r2(ms2, ts2)
            xl2 = np.linspace(ms2.min(), ms2.max(), 100)
            ax.scatter(ms2, ts2, s=56, color=c2, marker="s", zorder=3, label=f"QAOA ({opt2.upper()}) measured")
            ax.plot(xl2, s2 * xl2 + i2, ":", color=c2, linewidth=1.6, alpha=0.85, label=f"{opt2.upper()} linear fit")
            series_note += f"\nSecond series evals ({opt2}): {[_i(by_m2[int(mi)], 'num_objective_evals') for mi in ms2]}"

    ax.set_xlabel(r"Packages $M_{\mathrm{blocks}}$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        r"Runtime vs $M_{\mathrm{blocks}}$ with proportional shot budget"
        + f"\nLinear fit: $R^2 = {r2:.4f}$, slope ≈ {slope:.3f} s / package",
        fontsize=10,
    )
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.02, series_note, ha="center", fontsize=7, color="#374151")

    fig.subplots_adjust(bottom=0.22)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or f"runtime_vs_m_blocks_n{n}_p{p}_{opt}_k{k}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(
        f"  m={list(map(int, ms))}  runtime_sec={[float(x) for x in ts]}  "
        f"slope={slope:.6g}  R2={r2:.6g}"
    )


if __name__ == "__main__":
    main()
