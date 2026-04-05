#!/usr/bin/env python3
"""Line plot: wall-clock runtime vs N_local for classical ILP vs QAOA (COBYLA / SPSA).

Reads HEURISTICS/run_summaries.csv. Uses execution_target=local QAOA rows only (skips
Selene / random_batch). For each n, picks COBYLA and SPSA rows that share the same λ when
possible; otherwise falls back to the latest run_id per optimizer (subtitle notes this).

Y-axis is logarithmic by default because classical times are ~1e-2 s while QAOA is ~1e2 s.

Requires: matplotlib, numpy."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_algorithm_comparison_table import lambdas_close, load_rows, parse_lambda_tuple

REPO_SOLUTIONS = Path(__file__).resolve().parent

CLASSICAL_GREEN = "#2D8C3C"
QAOA_COBYLA = "#0066CC"
QAOA_SPSA = "#00356B"
SHELL_BG = "#F3F4F6"


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
    notes = row.get("notes") or ""
    return "execution_target=local" in notes


def run_id_sort_key(run_id: str) -> str:
    m = re.search(r"_(\d{8}T\d{6}Z)$", run_id.strip())
    return m.group(1) if m else run_id


def lam_key(raw: str) -> tuple[float, ...] | None:
    t = parse_lambda_tuple(raw or "")
    return t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--summaries",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv",
    )
    p.add_argument("--m", type=int, default=3, help="Fixed M_blocks (packages).")
    p.add_argument("--p", type=int, default=1, help="Fixed QAOA depth p.")
    p.add_argument(
        "--lambda",
        dest="lambda_str",
        default=None,
        help="If set, only QAOA rows whose λ equals this (after parsing) are used.",
    )
    p.add_argument(
        "--linear-y",
        action="store_true",
        help="Use linear seconds on y-axis (classical will hug the x-axis).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "plots",
    )
    p.add_argument("--output-name", default="runtime_vs_n_classical_qaoa_m3_p1.png")
    args = p.parse_args()

    target_lam: tuple[float, ...] | None = None
    if args.lambda_str:
        target_lam = parse_lambda_tuple(args.lambda_str)
        if target_lam is None:
            raise SystemExit("Could not parse --lambda")

    rows = load_rows(args.summaries)
    classical_rows = [
        r
        for r in rows
        if (r.get("algorithm") or "").strip() == "classical"
        and _i(r, "N_local") is not None
        and _i(r, "M_blocks") == args.m
    ]
    q_rows = [r for r in rows if is_local_qaoa(r) and _i(r, "M_blocks") == args.m and _i(r, "p") == args.p]

    by_n_classical: dict[int, dict[str, str]] = {}
    for r in classical_rows:
        n = _i(r, "N_local")
        if n is None:
            continue
        prev = by_n_classical.get(n)
        if prev is None or run_id_sort_key(r["run_id"]) > run_id_sort_key(prev["run_id"]):
            by_n_classical[n] = r

    cobyla_by_n: dict[int, list[dict[str, str]]] = {}
    spsa_by_n: dict[int, list[dict[str, str]]] = {}
    for r in q_rows:
        opt = (r.get("optimizer") or "").strip().lower()
        n = _i(r, "N_local")
        if n is None:
            continue
        if target_lam is not None:
            rk = lam_key(r.get("lambda") or "")
            if rk is None or not lambdas_close(rk, target_lam):
                continue
        if opt == "cobyla":
            cobyla_by_n.setdefault(n, []).append(r)
        else:
            spsa_by_n.setdefault(n, []).append(r)

    ns_sorted = sorted(
        set(by_n_classical) & set(cobyla_by_n) & set(spsa_by_n),
        key=int,
    )
    if not ns_sorted:
        raise SystemExit(
            "No n values with classical + local COBYLA + local SPSA for this filter. "
            "Try different --m/--p or omit --lambda."
        )

    ns_plot: list[int] = []
    classical_sec: list[float] = []
    cobyla_sec: list[float] = []
    spsa_sec: list[float] = []
    lambda_matched: list[bool] = []

    for n in ns_sorted:
        c_row = by_n_classical[n]
        c_opts = cobyla_by_n[n]
        s_opts = spsa_by_n[n]

        lam_c = {lam_key(r.get("lambda") or "") for r in c_opts}
        lam_s = {lam_key(r.get("lambda") or "") for r in s_opts}
        lam_c.discard(None)
        lam_s.discard(None)
        common = lam_c & lam_s

        matched = bool(common)
        if common:
            pick_lam = sorted(common, key=lambda t: tuple(t))[0]
            c_pick = max((r for r in c_opts if lam_key(r.get("lambda") or "") == pick_lam), key=lambda r: run_id_sort_key(r["run_id"]))
            s_pick = max((r for r in s_opts if lam_key(r.get("lambda") or "") == pick_lam), key=lambda r: run_id_sort_key(r["run_id"]))
        else:
            c_pick = max(c_opts, key=lambda r: run_id_sort_key(r["run_id"]))
            s_pick = max(s_opts, key=lambda r: run_id_sort_key(r["run_id"]))

        rt_cl = _f(c_row, "runtime_sec")
        rt_co = _f(c_pick, "runtime_sec")
        rt_sp = _f(s_pick, "runtime_sec")
        if rt_cl is None or rt_co is None or rt_sp is None:
            continue
        ns_plot.append(n)
        classical_sec.append(rt_cl)
        cobyla_sec.append(rt_co)
        spsa_sec.append(rt_sp)
        lambda_matched.append(matched)

    if not ns_plot:
        raise SystemExit("No complete runtime triples after filtering.")

    x = np.array(ns_plot, dtype=float)
    y_c = np.array(classical_sec, dtype=float)
    y_co = np.array(cobyla_sec, dtype=float)
    y_sp = np.array(spsa_sec, dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SHELL_BG)

    ax.plot(x, y_c, color=CLASSICAL_GREEN, marker="o", linewidth=2.0, markersize=7, label="Classical (ILP)")
    ax.plot(x, y_co, color=QAOA_COBYLA, marker="s", linewidth=2.0, markersize=7, label="QAOA COBYLA (local)")
    ax.plot(x, y_sp, color=QAOA_SPSA, marker="^", linewidth=2.0, markersize=7, label="QAOA SPSA (local)")

    if not args.linear_y:
        ax.set_yscale("log")

    ax.set_xlabel("N_local (coverages)")
    ax.set_ylabel("Wall time (seconds)" + ("" if args.linear_y else ", log scale"))
    subtitle = f"m={args.m}, p={args.p} · local QAOA only"
    if target_lam is not None:
        subtitle += " · λ filter applied"
    elif not all(lambda_matched):
        subtitle += " · at some n, COBYLA/SPSA used different λ (latest run each)"
    ax.set_title("Runtime vs subsample size: classical vs QAOA\n" + subtitle, fontsize=10)
    ax.grid(True, which="both", axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.grid(True, which="major", axis="x", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xticks(x)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.output_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} (n = {list(map(int, x))})")


if __name__ == "__main__":
    main()
