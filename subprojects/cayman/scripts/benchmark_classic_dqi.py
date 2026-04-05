#!/usr/bin/env python3
"""Benchmark parity-native classic DQI on subsampled insurance sizes.

Compares:
- classic_dqi (expected post-selected metrics)
- random_uniform baseline (same shot budget)
- optional exact parity optimum for small variable counts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_classic_pipeline import run_classic_dqi, score_f, score_s
from src.dqi_insurance_parity import build_insurance_parity_B_rhs
from src.insurance_model import load_ltm_instance, subsample_problem


OFFICIAL_NS_DEFAULT = (10, 20)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Directory with instance_*.csv files")
    ap.add_argument("--package", type=int, default=0, help="Package index")
    ap.add_argument(
        "--ns",
        type=int,
        nargs="*",
        default=list(OFFICIAL_NS_DEFAULT),
        help="Coverage counts to benchmark (default: 10 20)",
    )
    ap.add_argument("--skip-ns", type=int, nargs="*", default=[], help="Coverage counts to skip")
    ap.add_argument("--quick", action="store_true", help="Shortcut for ns = 10,20")
    ap.add_argument("--subsample-packages", type=int, default=0, help="Keep first k packages (0 = all)")
    ap.add_argument("--ell", type=int, default=1)
    ap.add_argument("--bp-iterations", type=int, default=1)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--allow-dirty-ancilla",
        action="store_true",
        help="Do not fail when ancilla bits are non-zero (strict by default).",
    )
    ap.add_argument(
        "--bruteforce-max-nvars",
        type=int,
        default=22,
        help="Compute exact parity optimum only when n_vars <= this threshold.",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: cayman/artifacts/benchmark_classic")
    return ap.parse_args()


def _load_problem(data_dir: Path, n_coverages: int, subsample_packages: int):
    problem = load_ltm_instance(data_dir)
    n_cov = min(int(n_coverages), int(problem.N))
    n_pkg = int(subsample_packages) if int(subsample_packages) > 0 else int(problem.M)
    n_pkg = min(n_pkg, int(problem.M))
    return subsample_problem(problem, n_cov, n_pkg)


def _random_uniform_expected(B: np.ndarray, v: np.ndarray, *, shots: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n_vars = int(B.shape[1])
    t0 = time.perf_counter()
    s_vals: list[float] = []
    f_vals: list[float] = []
    for _ in range(int(shots)):
        x = rng.integers(0, 2, size=n_vars, dtype=np.int8)
        xs = "".join(str(int(bit)) for bit in x.tolist())
        s_vals.append(float(score_s(B, v, xs)))
        f_vals.append(float(score_f(B, v, xs)))
    dt = time.perf_counter() - t0
    return {
        "expected_s": float(np.mean(s_vals)) if s_vals else 0.0,
        "expected_f": float(np.mean(f_vals)) if f_vals else 0.0,
        "runtime_s": float(dt),
        "cost": int(shots),
    }


def _bruteforce_opt_f(B: np.ndarray, v: np.ndarray, *, max_nvars: int) -> dict[str, float] | None:
    n_vars = int(B.shape[1])
    if n_vars > int(max_nvars):
        return None
    best_f = -float("inf")
    best_s = -float("inf")
    t0 = time.perf_counter()
    for mask in range(1 << n_vars):
        bits = "".join("1" if ((mask >> i) & 1) else "0" for i in range(n_vars))
        ff = float(score_f(B, v, bits))
        if ff > best_f:
            best_f = ff
            best_s = float(score_s(B, v, bits))
    dt = time.perf_counter() - t0
    return {"best_f": float(best_f), "best_s": float(best_s), "runtime_s": float(dt)}


def _plot_main(rows: list[dict], out_path: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    dqi_f = [float(r["classic_dqi"]["expected_f"]) if r["classic_dqi"]["expected_f"] is not None else np.nan for r in rows]
    rnd_f = [float(r["random_uniform"]["expected_f"]) for r in rows]
    best_f = [float(r["bruteforce"]["best_f"]) if r.get("bruteforce") is not None else np.nan for r in rows]
    keep = [float(r["classic_dqi"]["keep_rate"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), layout="constrained")

    axes[0].plot(ns, dqi_f, marker="o", linewidth=1.5, label="Classic DQI <f>")
    axes[0].plot(ns, rnd_f, marker="s", linewidth=1.2, label="Random uniform <f>")
    if not np.all(np.isnan(best_f)):
        axes[0].plot(ns, best_f, marker="^", linewidth=1.2, label="Exact best f (small n_vars)")
    axes[0].set_xlabel("Coverage count n")
    axes[0].set_ylabel("Parity objective f (higher is better)")
    axes[0].set_title("Classic DQI quality comparison")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(ns, keep, marker="o", linewidth=1.5, color="#E65100")
    axes[1].set_xlabel("Coverage count n")
    axes[1].set_ylabel("Post-selection keep rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Classic DQI post-selection rate")
    axes[1].grid(True, alpha=0.25)

    fig.savefig(out_path, dpi=150)


def _plot_runtime(rows: list[dict], out_path: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    dqi_t = [float(r["classic_dqi"]["runtime_s"]) for r in rows]
    rnd_t = [float(r["random_uniform"]["runtime_s"]) for r in rows]
    bf_t = [float(r["bruteforce"]["runtime_s"]) if r.get("bruteforce") is not None else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.plot(ns, dqi_t, marker="o", linewidth=1.6, label="Classic DQI runtime")
    ax.plot(ns, rnd_t, marker="s", linewidth=1.2, label="Random uniform runtime")
    if not np.all(np.isnan(bf_t)):
        ax.plot(ns, bf_t, marker="^", linewidth=1.2, label="Exact brute runtime (small n_vars)")
    ax.set_xlabel("Coverage count n")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Classic benchmark runtime")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)


def _plot_workflow_cost(rows: list[dict], out_path: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    dqi_cost = [float(r["classic_dqi"]["cost"]) for r in rows]
    rnd_cost = [float(r["random_uniform"]["cost"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.plot(ns, dqi_cost, marker="o", linewidth=1.6, label="Classic DQI cost (shots)")
    ax.plot(ns, rnd_cost, marker="s", linewidth=1.2, label="Random uniform cost (samples)")
    ax.set_xlabel("Coverage count n")
    ax.set_ylabel("Workflow cost")
    ax.set_title("Workflow cost trend")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)


def _plot_expected_s(rows: list[dict], out_path: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    dqi_s = [float(r["classic_dqi"]["expected_s"]) if r["classic_dqi"]["expected_s"] is not None else np.nan for r in rows]
    rnd_s = [float(r["random_uniform"]["expected_s"]) for r in rows]
    best_s = [float(r["bruteforce"]["best_s"]) if r.get("bruteforce") is not None else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.plot(ns, dqi_s, marker="o", linewidth=1.6, label="Classic DQI <s>")
    ax.plot(ns, rnd_s, marker="s", linewidth=1.2, label="Random uniform <s>")
    if not np.all(np.isnan(best_s)):
        ax.plot(ns, best_s, marker="^", linewidth=1.2, label="Exact best s (small n_vars)")
    ax.set_xlabel("Coverage count n")
    ax.set_ylabel("Expected satisfied checks <s>")
    ax.set_title("Parity-check satisfaction comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)


def _plot_gap(rows: list[dict], out_path: Path) -> None:
    ns = [int(r["n"]) for r in rows]
    dqi_gap: list[float] = []
    rnd_gap: list[float] = []
    for row in rows:
        bf = row.get("bruteforce")
        if bf is None:
            dqi_gap.append(np.nan)
            rnd_gap.append(np.nan)
            continue
        best_f = float(bf["best_f"])
        denom = max(abs(best_f), 1e-12)
        dqi_f = float(row["classic_dqi"]["expected_f"]) if row["classic_dqi"]["expected_f"] is not None else np.nan
        rnd_f = float(row["random_uniform"]["expected_f"])
        dqi_gap.append((best_f - dqi_f) / denom if not np.isnan(dqi_f) else np.nan)
        rnd_gap.append((best_f - rnd_f) / denom)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    ax.plot(ns, dqi_gap, marker="o", linewidth=1.6, label="Classic DQI normalized f-gap")
    ax.plot(ns, rnd_gap, marker="s", linewidth=1.2, label="Random normalized f-gap")
    ax.axhline(0.0, linestyle="--", linewidth=1.0, color="#1B5E20")
    ax.set_xlabel("Coverage count n")
    ax.set_ylabel("(f* - f) / max(|f*|, eps)")
    ax.set_title("Gap-to-exact comparison (where exact available)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=150)


def main() -> int:
    args = parse_args()
    ns = [int(x) for x in args.ns]
    if args.quick:
        ns = [n for n in (10, 20) if n not in set(args.skip_ns)]
    else:
        ns = [n for n in ns if n not in set(args.skip_ns)]
    if not ns:
        print("No sizes to run.", file=sys.stderr)
        return 1

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (ROOT / "artifacts" / "benchmark_classic")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    strict_ancilla = not bool(args.allow_dirty_ancilla)

    for n in ns:
        small = _load_problem(
            data_dir,
            n_coverages=int(n),
            subsample_packages=int(args.subsample_packages),
        )
        if int(args.package) < 0 or int(args.package) >= int(small.M):
            raise ValueError(f"package {args.package} out of range for M={small.M}")
        B, v = build_insurance_parity_B_rhs(small, int(args.package))

        t0 = time.perf_counter()
        classic = run_classic_dqi(
            B,
            v,
            ell=int(args.ell),
            bp_iterations=int(args.bp_iterations),
            shots=int(args.shots),
            seed=int(args.seed),
            strict_ancilla=strict_ancilla,
        )
        classic_runtime = time.perf_counter() - t0

        random_row = _random_uniform_expected(
            B,
            v,
            shots=int(args.shots),
            seed=int(args.seed) + 1000 + int(n),
        )
        brute = _bruteforce_opt_f(B, v, max_nvars=int(args.bruteforce_max_nvars))

        row = {
            "n": int(n),
            "n_vars": int(B.shape[1]),
            "n_checks": int(B.shape[0]),
            "package": int(args.package),
            "reference_value": None if brute is None else float(brute["best_f"]),
            "methods": {
                "classic_dqi": {
                    "value": None if classic.expected_f is None else float(classic.expected_f),
                    "time_s": float(classic_runtime),
                    "cost": int(args.shots),
                },
                "random_uniform": {
                    "value": float(random_row["expected_f"]),
                    "time_s": float(random_row["runtime_s"]),
                    "cost": int(random_row["cost"]),
                },
            },
            "classic_dqi": {
                "expected_s": classic.expected_s,
                "expected_f": classic.expected_f,
                "keep_rate": float(classic.keep_rate),
                "postselected_shots": int(classic.postselected_shots),
                "total_shots": int(classic.total_shots),
                "runtime_s": float(classic_runtime),
                "cost": int(args.shots),
                "num_qubits": int(classic.num_qubits),
            },
            "random_uniform": random_row,
            "bruteforce": brute,
        }
        rows.append(row)

        print(f"=== n={n}  n_vars={B.shape[1]}  n_checks={B.shape[0]} ===")
        print(
            f"  classic_dqi     <f>={classic.expected_f}  <s>={classic.expected_s}  "
            f"keep={classic.keep_rate:.4f}  time={classic_runtime:.3f}s"
        )
        print(
            f"  random_uniform  <f>={random_row['expected_f']:.6f}  <s>={random_row['expected_s']:.6f}  "
            f"time={random_row['runtime_s']:.3f}s"
        )
        if brute is None:
            print(f"  exact_best_f    skipped (n_vars>{int(args.bruteforce_max_nvars)})")
        else:
            print(
                f"  exact_best_f    f={brute['best_f']:.6f}  s={brute['best_s']:.6f}  "
                f"time={brute['runtime_s']:.3f}s"
            )

    json_path = out_dir / "classic_dqi_benchmark_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\nwrote:", json_path)

    plot_main = out_dir / "classic_dqi_benchmark_comparison.png"
    plot_runtime = out_dir / "classic_dqi_benchmark_runtime.png"
    plot_cost = out_dir / "classic_dqi_benchmark_workflow_cost.png"
    plot_s = out_dir / "classic_dqi_benchmark_expected_s.png"
    plot_gap = out_dir / "classic_dqi_benchmark_gap_to_exact.png"
    _plot_main(rows, plot_main)
    _plot_runtime(rows, plot_runtime)
    _plot_workflow_cost(rows, plot_cost)
    _plot_expected_s(rows, plot_s)
    _plot_gap(rows, plot_gap)
    print("wrote:", plot_main)
    print("wrote:", plot_runtime)
    print("wrote:", plot_cost)
    print("wrote:", plot_s)
    print("wrote:", plot_gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

