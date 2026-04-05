#!/usr/bin/env python3
"""Benchmark p=1 outer-loop optimizers (grid, random, COBYLA, SPSA) on the same QuboBlock.

Records **wall time**, **Selene evaluation count**, **optimizer objective**, and **best QUBO
energy** at the returned angles. Writes a **2×2 figure** (PNG) plus optional **CSV** for judges.

From ``qaoa_python/``::

    MPLBACKEND=Agg PYTHONPATH=src python scripts/benchmark_qaoa_optimizers.py \\
        --out qaoa_optimizer_benchmark.png --csv qaoa_optimizer_benchmark.csv

SciPy must be installed for **COBYLA**; if missing, that method is skipped.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_data_dir() -> Path:
    return _ROOT.parent / "Travelers" / "docs" / "data" / "YQH26_data"


@dataclass
class BenchmarkRow:
    method: str
    objective: float
    best_qubo_energy: float
    n_evaluations: int
    seconds: float
    gamma: float
    beta: float


def _load_block(args: argparse.Namespace):
    from src.insurance_model import load_ltm_instance, subsample_problem
    from src.qubo_block import build_qubo_block_for_package

    data_dir = args.data_dir.resolve()
    if not (data_dir / "instance_coverages.csv").is_file():
        raise FileNotFoundError(f"Missing CSVs under {data_dir}")

    problem = load_ltm_instance(data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(problem, args.subsample_coverages, args.subsample_packages)
    elif args.subsample_coverages > 0 or args.subsample_packages > 0:
        raise ValueError("Set both subsample N and M, or neither")

    m = args.package
    if m < 0 or m >= problem.M:
        raise ValueError(f"package {m} out of range M={problem.M}")

    return build_qubo_block_for_package(problem, m)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=_default_data_dir())
    ap.add_argument("--package", type=int, default=0)
    ap.add_argument("--subsample-coverages", type=int, default=10, metavar="N")
    ap.add_argument("--subsample-packages", type=int, default=3, metavar="M")
    ap.add_argument("--max-qubits", type=int, default=24)
    ap.add_argument("--shots", type=int, default=256, help="Shots per Selene evaluation")
    ap.add_argument(
        "--statistic",
        choices=("mean", "best"),
        default="mean",
        help="Objective each optimizer minimizes",
    )
    ap.add_argument("--seed-offset", type=int, default=0, help="Base seed passed to optimizers")
    ap.add_argument("--rng-seed", type=int, default=42, help="RNG seed for random + SPSA")

    ap.add_argument("--grid-ng", type=int, default=5, help="Grid resolution for γ")
    ap.add_argument("--grid-nb", type=int, default=5, help="Grid resolution for β")
    ap.add_argument("--random-samples", type=int, default=25, help="Random search trial count")
    ap.add_argument("--cobyla-maxiter", type=int, default=40)
    ap.add_argument("--spsa-maxiter", type=int, default=25)

    ap.add_argument("--out", type=Path, required=True, help="Output PNG path")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV path for numeric table")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title-suffix", type=str, default="", help="Append to figure suptitle")

    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print("Need matplotlib + numpy:  python -m pip install matplotlib", file=sys.stderr)
        return 1

    from src.qubo_qaoa import bruteforce_minimize_qubo
    from src.qubo_qaoa_optimize import (
        optimize_qaoa_p1_cobyla,
        optimize_qaoa_p1_grid,
        optimize_qaoa_p1_random,
        optimize_qaoa_p1_spsa,
    )

    block = _load_block(args)
    n = block.n_vars
    if n > args.max_qubits:
        print(f"n_vars={n} exceeds --max-qubits={args.max_qubits}", file=sys.stderr)
        return 1

    stat = args.statistic
    shots = int(args.shots)
    seed_off = int(args.seed_offset)
    rng_seed = int(args.rng_seed)

    bruteforce_e: float | None = None
    if n <= 20:
        bruteforce_e, _ = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset, max_n=n)

    rows: list[BenchmarkRow] = []

    def run(name: str, fn) -> None:
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        rows.append(
            BenchmarkRow(
                method=name,
                objective=float(res.objective_value),
                best_qubo_energy=float(res.stats_at_best.best_qubo_energy),
                n_evaluations=int(res.n_evaluations),
                seconds=float(dt),
                gamma=float(res.gamma),
                beta=float(res.beta),
            )
        )
        print(f"{name:12s}  obj={res.objective_value:.5g}  best_E={res.stats_at_best.best_qubo_energy:.5g}  "
              f"n_eval={res.n_evaluations}  time={dt:.2f}s  γ={res.gamma:.4f} β={res.beta:.4f}")

    print(f"block n={n}  shots/eval={shots}  statistic={stat}")
    if bruteforce_e is not None:
        print(f"bruteforce min E (block) = {bruteforce_e:.6g}")
    print("---")

    run(
        "grid",
        lambda: optimize_qaoa_p1_grid(
            block,
            n_gamma=args.grid_ng,
            n_beta=args.grid_nb,
            shots=shots,
            seed_offset=seed_off,
            statistic=stat,
            max_qubits=args.max_qubits,
        ),
    )

    run(
        "random",
        lambda: optimize_qaoa_p1_random(
            block,
            n_samples=args.random_samples,
            shots=shots,
            rng_seed=rng_seed,
            seed_offset=seed_off + 10_000,
            statistic=stat,
            max_qubits=args.max_qubits,
        ),
    )

    try:
        run(
            "cobyla",
            lambda: optimize_qaoa_p1_cobyla(
                block,
                shots=shots,
                seed_offset=seed_off + 20_000,
                statistic=stat,
                maxiter=args.cobyla_maxiter,
                max_qubits=args.max_qubits,
            ),
        )
    except ImportError:
        print("cobyla       SKIPPED (SciPy not available)", file=sys.stderr)

    run(
        "spsa",
        lambda: optimize_qaoa_p1_spsa(
            block,
            shots=shots,
            rng_seed=rng_seed + 1,
            seed_offset=seed_off + 30_000,
            statistic=stat,
            maxiter=args.spsa_maxiter,
            max_qubits=args.max_qubits,
        ),
    )

    names = [r.method for r in rows]
    color_map = {"grid": "#0066CC", "random": "#5BA3D6", "cobyla": "#00356B", "spsa": "#2D8C3C"}
    bar_colors = [color_map[r.method] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), layout="constrained")

    ax = axes[0, 0]
    ax.bar(names, [r.best_qubo_energy for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Best QUBO energy\n(at returned angles)")
    ax.set_title("Quality: lowest energy in winning run")
    if bruteforce_e is not None:
        ax.axhline(bruteforce_e, color="#2D8C3C", linestyle="--", linewidth=2, label="exact block min")
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(names, [r.objective for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel(f"Optimizer objective ({stat})")
    ax.set_title("What the outer loop minimized")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(names, [r.seconds for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Runtime (includes Selene + Guppy compile per eval)")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 1]
    ax.bar(names, [r.n_evaluations for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Selene evaluations")
    ax.set_title("Number of circuit runs (kernel builds)")
    ax.grid(True, axis="y", alpha=0.25)

    suf = f"  |  {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"p=1 QAOA optimizer benchmark  |  n={n} qubits  |  {shots} shots/eval  |  stat={stat}{suf}",
        fontsize=11,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    plt.close(fig)
    print(f"Wrote {args.out}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "method",
                    "objective",
                    "best_qubo_energy",
                    "n_evaluations",
                    "seconds",
                    "gamma",
                    "beta",
                    "bruteforce_min",
                ]
            )
            bf = "" if bruteforce_e is None else f"{bruteforce_e:.17g}"
            for r in rows:
                w.writerow(
                    [
                        r.method,
                        f"{r.objective:.17g}",
                        f"{r.best_qubo_energy:.17g}",
                        r.n_evaluations,
                        f"{r.seconds:.6f}",
                        f"{r.gamma:.17g}",
                        f"{r.beta:.17g}",
                        bf,
                    ]
                )
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
