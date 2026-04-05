#!/usr/bin/env python3
"""Plot QAOA quality on real LTM data (histogram + optional angle landscape).

From ``qaoa_python/``::

    PYTHONPATH=src python scripts/plot_qaoa_results.py histogram \\
        --gamma 0.7 --beta 0.5 --shots 800 --out qaoa_hist.png

    PYTHONPATH=src python scripts/plot_qaoa_results.py landscape \\
        --n-gamma 7 --n-beta 7 --shots 200 --out qaoa_landscape.png

``histogram``: weighted distribution of QUBO energies in the shot counts, plus
bruteforce minimum (when n is small), best sample, and sample mean ± SE.

``landscape``: heatmap of **mean** shot energy over a coarse ``(γ, β)`` grid (many Selene runs).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_data_dir() -> Path:
    return _ROOT.parent / "Travelers" / "docs" / "data" / "YQH26_data"


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

    block = build_qubo_block_for_package(problem, m)
    return problem, block


def _cmd_histogram(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Install matplotlib:  python -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from e

    from src.qubo_qaoa import bruteforce_minimize_qubo, run_qaoa_p1_on_block
    from src.qubo_qaoa_optimize import mean_sample_energy, sample_mean_energy_uncertainty

    _, block = _load_block(args)
    n = block.n_vars
    if n > args.max_qubits:
        print(f"n_vars={n} exceeds --max-qubits={args.max_qubits}", file=sys.stderr)
        return 1

    stats = run_qaoa_p1_on_block(
        block,
        args.gamma,
        args.beta,
        shots=int(args.shots),
        seed=int(args.seed),
        max_qubits=args.max_qubits,
    )

    energies: list[float] = []
    weights: list[float] = []
    for s, c in stats.bitstring_counts.items():
        x = np.array([float(int(ch)) for ch in s], dtype=float)
        energies.append(float(block.energy(x)))
        weights.append(float(c))

    e_arr = np.asarray(energies, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    w_arr /= float(np.sum(w_arr))

    unc = sample_mean_energy_uncertainty(block, stats)
    mean_e = mean_sample_energy(block, stats)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    n_bins = int(args.bins)
    lo = float(np.min(e_arr))
    hi = float(np.max(e_arr))
    pad = 0.05 * (hi - lo + 1e-9)
    bins = np.linspace(lo - pad, hi + pad, n_bins + 1)
    ax.hist(e_arr, bins=bins, weights=w_arr, color="#0066CC", edgecolor="white", linewidth=0.6, alpha=0.88)
    ax.axvline(mean_e, color="#2D2926", linestyle="-", linewidth=2, label=f"sample mean = {mean_e:.4g}")
    if unc.se_mean > 0 and not np.isnan(unc.se_mean):
        ax.axvspan(
            mean_e - 1.96 * unc.se_mean,
            mean_e + 1.96 * unc.se_mean,
            color="#2D2926",
            alpha=0.12,
            label="mean ± 1.96·SE",
        )
    ax.axvline(stats.best_qubo_energy, color="#00356B", linestyle="--", linewidth=2, label=f"best in sample = {stats.best_qubo_energy:.4g}")

    bf_e: float | None = None
    if args.bruteforce_max_n > 0 and n <= args.bruteforce_max_n:
        bf_e, _ = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset, max_n=n)
        ax.axvline(bf_e, color="#2D8C3C", linestyle="-", linewidth=2.5, label=f"exact min (block) = {bf_e:.4g}")

    ax.set_xlabel("QUBO energy  E(x) = xᵀQx + const")
    ax.set_ylabel("Fraction of shots")
    ax.set_title(
        f"p=1 QAOA  γ={args.gamma:.4g}  β={args.beta:.4g}  |  n={n} qubits  |  {args.shots} shots"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    if args.out:
        fig.savefig(args.out, dpi=int(args.dpi))
        print(f"Wrote {args.out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def _cmd_landscape(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Install matplotlib:  python -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from e

    from src.qubo_qaoa import run_qaoa_p1_on_block
    from src.qubo_qaoa_optimize import mean_sample_energy

    _, block = _load_block(args)
    n = block.n_vars
    if n > args.max_qubits:
        print(f"n_vars={n} exceeds --max-qubits={args.max_qubits}", file=sys.stderr)
        return 1

    ng, nb = int(args.n_gamma), int(args.n_beta)
    gammas = np.linspace(args.gamma_lo, args.gamma_hi, ng, dtype=float)
    betas = np.linspace(args.beta_lo, args.beta_hi, nb, dtype=float)
    Z = np.empty((nb, ng), dtype=float)
    total = ng * nb
    k = 0
    for j, b in enumerate(betas):
        for i, g in enumerate(gammas):
            stats = run_qaoa_p1_on_block(
                block,
                float(g),
                float(b),
                shots=int(args.shots),
                seed=int(args.seed + k),
                max_qubits=args.max_qubits,
            )
            Z[j, i] = mean_sample_energy(block, stats)
            k += 1
            if k % max(1, total // 10) == 0:
                print(f"  landscape {k}/{total} …", flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[gammas[0], gammas[-1], betas[0], betas[-1]],
        cmap="viridis",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Mean QUBO energy (shots)")
    ax.set_xlabel("γ  (rad)")
    ax.set_ylabel("β  (rad)")
    ax.set_title(f"p=1 mean energy landscape  |  n={n}  |  {args.shots} shots/cell  |  {ng}×{nb} grid")

    if args.out:
        fig.savefig(args.out, dpi=int(args.dpi))
        print(f"Wrote {args.out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=_default_data_dir())
    common.add_argument("--package", type=int, default=0)
    common.add_argument("--subsample-coverages", type=int, default=10, metavar="N")
    common.add_argument("--subsample-packages", type=int, default=3, metavar="M")
    common.add_argument("--max-qubits", type=int, default=24)
    common.add_argument("--bruteforce-max-n", type=int, default=18)
    common.add_argument("--seed", type=int, default=0)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    ph = sub.add_parser("histogram", parents=[common], help="Energy histogram for one (γ, β)")
    ph.add_argument("--gamma", type=float, required=True)
    ph.add_argument("--beta", type=float, required=True)
    ph.add_argument("--shots", type=int, default=800)
    ph.add_argument("--bins", type=int, default=32)
    ph.add_argument("--out", type=Path, default=None, help="PNG path (omit to open GUI)")
    ph.add_argument("--dpi", type=int, default=150)
    ph.set_defaults(_run=_cmd_histogram)

    pl = sub.add_parser("landscape", parents=[common], help="Mean-energy heatmap over (γ, β)")
    pl.add_argument("--n-gamma", type=int, default=7)
    pl.add_argument("--n-beta", type=int, default=7)
    pl.add_argument("--gamma-lo", type=float, default=0.0)
    pl.add_argument("--gamma-hi", type=float, default=math.pi)
    pl.add_argument("--beta-lo", type=float, default=0.0)
    pl.add_argument("--beta-hi", type=float, default=math.pi)
    pl.add_argument("--shots", type=int, default=200)
    pl.add_argument("--out", type=Path, default=None)
    pl.add_argument("--dpi", type=int, default=150)
    pl.set_defaults(_run=_cmd_landscape)

    args = ap.parse_args()
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())
