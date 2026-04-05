#!/usr/bin/env python3
"""Run the QAOA pipeline on real LTM data (load → QuboBlock → Selene → report).

Examples (from ``qaoa_python/`` with repo venv and guppylang installed).
**Common flags go on the subcommand** (``fixed-p1``, ``grid-p1``, …)::

    PYTHONPATH=src python scripts/run_qaoa_pipeline.py fixed-p1 \\
        --shots 512 --gamma 0.7 --beta 0.5

    PYTHONPATH=src python scripts/run_qaoa_pipeline.py grid-p1 \\
        --shots 256 --n-gamma 6 --n-beta 6 --statistic mean

    PYTHONPATH=src python scripts/run_qaoa_pipeline.py random-p2 \\
        --shots 512 --n-samples 24

    PYTHONPATH=src python scripts/run_qaoa_pipeline.py cobyla-p1 \\
        --subsample-coverages 0 --subsample-packages 0 --package 0 --max-qubits 24 --maxiter 80

Default ``--data-dir`` is ``../Travelers/docs/data/YQH26_data`` relative to ``qaoa_python``.
Default subsampling is **10 coverages × 3 packages** so the block fits comfortably under Selene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_data_dir() -> Path:
    return _ROOT.parent / "Travelers" / "docs" / "data" / "YQH26_data"


def _parse_stat(s: str):
    if s not in ("mean", "best"):
        raise argparse.ArgumentTypeError("statistic must be mean or best")
    return s


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Directory with instance_*.csv",
    )
    common.add_argument("--package", type=int, default=0, help="Package index m for the QUBO block")
    common.add_argument(
        "--subsample-coverages",
        type=int,
        default=10,
        metavar="N",
        help="Keep first N coverages (0 = full instance)",
    )
    common.add_argument(
        "--subsample-packages",
        type=int,
        default=3,
        metavar="M",
        help="Keep first M packages (0 = full instance)",
    )
    common.add_argument("--shots", type=int, default=512, help="Shots per Selene run")
    common.add_argument("--max-qubits", type=int, default=24, help="Abort if block dimension exceeds this")
    common.add_argument(
        "--statistic",
        type=_parse_stat,
        default="mean",
        help="Outer-loop objective: mean energy vs best observed bitstring energy",
    )
    common.add_argument(
        "--bruteforce-max-n",
        type=int,
        default=18,
        help="If block n_vars <= this, print exact QUBO minimum via enumeration (0 = skip)",
    )
    common.add_argument("--seed-offset", type=int, default=0, help="Base seed offset for Selene runs")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p_fix1 = sub.add_parser(
        "fixed-p1",
        parents=[common],
        help="Single p=1 run at given (γ, β) [radians]",
    )
    p_fix1.add_argument("--gamma", type=float, required=True)
    p_fix1.add_argument("--beta", type=float, required=True)

    p_fix2 = sub.add_parser("fixed-p2", parents=[common], help="Single p=2 run at (γ₁,β₁,γ₂,β₂) [radians]")
    p_fix2.add_argument("--gamma1", type=float, required=True)
    p_fix2.add_argument("--beta1", type=float, required=True)
    p_fix2.add_argument("--gamma2", type=float, required=True)
    p_fix2.add_argument("--beta2", type=float, required=True)

    p_g1 = sub.add_parser("grid-p1", parents=[common], help="Grid search on (γ, β)")
    p_g1.add_argument("--n-gamma", type=int, default=8)
    p_g1.add_argument("--n-beta", type=int, default=8)

    p_g2 = sub.add_parser("grid-p2", parents=[common], help="Grid on (γ₁,β₁,γ₂,β₂) — keep grids small")
    p_g2.add_argument("--n-gamma1", type=int, default=3)
    p_g2.add_argument("--n-beta1", type=int, default=3)
    p_g2.add_argument("--n-gamma2", type=int, default=3)
    p_g2.add_argument("--n-beta2", type=int, default=3)

    p_r1 = sub.add_parser("random-p1", parents=[common], help="Random search in [0,π]²")
    p_r1.add_argument("--n-samples", type=int, default=48)
    p_r1.add_argument("--rng-seed", type=int, default=0)

    p_r2 = sub.add_parser("random-p2", parents=[common], help="Random search in [0,π]⁴")
    p_r2.add_argument("--n-samples", type=int, default=32)
    p_r2.add_argument("--rng-seed", type=int, default=0)

    p_c1 = sub.add_parser("cobyla-p1", parents=[common], help="SciPy COBYLA on (γ, β)")
    p_c1.add_argument("--maxiter", type=int, default=60)

    p_c2 = sub.add_parser("cobyla-p2", parents=[common], help="SciPy COBYLA on (γ₁,β₁,γ₂,β₂)")
    p_c2.add_argument("--maxiter", type=int, default=80)

    p_s1 = sub.add_parser("spsa-p1", parents=[common], help="SPSA on (γ, β)")
    p_s1.add_argument("--maxiter", type=int, default=40)
    p_s1.add_argument("--rng-seed", type=int, default=0)

    args = ap.parse_args()

    from src.insurance_model import load_ltm_instance, subsample_problem
    from src.qubo_block import build_qubo_block_for_package
    from src.qubo_qaoa import bruteforce_minimize_qubo, run_qaoa_p1_on_block, run_qaoa_p2_on_block
    from src.qubo_qaoa_optimize import (
        optimize_qaoa_p1_cobyla,
        optimize_qaoa_p1_grid,
        optimize_qaoa_p1_random,
        optimize_qaoa_p1_spsa,
        optimize_qaoa_p2_cobyla,
        optimize_qaoa_p2_grid,
        optimize_qaoa_p2_random,
    )

    data_dir = args.data_dir.resolve()
    if not (data_dir / "instance_coverages.csv").is_file():
        print(f"Missing CSVs under {data_dir}", file=sys.stderr)
        return 1

    problem = load_ltm_instance(data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(problem, args.subsample_coverages, args.subsample_packages)
    elif args.subsample_coverages > 0 or args.subsample_packages > 0:
        print("Set both --subsample-coverages and --subsample-packages, or neither.", file=sys.stderr)
        return 1

    m = args.package
    if m < 0 or m >= problem.M:
        print(f"package {m} out of range M={problem.M}", file=sys.stderr)
        return 1

    block = build_qubo_block_for_package(problem, m)
    n = block.n_vars
    print(f"data_dir     = {data_dir}")
    print(f"package m    = {m}  (M={problem.M} after subsampling)")
    print(f"block n_vars = {n}  (coverages+slack in this block)")
    print(f"shots        = {args.shots}  statistic = {args.statistic}")

    if args.bruteforce_max_n > 0 and n <= args.bruteforce_max_n:
        e_bf, x_bf = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset, max_n=n)
        print(f"bruteforce min E = {e_bf:.6g}  (exact for this block)")
    else:
        print("bruteforce     = skipped (n too large or --bruteforce-max-n 0)")

    seed_off = int(args.seed_offset)
    stat = args.statistic

    if args.mode == "fixed-p1":
        st = run_qaoa_p1_on_block(
            block,
            args.gamma,
            args.beta,
            shots=args.shots,
            seed=seed_off,
            max_qubits=args.max_qubits,
        )
        print(f"γ={args.gamma:.6g} β={args.beta:.6g}  best_bitstring={st.best_bitstring}  best_E={st.best_qubo_energy:.6g}")
    elif args.mode == "fixed-p2":
        st = run_qaoa_p2_on_block(
            block,
            args.gamma1,
            args.beta1,
            args.gamma2,
            args.beta2,
            shots=args.shots,
            seed=seed_off,
            max_qubits=args.max_qubits,
        )
        print(
            f"γ₁={args.gamma1:.6g} β₁={args.beta1:.6g} γ₂={args.gamma2:.6g} β₂={args.beta2:.6g}  "
            f"best_bitstring={st.best_bitstring}  best_E={st.best_qubo_energy:.6g}"
        )
    elif args.mode == "grid-p1":
        res = optimize_qaoa_p1_grid(
            block,
            n_gamma=args.n_gamma,
            n_beta=args.n_beta,
            shots=args.shots,
            seed_offset=seed_off,
            statistic=stat,
            max_qubits=args.max_qubits,
        )
        print(
            f"best γ={res.gamma:.6g} β={res.beta:.6g}  objective({stat})={res.objective_value:.6g}  "
            f"n_eval={res.n_evaluations}  best_E_sample={res.stats_at_best.best_qubo_energy:.6g}  "
            f"best_bits={res.stats_at_best.best_bitstring}"
        )
    elif args.mode == "grid-p2":
        ntot = args.n_gamma1 * args.n_beta1 * args.n_gamma2 * args.n_beta2
        print(f"grid size = {ntot} evaluations …")
        res = optimize_qaoa_p2_grid(
            block,
            n_gamma1=args.n_gamma1,
            n_beta1=args.n_beta1,
            n_gamma2=args.n_gamma2,
            n_beta2=args.n_beta2,
            shots=args.shots,
            seed_offset=seed_off,
            statistic=stat,
            max_qubits=args.max_qubits,
        )
        print(
            f"best γ₁={res.gamma1:.6g} β₁={res.beta1:.6g} γ₂={res.gamma2:.6g} β₂={res.beta2:.6g}  "
            f"objective({stat})={res.objective_value:.6g}  n_eval={res.n_evaluations}  "
            f"best_E={res.stats_at_best.best_qubo_energy:.6g}  bits={res.stats_at_best.best_bitstring}"
        )
    elif args.mode == "random-p1":
        res = optimize_qaoa_p1_random(
            block,
            n_samples=args.n_samples,
            shots=args.shots,
            rng_seed=args.rng_seed,
            seed_offset=seed_off,
            statistic=stat,
            max_qubits=args.max_qubits,
        )
        print(
            f"best γ={res.gamma:.6g} β={res.beta:.6g}  objective({stat})={res.objective_value:.6g}  "
            f"n_eval={res.n_evaluations}  best_E={res.stats_at_best.best_qubo_energy:.6g}"
        )
    elif args.mode == "random-p2":
        res = optimize_qaoa_p2_random(
            block,
            n_samples=args.n_samples,
            shots=args.shots,
            rng_seed=args.rng_seed,
            seed_offset=seed_off,
            statistic=stat,
            max_qubits=args.max_qubits,
        )
        print(
            f"best (γ₁,β₁,γ₂,β₂)=({res.gamma1:.6g},{res.beta1:.6g},{res.gamma2:.6g},{res.beta2:.6g})  "
            f"objective={res.objective_value:.6g}  best_E={res.stats_at_best.best_qubo_energy:.6g}"
        )
    elif args.mode == "cobyla-p1":
        try:
            res = optimize_qaoa_p1_cobyla(
                block,
                shots=args.shots,
                seed_offset=seed_off,
                statistic=stat,
                maxiter=args.maxiter,
                max_qubits=args.max_qubits,
            )
        except ImportError as e:
            print(f"COBYLA needs SciPy: {e}", file=sys.stderr)
            return 1
        print(
            f"best γ={res.gamma:.6g} β={res.beta:.6g}  objective({stat})={res.objective_value:.6g}  "
            f"n_eval={res.n_evaluations}  best_E={res.stats_at_best.best_qubo_energy:.6g}"
        )
    elif args.mode == "cobyla-p2":
        try:
            res = optimize_qaoa_p2_cobyla(
                block,
                shots=args.shots,
                seed_offset=seed_off,
                statistic=stat,
                maxiter=args.maxiter,
                max_qubits=args.max_qubits,
            )
        except ImportError as e:
            print(f"COBYLA needs SciPy: {e}", file=sys.stderr)
            return 1
        print(
            f"best (γ₁,β₁,γ₂,β₂)=({res.gamma1:.6g},{res.beta1:.6g},{res.gamma2:.6g},{res.beta2:.6g})  "
            f"objective={res.objective_value:.6g}  n_eval={res.n_evaluations}"
        )
    elif args.mode == "spsa-p1":
        res = optimize_qaoa_p1_spsa(
            block,
            shots=args.shots,
            rng_seed=args.rng_seed,
            seed_offset=seed_off,
            statistic=stat,
            maxiter=args.maxiter,
            max_qubits=args.max_qubits,
        )
        print(
            f"best γ={res.gamma:.6g} β={res.beta:.6g}  objective({stat})={res.objective_value:.6g}  "
            f"n_eval={res.n_evaluations}  best_E={res.stats_at_best.best_qubo_energy:.6g}"
        )
    else:
        print(f"Unknown mode {args.mode!r}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
