#!/usr/bin/env python3
"""CLI wrapper for running DQI on matrix or insurance QUBO blocks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_benchmarks import benchmark_dqi_pipeline
from src.dqi_visualize import plot_bitstring_histogram, plot_convergence
from src.run_dqi import run_dqi_with_details


def _load_q_from_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        q = np.load(path)
    else:
        q = np.loadtxt(path, delimiter=",")
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a square matrix")
    return (q + q.T) * 0.5


def _build_block_from_ltm(args: argparse.Namespace):
    from src.insurance_model import load_ltm_instance, subsample_problem
    from src.qubo_block import build_qubo_block_for_package

    problem = load_ltm_instance(args.data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(problem, args.subsample_coverages, args.subsample_packages)
    block = build_qubo_block_for_package(problem, package_index=args.package, penalty_weight=args.penalty_weight)
    return block


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["matrix", "ltm-block"], default="matrix")
    ap.add_argument("--q-path", type=Path, default=None, help="Path to .npy or .csv Q matrix")
    ap.add_argument("--data-dir", type=Path, default=None, help="LTM CSV directory for ltm-block source")
    ap.add_argument("--package", type=int, default=0, help="Package index for ltm-block source")
    ap.add_argument("--penalty-weight", type=float, default=None, help="Optional lambda override")
    ap.add_argument("--subsample-coverages", type=int, default=0)
    ap.add_argument("--subsample-packages", type=int, default=0)

    ap.add_argument("--p", type=int, default=1, help="DQI depth/layers (fixed angles, one shot)")
    ap.add_argument("--statistic", choices=["mean", "best"], default="mean")
    ap.add_argument(
        "--mixer",
        choices=["rx", "h"],
        default="h",
        help="Mixer after each cost layer; 'h' uses Hadamard interference (default).",
    )
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-qubits", type=int, default=50)

    ap.add_argument(
        "--execution",
        choices=["local", "selene", "nexus-selene", "nexus-helios"],
        default="nexus-selene",
        help="Default nexus-selene; use local/selene for on-machine emulation.",
    )
    ap.add_argument("--nexus-hugr-name", default="dqi-hugr", help="HUGR upload name prefix (eval index appended).")
    ap.add_argument("--nexus-job-name", default="dqi-execute", help="Execute job name prefix (eval index appended).")
    ap.add_argument(
        "--nexus-helios-system",
        default="Helios-1",
        help="HeliosConfig.system_name when --execution nexus-helios (hardware or emulator name).",
    )
    ap.add_argument(
        "--nexus-no-timeout",
        action="store_true",
        help="Do not cap wait time for Nexus execute (timeout=None).",
    )
    ap.add_argument(
        "--nexus-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for each Nexus job (ignored if --nexus-no-timeout).",
    )

    ap.add_argument("--benchmark", action="store_true", help="Run baseline comparisons")
    ap.add_argument("--no-qaoa-baseline", action="store_true")
    ap.add_argument("--random-samples", type=int, default=4096)
    ap.add_argument("--bruteforce-max-n", type=int, default=20)

    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--save-plots", action="store_true", help="Write convergence and histogram PNGs")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "matrix":
        if args.q_path is None:
            raise ValueError("--q-path is required when --source matrix")
        target = _load_q_from_file(args.q_path)
    else:
        if args.data_dir is None:
            raise ValueError("--data-dir is required when --source ltm-block")
        target = _build_block_from_ltm(args)

    nexus_timeout = None if args.nexus_no_timeout else float(args.nexus_timeout)
    exec_key = args.execution.replace("-", "_")

    best_x, best_value, meta = run_dqi_with_details(
        target,
        p=args.p,
        shots=args.shots,
        seed=args.seed,
        statistic=args.statistic,
        mixer=args.mixer,
        max_qubits=args.max_qubits,
        execution=exec_key,
        nexus_hugr_name=args.nexus_hugr_name,
        nexus_job_name=args.nexus_job_name,
        nexus_helios_system=args.nexus_helios_system,
        nexus_timeout=nexus_timeout,
    )

    print("=== DQI Result ===")
    print("best bitstring:", meta.bitstring)
    print("best solution:", best_x.astype(int).tolist())
    print("best value:", best_value)
    print("hamming weight (full):", meta.hamming_weight_full)
    if meta.hamming_weight_coverage is not None:
        print("hamming weight (coverage bits):", meta.hamming_weight_coverage)
    print("circuit executions:", meta.run_result.n_evaluations)

    if args.save_plots:
        conv_path = args.out_dir / "dqi_objective.png"
        hist_path = args.out_dir / "dqi_histogram.png"
        plot_convergence(meta.run_result.history, out_path=conv_path, title="DQI one-shot objective")
        plot_bitstring_histogram(meta.run_result.stats_at_best.bitstring_counts, out_path=hist_path)
        print("wrote:", conv_path)
        print("wrote:", hist_path)

    if args.benchmark:
        bench = benchmark_dqi_pipeline(
            target,
            p=args.p,
            shots=args.shots,
            dqi_seed=args.seed,
            random_seed=args.seed,
            brute_force_max_n=args.bruteforce_max_n,
            random_samples=args.random_samples,
            include_qaoa_baseline=not args.no_qaoa_baseline,
            mixer=args.mixer,
        )
        print("\n=== Benchmarks ===")
        for name, res in bench.items():
            print(
                f"{name:10s}  best={res.best_value: .6f}  "
                f"time={res.runtime_sec: .3f}s  approx_ratio={res.approximation_ratio}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
