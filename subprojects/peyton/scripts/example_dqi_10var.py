#!/usr/bin/env python3
"""Example DQI run on a 10-variable insurance-style QUBO."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_benchmarks import benchmark_dqi_pipeline
from src.dqi_core import build_hamming_weight_penalty_qubo
from src.dqi_visualize import plot_bitstring_histogram, plot_convergence
from src.run_dqi import run_dqi_with_details


def build_example_qubo(n: int = 10, target_weight: int = 4) -> np.ndarray:
    """Construct a reproducible 10-variable QUBO with hamming-weight regularization."""
    rng = np.random.default_rng(26)
    linear_profit = rng.uniform(0.8, 2.0, size=n)
    base_Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        # Minimize negative profit => maximize profit.
        base_Q[i, i] -= float(linear_profit[i])
    # Add small pairwise structure.
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.uniform(-0.25, 0.45))
            base_Q[i, j] += w
            base_Q[j, i] += w
    # Encourage selecting approximately target_weight coverages.
    base_Q += build_hamming_weight_penalty_qubo(n=n, target_weight=target_weight, penalty=1.4)
    return (base_Q + base_Q.T) * 0.5


def main() -> int:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    Q = build_example_qubo(n=10, target_weight=4)
    # Single parity layer: Rz(gamma * c_i) with fixed gamma=1 (no classical angle search).
    # Use normalize_phase_c=False for raw Q diagonal as c_i; True scales c by max|diag(Q)|.
    x_best, value, meta = run_dqi_with_details(
        Q,
        p=1,
        optimizer="cobyla",
        variational=False,
        fixed_gammas=[1.0],
        shots=768,
        seed=7,
        maxiter=35,
        statistic="mean",
        mixer="rx",
        max_qubits=50,
        normalize_phase_c=False,
        ##############################
        execution="nexus_selene",
        ##############################
    )

    print("=== DQI (10-variable) ===")
    print("best bitstring:", meta.bitstring)
    print("best x:", x_best.astype(int).tolist())
    print("best value:", value)
    print("hamming weight:", meta.hamming_weight_full)
    print("optimizer evaluations:", meta.optimizer_result.n_evaluations)

    conv_path = out_dir / "dqi_convergence_10var.png"
    hist_path = out_dir / "dqi_histogram_10var.png"
    plot_convergence(meta.optimizer_result.history, out_path=conv_path, title="DQI convergence (10 vars)")
    plot_bitstring_histogram(
        meta.optimizer_result.stats_at_best.bitstring_counts,
        out_path=hist_path,
        title="DQI sampled bitstrings (10 vars)",
    )
    print("wrote:", conv_path)
    print("wrote:", hist_path)

    bench = benchmark_dqi_pipeline(
        Q,
        p=2,
        optimizer="cobyla",
        shots=512,
        dqi_seed=11,
        random_seed=12,
        brute_force_max_n=20,
        random_samples=2500,
        include_qaoa_baseline=True,
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
