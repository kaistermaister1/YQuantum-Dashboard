#!/usr/bin/env python3
"""Example DQI run on a 10-variable insurance-style QUBO."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set before any matplotlib import (via src.dqi_visualize).
os.environ.setdefault("MPLBACKEND", "TkAgg")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_benchmarks import benchmark_dqi_pipeline
from src.dqi_core import array_to_bitstring, build_hamming_weight_penalty_qubo
from src.dqi_visualize import LiveBenchmarkComparison, plot_bitstring_histogram, plot_convergence


def build_example_qubo(n: int = 10, target_weight: int = 4) -> tuple[np.ndarray, list[str]]:
    """Construct a reproducible QUBO and human-readable labels for each binary variable (bundle slot)."""
    rng = np.random.default_rng(26)
    linear_profit = rng.uniform(0.8, 2.0, size=n)
    base_Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        base_Q[i, i] -= float(linear_profit[i])
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.uniform(-0.25, 0.45))
            base_Q[i, j] += w
            base_Q[j, i] += w
    base_Q += build_hamming_weight_penalty_qubo(n=n, target_weight=target_weight, penalty=1.4)
    Q = (base_Q + base_Q.T) * 0.5
    bundle_names = [f"Bundle-{i}" for i in range(n)]
    return Q, bundle_names


def selected_bundle_labels(bitstring: str, names: list[str]) -> list[str]:
    """Map a 0/1 string to the list of selected bundle names (index i = variable x_i)."""
    return [names[i] for i, ch in enumerate(bitstring) if ch == "1"]


def main() -> int:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    Q, bundle_names = build_example_qubo(n=10, target_weight=4)

    live = LiveBenchmarkComparison(
        title="Best QUBO objective (lower is better) — chart updates as each method finishes",
    )
    bench = benchmark_dqi_pipeline(
        Q,
        p=2,
        shots=768,
        dqi_seed=7,
        random_seed=12,
        brute_force_max_n=20,
        random_samples=2500,
        include_qaoa_baseline=True,
        mixer="h",
        progress_callback=live.update,
    )

    live.savefig(out_dir / "dqi_benchmark_comparison.png")

    dqi = bench["dqi"]
    bits = dqi.extra["bitstring"]
    picked = selected_bundle_labels(bits, bundle_names)
    idx_on = [i for i, ch in enumerate(bits) if ch == "1"]

    print("=== DQI best sample ===")
    print("best bitstring:", bits)
    print("selected variable indices (1 = include):", idx_on)
    print("selected bundles:", ", ".join(picked) if picked else "(none)")

    if "bruteforce" in bench:
        bf = bench["bruteforce"]
        opt_bits = array_to_bitstring(bf.best_solution)
        print("\n=== Brute-force optimum (reference) ===")
        print("best bitstring:", opt_bits)
        print(
            "optimal bundles:",
            ", ".join(selected_bundle_labels(opt_bits, bundle_names)),
        )

    if "qaoa" in bench:
        qbits = bench["qaoa"].extra.get("best_bitstring", "")
        if qbits:
            print("\n=== QAOA baseline best ===")
            print("best bitstring:", qbits)
            print(
                "bundles:",
                ", ".join(selected_bundle_labels(qbits, bundle_names)),
            )

    print("\n=== All methods ===")
    for name, res in bench.items():
        print(
            f"{name:10s}  best={res.best_value: .6f}  "
            f"time={res.runtime_sec: .3f}s  approx_ratio={res.approximation_ratio}"
        )

    conv_path = out_dir / "dqi_objective_10var.png"
    hist_path = out_dir / "dqi_histogram_10var.png"
    plot_convergence(
        dqi.extra["history"], out_path=conv_path, title="DQI one-shot objective (10 vars)"
    )
    plot_bitstring_histogram(
        dqi.extra["bitstring_counts"],
        out_path=hist_path,
        title="DQI sampled bitstrings (10 vars)",
    )
    print("\nwrote:", conv_path)
    print("wrote:", hist_path)
    print("wrote:", out_dir / "dqi_benchmark_comparison.png")

    print("\nClose the live comparison window to exit (or it will stay open).")
    live.show(block=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
