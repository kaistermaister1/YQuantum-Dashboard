#!/usr/bin/env python3
"""Replicate key DQI artifact plot styles for classic DQI.

Generates classical-DQI counterparts of:
- artifacts/dqi_convergence_10var.png
- artifacts/benchmark_official/dqi_official_sizes_comparison.png
- artifacts/benchmark_official/test_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_classic_pipeline import run_classic_dqi, score_f
from src.dqi_insurance_parity import build_insurance_parity_B_rhs
from src.dqi_visualize import plot_convergence, plot_scaling_benchmark
from src.insurance_model import load_ltm_instance, subsample_problem


def _default_data_dir() -> Path:
    return ROOT.parent / "will" / "Travelers" / "docs" / "data" / "YQH26_data"


def _load_problem(data_dir: Path, n_coverages: int, subsample_packages: int):
    problem = load_ltm_instance(data_dir)
    n_cov = min(int(n_coverages), int(problem.N))
    n_pkg = int(subsample_packages) if int(subsample_packages) > 0 else int(problem.M)
    n_pkg = min(n_pkg, int(problem.M))
    return subsample_problem(problem, n_cov, n_pkg)


def _random_baseline_f(B: np.ndarray, v: np.ndarray, *, samples: int, seed: int) -> tuple[float, int, float]:
    rng = np.random.default_rng(int(seed))
    n_vars = int(B.shape[1])
    best_f = -float("inf")
    t0 = time.perf_counter()
    for _ in range(int(samples)):
        x = rng.integers(0, 2, size=n_vars, dtype=np.int8)
        f_val = float(score_f(B, v, "".join(str(int(b)) for b in x.tolist())))
        if f_val > best_f:
            best_f = f_val
    dt = time.perf_counter() - t0
    return float(best_f), int(samples), float(dt)


def _multistart_local_search_f(
    B: np.ndarray,
    v: np.ndarray,
    *,
    restarts: int,
    seed: int,
) -> tuple[float, int, float]:
    rng = np.random.default_rng(int(seed))
    n_vars = int(B.shape[1])
    eval_count = 0

    def eval_bits(bits: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        return float(score_f(B, v, "".join(str(int(b)) for b in bits.tolist())))

    best_f = -float("inf")
    t0 = time.perf_counter()
    for _ in range(int(restarts)):
        x = rng.integers(0, 2, size=n_vars, dtype=np.int8)
        cur = eval_bits(x)
        improved = True
        while improved:
            improved = False
            best_nb = cur
            best_i: int | None = None
            for i in range(n_vars):
                x2 = x.copy()
                x2[i] = np.int8(1 - int(x2[i]))
                fv = eval_bits(x2)
                if fv > best_nb + 1e-15:
                    best_nb = fv
                    best_i = i
            if best_i is not None:
                x[best_i] = np.int8(1 - int(x[best_i]))
                cur = best_nb
                improved = True
        if cur > best_f:
            best_f = cur
    dt = time.perf_counter() - t0
    return float(best_f), int(eval_count), float(dt)


def _bruteforce_best_f(B: np.ndarray, v: np.ndarray, *, max_nvars: int) -> float | None:
    n_vars = int(B.shape[1])
    if n_vars > int(max_nvars):
        return None
    best_f = -float("inf")
    for mask in range(1 << n_vars):
        bits = "".join("1" if ((mask >> i) & 1) else "0" for i in range(n_vars))
        fv = float(score_f(B, v, bits))
        if fv > best_f:
            best_f = fv
    return float(best_f)


def _classic_row(
    *,
    data_dir: Path,
    n_coverages: int,
    package: int,
    subsample_packages: int,
    ell: int,
    bp_iterations: int,
    shots: int,
    seed: int,
    bruteforce_max_nvars: int,
) -> dict:
    small = _load_problem(data_dir, n_coverages=n_coverages, subsample_packages=subsample_packages)
    if package < 0 or package >= int(small.M):
        raise ValueError(f"package {package} out of range for M={small.M}")
    B, v = build_insurance_parity_B_rhs(small, package)

    t0 = time.perf_counter()
    classic = run_classic_dqi(
        B,
        v,
        ell=int(ell),
        bp_iterations=int(bp_iterations),
        shots=int(shots),
        seed=int(seed),
        strict_ancilla=True,
    )
    dqi_t = time.perf_counter() - t0
    dqi_f = float(classic.expected_f) if classic.expected_f is not None else -float("inf")

    rnd_f, rnd_cost, rnd_t = _random_baseline_f(B, v, samples=int(shots), seed=int(seed) + 111)
    ls_restarts = max(8, min(128, max(16, int(B.shape[1]))))
    ls_f, ls_cost, ls_t = _multistart_local_search_f(B, v, restarts=ls_restarts, seed=int(seed) + 222)
    best_f = _bruteforce_best_f(B, v, max_nvars=int(bruteforce_max_nvars))

    # plot_scaling_benchmark is minimization-oriented; negate f so "lower is better".
    reference_value = None if best_f is None else -float(best_f)
    return {
        "n": int(n_coverages),
        "n_vars": int(B.shape[1]),
        "reference_value": reference_value,
        "methods": {
            "dqi": {"value": -float(dqi_f), "time_s": float(dqi_t), "cost": float(shots)},
            "random": {"value": -float(rnd_f), "time_s": float(rnd_t), "cost": float(rnd_cost)},
            "local_search": {"value": -float(ls_f), "time_s": float(ls_t), "cost": float(ls_cost)},
        },
    }


def _convergence_history(
    *,
    data_dir: Path,
    package: int,
    subsample_packages: int,
    n_coverages: int,
    ell: int,
    bp_iterations: int,
    shots: int,
    seed: int,
    n_eval: int,
) -> list[float]:
    small = _load_problem(data_dir, n_coverages=n_coverages, subsample_packages=subsample_packages)
    B, v = build_insurance_parity_B_rhs(small, package)
    hist: list[float] = []
    for i in range(int(n_eval)):
        res = run_classic_dqi(
            B,
            v,
            ell=int(ell),
            bp_iterations=int(bp_iterations),
            shots=int(shots),
            seed=int(seed) + i,
            strict_ancilla=True,
        )
        f_val = float(res.expected_f) if res.expected_f is not None else -1e9
        hist.append(-f_val)  # minimization-style objective for visual parity
    return hist


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_default_data_dir())
    ap.add_argument("--package", type=int, default=0)
    ap.add_argument("--subsample-packages", type=int, default=2)
    ap.add_argument("--ell", type=int, default=1)
    ap.add_argument("--bp-iterations", type=int, default=1)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bruteforce-max-nvars", type=int, default=22)
    ap.add_argument("--official-ns", type=int, nargs="*", default=[10], help="Rows for official-style comparison")
    ap.add_argument("--test-ns", type=int, nargs="*", default=[10, 50], help="Rows for test-style comparison")
    ap.add_argument("--convergence-evals", type=int, default=35)
    ap.add_argument("--convergence-n", type=int, default=10)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    out_root = args.out_root
    out_bench = out_root / "benchmark_official"
    out_root.mkdir(parents=True, exist_ok=True)
    out_bench.mkdir(parents=True, exist_ok=True)

    # 1) Convergence counterpart
    hist = _convergence_history(
        data_dir=data_dir,
        package=int(args.package),
        subsample_packages=int(args.subsample_packages),
        n_coverages=int(args.convergence_n),
        ell=int(args.ell),
        bp_iterations=int(args.bp_iterations),
        shots=int(args.shots),
        seed=int(args.seed),
        n_eval=int(args.convergence_evals),
    )
    conv_path = out_root / "classic_dqi_convergence_10var.png"
    plot_convergence(hist, out_path=conv_path, title="Classic DQI convergence (10 vars)")
    print("wrote:", conv_path)

    # 2) Official comparison counterpart
    official_rows = [
        _classic_row(
            data_dir=data_dir,
            n_coverages=int(n),
            package=int(args.package),
            subsample_packages=int(args.subsample_packages),
            ell=int(args.ell),
            bp_iterations=int(args.bp_iterations),
            shots=int(args.shots),
            seed=int(args.seed) + int(n),
            bruteforce_max_nvars=int(args.bruteforce_max_nvars),
        )
        for n in args.official_ns
    ]
    official_plot = out_bench / "classic_dqi_official_sizes_comparison.png"
    plot_scaling_benchmark(
        official_rows,
        method_keys=("dqi", "random", "local_search"),
        out_path=official_plot,
        title_prefix="Classic official sizes benchmark",
    )
    print("wrote:", official_plot)

    # 3) test_plot counterpart
    test_rows = [
        _classic_row(
            data_dir=data_dir,
            n_coverages=int(n),
            package=int(args.package),
            subsample_packages=int(args.subsample_packages),
            ell=int(args.ell),
            bp_iterations=int(args.bp_iterations),
            shots=int(args.shots),
            seed=int(args.seed) + 1000 + int(n),
            bruteforce_max_nvars=int(args.bruteforce_max_nvars),
        )
        for n in args.test_ns
    ]
    test_plot = out_bench / "classic_test_plot.png"
    plot_scaling_benchmark(
        test_rows,
        method_keys=("dqi", "random", "local_search"),
        out_path=test_plot,
        title_prefix="Classic DQI vs classical baselines",
    )
    print("wrote:", test_plot)

    # Save row data for reproducibility
    json_out = out_bench / "classic_replicated_plot_rows.json"
    json_out.write_text(
        json.dumps(
            {
                "official_rows": official_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote:", json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

