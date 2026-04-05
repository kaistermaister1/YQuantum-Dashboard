#!/usr/bin/env python3
"""Benchmark DQI on official instance sizes vs classical baselines and save comparison plots.

Each size ``n`` is the number of **coverages** subsampled from the LTM instance; the QUBO is
built with ``qubo_block.build_qubo_block_for_package`` (so the matrix dimension is
``n_coverage + n_slack`` for that block, usually >= ``n``).

Requires ``--data-dir`` pointing at LTM ``instance_*.csv`` files.

Nexus **Selene** statevector jobs are capped at **26 qubits** per Quantinuum. If a
``QuboBlock`` has ``n_vars > 26``, DQI on ``--execution nexus-selene`` is skipped for
that row (classical baselines still run) unless you use ``--classical-only`` or another
backend (e.g. ``nexus-helios``).

Examples:

  python scripts/benchmark_dqi_official_sizes.py --data-dir path/to/ltm_csvs
  python scripts/benchmark_dqi_official_sizes.py --data-dir path/to/ltm --ns 10 21 20 50 200
  python scripts/benchmark_dqi_official_sizes.py --data-dir path/to/ltm --classical-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Non-interactive backend for headless runs
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_benchmarks import (
    benchmark_dqi_pipeline,
    brute_force_qubo,
    multistart_local_search_qubo,
    random_sampling_baseline,
)
from src.dqi_visualize import plot_scaling_benchmark
from src.insurance_model import load_ltm_instance, subsample_problem
from src.qubo_block import QuboBlock, build_qubo_block_for_package


OFFICIAL_NS_DEFAULT = (10, 20)

# Nexus API: Selene / SelenePlus statevector execute rejects programs above this qubit count.
NEXUS_SELENE_STATEVECTOR_MAX_QUBITS = 26

SELENE_LOCAL_FAIL_HELP = """
Local DQI uses the Selene emulator (Zig + Windows SDK). This error usually means the
SDK/MSVC environment is missing or not visible to Zig.

Fix options:
  1) Install "Desktop development with C++" (Visual Studio or Build Tools) and the
     Windows 10/11 SDK, then open a Developer shell and rerun with --execution local.
  2) Use Quantinuum Nexus (default for this script): omit --execution or use
     --execution nexus-selene
  3) Skip quantum and plot classical baselines only: --classical-only

Docs: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
"""


def build_block_for_coverage_count(
    data_dir: Path,
    n_coverages: int,
    *,
    package_index: int,
    penalty_weight: float | None,
    subsample_packages: int,
) -> tuple[QuboBlock, object]:
    """Load LTM data, subsample, return ``(package_block, subsampled_problem)``."""
    problem = load_ltm_instance(data_dir)
    n_cov = min(int(n_coverages), int(problem.N))
    if n_cov < 1:
        raise ValueError(f"n_coverages must be >= 1 after cap (got {n_coverages}, N={problem.N})")
    n_pkg = int(subsample_packages) if int(subsample_packages) > 0 else int(problem.M)
    n_pkg = min(n_pkg, int(problem.M))
    small = subsample_problem(problem, n_cov, n_pkg)
    if package_index < 0 or package_index >= small.M:
        raise ValueError(f"package_index {package_index} out of range for subsampled M={small.M}")
    block = build_qubo_block_for_package(
        small,
        package_index=int(package_index),
        penalty_weight=penalty_weight,
    )
    return block, small


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with LTM instance_*.csv files (see insurance_model.load_ltm_instance)",
    )
    ap.add_argument("--package", type=int, default=0, help="Package index for QuboBlock")
    ap.add_argument(
        "--penalty-weight",
        type=float,
        default=None,
        help="QUBO penalty lambda (default: qubo_block.default_penalty_weight)",
    )
    ap.add_argument(
        "--subsample-packages",
        type=int,
        default=0,
        help="Keep first k packages after load (0 = keep all M; used with subsampled coverages)",
    )
    ap.add_argument(
        "--ns",
        type=int,
        nargs="*",
        default=list(OFFICIAL_NS_DEFAULT),
        help="Coverage counts to subsample, run order (default: 10 20)",
    )
    ap.add_argument("--skip-ns", type=int, nargs="*", default=[], help="Sizes to skip")
    ap.add_argument("--quick", action="store_true", help="Only n=10 and n=20")
    ap.add_argument("--p", type=int, default=None, help="DQI layers (default: 2 if n<=21 else 1)")
    ap.add_argument("--shots", type=int, default=512, help="DQI shots; random baseline matches this budget")
    ap.add_argument("--dqi-seed", type=int, default=7)
    ap.add_argument("--random-seed", type=int, default=12)
    ap.add_argument(
        "--execution",
        choices=["local", "selene", "nexus-selene", "nexus-helios"],
        default="nexus-selene",
        help="Default nexus-selene; use local/selene for on-machine Selene (see dqi_backends).",
    )
    ap.add_argument("--max-qubits", type=int, default=256, help="Must be >= largest n")
    ap.add_argument("--mixer", choices=["h", "rx"], default="h")
    ap.add_argument("--statistic", choices=["mean", "best"], default="mean")
    ap.add_argument(
        "--dicke-k",
        type=int,
        default=None,
        help=(
            "Coverage-register Dicke Hamming weight for DQI parity mode. "
            "Default: min(K, n_coverages_in_row)."
        ),
    )
    ap.add_argument(
        "--bp-decoder",
        action="store_true",
        help="Use dqi-main-compatible BP pipeline (Qiskit local simulation only).",
    )
    ap.add_argument("--bp-iterations", type=int, default=1, help="Belief-propagation iterations in --bp-decoder mode.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: cayman/artifacts/benchmark_official")
    ap.add_argument("--include-qaoa", action="store_true", help="QAOA baseline for n<=50 only")
    ap.add_argument(
        "--local-search-restarts",
        type=int,
        default=None,
        help="Override multistart count for classical 1-opt baseline",
    )
    ap.add_argument(
        "--classical-only",
        action="store_true",
        help="Do not run Guppy/Selene; only random + multistart 1-opt (+ brute if n<=22).",
    )
    ap.add_argument(
        "--nexus-no-timeout",
        action="store_true",
        help="Do not cap wait time for each Nexus execute (timeout=None).",
    )
    ap.add_argument(
        "--nexus-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait per DQI Nexus job (default 1800; use --nexus-no-timeout for no cap).",
    )
    ap.add_argument(
        "--nexus-max-cost",
        type=float,
        default=None,
        help="Optional Nexus max_cost (required by some Helios systems).",
    )
    ap.add_argument(
        "--nexus-hugr-name",
        default="dqi-hugr",
        help="HUGR package name prefix on Nexus (same as run_dqi_cli).",
    )
    ap.add_argument(
        "--nexus-job-name",
        default="dqi-execute",
        help="Execute job name prefix on Nexus.",
    )
    ap.add_argument(
        "--nexus-helios-system",
        default="Helios-1",
        help="HeliosConfig.system_name when using nexus-helios.",
    )
    ap.add_argument(
        "--nexus-selene-max-qubits",
        type=int,
        default=NEXUS_SELENE_STATEVECTOR_MAX_QUBITS,
        help=(
            "If n_vars exceeds this and execution is nexus-selene, skip DQI for that row "
            f"(default {NEXUS_SELENE_STATEVECTOR_MAX_QUBITS}, per Nexus Selene statevector limit)."
        ),
    )
    return ap.parse_args()


def _classical_methods_for_qubo(
    Q: np.ndarray,
    *,
    constant_offset: float = 0.0,
    shots: int,
    random_seed: int,
    brute_force_max_n: int,
    local_search_restarts: int | None,
) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Random + multistart local search; optional exact optimum for small n."""
    q = np.asarray(Q, dtype=float)
    const = float(constant_offset)
    n_q = q.shape[0]
    restarts = (
        int(local_search_restarts)
        if local_search_restarts is not None
        else max(8, min(128, max(16, n_q)))
    )

    t1 = time.perf_counter()
    _, rnd_val, rnd_ne = random_sampling_baseline(
        q,
        n_samples=int(shots),
        rng_seed=int(random_seed),
        constant_offset=const,
    )
    rnd_t = time.perf_counter() - t1

    t2 = time.perf_counter()
    _, ls_val, ls_ne = multistart_local_search_qubo(
        q,
        n_restarts=restarts,
        rng_seed=int(random_seed) + 911,
        constant_offset=const,
    )
    ls_t = time.perf_counter() - t2

    ref_val: float | None = None
    try:
        _, bf_val = brute_force_qubo(
            q,
            constant_offset=const,
            max_n=brute_force_max_n,
        )
        ref_val = float(bf_val)
    except ValueError:
        pass

    methods = {
        "random": {"value": float(rnd_val), "time_s": float(rnd_t), "cost": float(rnd_ne)},
        "local_search": {"value": float(ls_val), "time_s": float(ls_t), "cost": float(ls_ne)},
    }
    return ref_val, methods


def main() -> int:
    args = parse_args()
    ns = [int(x) for x in args.ns]
    if args.quick:
        ns = [n for n in (10, 20) if n not in set(args.skip_ns)]
    else:
        ns = [n for n in ns if n not in set(args.skip_ns)]

    if not ns:
        print("No instance sizes to run.", file=sys.stderr)
        return 1

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"data dir not found: {data_dir}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (ROOT / "artifacts" / "benchmark_official")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_n = max(ns)
    max_qubits = max(int(args.max_qubits), max_n)

    rows: list[dict] = []
    summary: list[dict] = []
    classical_only = bool(args.classical_only)
    plot_methods: tuple[str, ...] = (
        ("random", "local_search") if classical_only else ("dqi", "random", "local_search")
    )
    plot_title = (
        "Official sizes - classical baselines only"
        if classical_only
        else "Official sizes benchmark"
    )

    exec_key = str(args.execution).replace("-", "_")
    local_selene = exec_key in ("local", "selene")
    if args.bp_decoder and not local_selene:
        raise ValueError("--bp-decoder requires --execution local (Qiskit simulation path).")
    nexus_timeout = None if args.nexus_no_timeout else float(args.nexus_timeout)
    selene_limit = max(1, int(args.nexus_selene_max_qubits))

    for n in ns:
        p = int(args.p) if args.p is not None else (2 if n <= 21 else 1)
        block, small_problem = build_block_for_coverage_count(
            data_dir,
            n,
            package_index=int(args.package),
            penalty_weight=args.penalty_weight,
            subsample_packages=int(args.subsample_packages),
        )
        n_vars = int(block.n_vars)
        max_qubits = max(max_qubits, n_vars)

        dqi_skipped_selene = False

        if classical_only:
            ref_val, methods = _classical_methods_for_qubo(
                block.Q,
                constant_offset=float(block.constant_offset),
                shots=int(args.shots),
                random_seed=int(args.random_seed),
                brute_force_max_n=22,
                local_search_restarts=args.local_search_restarts,
            )
        elif exec_key == "nexus_selene" and n_vars > selene_limit:
            dqi_skipped_selene = True
            print(
                f"Skipping DQI for coverages={n} (n_vars={n_vars}): Nexus Selene statevector "
                f"limit is {selene_limit} qubits. Classical baselines only for this row. "
                "Use --execution nexus-helios, --classical-only, or smaller --ns.\n",
                file=sys.stderr,
            )
            ref_val, methods = _classical_methods_for_qubo(
                block.Q,
                constant_offset=float(block.constant_offset),
                shots=int(args.shots),
                random_seed=int(args.random_seed),
                brute_force_max_n=22,
                local_search_restarts=args.local_search_restarts,
            )
        else:
            try:
                bench = benchmark_dqi_pipeline(
                    block,
                    p=p,
                    shots=int(args.shots),
                    dqi_seed=int(args.dqi_seed),
                    random_seed=int(args.random_seed),
                    brute_force_max_n=22,
                    random_samples=int(args.shots),
                    include_qaoa_baseline=bool(args.include_qaoa),
                    include_local_search_baseline=True,
                    local_search_restarts=args.local_search_restarts,
                    mixer=str(args.mixer),
                    statistic=str(args.statistic),
                    execution=exec_key,
                    max_qubits=max_qubits,
                    nexus_hugr_name=f"{args.nexus_hugr_name}-cov{n}-p{args.package}",
                    nexus_job_name=f"{args.nexus_job_name}-cov{n}-p{args.package}",
                    nexus_helios_system=str(args.nexus_helios_system),
                    nexus_timeout=nexus_timeout,
                    nexus_max_cost=args.nexus_max_cost,
                    insurance_parity=(small_problem, int(args.package)),
                    dicke_k=(
                        int(args.dicke_k)
                        if args.dicke_k is not None
                        else int(min(small_problem.max_options_per_package, small_problem.N))
                    ),
                    use_bp_decoder=bool(args.bp_decoder),
                    bp_iterations=int(args.bp_iterations),
                )
            except TimeoutError:
                print(
                    "\nNexus wait timed out (queue or compile can exceed the default cap). Try:\n"
                    "  --nexus-timeout 3600\n"
                    "  --nexus-no-timeout\n",
                    file=sys.stderr,
                )
                raise
            except RuntimeError as exc:
                err = str(exc).lower().replace("_", "")
                if local_selene and ("windowssdk" in err or "zig" in err):
                    print(SELENE_LOCAL_FAIL_HELP, file=sys.stderr)
                raise

            ref_val = float(bench["bruteforce"].best_value) if "bruteforce" in bench else None

            def pack(key: str) -> dict[str, float]:
                r = bench[key]
                extra = r.extra or {}
                cost = float(extra.get("n_energy_evaluations", extra.get("n_samples", 0)))
                return {
                    "value": float(r.best_value),
                    "time_s": float(r.runtime_sec),
                    "cost": cost,
                }

            methods = {
                "dqi": pack("dqi"),
                "random": pack("random"),
                "local_search": pack("local_search"),
            }

        row = {
            "n": n,
            "n_vars": n_vars,
            "p": p,
            "package": int(args.package),
            "reference_value": ref_val,
            "methods": methods,
            "dqi_skipped_selene_qubit_cap": dqi_skipped_selene,
        }
        rows.append(row)

        sum_row: dict = {
            "n": n,
            "n_vars": n_vars,
            "package": int(args.package),
            "p": p,
            "reference_optimal": ref_val,
            "random": methods["random"],
            "local_search": methods["local_search"],
            "dqi_skipped_selene_qubit_cap": dqi_skipped_selene,
        }
        if not classical_only and "dqi" in methods:
            sum_row["dqi"] = methods["dqi"]
        summary.append(sum_row)

        print(f"=== coverages={n}  n_vars={n_vars}  package={args.package}  (p={p}) ===")
        for name, m in methods.items():
            print(
                f"  {name:12s}  objective={m['value']:.6f}  "
                f"time={m['time_s']:.4f}s  cost={int(m['cost'])}"
            )
        if ref_val is not None:
            print(f"  {'exact':12s}  objective={ref_val:.6f}")

    plot_path = out_dir / "dqi_official_sizes_comparison.png"
    plot_scaling_benchmark(
        rows,
        method_keys=plot_methods,
        out_path=plot_path,
        title_prefix=plot_title,
    )
    print("\nwrote:", plot_path)

    json_path = out_dir / "dqi_official_sizes_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote:", json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
