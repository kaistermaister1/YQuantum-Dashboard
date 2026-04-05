#!/usr/bin/env python3
"""Benchmark DQI on official instance sizes vs classical baselines and save comparison plots.

Instance sizes default to n ∈ {10, 21, 20, 50, 200} (same family as ``example_dqi_10var`` QUBOs).
Reports solution quality (objective relative to exact optimum when n ≤ 22, else best-of-methods),
wall time, and workflow cost (shots vs classical energy evaluations).

Examples:

  python scripts/benchmark_dqi_official_sizes.py
  python scripts/benchmark_dqi_official_sizes.py --skip-ns 200
  python scripts/benchmark_dqi_official_sizes.py --execution local --quick
  python scripts/benchmark_dqi_official_sizes.py --classical-only
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
from src.dqi_core import build_hamming_weight_penalty_qubo
from src.dqi_visualize import plot_scaling_benchmark


OFFICIAL_NS_DEFAULT = (10, 21, 20, 50, 200)

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


def build_official_qubo(n: int, *, rng_seed: int = 26) -> np.ndarray:
    """Reproducible insurance-style QUBO: linear terms, random couplings, Hamming penalty."""
    rng = np.random.default_rng(int(rng_seed) + n * 1_000_003)
    target_weight = max(1, min(n - 1, int(round(0.4 * n)))) if n > 1 else 1
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
    return (base_Q + base_Q.T) * 0.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--ns",
        type=int,
        nargs="*",
        default=list(OFFICIAL_NS_DEFAULT),
        help="Instance sizes in run order (default: 10 21 20 50 200)",
    )
    ap.add_argument("--skip-ns", type=int, nargs="*", default=[], help="Sizes to skip")
    ap.add_argument("--quick", action="store_true", help="Only n=10 and n=20")
    ap.add_argument("--p", type=int, default=None, help="DQI layers (default: 2 if n≤21 else 1)")
    ap.add_argument("--shots", type=int, default=512, help="DQI shots; random baseline matches this budget")
    ap.add_argument("--dqi-seed", type=int, default=7)
    ap.add_argument("--random-seed", type=int, default=12)
    ap.add_argument(
        "--execution",
        choices=["local", "selene", "nexus-selene", "nexus-helios"],
        default="nexus-selene",
        help="Default nexus-selene; use local/selene for on-machine Selene (see dqi_backends).",
    )
    ap.add_argument("--max-qubits", type=int, default=256, help="Must be ≥ largest n")
    ap.add_argument("--mixer", choices=["h", "rx"], default="h")
    ap.add_argument("--statistic", choices=["mean", "best"], default="mean")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: cayman/artifacts/benchmark_official")
    ap.add_argument("--include-qaoa", action="store_true", help="QAOA baseline for n≤50 only")
    ap.add_argument(
        "--local-search-restarts",
        type=int,
        default=None,
        help="Override multistart count for classical 1-opt baseline",
    )
    ap.add_argument(
        "--classical-only",
        action="store_true",
        help="Do not run Guppy/Selene; only random + multistart 1-opt (+ brute if n≤22).",
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
    return ap.parse_args()


def _classical_methods_for_qubo(
    Q: np.ndarray,
    *,
    shots: int,
    random_seed: int,
    brute_force_max_n: int,
    local_search_restarts: int | None,
) -> tuple[float | None, dict[str, dict[str, float]]]:
    """Random + multistart local search; optional exact optimum for small n."""
    q = np.asarray(Q, dtype=float)
    const = 0.0
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
        _, bf_val = brute_force_qubo(q, constant_offset=const, max_n=brute_force_max_n)
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
        "Official sizes — classical baselines only"
        if classical_only
        else "Official sizes benchmark"
    )

    exec_key = str(args.execution).replace("-", "_")
    local_selene = exec_key in ("local", "selene")
    nexus_timeout = None if args.nexus_no_timeout else float(args.nexus_timeout)

    for n in ns:
        p = int(args.p) if args.p is not None else (2 if n <= 21 else 1)
        Q = build_official_qubo(n)

        if classical_only:
            ref_val, methods = _classical_methods_for_qubo(
                Q,
                shots=int(args.shots),
                random_seed=int(args.random_seed),
                brute_force_max_n=22,
                local_search_restarts=args.local_search_restarts,
            )
        else:
            try:
                bench = benchmark_dqi_pipeline(
                    Q,
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
                    nexus_hugr_name=f"{args.nexus_hugr_name}-n{n}",
                    nexus_job_name=f"{args.nexus_job_name}-n{n}",
                    nexus_helios_system=str(args.nexus_helios_system),
                    nexus_timeout=nexus_timeout,
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

        row = {"n": n, "p": p, "reference_value": ref_val, "methods": methods}
        rows.append(row)

        sum_row: dict = {
            "n": n,
            "p": p,
            "reference_optimal": ref_val,
            "random": methods["random"],
            "local_search": methods["local_search"],
        }
        if not classical_only:
            sum_row["dqi"] = methods["dqi"]
        summary.append(sum_row)

        print(f"=== n={n} (p={p}) ===")
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
