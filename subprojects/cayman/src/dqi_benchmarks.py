"""Benchmark helpers for DQI, random search, brute force, and optional QAOA baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.dqi_core import bitstring_to_array, qubo_energy
from src.run_dqi import run_dqi_with_details


@dataclass
class BenchmarkResult:
    method: str
    best_solution: np.ndarray
    best_value: float
    runtime_sec: float
    approximation_ratio: float | None
    extra: dict[str, Any]


def brute_force_qubo(
    Q: np.ndarray,
    *,
    constant_offset: float = 0.0,
    max_n: int = 22,
) -> tuple[np.ndarray, float]:
    """Exact minimization for small QUBOs."""
    q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    if n > max_n:
        raise ValueError(f"Brute force disabled for n={n} > max_n={max_n}")
    best_val = float("inf")
    best = np.zeros(n, dtype=float)
    for k in range(1 << n):
        x = np.array([float((k >> i) & 1) for i in range(n)], dtype=float)
        v = qubo_energy(x, q, constant_offset=constant_offset)
        if v < best_val:
            best_val = float(v)
            best = x.copy()
    return best, float(best_val)


def random_sampling_baseline(
    Q: np.ndarray,
    *,
    n_samples: int = 4096,
    rng_seed: int = 0,
    constant_offset: float = 0.0,
) -> tuple[np.ndarray, float, int]:
    """Classical random bitstring baseline. Returns ``(x_best, energy, n_energy_evaluations)``."""
    q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    rng = np.random.default_rng(int(rng_seed))
    best_val = float("inf")
    best = np.zeros(n, dtype=float)
    n_eval = 0
    for _ in range(int(n_samples)):
        x = rng.integers(0, 2, size=n).astype(float)
        v = qubo_energy(x, q, constant_offset=constant_offset)
        n_eval += 1
        if v < best_val:
            best_val = float(v)
            best = x.copy()
    return best, float(best_val), n_eval


def local_search_1opt_qubo(
    Q: np.ndarray,
    x0: np.ndarray,
    *,
    constant_offset: float = 0.0,
) -> tuple[np.ndarray, float, int]:
    """Greedy single-bit flips until a local minimum. Returns ``(x, energy, n_energy_evaluations)``."""
    q = np.asarray(Q, dtype=float)
    x = np.asarray(x0, dtype=float).ravel().copy()
    n_eval = 0

    def eval_at(xx: np.ndarray) -> float:
        nonlocal n_eval
        n_eval += 1
        return float(qubo_energy(xx, q, constant_offset=constant_offset))

    current = eval_at(x)
    while True:
        best_i: int | None = None
        best_val = current
        for i in range(q.shape[0]):
            x2 = x.copy()
            x2[i] = 1.0 - x2[i]
            v = eval_at(x2)
            if v < best_val - 1e-15:
                best_val = v
                best_i = i
        if best_i is None:
            break
        x[best_i] = 1.0 - x[best_i]
        current = best_val
    return x, float(current), n_eval


def multistart_local_search_qubo(
    Q: np.ndarray,
    *,
    n_restarts: int,
    rng_seed: int,
    constant_offset: float = 0.0,
) -> tuple[np.ndarray, float, int]:
    """Several random starts + 1-opt local search (strong scalable classical baseline)."""
    q = np.asarray(Q, dtype=float)
    n = q.shape[0]
    rng = np.random.default_rng(int(rng_seed))
    best_val = float("inf")
    best = np.zeros(n, dtype=float)
    total_eval = 0
    for _ in range(int(n_restarts)):
        x0 = rng.integers(0, 2, size=n).astype(float)
        x, val, ne = local_search_1opt_qubo(q, x0, constant_offset=constant_offset)
        total_eval += ne
        if val < best_val:
            best_val = val
            best = x.copy()
    return best, float(best_val), total_eval


def _approx_ratio(best_val: float, ref_val: float) -> float | None:
    # For minimization: ratio 1 means optimal (or reference-equal), >1 worse.
    denom = abs(ref_val)
    if denom < 1e-12:
        return None
    return float(best_val / ref_val)


def benchmark_dqi_pipeline(
    Q_or_block: Any,
    *,
    p: int = 1,
    shots: int = 512,
    dqi_seed: int = 0,
    random_seed: int = 1,
    brute_force_max_n: int = 22,
    random_samples: int = 4096,
    include_qaoa_baseline: bool = True,
    include_local_search_baseline: bool = False,
    local_search_restarts: int | None = None,
    mixer: str = "h",
    statistic: str = "mean",
    execution: str = "nexus_selene",
    max_qubits: int = 50,
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
    nexus_max_cost: float | None = None,
    insurance_parity: tuple[Any, int] | None = None,
    dicke_k: int | None = None,
    use_bp_decoder: bool = False,
    bp_iterations: int = 1,
    progress_callback: Callable[[dict[str, BenchmarkResult]], None] | None = None,
) -> dict[str, BenchmarkResult]:
    """Run DQI + baselines and return comparable metrics.

    If ``progress_callback`` is set, it is invoked with a shallow copy of ``results`` after
    each method completes (before ``approximation_ratio`` is filled, except the optional
    final call — see below).

    After all ratios are computed, ``progress_callback`` is invoked one last time with the
    final mapping (if it was provided).
    """

    def _notify() -> None:
        if progress_callback is not None:
            progress_callback(dict(results))
    q = np.asarray(getattr(Q_or_block, "Q", Q_or_block), dtype=float)
    const = float(getattr(Q_or_block, "constant_offset", 0.0))

    results: dict[str, BenchmarkResult] = {}

    t0 = time.perf_counter()
    dqi_x, dqi_val, dqi_meta = run_dqi_with_details(
        Q_or_block,
        p=p,
        shots=shots,
        seed=dqi_seed,
        mixer=mixer,
        statistic=statistic,  # type: ignore[arg-type]
        execution=execution,
        max_qubits=max_qubits,
        nexus_hugr_name=nexus_hugr_name,
        nexus_job_name=nexus_job_name,
        nexus_helios_system=nexus_helios_system,
        nexus_timeout=nexus_timeout,
        nexus_max_cost=nexus_max_cost,
        insurance_parity=insurance_parity,
        dicke_k=dicke_k,
        use_bp_decoder=use_bp_decoder,
        bp_iterations=bp_iterations,
    )
    dqi_runtime = time.perf_counter() - t0
    results["dqi"] = BenchmarkResult(
        method="dqi",
        best_solution=dqi_x,
        best_value=float(dqi_val),
        runtime_sec=float(dqi_runtime),
        approximation_ratio=None,
        extra={
            "history": dqi_meta.run_result.history,
            "n_evaluations": dqi_meta.run_result.n_evaluations,
            "n_energy_evaluations": int(shots),
            "bitstring": dqi_meta.bitstring,
            "bitstring_counts": dqi_meta.run_result.stats_at_best.bitstring_counts,
            "post_selection_rate": dqi_meta.run_result.stats_at_best.post_selection_rate,
        },
    )
    _notify()

    t1 = time.perf_counter()
    rnd_x, rnd_val, rnd_ne = random_sampling_baseline(
        q,
        n_samples=random_samples,
        rng_seed=random_seed,
        constant_offset=const,
    )
    rnd_runtime = time.perf_counter() - t1
    results["random"] = BenchmarkResult(
        method="random",
        best_solution=rnd_x,
        best_value=float(rnd_val),
        runtime_sec=float(rnd_runtime),
        approximation_ratio=None,
        extra={"n_samples": int(random_samples), "n_energy_evaluations": int(rnd_ne)},
    )
    _notify()

    if include_local_search_baseline:
        n_q = q.shape[0]
        restarts = (
            int(local_search_restarts)
            if local_search_restarts is not None
            else max(8, min(128, max(16, n_q)))
        )
        t_ls = time.perf_counter()
        ls_x, ls_val, ls_ne = multistart_local_search_qubo(
            q,
            n_restarts=restarts,
            rng_seed=int(random_seed) + 911,
            constant_offset=const,
        )
        ls_runtime = time.perf_counter() - t_ls
        results["local_search"] = BenchmarkResult(
            method="local_search",
            best_solution=ls_x,
            best_value=float(ls_val),
            runtime_sec=float(ls_runtime),
            approximation_ratio=None,
            extra={"n_restarts": restarts, "n_energy_evaluations": int(ls_ne)},
        )
        _notify()

    brute_val: float | None = None
    try:
        t2 = time.perf_counter()
        bf_x, bf_val = brute_force_qubo(q, constant_offset=const, max_n=brute_force_max_n)
        bf_runtime = time.perf_counter() - t2
        brute_val = float(bf_val)
        results["bruteforce"] = BenchmarkResult(
            method="bruteforce",
            best_solution=bf_x,
            best_value=float(bf_val),
            runtime_sec=float(bf_runtime),
            approximation_ratio=1.0,
            extra={"max_n": int(brute_force_max_n)},
        )
        _notify()
    except ValueError:
        pass

    if include_qaoa_baseline and q.shape[0] <= 50:
        try:
            from src.qubo_qaoa_optimize import optimize_qaoa_p1_random
            from src.qubo_block import QuboBlock

            q_block = QuboBlock(
                package_index=0,
                Q=q,
                n_coverage=q.shape[0],
                n_slack=0,
                coverage_offset=0,
                penalty_weight=1.0,
                constant_offset=const,
            )
            t3 = time.perf_counter()
            qaoa_res = optimize_qaoa_p1_random(
                q_block,
                n_samples=24,
                shots=max(256, shots // 2),
                rng_seed=dqi_seed + 99,
                seed_offset=dqi_seed + 1000,
                statistic="mean",
                max_qubits=min(50, int(max_qubits)),
            )
            qaoa_runtime = time.perf_counter() - t3
            qaoa_bits = qaoa_res.stats_at_best.best_bitstring
            qaoa_x = bitstring_to_array(qaoa_bits)
            qaoa_val = float(qaoa_res.stats_at_best.best_qubo_energy)
            results["qaoa"] = BenchmarkResult(
                method="qaoa",
                best_solution=qaoa_x,
                best_value=qaoa_val,
                runtime_sec=float(qaoa_runtime),
                approximation_ratio=None,
                extra={"best_bitstring": qaoa_bits, "n_evaluations": qaoa_res.n_evaluations},
            )
            _notify()
        except Exception:
            # Baseline is optional.
            pass

    if brute_val is not None:
        for key, val in results.items():
            if key == "bruteforce":
                continue
            val.approximation_ratio = _approx_ratio(val.best_value, brute_val)

    _notify()
    return results
