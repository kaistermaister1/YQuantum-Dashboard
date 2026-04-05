"""Public DQI API and QuboBlock adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.dqi_core import bitstring_to_array, hamming_weight
from src.dqi_optimize import DqiOptimizationResult, optimize_dqi


@dataclass
class DqiRunMetadata:
    """Extended metadata returned alongside best solution/value."""

    optimizer_result: DqiOptimizationResult
    bitstring: str
    hamming_weight_full: int
    hamming_weight_coverage: int | None
    n_coverage: int | None
    n_slack: int | None
    coverage_bits: str | None
    constant_offset: float


def _extract_qubo_and_meta(Q_or_block: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Accept either a Q matrix or a qubo_block.QuboBlock instance."""
    if isinstance(Q_or_block, np.ndarray):
        return np.asarray(Q_or_block, dtype=float), {
            "constant_offset": 0.0,
            "n_coverage": None,
            "n_slack": None,
        }

    q_attr = getattr(Q_or_block, "Q", None)
    if q_attr is None:
        raise TypeError("Q must be a numpy matrix or an object with attribute .Q")

    q = np.asarray(q_attr, dtype=float)
    meta = {
        "constant_offset": float(getattr(Q_or_block, "constant_offset", 0.0)),
        "n_coverage": getattr(Q_or_block, "n_coverage", None),
        "n_slack": getattr(Q_or_block, "n_slack", None),
    }
    return q, meta


def run_dqi(
    Q: Any,
    p: int,
    optimizer: str,
    *,
    shots: int = 512,
    seed: int = 0,
    rng_seed: int = 0,
    maxiter: int = 60,
    n_samples: int = 64,
    statistic: str = "mean",
    mixer: str = "rx",
    max_qubits: int = 50,
) -> tuple[np.ndarray, float]:
    """Run DQI and return `(best_solution, value)` as requested."""
    q, meta = _extract_qubo_and_meta(Q)
    res = optimize_dqi(
        q,
        p=p,
        optimizer=optimizer,  # type: ignore[arg-type]
        statistic=statistic,  # type: ignore[arg-type]
        shots=shots,
        seed=seed,
        rng_seed=rng_seed,
        maxiter=maxiter,
        n_samples=n_samples,
        mixer=mixer,
        max_qubits=max_qubits,
        constant_offset=float(meta["constant_offset"]),
    )
    best_x = bitstring_to_array(res.stats_at_best.best_bitstring)
    return best_x, float(res.stats_at_best.best_value)


def run_dqi_with_details(
    Q: Any,
    p: int,
    optimizer: str,
    *,
    shots: int = 512,
    seed: int = 0,
    rng_seed: int = 0,
    maxiter: int = 60,
    n_samples: int = 64,
    statistic: str = "mean",
    mixer: str = "rx",
    max_qubits: int = 50,
) -> tuple[np.ndarray, float, DqiRunMetadata]:
    """Run DQI and return `(best_solution, value, metadata)`."""
    q, meta = _extract_qubo_and_meta(Q)
    res = optimize_dqi(
        q,
        p=p,
        optimizer=optimizer,  # type: ignore[arg-type]
        statistic=statistic,  # type: ignore[arg-type]
        shots=shots,
        seed=seed,
        rng_seed=rng_seed,
        maxiter=maxiter,
        n_samples=n_samples,
        mixer=mixer,
        max_qubits=max_qubits,
        constant_offset=float(meta["constant_offset"]),
    )

    bitstring = res.stats_at_best.best_bitstring
    best_x = bitstring_to_array(bitstring)
    value = float(res.stats_at_best.best_value)

    n_cov = meta["n_coverage"]
    coverage_bits: str | None = None
    hw_cov: int | None = None
    if isinstance(n_cov, int) and n_cov > 0:
        coverage_bits = bitstring[:n_cov]
        hw_cov = hamming_weight(bitstring_to_array(coverage_bits))

    details = DqiRunMetadata(
        optimizer_result=res,
        bitstring=bitstring,
        hamming_weight_full=hamming_weight(best_x),
        hamming_weight_coverage=hw_cov,
        n_coverage=n_cov if isinstance(n_cov, int) else None,
        n_slack=meta["n_slack"] if isinstance(meta["n_slack"], int) else None,
        coverage_bits=coverage_bits,
        constant_offset=float(meta["constant_offset"]),
    )
    return best_x, value, details
