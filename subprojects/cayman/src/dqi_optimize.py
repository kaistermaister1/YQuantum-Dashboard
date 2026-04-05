"""Single-shot DQI: one circuit execution; no classical angle optimization loop.

Default mode is fixed-angle QUBO/Ising phase plus mixer (QAOA-shaped).  Optional ``B`` /
``parity_rhs`` enable XOR syndrome qubits and post-selection; optional ``dicke_k`` /
``n_coverage`` prepare a Dicke state on coverage bits (Bärtschi–Eidenbenz schedule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.dqi_core import DqiSampleStats, bitstring_to_array, qubo_energy, sample_dqi

Statistic = Literal["mean", "best"]


@dataclass
class DqiRunResult:
    """Fixed-angle DQI run: one kernel execution and sample statistics."""

    gammas: list[float]
    betas: list[float]
    objective_value: float
    statistic: Statistic
    stats_at_best: DqiSampleStats
    n_evaluations: int
    history: list[float]


def mean_sample_energy(Q: np.ndarray, stats: DqiSampleStats, constant_offset: float = 0.0) -> float:
    """Compute expected QUBO value under the empirical sample histogram."""
    total = int(sum(stats.bitstring_counts.values()))
    if total <= 0:
        return float("nan")
    agg = 0.0
    for bitstring, count in stats.bitstring_counts.items():
        x = bitstring_to_array(bitstring)
        agg += float(count) * qubo_energy(x, Q, constant_offset=constant_offset)
    return float(agg / total)


def _objective(
    Q: np.ndarray,
    stats: DqiSampleStats,
    *,
    statistic: Statistic,
    constant_offset: float,
) -> float:
    if statistic == "mean":
        return mean_sample_energy(Q, stats, constant_offset=constant_offset)
    if statistic == "best":
        return float(stats.best_value)
    raise ValueError(f"Unsupported statistic: {statistic}")


def _default_angles(p: int) -> tuple[list[float], list[float]]:
    half_pi = 0.5 * math.pi
    return [half_pi] * int(p), [half_pi] * int(p)


def run_dqi_oneshot(
    Q: np.ndarray,
    p: int,
    *,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    statistic: Statistic = "mean",
    shots: int = 512,
    seed: int = 0,
    mixer: str = "h",
    max_qubits: int = 50,
    constant_offset: float = 0.0,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
    B: np.ndarray | None = None,
    parity_rhs: np.ndarray | None = None,
    dicke_k: int | None = None,
    n_coverage: int | None = None,
) -> DqiRunResult:
    """Run DQI once with fixed angles (interference from cost phases + mixer, default Hadamard).

    There is no classical optimization loop; angles default to π/2 per layer unless ``gammas`` /
    ``betas`` are provided.
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    q = np.asarray(Q, dtype=float)
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")

    if gammas is None and betas is None:
        g_list, b_list = _default_angles(p)
    else:
        if gammas is None or betas is None:
            raise ValueError("Provide both gammas and betas, or neither for defaults")
        g_list = [float(v) for v in gammas]
        b_list = [float(v) for v in betas]
        if len(g_list) != p or len(b_list) != p:
            raise ValueError(f"gammas and betas must each have length p={p}")

    stats = sample_dqi(
        q,
        gammas=g_list,
        betas=b_list,
        shots=int(shots),
        seed=int(seed),
        mixer=mixer,
        max_qubits=max_qubits,
        constant_offset=constant_offset,
        execution=execution,
        nexus_hugr_name=nexus_hugr_name,
        nexus_job_name=nexus_job_name,
        nexus_helios_system=nexus_helios_system,
        nexus_timeout=nexus_timeout,
        eval_tag="",
        B=B,
        parity_rhs=parity_rhs,
        dicke_k=dicke_k,
        n_coverage=n_coverage,
    )
    obj = _objective(q, stats, statistic=statistic, constant_offset=constant_offset)
    return DqiRunResult(
        gammas=g_list,
        betas=b_list,
        objective_value=float(obj),
        statistic=statistic,
        stats_at_best=stats,
        n_evaluations=1,
        history=[float(obj)],
    )
