"""Classical optimization loops for DQI angle parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.dqi_core import DqiSampleStats, bitstring_to_array, qubo_energy, sample_dqi

Statistic = Literal["mean", "best"]
OptimizerName = Literal["random", "cobyla", "spsa"]


@dataclass
class DqiOptimizationResult:
    """Best parameter set and associated sampling statistics."""

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


def _scipy_minimize():
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("COBYLA requires SciPy: pip install scipy>=1.10") from exc
    return minimize


def _clip_params(theta: np.ndarray, p: int) -> np.ndarray:
    out = np.asarray(theta, dtype=float).copy()
    out[:p] = np.clip(out[:p], 0.0, math.pi)
    out[p:] = np.clip(out[p:], 0.0, math.pi)
    return out


def optimize_dqi(
    Q: np.ndarray,
    p: int,
    *,
    optimizer: OptimizerName = "cobyla",
    statistic: Statistic = "mean",
    shots: int = 512,
    seed: int = 0,
    rng_seed: int = 0,
    maxiter: int = 60,
    n_samples: int = 64,
    spsa_a: float = 0.15,
    spsa_c: float = 0.12,
    spsa_alpha: float = 0.602,
    spsa_gamma: float = 0.101,
    spsa_A: float = 10.0,
    mixer: str = "rx",
    max_qubits: int = 50,
    constant_offset: float = 0.0,
    execution: str = "local",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
) -> DqiOptimizationResult:
    """Optimize DQI angles for a fixed QUBO matrix."""
    if p < 1:
        raise ValueError("p must be >= 1")
    q = np.asarray(Q, dtype=float)
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")

    history: list[float] = []
    eval_idx = 0

    def evaluate(theta: np.ndarray) -> tuple[float, DqiSampleStats]:
        nonlocal eval_idx
        vec = _clip_params(theta, p)
        gammas = [float(v) for v in vec[:p]]
        betas = [float(v) for v in vec[p:]]
        stats = sample_dqi(
            q,
            gammas=gammas,
            betas=betas,
            shots=int(shots),
            seed=int(seed + eval_idx),
            mixer=mixer,
            max_qubits=max_qubits,
            constant_offset=constant_offset,
            execution=execution,
            nexus_hugr_name=nexus_hugr_name,
            nexus_job_name=nexus_job_name,
            nexus_helios_system=nexus_helios_system,
            nexus_timeout=nexus_timeout,
            eval_tag=str(eval_idx),
        )
        obj = _objective(q, stats, statistic=statistic, constant_offset=constant_offset)
        history.append(float(obj))
        eval_idx += 1
        return float(obj), stats

    theta0 = np.full(2 * p, 0.5 * math.pi, dtype=float)
    best_obj = float("inf")
    best_theta = theta0.copy()
    best_stats: DqiSampleStats | None = None

    def accept(theta: np.ndarray, obj: float, stats: DqiSampleStats) -> None:
        nonlocal best_obj, best_theta, best_stats
        if obj < best_obj:
            best_obj = float(obj)
            best_theta = _clip_params(theta, p)
            best_stats = stats

    if optimizer == "random":
        rng = np.random.default_rng(int(rng_seed))
        for _ in range(int(n_samples)):
            theta = rng.uniform(0.0, math.pi, size=2 * p)
            obj, stats = evaluate(theta)
            accept(theta, obj, stats)

    elif optimizer == "cobyla":
        minimize = _scipy_minimize()

        def fun(theta: np.ndarray) -> float:
            obj, stats = evaluate(theta)
            accept(theta, obj, stats)
            return float(obj)

        bounds = [(0.0, math.pi)] * (2 * p)
        minimize(
            fun,
            theta0,
            method="COBYLA",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )

    elif optimizer == "spsa":
        rng = np.random.default_rng(int(rng_seed))
        theta = theta0.copy()
        for k in range(int(maxiter)):
            a_k = float(spsa_a / (k + 1 + float(spsa_A)) ** spsa_alpha)
            c_k = float(spsa_c / (k + 1) ** spsa_gamma)
            delta = rng.choice([-1.0, 1.0], size=2 * p).astype(float)
            theta_plus = _clip_params(theta + c_k * delta, p)
            theta_minus = _clip_params(theta - c_k * delta, p)
            y_plus, stats_plus = evaluate(theta_plus)
            accept(theta_plus, y_plus, stats_plus)
            y_minus, stats_minus = evaluate(theta_minus)
            accept(theta_minus, y_minus, stats_minus)
            g_hat = np.where(delta == 0.0, 0.0, (y_plus - y_minus) / (2.0 * c_k * delta))
            theta = _clip_params(theta - a_k * g_hat, p)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if best_stats is None:
        # Fallback for any unexpected empty loop configuration.
        obj, stats = evaluate(theta0)
        accept(theta0, obj, stats)
        assert best_stats is not None

    return DqiOptimizationResult(
        gammas=[float(v) for v in best_theta[:p]],
        betas=[float(v) for v in best_theta[p:]],
        objective_value=float(best_obj),
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=int(eval_idx),
        history=history,
    )
