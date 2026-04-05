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


def _clip_params(theta: np.ndarray, p: int, *, parity: bool) -> np.ndarray:
    out = np.asarray(theta, dtype=float).copy()
    out[:p] = np.clip(out[:p], 0.0, math.pi)
    if not parity:
        out[p : 2 * p] = np.clip(out[p : 2 * p], 0.0, math.pi)
    return out


def run_dqi_fixed_angles(
    Q: np.ndarray,
    gammas: list[float],
    *,
    B: np.ndarray | None = None,
    v: np.ndarray | None = None,
    phase_c: np.ndarray | None = None,
    normalize_phase_c: bool = True,
    legacy_ising: bool = False,
    betas: list[float] | None = None,
    statistic: Statistic = "mean",
    shots: int = 512,
    seed: int = 0,
    mixer: str = "rx",
    max_qubits: int = 50,
    constant_offset: float = 0.0,
    execution: str = "local",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
) -> DqiOptimizationResult:
    """Run DQI once with fixed layer angles (no classical optimization).

    **Parity / Travelers ansatz** (``legacy_ising=False``): only ``gammas`` are used; each layer
    applies ``Rz(gamma_l * c_i)`` with ``c`` from ``Q`` (or ``phase_c``). ``betas`` are ignored.

    **Legacy Ising ansatz** (``legacy_ising=True``): pass ``betas`` with the same length as
    ``gammas`` (alternating cost and mixer layers). If ``betas`` is omitted, uses
    ``(π/2)`` per layer to match the default COBYLA initial point in ``optimize_dqi``.
    """
    q = np.asarray(Q, dtype=float)
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")
    p = len(gammas)
    if p < 1:
        raise ValueError("gammas must be non-empty")

    parity_mode = not legacy_ising
    if parity_mode:
        betas_use = [0.0] * p
    else:
        if betas is None:
            betas_use = [0.5 * math.pi] * p
        else:
            if len(betas) != p:
                raise ValueError(f"betas must have length {p} (same as gammas), got {len(betas)}")
            betas_use = [float(b) for b in betas]

    stats = sample_dqi(
        q,
        gammas=[float(g) for g in gammas],
        betas=betas_use,
        B=B,
        v=v,
        phase_c=phase_c,
        normalize_phase_c=normalize_phase_c,
        legacy_ising=legacy_ising,
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
        eval_tag="fixed",
    )
    obj = _objective(q, stats, statistic=statistic, constant_offset=constant_offset)
    return DqiOptimizationResult(
        gammas=[float(g) for g in gammas],
        betas=[0.0] * p if parity_mode else betas_use,
        objective_value=float(obj),
        statistic=statistic,
        stats_at_best=stats,
        n_evaluations=1,
        history=[float(obj)],
    )


def optimize_dqi(
    Q: np.ndarray,
    p: int,
    *,
    B: np.ndarray | None = None,
    v: np.ndarray | None = None,
    phase_c: np.ndarray | None = None,
    normalize_phase_c: bool = True,
    legacy_ising: bool = False,
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
    """Optimize DQI angles for a fixed QUBO matrix.

    By default uses the Travelers **parity / max-XORSAT** ansatz (optionally with ``B y = v mod 2``);
    only ``gammas`` are optimized. Set ``legacy_ising=True`` for the old Ising / QAOA-style ansatz
    (both ``gammas`` and ``betas``).
    """
    if p < 1:
        raise ValueError("p must be >= 1")
    q = np.asarray(Q, dtype=float)
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")

    parity_mode = not legacy_ising
    if parity_mode:
        if B is not None and v is None:
            raise ValueError("v is required when B is provided")
    n_theta = p if parity_mode else 2 * p

    history: list[float] = []
    eval_idx = 0

    def evaluate(theta: np.ndarray) -> tuple[float, DqiSampleStats]:
        nonlocal eval_idx
        vec = _clip_params(theta, p, parity=parity_mode)
        gammas = [float(v_) for v_ in vec[:p]]
        betas = [0.0] * p if parity_mode else [float(v_) for v_ in vec[p : 2 * p]]
        stats = sample_dqi(
            q,
            gammas=gammas,
            betas=betas if not parity_mode else [0.0] * p,
            B=B,
            v=v,
            phase_c=phase_c,
            normalize_phase_c=normalize_phase_c,
            legacy_ising=legacy_ising,
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

    theta0 = np.full(n_theta, 0.5 * math.pi, dtype=float)
    best_obj = float("inf")
    best_theta = theta0.copy()
    best_stats: DqiSampleStats | None = None

    def accept(theta: np.ndarray, obj: float, stats: DqiSampleStats) -> None:
        nonlocal best_obj, best_theta, best_stats
        if obj < best_obj:
            best_obj = float(obj)
            best_theta = _clip_params(theta, p, parity=parity_mode)
            best_stats = stats

    if optimizer == "random":
        rng = np.random.default_rng(int(rng_seed))
        for _ in range(int(n_samples)):
            theta = rng.uniform(0.0, math.pi, size=n_theta)
            obj, stats = evaluate(theta)
            accept(theta, obj, stats)

    elif optimizer == "cobyla":
        minimize = _scipy_minimize()

        def fun(theta: np.ndarray) -> float:
            obj, stats = evaluate(theta)
            accept(theta, obj, stats)
            return float(obj)

        bounds = [(0.0, math.pi)] * n_theta
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
            delta = rng.choice([-1.0, 1.0], size=n_theta).astype(float)
            theta_plus = _clip_params(theta + c_k * delta, p, parity=parity_mode)
            theta_minus = _clip_params(theta - c_k * delta, p, parity=parity_mode)
            y_plus, stats_plus = evaluate(theta_plus)
            accept(theta_plus, y_plus, stats_plus)
            y_minus, stats_minus = evaluate(theta_minus)
            accept(theta_minus, y_minus, stats_minus)
            g_hat = np.where(delta == 0.0, 0.0, (y_plus - y_minus) / (2.0 * c_k * delta))
            theta = _clip_params(theta - a_k * g_hat, p, parity=parity_mode)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    if best_stats is None:
        # Fallback for any unexpected empty loop configuration.
        obj, stats = evaluate(theta0)
        accept(theta0, obj, stats)
        assert best_stats is not None

    return DqiOptimizationResult(
        gammas=[float(v_) for v_ in best_theta[:p]],
        betas=[0.0] * p if parity_mode else [float(v_) for v_ in best_theta[p : 2 * p]],
        objective_value=float(best_obj),
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=int(eval_idx),
        history=history,
    )
