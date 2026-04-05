"""Classical outer loop over QAOA angles.

Each candidate tuple runs Selene via :func:`run_qaoa_p1_on_block` or
:func:`run_qaoa_p2_on_block` (new Guppy kernel per evaluation).

**Methods**

* **Grid / random** — :func:`optimize_qaoa_p1_grid`, :func:`optimize_qaoa_p1_random`, etc.
* **COBYLA** — derivative-free constrained optimization via ``scipy.optimize.minimize`` (needs SciPy).
* **SPSA** — simultaneous perturbation stochastic approximation (2 function evals per iteration;
  suited to **noisy** shot-based objectives; pure NumPy).

**Objective** (minimize):

* ``statistic="mean"`` — sample mean QUBO energy
  :math:`\\sum_s (c_s / N_{\\mathrm{shots}})\\, E(x_s)` (standard QAOA classical surrogate).
* ``statistic="best"`` — lowest energy among observed bitstrings in that run
  (useful for very peaked distributions; noisier for comparing angles).

**Uncertainty (objective, not angles)**

* :func:`sample_mean_energy_uncertainty` — from ``bitstring_counts``, returns the sample mean,
  an estimated s.d. of one shot, and **SE(mean) = std / √N** for error bars on the *mean energy*
  at fixed angles (i.i.d. shot approximation).
* **Angles** — use :func:`repeat_optimize_qaoa_p1` / :func:`repeat_optimize_qaoa_p2` to re-run an
  optimizer with separated Selene / RNG seeds, then call :meth:`QaoaP1RepeatSummary.angle_percentiles`
  (or p2 equivalent) for percentile bands on angles and reported objectives.

Angles are in **radians**. Default search box is ``[0, \\pi]`` per angle (common for many QUBOs;
widen bounds if your landscape needs it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.qubo_qaoa import (
    QaoaP1SampleStats,
    QaoaP2SampleStats,
    run_qaoa_p1_on_block,
    run_qaoa_p2_on_block,
)

if TYPE_CHECKING:
    from src.qubo_block import QuboBlock

Statistic = Literal["mean", "best"]


def _linspace_bounds(lo: float, hi: float, n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([(lo + hi) * 0.5], dtype=float)
    return np.linspace(lo, hi, n, dtype=float)


def mean_sample_energy(block: "QuboBlock", stats: QaoaP1SampleStats | QaoaP2SampleStats) -> float:
    """Sample mean QUBO energy :math:`\\sum_s (c_s / N) E(x_s)` with :meth:`QuboBlock.energy`."""
    denom = sum(stats.bitstring_counts.values())
    if denom <= 0:
        return float("nan")
    total = 0.0
    for s, c in stats.bitstring_counts.items():
        x = np.array([float(int(ch)) for ch in s], dtype=float)
        total += float(c) * block.energy(x)
    return total / float(denom)


@dataclass(frozen=True)
class SampleMeanEnergyUncertainty:
    """Uncertainty for the **sample-mean** QUBO energy from one histogram (fixed angles).

    Treats each shot as i.i.d. with the empirical distribution implied by ``bitstring_counts``.
    Then :math:`\\mathrm{SE}(\\bar E) \\approx \\sigma / \\sqrt{N}` where :math:`\\sigma^2` is the
    weighted variance of per-shot energies across bitstrings.

    Use ``se_mean`` for error bars on the *objective* when ``statistic="mean"``. For a rough
    95% band (Gaussian), ``mean ± 1.96 * se_mean``.
    """

    mean: float
    std_iid: float
    se_mean: float
    n_shots: int


def sample_mean_energy_uncertainty(
    block: "QuboBlock",
    stats: QaoaP1SampleStats | QaoaP2SampleStats,
) -> SampleMeanEnergyUncertainty:
    """Mean energy plus **standard error of the mean** from ``stats.bitstring_counts``."""
    denom = int(sum(stats.bitstring_counts.values()))
    if denom <= 0:
        return SampleMeanEnergyUncertainty(
            mean=float("nan"),
            std_iid=float("nan"),
            se_mean=float("nan"),
            n_shots=0,
        )

    energies: list[float] = []
    weights: list[float] = []
    for s, c in stats.bitstring_counts.items():
        x = np.array([float(int(ch)) for ch in s], dtype=float)
        e = float(block.energy(x))
        energies.append(e)
        weights.append(float(c))

    w = np.asarray(weights, dtype=float)
    w /= float(np.sum(w))
    e_arr = np.asarray(energies, dtype=float)
    mean = float(np.dot(w, e_arr))
    var = float(np.dot(w, (e_arr - mean) ** 2))
    std_iid = math.sqrt(var) if var > 0.0 else 0.0
    se_mean = std_iid / math.sqrt(float(denom))
    return SampleMeanEnergyUncertainty(
        mean=mean,
        std_iid=std_iid,
        se_mean=se_mean,
        n_shots=denom,
    )


def _objective(
    block: "QuboBlock",
    stats: QaoaP1SampleStats | QaoaP2SampleStats,
    statistic: Statistic,
) -> float:
    if statistic == "mean":
        return float(mean_sample_energy(block, stats))
    if statistic == "best":
        return float(stats.best_qubo_energy)
    raise ValueError(f"statistic must be 'mean' or 'best', got {statistic!r}")


def _clip_vec(x: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def _scipy_minimize():
    try:
        from scipy.optimize import minimize
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "COBYLA optimization requires SciPy. Install with: pip install scipy>=1.10"
        ) from e
    return minimize


@dataclass
class QaoaP1OptimizationResult:
    """Best angles from an outer loop (p = 1)."""

    gamma: float
    beta: float
    objective_value: float
    statistic: Statistic
    stats_at_best: QaoaP1SampleStats
    n_evaluations: int


@dataclass
class QaoaP2OptimizationResult:
    """Best angles from an outer loop (p = 2)."""

    gamma1: float
    beta1: float
    gamma2: float
    beta2: float
    objective_value: float
    statistic: Statistic
    stats_at_best: QaoaP2SampleStats
    n_evaluations: int


def optimize_qaoa_p1_grid(
    block: "QuboBlock",
    *,
    n_gamma: int = 8,
    n_beta: int = 8,
    gamma_bounds: tuple[float, float] = (0.0, math.pi),
    beta_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    max_qubits: int = 24,
) -> QaoaP1OptimizationResult:
    """Exhaustive grid over ``γ × β``; return the tuple minimizing the chosen statistic."""
    gammas = _linspace_bounds(gamma_bounds[0], gamma_bounds[1], n_gamma)
    betas = _linspace_bounds(beta_bounds[0], beta_bounds[1], n_beta)

    best_obj = float("inf")
    best_gamma = float(gammas[0])
    best_beta = float(betas[0])
    best_stats: QaoaP1SampleStats | None = None
    eval_idx = 0

    for g in gammas:
        for b in betas:
            stats = run_qaoa_p1_on_block(
                block,
                float(g),
                float(b),
                shots=int(shots),
                seed=int(seed_offset + eval_idx),
                max_qubits=max_qubits,
            )
            obj = _objective(block, stats, statistic)
            eval_idx += 1
            if obj < best_obj:
                best_obj = obj
                best_gamma = float(g)
                best_beta = float(b)
                best_stats = stats

    assert best_stats is not None
    return QaoaP1OptimizationResult(
        gamma=best_gamma,
        beta=best_beta,
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


def optimize_qaoa_p1_random(
    block: "QuboBlock",
    *,
    n_samples: int = 48,
    gamma_bounds: tuple[float, float] = (0.0, math.pi),
    beta_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    rng_seed: int = 0,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    max_qubits: int = 24,
) -> QaoaP1OptimizationResult:
    """Uniform random angles; return the sample minimizing the chosen statistic."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(int(rng_seed))

    best_obj = float("inf")
    best_gamma = 0.0
    best_beta = 0.0
    best_stats: QaoaP1SampleStats | None = None

    for t in range(n_samples):
        g = float(rng.uniform(gamma_bounds[0], gamma_bounds[1]))
        b = float(rng.uniform(beta_bounds[0], beta_bounds[1]))
        stats = run_qaoa_p1_on_block(
            block,
            g,
            b,
            shots=int(shots),
            seed=int(seed_offset + t),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        if obj < best_obj:
            best_obj = obj
            best_gamma = g
            best_beta = b
            best_stats = stats

    assert best_stats is not None
    return QaoaP1OptimizationResult(
        gamma=best_gamma,
        beta=best_beta,
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=n_samples,
    )


def optimize_qaoa_p2_grid(
    block: "QuboBlock",
    *,
    n_gamma1: int = 4,
    n_beta1: int = 4,
    n_gamma2: int = 4,
    n_beta2: int = 4,
    gamma1_bounds: tuple[float, float] = (0.0, math.pi),
    beta1_bounds: tuple[float, float] = (0.0, math.pi),
    gamma2_bounds: tuple[float, float] = (0.0, math.pi),
    beta2_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    max_qubits: int = 24,
) -> QaoaP2OptimizationResult:
    """Exhaustive grid over ``γ₁ × β₁ × γ₂ × β₂``."""
    G1 = _linspace_bounds(gamma1_bounds[0], gamma1_bounds[1], n_gamma1)
    B1 = _linspace_bounds(beta1_bounds[0], beta1_bounds[1], n_beta1)
    G2 = _linspace_bounds(gamma2_bounds[0], gamma2_bounds[1], n_gamma2)
    B2 = _linspace_bounds(beta2_bounds[0], beta2_bounds[1], n_beta2)

    best_obj = float("inf")
    best = (0.0, 0.0, 0.0, 0.0)
    best_stats: QaoaP2SampleStats | None = None
    eval_idx = 0

    for g1, b1, g2, b2 in product(G1, B1, G2, B2):
        stats = run_qaoa_p2_on_block(
            block,
            float(g1),
            float(b1),
            float(g2),
            float(b2),
            shots=int(shots),
            seed=int(seed_offset + eval_idx),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        eval_idx += 1
        if obj < best_obj:
            best_obj = obj
            best = (float(g1), float(b1), float(g2), float(b2))
            best_stats = stats

    assert best_stats is not None
    return QaoaP2OptimizationResult(
        gamma1=best[0],
        beta1=best[1],
        gamma2=best[2],
        beta2=best[3],
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


def optimize_qaoa_p2_random(
    block: "QuboBlock",
    *,
    n_samples: int = 64,
    gamma1_bounds: tuple[float, float] = (0.0, math.pi),
    beta1_bounds: tuple[float, float] = (0.0, math.pi),
    gamma2_bounds: tuple[float, float] = (0.0, math.pi),
    beta2_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    rng_seed: int = 0,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    max_qubits: int = 24,
) -> QaoaP2OptimizationResult:
    """Uniform random angles in the four boxes; return the best sample."""
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(int(rng_seed))

    best_obj = float("inf")
    best = (0.0, 0.0, 0.0, 0.0)
    best_stats: QaoaP2SampleStats | None = None

    for t in range(n_samples):
        g1 = float(rng.uniform(*gamma1_bounds))
        b1 = float(rng.uniform(*beta1_bounds))
        g2 = float(rng.uniform(*gamma2_bounds))
        b2 = float(rng.uniform(*beta2_bounds))
        stats = run_qaoa_p2_on_block(
            block,
            g1,
            b1,
            g2,
            b2,
            shots=int(shots),
            seed=int(seed_offset + t),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        if obj < best_obj:
            best_obj = obj
            best = (g1, b1, g2, b2)
            best_stats = stats

    assert best_stats is not None
    return QaoaP2OptimizationResult(
        gamma1=best[0],
        beta1=best[1],
        gamma2=best[2],
        beta2=best[3],
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=n_samples,
    )


def optimize_qaoa_p1_cobyla(
    block: "QuboBlock",
    *,
    x0: tuple[float, float] | None = None,
    gamma_bounds: tuple[float, float] = (0.0, math.pi),
    beta_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    maxiter: int = 60,
    max_qubits: int = 24,
) -> QaoaP1OptimizationResult:
    """Derivative-free **COBYLA** (SciPy) on ``(γ, β)``.

    Tracks the **best** noisy objective seen (not necessarily SciPy’s final iterate).
    Angles passed to the emulator are **clipped** to the given bounds each evaluation.
    """
    minimize = _scipy_minimize()
    bounds_scipy = [gamma_bounds, beta_bounds]
    if x0 is None:
        x0_arr = np.array(
            [
                0.5 * (gamma_bounds[0] + gamma_bounds[1]),
                0.5 * (beta_bounds[0] + beta_bounds[1]),
            ],
            dtype=float,
        )
    else:
        x0_arr = _clip_vec(np.array(x0, dtype=float), bounds_scipy)

    eval_idx = 0
    best_obj = float("inf")
    best_gamma = float(x0_arr[0])
    best_beta = float(x0_arr[1])
    best_stats: QaoaP1SampleStats | None = None

    def fun(x: np.ndarray) -> float:
        nonlocal eval_idx, best_obj, best_gamma, best_beta, best_stats
        v = _clip_vec(np.asarray(x, dtype=float), bounds_scipy)
        stats = run_qaoa_p1_on_block(
            block,
            float(v[0]),
            float(v[1]),
            shots=int(shots),
            seed=int(seed_offset + eval_idx),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        eval_idx += 1
        if obj < best_obj:
            best_obj = obj
            best_gamma = float(v[0])
            best_beta = float(v[1])
            best_stats = stats
        return float(obj)

    minimize(fun, x0_arr, method="COBYLA", bounds=bounds_scipy, options={"maxiter": int(maxiter)})

    assert best_stats is not None
    return QaoaP1OptimizationResult(
        gamma=best_gamma,
        beta=best_beta,
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


def optimize_qaoa_p1_spsa(
    block: "QuboBlock",
    *,
    x0: tuple[float, float] | None = None,
    gamma_bounds: tuple[float, float] = (0.0, math.pi),
    beta_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    rng_seed: int = 0,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    maxiter: int = 40,
    a: float = 0.15,
    c: float = 0.12,
    alpha: float = 0.602,
    gamma_spsa: float = 0.101,
    stability_A: float = 10.0,
    max_qubits: int = 24,
) -> QaoaP1OptimizationResult:
    """**2SPSA** minimization of the noisy shot objective over ``(γ, β)``.

    Uses the standard gain sequences ``a_k = a / (k + 1 + A)^α``,
    ``c_k = c / (k + 1)^γ`` with Rademacher perturbations ``Δ_i ∈ {−1, +1}``.
    After each update, ``θ`` is projected into the bound box.
    """
    bounds = [gamma_bounds, beta_bounds]
    rng = np.random.default_rng(int(rng_seed))
    theta = (
        _clip_vec(
            np.array(
                [
                    0.5 * (gamma_bounds[0] + gamma_bounds[1]),
                    0.5 * (beta_bounds[0] + beta_bounds[1]),
                ],
                dtype=float,
            ),
            bounds,
        )
        if x0 is None
        else _clip_vec(np.array(x0, dtype=float), bounds)
    )

    eval_idx = 0
    best_obj = float("inf")
    best_gamma = float(theta[0])
    best_beta = float(theta[1])
    best_stats: QaoaP1SampleStats | None = None

    def eval_p1(g: float, b: float) -> tuple[float, QaoaP1SampleStats]:
        nonlocal eval_idx, best_obj, best_gamma, best_beta, best_stats
        stats = run_qaoa_p1_on_block(
            block,
            float(g),
            float(b),
            shots=int(shots),
            seed=int(seed_offset + eval_idx),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        eval_idx += 1
        if obj < best_obj:
            best_obj = obj
            best_gamma = float(g)
            best_beta = float(b)
            best_stats = stats
        return obj, stats

    for k in range(int(maxiter)):
        a_k = float(a / (k + 1 + float(stability_A)) ** alpha)
        c_k = float(c / (k + 1) ** gamma_spsa)
        delta = rng.choice([-1.0, 1.0], size=2).astype(float)
        tp = _clip_vec(theta + c_k * delta, bounds)
        tm = _clip_vec(theta - c_k * delta, bounds)
        yp, _ = eval_p1(float(tp[0]), float(tp[1]))
        ym, _ = eval_p1(float(tm[0]), float(tm[1]))
        g_hat = np.where(delta == 0.0, 0.0, (yp - ym) / (2.0 * c_k * delta))
        theta = _clip_vec(theta - a_k * g_hat, bounds)

    assert best_stats is not None
    return QaoaP1OptimizationResult(
        gamma=best_gamma,
        beta=best_beta,
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


def optimize_qaoa_p2_cobyla(
    block: "QuboBlock",
    *,
    x0: tuple[float, float, float, float] | None = None,
    gamma1_bounds: tuple[float, float] = (0.0, math.pi),
    beta1_bounds: tuple[float, float] = (0.0, math.pi),
    gamma2_bounds: tuple[float, float] = (0.0, math.pi),
    beta2_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    maxiter: int = 80,
    max_qubits: int = 24,
) -> QaoaP2OptimizationResult:
    """COBYLA on ``(γ₁, β₁, γ₂, β₂)``."""
    minimize = _scipy_minimize()
    bounds_scipy = [gamma1_bounds, beta1_bounds, gamma2_bounds, beta2_bounds]
    if x0 is None:
        x0_arr = np.array([(lo + hi) * 0.5 for lo, hi in bounds_scipy], dtype=float)
    else:
        x0_arr = _clip_vec(np.array(x0, dtype=float), bounds_scipy)

    eval_idx = 0
    best_obj = float("inf")
    best = tuple(float(x) for x in x0_arr)
    best_stats: QaoaP2SampleStats | None = None

    def fun(x: np.ndarray) -> float:
        nonlocal eval_idx, best_obj, best, best_stats
        v = _clip_vec(np.asarray(x, dtype=float), bounds_scipy)
        stats = run_qaoa_p2_on_block(
            block,
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(v[3]),
            shots=int(shots),
            seed=int(seed_offset + eval_idx),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        eval_idx += 1
        if obj < best_obj:
            best_obj = obj
            best = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            best_stats = stats
        return float(obj)

    minimize(fun, x0_arr, method="COBYLA", bounds=bounds_scipy, options={"maxiter": int(maxiter)})

    assert best_stats is not None
    return QaoaP2OptimizationResult(
        gamma1=best[0],
        beta1=best[1],
        gamma2=best[2],
        beta2=best[3],
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


def optimize_qaoa_p2_spsa(
    block: "QuboBlock",
    *,
    x0: tuple[float, float, float, float] | None = None,
    gamma1_bounds: tuple[float, float] = (0.0, math.pi),
    beta1_bounds: tuple[float, float] = (0.0, math.pi),
    gamma2_bounds: tuple[float, float] = (0.0, math.pi),
    beta2_bounds: tuple[float, float] = (0.0, math.pi),
    shots: int = 256,
    rng_seed: int = 0,
    seed_offset: int = 0,
    statistic: Statistic = "mean",
    maxiter: int = 40,
    a: float = 0.12,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma_spsa: float = 0.101,
    stability_A: float = 10.0,
    max_qubits: int = 24,
) -> QaoaP2OptimizationResult:
    """2SPSA on four QAOA angles (p = 2)."""
    bounds = [gamma1_bounds, beta1_bounds, gamma2_bounds, beta2_bounds]
    rng = np.random.default_rng(int(rng_seed))
    if x0 is None:
        theta = _clip_vec(np.array([(lo + hi) * 0.5 for lo, hi in bounds], dtype=float), bounds)
    else:
        theta = _clip_vec(np.array(x0, dtype=float), bounds)

    eval_idx = 0
    best_obj = float("inf")
    best = tuple(float(x) for x in theta)
    best_stats: QaoaP2SampleStats | None = None

    def eval_p2(v: np.ndarray) -> float:
        nonlocal eval_idx, best_obj, best, best_stats
        stats = run_qaoa_p2_on_block(
            block,
            float(v[0]),
            float(v[1]),
            float(v[2]),
            float(v[3]),
            shots=int(shots),
            seed=int(seed_offset + eval_idx),
            max_qubits=max_qubits,
        )
        obj = _objective(block, stats, statistic)
        eval_idx += 1
        if obj < best_obj:
            best_obj = obj
            best = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
            best_stats = stats
        return float(obj)

    for k in range(int(maxiter)):
        a_k = float(a / (k + 1 + float(stability_A)) ** alpha)
        c_k = float(c / (k + 1) ** gamma_spsa)
        delta = rng.choice([-1.0, 1.0], size=4).astype(float)
        tp = _clip_vec(theta + c_k * delta, bounds)
        tm = _clip_vec(theta - c_k * delta, bounds)
        yp = eval_p2(tp)
        ym = eval_p2(tm)
        g_hat = np.where(delta == 0.0, 0.0, (yp - ym) / (2.0 * c_k * delta))
        theta = _clip_vec(theta - a_k * g_hat, bounds)

    assert best_stats is not None
    return QaoaP2OptimizationResult(
        gamma1=best[0],
        beta1=best[1],
        gamma2=best[2],
        beta2=best[3],
        objective_value=best_obj,
        statistic=statistic,
        stats_at_best=best_stats,
        n_evaluations=eval_idx,
    )


OptimizerP1Method = Literal["grid", "random", "cobyla", "spsa"]
OptimizerP2Method = Literal["grid", "random", "cobyla", "spsa"]

# Selene seeds inside one run are ``seed_offset + 0, 1, …``; keep repeats disjoint.
_DEFAULT_SEED_STRIDE = 10_000_000


@dataclass
class QaoaP1RepeatSummary:
    """``n_repeats`` independent outer optimizations (p = 1); use for angle / objective bands."""

    results: list[QaoaP1OptimizationResult]
    method: OptimizerP1Method
    seed_stride: int

    @property
    def gamma(self) -> np.ndarray:
        return np.array([r.gamma for r in self.results], dtype=float)

    @property
    def beta(self) -> np.ndarray:
        return np.array([r.beta for r in self.results], dtype=float)

    @property
    def objective_value(self) -> np.ndarray:
        return np.array([r.objective_value for r in self.results], dtype=float)

    def angle_percentiles(
        self,
        qs: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
    ) -> dict[str, dict[str, float]]:
        """Percentiles over repeats for ``γ``, ``β``, and the reported objective value."""
        out: dict[str, dict[str, float]] = {}
        for name, arr in (
            ("gamma", self.gamma),
            ("beta", self.beta),
            ("objective_value", self.objective_value),
        ):
            out[name] = {_pct_key(q): float(np.percentile(arr, q)) for q in qs}
        return out


@dataclass
class QaoaP2RepeatSummary:
    """``n_repeats`` independent outer optimizations (p = 2)."""

    results: list[QaoaP2OptimizationResult]
    method: OptimizerP2Method
    seed_stride: int

    @property
    def gamma1(self) -> np.ndarray:
        return np.array([r.gamma1 for r in self.results], dtype=float)

    @property
    def beta1(self) -> np.ndarray:
        return np.array([r.beta1 for r in self.results], dtype=float)

    @property
    def gamma2(self) -> np.ndarray:
        return np.array([r.gamma2 for r in self.results], dtype=float)

    @property
    def beta2(self) -> np.ndarray:
        return np.array([r.beta2 for r in self.results], dtype=float)

    @property
    def objective_value(self) -> np.ndarray:
        return np.array([r.objective_value for r in self.results], dtype=float)

    def angle_percentiles(
        self,
        qs: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
    ) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, arr in (
            ("gamma1", self.gamma1),
            ("beta1", self.beta1),
            ("gamma2", self.gamma2),
            ("beta2", self.beta2),
            ("objective_value", self.objective_value),
        ):
            out[name] = {_pct_key(q): float(np.percentile(arr, q)) for q in qs}
        return out


def _pct_key(q: float) -> str:
    return f"p{int(q)}" if float(q).is_integer() else f"p{q}"


def repeat_optimize_qaoa_p1(
    block: "QuboBlock",
    *,
    method: OptimizerP1Method,
    n_repeats: int,
    base_seed_offset: int = 0,
    base_rng_seed: int = 0,
    seed_stride: int = _DEFAULT_SEED_STRIDE,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> QaoaP1RepeatSummary:
    """Run the chosen p = 1 optimizer ``n_repeats`` times with disjoint emulator seeds.

    ``seed_offset`` for repeat ``i`` is ``base_seed_offset + i * seed_stride`` (COBYLA / grid /
    random / SPSA all use this for Selene). Methods that also take ``rng_seed`` (random, SPSA) use
    ``base_rng_seed + i * 9973``.

    Pass options (``shots``, ``n_gamma``, ``maxiter``, …) via ``optimizer_kwargs``. Do not put
    ``seed_offset`` or ``rng_seed`` there; they are set here.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    kw = dict(optimizer_kwargs or {})
    kw.pop("seed_offset", None)
    kw.pop("rng_seed", None)

    dispatch = {
        "grid": optimize_qaoa_p1_grid,
        "random": optimize_qaoa_p1_random,
        "cobyla": optimize_qaoa_p1_cobyla,
        "spsa": optimize_qaoa_p1_spsa,
    }[method]

    results: list[QaoaP1OptimizationResult] = []
    for i in range(n_repeats):
        so = int(base_seed_offset + i * int(seed_stride))
        rs = int(base_rng_seed + i * 9973)
        if method in ("grid", "cobyla"):
            results.append(dispatch(block, seed_offset=so, **kw))
        else:
            results.append(dispatch(block, seed_offset=so, rng_seed=rs, **kw))

    return QaoaP1RepeatSummary(results=results, method=method, seed_stride=int(seed_stride))


def repeat_optimize_qaoa_p2(
    block: "QuboBlock",
    *,
    method: OptimizerP2Method,
    n_repeats: int,
    base_seed_offset: int = 0,
    base_rng_seed: int = 0,
    seed_stride: int = _DEFAULT_SEED_STRIDE,
    optimizer_kwargs: dict[str, Any] | None = None,
) -> QaoaP2RepeatSummary:
    """Same as :func:`repeat_optimize_qaoa_p1` for depth-2 optimizers."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    kw = dict(optimizer_kwargs or {})
    kw.pop("seed_offset", None)
    kw.pop("rng_seed", None)

    dispatch = {
        "grid": optimize_qaoa_p2_grid,
        "random": optimize_qaoa_p2_random,
        "cobyla": optimize_qaoa_p2_cobyla,
        "spsa": optimize_qaoa_p2_spsa,
    }[method]

    results: list[QaoaP2OptimizationResult] = []
    for i in range(n_repeats):
        so = int(base_seed_offset + i * int(seed_stride))
        rs = int(base_rng_seed + i * 9973)
        if method in ("grid", "cobyla"):
            results.append(dispatch(block, seed_offset=so, **kw))
        else:
            results.append(dispatch(block, seed_offset=so, rng_seed=rs, **kw))

    return QaoaP2RepeatSummary(results=results, method=method, seed_stride=int(seed_stride))
