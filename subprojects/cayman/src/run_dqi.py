"""Public DQI API and QuboBlock adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.dqi_core import bitstring_to_array, hamming_weight
from src.dqi_insurance_parity import build_insurance_parity_B_rhs
from src.dqi_optimize import DqiRunResult, run_dqi_oneshot

try:
    from insurance_model import BundlingProblem
except ImportError:
    from src.insurance_model import BundlingProblem


@dataclass
class DqiRunMetadata:
    """Extended metadata returned alongside best solution/value."""

    run_result: DqiRunResult
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
    *,
    shots: int = 512,
    seed: int = 0,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    statistic: str = "mean",
    mixer: str = "h",
    max_qubits: int = 50,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
    B: np.ndarray | None = None,
    parity_rhs: np.ndarray | None = None,
    dicke_k: int | None = None,
    n_coverage: int | None = None,
    insurance_parity: tuple[BundlingProblem, int] | None = None,
) -> tuple[np.ndarray, float]:
    """Run DQI once; default ``execution`` is ``nexus_selene`` (use ``local`` for on-machine Selene).

    Set ``insurance_parity=(problem, package_index)`` to build a GF(2) parity matrix from the same
    slack layout as :class:`qubo_block.QuboBlock`, append syndrome qubits, and post-select algebraically
    consistent shots.  Use ``dicke_k`` together with ``n_coverage`` (or rely on ``problem.N`` when
    ``insurance_parity`` is set) to prepare a Dicke state on the coverage qubits before parity encoding.
    """
    q, meta = _extract_qubo_and_meta(Q)
    B_use, rhs_use, n_cov_use = B, parity_rhs, n_coverage
    if insurance_parity is not None:
        prob, m = insurance_parity
        if B_use is None:
            B_use, rhs_use = build_insurance_parity_B_rhs(prob, m)
        if n_cov_use is None:
            n_cov_use = int(prob.N)
    res = run_dqi_oneshot(
        q,
        p=p,
        gammas=gammas,
        betas=betas,
        statistic=statistic,  # type: ignore[arg-type]
        shots=shots,
        seed=seed,
        mixer=mixer,
        max_qubits=max_qubits,
        constant_offset=float(meta["constant_offset"]),
        execution=execution,
        nexus_hugr_name=nexus_hugr_name,
        nexus_job_name=nexus_job_name,
        nexus_helios_system=nexus_helios_system,
        nexus_timeout=nexus_timeout,
        B=B_use,
        parity_rhs=rhs_use,
        dicke_k=dicke_k,
        n_coverage=n_cov_use,
    )
    best_x = bitstring_to_array(res.stats_at_best.best_bitstring)
    return best_x, float(res.stats_at_best.best_value)


def run_dqi_with_details(
    Q: Any,
    p: int,
    *,
    shots: int = 512,
    seed: int = 0,
    gammas: Sequence[float] | None = None,
    betas: Sequence[float] | None = None,
    statistic: str = "mean",
    mixer: str = "h",
    max_qubits: int = 50,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
    B: np.ndarray | None = None,
    parity_rhs: np.ndarray | None = None,
    dicke_k: int | None = None,
    n_coverage: int | None = None,
    insurance_parity: tuple[BundlingProblem, int] | None = None,
) -> tuple[np.ndarray, float, DqiRunMetadata]:
    """Run DQI once; returns metadata. Default ``execution`` is ``nexus_selene``.

    Same options as :func:`run_dqi`.
    """
    q, meta = _extract_qubo_and_meta(Q)
    B_use, rhs_use, n_cov_use = B, parity_rhs, n_coverage
    if insurance_parity is not None:
        prob, m = insurance_parity
        if B_use is None:
            B_use, rhs_use = build_insurance_parity_B_rhs(prob, m)
        if n_cov_use is None:
            n_cov_use = int(prob.N)
    res = run_dqi_oneshot(
        q,
        p=p,
        gammas=gammas,
        betas=betas,
        statistic=statistic,  # type: ignore[arg-type]
        shots=shots,
        seed=seed,
        mixer=mixer,
        max_qubits=max_qubits,
        constant_offset=float(meta["constant_offset"]),
        execution=execution,
        nexus_hugr_name=nexus_hugr_name,
        nexus_job_name=nexus_job_name,
        nexus_helios_system=nexus_helios_system,
        nexus_timeout=nexus_timeout,
        B=B_use,
        parity_rhs=rhs_use,
        dicke_k=dicke_k,
        n_coverage=n_cov_use,
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
        run_result=res,
        bitstring=bitstring,
        hamming_weight_full=hamming_weight(best_x),
        hamming_weight_coverage=hw_cov,
        n_coverage=n_cov if isinstance(n_cov, int) else None,
        n_slack=meta["n_slack"] if isinstance(meta["n_slack"], int) else None,
        coverage_bits=coverage_bits,
        constant_offset=float(meta["constant_offset"]),
    )
    return best_x, value, details
