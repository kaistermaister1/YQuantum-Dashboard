"""Public DQI API and QuboBlock adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.dqi_core import bitstring_to_array, hamming_weight
from src.dqi_optimize import DqiOptimizationResult, optimize_dqi, run_dqi_fixed_angles


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
    B: np.ndarray | None = None,
    v: np.ndarray | None = None,
    phase_c: np.ndarray | None = None,
    normalize_phase_c: bool = True,
    legacy_ising: bool = False,
    variational: bool = True,
    fixed_gammas: list[float] | None = None,
    fixed_betas: list[float] | None = None,
    shots: int = 512,
    seed: int = 0,
    rng_seed: int = 0,
    maxiter: int = 60,
    n_samples: int = 64,
    statistic: str = "mean",
    mixer: str = "rx",
    max_qubits: int = 50,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
) -> tuple[np.ndarray, float]:
    """Run DQI and return `(best_solution, value)` as requested.

    Optional ``B`` and ``v`` encode XOR constraints ``B y = v (mod 2)`` (Travelers DQI pipeline).
    Omit both for unconstrained parity phase encoding only (constraints in ``Q`` are **not**
    inferred from the QUBO automatically; supply ``B`` and ``v`` from an ILP→max-XORSAT
    conversion when you need them in the circuit). Set ``legacy_ising=True`` for the previous
    Ising variational circuit.

    Set ``variational=False`` to run a single circuit evaluation with fixed angles: by default
    ``gammas = [1.0] * p`` (parity: one ``Rz(gamma * c_i)`` block per layer). Pass ``fixed_gammas``
    to override (length must equal ``p``). For ``legacy_ising=True``, pass ``fixed_betas`` of
    length ``p`` or omit to use ``(π/2)`` per layer.
    """
    q, meta = _extract_qubo_and_meta(Q)
    if variational:
        res = optimize_dqi(
            q,
            p=p,
            optimizer=optimizer,  # type: ignore[arg-type]
            statistic=statistic,  # type: ignore[arg-type]
            B=B,
            v=v,
            phase_c=phase_c,
            normalize_phase_c=normalize_phase_c,
            legacy_ising=legacy_ising,
            shots=shots,
            seed=seed,
            rng_seed=rng_seed,
            maxiter=maxiter,
            n_samples=n_samples,
            mixer=mixer,
            max_qubits=max_qubits,
            constant_offset=float(meta["constant_offset"]),
            execution=execution,
            nexus_hugr_name=nexus_hugr_name,
            nexus_job_name=nexus_job_name,
            nexus_helios_system=nexus_helios_system,
            nexus_timeout=nexus_timeout,
        )
    else:
        if fixed_gammas is None:
            gammas = [1.0] * int(p)
        else:
            if len(fixed_gammas) != int(p):
                raise ValueError(f"fixed_gammas must have length p={p}, got {len(fixed_gammas)}")
            gammas = [float(x) for x in fixed_gammas]
        if legacy_ising and fixed_betas is not None and len(fixed_betas) != int(p):
            raise ValueError(f"fixed_betas must have length p={p}, got {len(fixed_betas)}")
        betas_arg = None if fixed_betas is None else [float(x) for x in fixed_betas]
        res = run_dqi_fixed_angles(
            q,
            gammas,
            B=B,
            v=v,
            phase_c=phase_c,
            normalize_phase_c=normalize_phase_c,
            legacy_ising=legacy_ising,
            betas=betas_arg,
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
        )
    best_x = bitstring_to_array(res.stats_at_best.best_bitstring)
    return best_x, float(res.stats_at_best.best_value)


def run_dqi_with_details(
    Q: Any,
    p: int,
    optimizer: str,
    *,
    B: np.ndarray | None = None,
    v: np.ndarray | None = None,
    phase_c: np.ndarray | None = None,
    normalize_phase_c: bool = True,
    legacy_ising: bool = False,
    variational: bool = True,
    fixed_gammas: list[float] | None = None,
    fixed_betas: list[float] | None = None,
    shots: int = 512,
    seed: int = 0,
    rng_seed: int = 0,
    maxiter: int = 60,
    n_samples: int = 64,
    statistic: str = "mean",
    mixer: str = "rx",
    max_qubits: int = 50,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
) -> tuple[np.ndarray, float, DqiRunMetadata]:
    """Run DQI and return `(best_solution, value, metadata)`."""
    q, meta = _extract_qubo_and_meta(Q)
    if variational:
        res = optimize_dqi(
            q,
            p=p,
            optimizer=optimizer,  # type: ignore[arg-type]
            statistic=statistic,  # type: ignore[arg-type]
            B=B,
            v=v,
            phase_c=phase_c,
            normalize_phase_c=normalize_phase_c,
            legacy_ising=legacy_ising,
            shots=shots,
            seed=seed,
            rng_seed=rng_seed,
            maxiter=maxiter,
            n_samples=n_samples,
            mixer=mixer,
            max_qubits=max_qubits,
            constant_offset=float(meta["constant_offset"]),
            execution=execution,
            nexus_hugr_name=nexus_hugr_name,
            nexus_job_name=nexus_job_name,
            nexus_helios_system=nexus_helios_system,
            nexus_timeout=nexus_timeout,
        )
    else:
        if fixed_gammas is None:
            gammas = [1.0] * int(p)
        else:
            if len(fixed_gammas) != int(p):
                raise ValueError(f"fixed_gammas must have length p={p}, got {len(fixed_gammas)}")
            gammas = [float(x) for x in fixed_gammas]
        if legacy_ising and fixed_betas is not None and len(fixed_betas) != int(p):
            raise ValueError(f"fixed_betas must have length p={p}, got {len(fixed_betas)}")
        betas_arg = None if fixed_betas is None else [float(x) for x in fixed_betas]
        res = run_dqi_fixed_angles(
            q,
            gammas,
            B=B,
            v=v,
            phase_c=phase_c,
            normalize_phase_c=normalize_phase_c,
            legacy_ising=legacy_ising,
            betas=betas_arg,
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
