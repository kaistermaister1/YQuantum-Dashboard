"""Standalone pipeline API for parity-native classic DQI."""

from __future__ import annotations

import numpy as np

from src.dqi_classic_core import run_classic_dqi_histogram
from src.dqi_classic_types import ClassicDQIInputs, ClassicDQIResult


def score_s(B: np.ndarray, v: np.ndarray, x: str | list[int]) -> int:
    """Number of satisfied parity checks."""
    B_bin = (np.asarray(B, dtype=np.uint8) & 1).astype(int)
    v_bin = (np.asarray(v, dtype=np.uint8) & 1).astype(int).reshape(-1)
    if isinstance(x, str):
        if not all(c in "01" for c in x):
            raise ValueError("Bitstring must contain only '0'/'1'")
        x_bits = np.array([int(ch) for ch in x], dtype=int)
    else:
        x_bits = np.asarray(x, dtype=int).reshape(-1)
    if x_bits.shape[0] != B_bin.shape[1]:
        raise ValueError(
            f"Mismatch: expected {B_bin.shape[1]} variables, got {x_bits.shape[0]}",
        )
    lhs = (B_bin @ x_bits) % 2
    return int(np.sum(lhs == v_bin))


def score_f(B: np.ndarray, v: np.ndarray, x: str | list[int]) -> int:
    """f(B, v, x) = sum_i (-1)^(B_i*x + v_i)."""
    B_bin = (np.asarray(B, dtype=np.uint8) & 1).astype(int)
    v_bin = (np.asarray(v, dtype=np.uint8) & 1).astype(int).reshape(-1)
    if isinstance(x, str):
        if not all(c in "01" for c in x):
            raise ValueError("Bitstring must contain only '0'/'1'")
        x_bits = np.array([int(ch) for ch in x], dtype=int)
    else:
        x_bits = np.asarray(x, dtype=int).reshape(-1)
    if x_bits.shape[0] != B_bin.shape[1]:
        raise ValueError(
            f"Mismatch: expected {B_bin.shape[1]} variables, got {x_bits.shape[0]}",
        )
    lhs = (B_bin @ x_bits) % 2
    return int(np.sum(np.where(lhs == v_bin, 1, -1)))


def _expected_scores(
    B: np.ndarray,
    v: np.ndarray,
    postselected_counts: dict[str, int],
) -> tuple[float | None, float | None]:
    shots = int(sum(int(c) for c in postselected_counts.values()))
    if shots == 0:
        return None, None
    s_total = 0.0
    f_total = 0.0
    for bitstring, count in postselected_counts.items():
        cc = int(count)
        s_total += float(score_s(B, v, bitstring)) * cc
        f_total += float(score_f(B, v, bitstring)) * cc
    return s_total / shots, f_total / shots


def run_classic_dqi(
    B: np.ndarray,
    v: np.ndarray,
    *,
    ell: int = 1,
    bp_iterations: int = 1,
    shots: int = 4096,
    seed: int = 12345,
    strict_ancilla: bool = True,
) -> ClassicDQIResult:
    """Run parity-native classic DQI and return scored postselected results."""
    inputs = ClassicDQIInputs(
        B=B,
        v=v,
        ell=int(ell),
        bp_iterations=int(bp_iterations),
        shots=int(shots),
        seed=int(seed),
        strict_ancilla=bool(strict_ancilla),
    )
    postselected_counts, keep_rate, raw_counts, metadata = run_classic_dqi_histogram(
        inputs.B,
        inputs.v,
        ell=inputs.ell,
        bp_iterations=inputs.bp_iterations,
        shots=inputs.shots,
        seed=inputs.seed,
        strict_ancilla=inputs.strict_ancilla,
    )
    expected_s, expected_f = _expected_scores(inputs.B, inputs.v, postselected_counts)
    return ClassicDQIResult(
        postselected_counts=postselected_counts,
        keep_rate=float(keep_rate),
        expected_f=expected_f,
        expected_s=expected_s,
        raw_counts=raw_counts,
        postselected_shots=int(metadata["postselected_shots"]),
        total_shots=int(metadata["total_shots"]),
        ell=inputs.ell,
        bp_iterations=inputs.bp_iterations,
        shots=inputs.shots,
        seed=inputs.seed,
        strict_ancilla=inputs.strict_ancilla,
        num_qubits=int(metadata["num_qubits"]),
        gate_counts=dict(metadata["gate_counts"]),
    )

