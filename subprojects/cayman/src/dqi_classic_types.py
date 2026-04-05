"""Dataclasses for standalone classic DQI runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ClassicDQIInputs:
    """Inputs for a parity-native classic DQI run."""

    B: np.ndarray
    v: np.ndarray
    ell: int = 1
    bp_iterations: int = 1
    shots: int = 4096
    seed: int = 12345
    strict_ancilla: bool = True

    def __post_init__(self) -> None:
        self.B = (np.asarray(self.B, dtype=np.uint8) & 1).astype(np.uint8)
        self.v = (np.asarray(self.v, dtype=np.uint8) & 1).astype(np.uint8).reshape(-1)
        if self.B.ndim != 2:
            raise ValueError("B must be a 2D matrix")
        if self.v.ndim != 1:
            raise ValueError("v must be a 1D vector")
        if self.B.shape[0] != self.v.shape[0]:
            raise ValueError(
                f"B rows ({self.B.shape[0]}) must match v length ({self.v.shape[0]})",
            )
        if int(self.ell) < 1:
            raise ValueError("ell must be >= 1")
        if int(self.bp_iterations) < 0:
            raise ValueError("bp_iterations must be >= 0")
        if int(self.shots) < 1:
            raise ValueError("shots must be >= 1")


@dataclass(slots=True)
class ClassicDQIResult:
    """Result bundle for standalone classic DQI."""

    postselected_counts: dict[str, int]
    keep_rate: float
    expected_f: float | None
    expected_s: float | None
    raw_counts: dict[str, int]
    postselected_shots: int
    total_shots: int
    ell: int
    bp_iterations: int
    shots: int
    seed: int
    strict_ancilla: bool
    num_qubits: int
    gate_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "postselected_counts": {str(k): int(v) for k, v in self.postselected_counts.items()},
            "keep_rate": float(self.keep_rate),
            "expected_f": None if self.expected_f is None else float(self.expected_f),
            "expected_s": None if self.expected_s is None else float(self.expected_s),
            "raw_counts": {str(k): int(v) for k, v in self.raw_counts.items()},
            "postselected_shots": int(self.postselected_shots),
            "total_shots": int(self.total_shots),
            "ell": int(self.ell),
            "bp_iterations": int(self.bp_iterations),
            "shots": int(self.shots),
            "seed": int(self.seed),
            "strict_ancilla": bool(self.strict_ancilla),
            "num_qubits": int(self.num_qubits),
            "gate_counts": {str(k): int(v) for k, v in self.gate_counts.items()},
        }

