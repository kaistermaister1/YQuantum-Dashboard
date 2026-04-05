"""Core DQI utilities: QUBO math, Guppy circuit generation, and sampling."""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


def qubo_energy(x: np.ndarray, Q: np.ndarray, constant_offset: float = 0.0) -> float:
    """Compute f(x) = x^T Q x + constant_offset for binary x."""
    x_vec = np.asarray(x, dtype=float).ravel()
    q_mat = np.asarray(Q, dtype=float)
    if q_mat.shape[0] != q_mat.shape[1]:
        raise ValueError("Q must be square")
    if x_vec.shape[0] != q_mat.shape[0]:
        raise ValueError("x length must match Q dimension")
    return float(x_vec @ q_mat @ x_vec) + float(constant_offset)


def hamming_weight(x: np.ndarray) -> int:
    """Return Hamming weight of a binary vector."""
    x_vec = np.asarray(x, dtype=float).ravel()
    return int(np.sum(x_vec > 0.5))


def build_hamming_weight_penalty_qubo(n: int, target_weight: int, penalty: float) -> np.ndarray:
    """Build Q for penalty * (sum_i x_i - target_weight)^2."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    k = float(target_weight)
    lam = float(penalty)
    Q = np.zeros((n, n), dtype=float)

    # For binary x, sum_i x_i^2 = sum_i x_i.
    for i in range(n):
        Q[i, i] += lam * (1.0 - 2.0 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += lam
            Q[j, i] += lam
    return Q


def qubo_to_ising(Q: np.ndarray) -> tuple[float, np.ndarray, list[tuple[int, int, float]]]:
    """Map symmetric QUBO to Ising coefficients with x_i=(1-Z_i)/2.

    Returns ``(c_I, c_z, zz)`` such that:
        H = c_I I + sum_i c_z[i] Z_i + sum_{i<j} w_ij Z_i Z_j
    and ``<x|H|x> = x^T Q x`` for computational-basis bitstring x.
    """
    q = np.asarray(Q, dtype=float)
    q = (q + q.T) * 0.5
    n = q.shape[0]
    c_i = 0.0
    c_z = np.zeros(n, dtype=float)
    zz: list[tuple[int, int, float]] = []

    for i in range(n):
        c_i += q[i, i] / 2.0
        c_z[i] += -q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            qij = q[i, j]
            if abs(qij) < 1e-15:
                continue
            c_i += qij / 2.0
            c_z[i] += -qij / 2.0
            c_z[j] += -qij / 2.0
            zz.append((i, j, qij / 2.0))
    return c_i, c_z, zz


def ising_energy_for_bitstring(
    x: np.ndarray, c_i: float, c_z: np.ndarray, zz: list[tuple[int, int, float]]
) -> float:
    """Evaluate Ising energy at binary bitstring x."""
    x_vec = np.asarray(x, dtype=float).ravel()
    z = np.where(x_vec < 0.5, 1.0, -1.0)
    val = float(c_i + np.dot(c_z, z))
    for i, j, w in zz:
        val += float(w * z[i] * z[j])
    return float(val)


@dataclass
class DqiSampleStats:
    """Sample statistics for one DQI run."""

    n_qubits: int
    p: int
    shots: int
    gammas: list[float]
    betas: list[float]
    bitstring_counts: dict[str, int]
    best_bitstring: str
    best_value: float
    constant_offset: float


def _angle_literal(phi_radians: float) -> str:
    """Convert radians to Guppy angle literal (half-turn convention)."""
    return repr(2.0 * float(phi_radians) / math.pi)


def _append_cost_layer(
    lines: list[str], n: int, c_z: np.ndarray, zz: list[tuple[int, int, float]], gamma: float
) -> None:
    # RZZ via CX-RZ-CX decomposition.
    for i, j, w in zz:
        if abs(w) < 1e-15:
            continue
        lit = _angle_literal(gamma * float(w))
        lines.append(f"    cx(q{i}, q{j})")
        lines.append(f"    rz(q{j}, angle({lit}))")
        lines.append(f"    cx(q{i}, q{j})")
    for k in range(n):
        ck = float(c_z[k])
        if abs(ck) < 1e-15:
            continue
        lit = _angle_literal(gamma * ck)
        lines.append(f"    rz(q{k}, angle({lit}))")


def _append_mixer_layer(lines: list[str], n: int, beta: float, mixer: str) -> None:
    if mixer == "rx":
        lit = _angle_literal(beta)
        for i in range(n):
            lines.append(f"    rx(q{i}, angle({lit}))")
    elif mixer == "h":
        for i in range(n):
            lines.append(f"    h(q{i})")
    else:
        raise ValueError("mixer must be 'rx' or 'h'")


def _build_dqi_source(
    n: int,
    c_z: np.ndarray,
    zz: list[tuple[int, int, float]],
    gammas: list[float],
    betas: list[float],
    mixer: str,
) -> str:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    lines: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rx, rz",
        "",
        "@guppy",
        "def dqi_kernel() -> None:",
    ]
    for i in range(n):
        lines.append(f"    q{i} = qubit()")
    for i in range(n):
        lines.append(f"    h(q{i})")

    for g, b in zip(gammas, betas):
        _append_cost_layer(lines, n, c_z, zz, gamma=float(g))
        _append_mixer_layer(lines, n, beta=float(b), mixer=mixer)

    for i in range(n):
        lines.append(f'    result("m{i}", measure(q{i}))')
    return "\n".join(lines) + "\n"


def sample_dqi(
    Q: np.ndarray,
    gammas: list[float],
    betas: list[float],
    *,
    shots: int = 512,
    seed: int = 0,
    mixer: str = "rx",
    max_qubits: int = 50,
    constant_offset: float = 0.0,
) -> DqiSampleStats:
    """Run a DQI circuit with fixed parameters and return sampled statistics."""
    try:
        import guppylang  # noqa: F401
    except ImportError as exc:
        raise ImportError("Guppy is required: pip install guppylang") from exc

    q = np.asarray(Q, dtype=float)
    q = (q + q.T) * 0.5
    n = q.shape[0]
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")
    if n > int(max_qubits):
        raise ValueError(f"n_qubits={n} exceeds max_qubits={max_qubits}")
    if len(gammas) < 1 or len(betas) < 1:
        raise ValueError("At least one layer is required")

    _, c_z, zz = qubo_to_ising(q)
    src = _build_dqi_source(n, c_z, zz, list(gammas), list(betas), mixer=mixer)

    fd, path = tempfile.mkstemp(suffix="_dqi_guppy.py", text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(src)
        spec = importlib.util.spec_from_file_location("dqi_dyn", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load generated DQI module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = mod.dqi_kernel
        kernel.check()

        emu = kernel.emulator(n_qubits=n).with_shots(int(shots)).with_seed(int(seed))
        res = emu.run()

        def shot_to_str(shot: Any) -> str:
            d = shot.as_dict()
            return "".join(str(int(d[f"m{i}"])) for i in range(n))

        counts = Counter(shot_to_str(s) for s in res.results)
        bitstring_counts = dict(counts.most_common())

        best_s: str | None = None
        best_val = float("inf")
        for s in counts:
            x = np.array([float(int(ch)) for ch in s], dtype=float)
            e = qubo_energy(x, q, constant_offset=constant_offset)
            if e < best_val:
                best_val = e
                best_s = s
        assert best_s is not None

        return DqiSampleStats(
            n_qubits=n,
            p=len(gammas),
            shots=int(shots),
            gammas=[float(v) for v in gammas],
            betas=[float(v) for v in betas],
            bitstring_counts=bitstring_counts,
            best_bitstring=best_s,
            best_value=float(best_val),
            constant_offset=float(constant_offset),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def bitstring_to_array(bitstring: str) -> np.ndarray:
    """Convert a bitstring like '101' to float array [1., 0., 1.]."""
    return np.array([float(int(ch)) for ch in bitstring], dtype=float)
