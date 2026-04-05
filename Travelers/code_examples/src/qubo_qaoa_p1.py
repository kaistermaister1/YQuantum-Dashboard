"""QAOA (p = 1 and p = 2) on one :class:`QuboBlock` using Guppy + Selene (Quantinuum stack).

Maps binary QUBO  ``E(x) = xᵀ Q x + const``  to a **commuting Z Hamiltonian** via
``x_i ↦ (I - Z_i)/2``, then runs alternating cost ``exp(-i γ H_C)`` and mixer
``exp(-i β Σ_i X_i)`` layers on ``|+⟩^{⊗ n}`` (standard QAOA order).

Angle literals for ``rz`` / ``rx`` follow the same **π-scaled** convention as the
hackathon template: use ``2 * phase_radians / π`` inside ``angle(...)`` so that a
full ``exp(-i φ Z)`` (single-qubit) is obtained when the underlying gate matches
the usual ``Rz(2φ) = exp(-i φ Z)`` style (see notebook ``will/qaoa_guppy_template.ipynb``).

The Selene entrypoint is **zero-argument**; this module **code-generates** a fresh
``@guppy`` kernel for each angle tuple (acceptable for small outer optimization loops).

* :func:`run_qaoa_p1_on_block` — one cost + one mixer.
* :func:`run_qaoa_p2_on_block` — two cost + two mixer layers ``(γ₁,β₁,γ₂,β₂)``.
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.qubo_block import QuboBlock


def qubo_to_ising_pauli_coefficients(Q: np.ndarray) -> tuple[float, np.ndarray, list[tuple[int, int, float]]]:
    """Symmetric QUBO → diagonal Ising form in the Z basis (no constant from QUBO offset).

    Returns ``(c_I, c_Z, zz)`` where::

        H = c_I I + Σ_k c_Z[k] Z_k + Σ_{i<j} w_ij Z_i Z_j

    and for computational basis ``|x⟩`` (x_k ∈ {0,1}), ``⟨x|H|x⟩ = xᵀ Q x``.

    ``Q`` is symmetrized; diagonal and off-diagonal are handled as in ``qubo_block``.
    """
    Q = np.asarray(Q, dtype=float)
    Q = (Q + Q.T) * 0.5
    n = Q.shape[0]
    c_I = 0.0
    c_z = np.zeros(n, dtype=float)
    zz: list[tuple[int, int, float]] = []

    for i in range(n):
        c_I += Q[i, i] / 2.0
        c_z[i] += -Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            if qij == 0.0:
                continue
            c_I += qij / 2.0
            c_z[i] += -qij / 2.0
            c_z[j] += -qij / 2.0
            zz.append((i, j, qij / 2.0))

    return c_I, c_z, zz


def ising_energy_for_bitstring(
    x: np.ndarray,
    c_I: float,
    c_z: np.ndarray,
    zz: list[tuple[int, int, float]],
) -> float:
    """⟨x|H|x⟩ with Z eigenvalues +1 for bit 0 and −1 for bit 1."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    ez = np.where(x < 0.5, 1.0, -1.0)
    e = c_I + float(np.dot(c_z, ez))
    for i, j, w in zz:
        e += w * ez[i] * ez[j]
    return float(e)


def bruteforce_minimize_qubo(
    Q: np.ndarray,
    *,
    constant_offset: float = 0.0,
    max_n: int = 16,
) -> tuple[float, np.ndarray]:
    """Exact minimum of ``xᵀ Q x + constant_offset`` over ``x ∈ {0,1}ⁿ`` (small ``n`` only).

    Symmetrizes ``Q``. Intended for **verification** on toy blocks (``n`` up to ``max_n``).
    Returns ``(min_energy, x_best)`` with ``x_best`` a length-``n`` float vector of 0/1.
    """
    Q = np.asarray(Q, dtype=float)
    Q = (Q + Q.T) * 0.5
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(f"n={n} exceeds max_n={max_n} for brute force")
    best_e = float("inf")
    best_x = np.zeros(n, dtype=float)
    for k in range(1 << n):
        x = np.array([float((k >> i) & 1) for i in range(n)], dtype=float)
        e = float(x @ Q @ x) + float(constant_offset)
        if e < best_e - 1e-15:
            best_e = e
            best_x = x.copy()
    return best_e, best_x


def verify_ising_matches_qubo(Q: np.ndarray, *, rng: np.random.Generator | None = None) -> None:
    """Assert Ising energy equals ``xᵀ Q x`` for random binary ``x``."""
    rng = rng or np.random.default_rng(0)
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    c_I, c_z, zz = qubo_to_ising_pauli_coefficients(Q)
    for _ in range(20):
        x = rng.integers(0, 2, size=n).astype(float)
        e1 = float(x @ Q @ x)
        e2 = ising_energy_for_bitstring(x, c_I, c_z, zz)
        np.testing.assert_allclose(e1, e2, rtol=0, atol=1e-9)


def _angle_literal_from_radians(phi: float) -> str:
    """Literal inside ``angle(...)`` for ``rz``/``rx`` (see module docstring)."""
    v = 2.0 * float(phi) / math.pi
    return repr(v)


def _guppy_append_cost_layer(
    lines: list[str],
    n: int,
    c_z: np.ndarray,
    zz: list[tuple[int, int, float]],
    gamma: float,
) -> None:
    """Append ``exp(-i γ H_C)`` as commuting CX–Rz–CX (ZZ) and Rz (Z) terms."""
    for i, j, w in zz:
        if abs(w) < 1e-15:
            continue
        phi = gamma * w
        lit = _angle_literal_from_radians(phi)
        lines.append(f"    cx(q{i}, q{j})")
        lines.append(f"    rz(q{j}, angle({lit}))")
        lines.append(f"    cx(q{i}, q{j})")

    for k in range(n):
        if abs(c_z[k]) < 1e-15:
            continue
        phi = gamma * float(c_z[k])
        lit = _angle_literal_from_radians(phi)
        lines.append(f"    rz(q{k}, angle({lit}))")


def _guppy_append_mixer_layer(lines: list[str], n: int, beta_var: str) -> None:
    """Append ``exp(-i β Σ_i X_i)`` as ``Rx`` on each line (``beta_var`` is a Guppy angle id)."""
    for i in range(n):
        lines.append(f"    rx(q{i}, {beta_var})")


def _build_guppy_p1_source(
    n: int,
    c_z: np.ndarray,
    zz: list[tuple[int, int, float]],
    gamma: float,
    beta: float,
) -> str:
    """Generate ``@guppy`` kernel source (p=1). ``gamma``, ``beta`` in radians."""
    b_lit = _angle_literal_from_radians(beta)

    lines: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rx, rz",
        "",
        "@guppy",
        "def qaoa_p1_kernel() -> None:",
        f"    g_beta = angle({b_lit})",
    ]
    for i in range(n):
        lines.append(f"    q{i} = qubit()")

    for i in range(n):
        lines.append(f"    h(q{i})")

    _guppy_append_cost_layer(lines, n, c_z, zz, gamma)
    _guppy_append_mixer_layer(lines, n, "g_beta")

    for i in range(n):
        lines.append(f'    result("m{i}", measure(q{i}))')

    return "\n".join(lines) + "\n"


def _build_guppy_p2_source(
    n: int,
    c_z: np.ndarray,
    zz: list[tuple[int, int, float]],
    gamma1: float,
    beta1: float,
    gamma2: float,
    beta2: float,
) -> str:
    """Generate ``@guppy`` kernel source (p=2). All angles in **radians**."""
    b1_lit = _angle_literal_from_radians(beta1)
    b2_lit = _angle_literal_from_radians(beta2)

    lines: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rx, rz",
        "",
        "@guppy",
        "def qaoa_p2_kernel() -> None:",
        f"    g_beta1 = angle({b1_lit})",
        f"    g_beta2 = angle({b2_lit})",
    ]
    for i in range(n):
        lines.append(f"    q{i} = qubit()")

    for i in range(n):
        lines.append(f"    h(q{i})")

    _guppy_append_cost_layer(lines, n, c_z, zz, gamma1)
    _guppy_append_mixer_layer(lines, n, "g_beta1")
    _guppy_append_cost_layer(lines, n, c_z, zz, gamma2)
    _guppy_append_mixer_layer(lines, n, "g_beta2")

    for i in range(n):
        lines.append(f'    result("m{i}", measure(q{i}))')

    return "\n".join(lines) + "\n"


@dataclass
class QaoaP1SampleStats:
    """Summary of one p=1 run."""

    n_qubits: int
    shots: int
    bitstring_counts: dict[str, int]
    best_bitstring: str
    best_qubo_energy: float
    constant_offset: float


@dataclass
class QaoaP2SampleStats:
    """Summary of one p=2 run."""

    n_qubits: int
    shots: int
    bitstring_counts: dict[str, int]
    best_bitstring: str
    best_qubo_energy: float
    constant_offset: float


def run_qaoa_p1_on_block(
    block: "QuboBlock",
    gamma: float,
    beta: float,
    *,
    shots: int = 512,
    seed: int = 0,
    max_qubits: int = 24,
) -> QaoaP1SampleStats:
    """Run p=1 QAOA (Selene) for one :class:`QuboBlock`; return sample stats.

    ``gamma`` and ``beta`` are in **radians** (standard QAOA ``U_C(γ)``, ``U_B(β)``).

    **Note:** Ignores global phase from ``c_I I`` in the cost (does not affect samples).
    Reported ``best_qubo_energy`` uses :meth:`QuboBlock.energy` including ``constant_offset``.
    """
    from src.qubo_block import QuboBlock as _QB

    if not isinstance(block, _QB):
        raise TypeError("block must be a QuboBlock")

    Q = np.asarray(block.Q, dtype=float)
    n = Q.shape[0]
    if n > max_qubits:
        raise ValueError(f"n_qubits={n} exceeds max_qubits={max_qubits}")

    _c_I, c_z, zz = qubo_to_ising_pauli_coefficients(Q)

    src = _build_guppy_p1_source(n, c_z, zz, gamma, beta)
    # Guppy re-reads source at compile time — keep the temp file until ``run()`` finishes.
    fd, path = tempfile.mkstemp(suffix="_qaoa_p1_guppy.py", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("qaoa_p1_dyn", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load generated Guppy module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = mod.qaoa_p1_kernel
        kernel.check()

        bitstring_counts, best_s, best_e = _run_qaoa_sample_stats(
            block, kernel=kernel, n=n, shots=shots, seed=seed
        )
        return QaoaP1SampleStats(
            n_qubits=n,
            shots=int(shots),
            bitstring_counts=bitstring_counts,
            best_bitstring=best_s,
            best_qubo_energy=best_e,
            constant_offset=float(block.constant_offset),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_qaoa_sample_stats(
    block: "QuboBlock",
    *,
    kernel,
    n: int,
    shots: int,
    seed: int,
) -> tuple[dict[str, int], str, float]:
    """Shared Selene run: counts, best bitstring, best QUBO energy (with block offset)."""
    emu = kernel.emulator(n_qubits=n).with_shots(int(shots)).with_seed(int(seed))
    res = emu.run()

    def shot_to_str(shot) -> str:
        d = shot.as_dict()
        return "".join(str(int(d[f"m{i}"])) for i in range(n))

    counts = Counter(shot_to_str(s) for s in res.results)
    bitstring_counts = dict(counts.most_common())

    best_s: str | None = None
    best_e = float("inf")
    for s, _ in counts.items():
        x = np.array([float(int(c)) for c in s], dtype=float)
        e = block.energy(x)
        if e < best_e:
            best_e = e
            best_s = s

    assert best_s is not None
    return bitstring_counts, best_s, best_e


def run_qaoa_p2_on_block(
    block: "QuboBlock",
    gamma1: float,
    beta1: float,
    gamma2: float,
    beta2: float,
    *,
    shots: int = 512,
    seed: int = 0,
    max_qubits: int = 24,
) -> QaoaP2SampleStats:
    """Run p=2 QAOA (Selene) after ``H^{⊗n}``: cost ``γ₁``, mixer ``β₁``, cost ``γ₂``, mixer ``β₂``.

    Matches the usual QAOA depth-2 order (innermost near ``|+⟩^{⊗n}`` is ``exp(-iγ₁ H_C)``).
    All angles are in **radians**. Same QUBO → Ising map and energy reporting as p=1.
    """
    from src.qubo_block import QuboBlock as _QB

    if not isinstance(block, _QB):
        raise TypeError("block must be a QuboBlock")

    Q = np.asarray(block.Q, dtype=float)
    n = Q.shape[0]
    if n > max_qubits:
        raise ValueError(f"n_qubits={n} exceeds max_qubits={max_qubits}")

    _c_I, c_z, zz = qubo_to_ising_pauli_coefficients(Q)

    src = _build_guppy_p2_source(n, c_z, zz, gamma1, beta1, gamma2, beta2)
    fd, path = tempfile.mkstemp(suffix="_qaoa_p2_guppy.py", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("qaoa_p2_dyn", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to load generated Guppy module")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = mod.qaoa_p2_kernel
        kernel.check()

        bitstring_counts, best_s, best_e = _run_qaoa_sample_stats(
            block, kernel=kernel, n=n, shots=shots, seed=seed
        )
        return QaoaP2SampleStats(
            n_qubits=n,
            shots=int(shots),
            bitstring_counts=bitstring_counts,
            best_bitstring=best_s,
            best_qubo_energy=best_e,
            constant_offset=float(block.constant_offset),
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
