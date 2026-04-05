"""Dicke-state preparation (Bärtschi & Eidenbenz, arXiv:1904.07358) as Guppy lines.

Ported from ``qrisp.alg_primitives.dicke_state_prep`` (Eclipse Qrisp).  Requires the
initial computational subspace state with ``k`` ones on the **rightmost** ``k`` qubits
of the block (i.e. indices ``n-k .. n-1`` are set to ``|1\\rangle`` before the circuit).
"""

from __future__ import annotations

import math
from collections.abc import Callable


def _emit_cry(
    lines: list[str],
    ctrl: int,
    tgt: int,
    theta: float,
    *,
    angle_literal: Callable[[float], str],
) -> None:
    """CRY(θ): apply ``Ry(θ)`` on ``tgt`` when ``ctrl`` is ``|1⟩`` (same as Qiskit)."""
    lines.append(f"    ry(q{tgt}, angle({angle_literal(theta / 2.0)}))")
    lines.append(f"    cx(q{ctrl}, q{tgt})")
    lines.append(f"    ry(q{tgt}, angle({angle_literal(-theta / 2.0)}))")
    lines.append(f"    cx(q{ctrl}, q{tgt})")


def _emit_ccry(
    lines: list[str],
    c1: int,
    c2: int,
    tgt: int,
    theta: float,
    *,
    angle_literal: Callable[[float], str],
) -> None:
    """Two-control Ry on ``tgt`` (active when both controls are ``|1⟩``).

    Same six-``cx``/``ry`` block used in several textbooks; matches Qrisp's two-control
    ``ry`` in ``split_cycle_shift`` up to global phase on the active subspace.
    """
    lines.append(f"    ry(q{tgt}, angle({angle_literal(theta / 2.0)}))")
    lines.append(f"    cx(q{c2}, q{tgt})")
    lines.append(f"    ry(q{tgt}, angle({angle_literal(-theta / 2.0)}))")
    lines.append(f"    cx(q{c1}, q{tgt})")
    lines.append(f"    ry(q{tgt}, angle({angle_literal(theta / 2.0)}))")
    lines.append(f"    cx(q{c2}, q{tgt})")
    lines.append(f"    cx(q{c1}, q{tgt})")


def _split_cycle_shift(
    q_map: list[int],
    high_index: int,
    low_index: int,
    lines: list[str],
    *,
    angle_literal: Callable[[float], str],
) -> None:
    """Qrisp ``split_cycle_shift`` with 0-based ``q_map[j]`` → global qubit index."""
    index_range = [high_index - i for i in range(low_index)]
    for index in index_range:
        param = 2.0 * math.acos(math.sqrt((high_index - index + 1) / (high_index)))
        if index == high_index:
            a = q_map[high_index - 2]
            b = q_map[high_index - 1]
            lines.append(f"    cx(q{a}, q{b})")
            _emit_cry(lines, b, a, param, angle_literal=angle_literal)
            lines.append(f"    cx(q{a}, q{b})")
        else:
            mid = q_map[index - 2]
            hi = q_map[high_index - 1]
            ix = q_map[index - 1]
            lines.append(f"    cx(q{mid}, q{hi})")
            _emit_ccry(lines, hi, ix, mid, param, angle_literal=angle_literal)
            lines.append(f"    cx(q{mid}, q{hi})")


def append_dicke_state_lines(
    lines: list[str],
    q_map: list[int],
    k: int,
    angle_literal: Callable[[float], str],
) -> None:
    """Prepare ``|D_n^k⟩`` on qubits ``q_map[0]..q_map[n-1]`` (``n = len(q_map)``).

    Preconditions: ``k`` in ``[0, n]``.  The physical qubits must start in ``|0…0⟩``;
    this routine applies ``X`` to the rightmost ``k`` positions **before** the Qrisp
    schedule (matching ``dicke_state`` examples).
    """
    n = len(q_map)
    if not (0 <= k <= n):
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return
    if k == n:
        for j in range(n):
            lines.append(f"    x(q{q_map[j]})")
        return

    for j in range(n - k, n):
        lines.append(f"    x(q{q_map[j]})")

    for index2 in reversed(range(k + 1, n + 1)):
        _split_cycle_shift(q_map, index2, k, lines, angle_literal=angle_literal)

    for index in reversed(range(2, k + 1)):
        _split_cycle_shift(q_map, index, index - 1, lines, angle_literal=angle_literal)
