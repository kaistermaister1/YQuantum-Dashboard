"""Core DQI utilities: QUBO math, Guppy circuit generation, and sampling."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.dqi_backends import (
    bitstring_counts_from_shots,
    normalize_execution,
    run_kernel_local,
    run_kernel_nexus,
)


def import_guppylang_with_workaround() -> None:
    """Import ``guppylang`` (pulls in wasmtime). Applies a Windows-on-ARM64 shim when needed.

    PyPI ``wasmtime`` wheels often ship only ``win32-x86_64/_wasmtime.dll``, while
    ``platform.machine()`` on Windows ARM64 is ``ARM64``, so ``wasmtime._ffi`` looks for a
    missing ``win32-aarch64`` folder. We temporarily report ``AMD64`` so wasmtime loads the
    bundled x86_64 DLL. That only works if this **Python process is x64** (installer:
    "Windows installer (64-bit)" from python.org, which runs under emulation on ARM PCs).
    Native **ARM64** Python cannot load an x86_64 DLL; use x64 Python in that case.
    """
    import platform

    restore_machine: Callable[[], str] | None = None
    if sys.platform == "win32" and platform.machine() in ("ARM64", "arm64"):
        spec = importlib.util.find_spec("wasmtime")
        root: Path | None = None
        if spec is not None:
            if spec.origin:
                root = Path(spec.origin).parent
            elif spec.submodule_search_locations:
                root = Path(next(iter(spec.submodule_search_locations)))
        if root is not None:
            aarch_dll = root / "win32-aarch64" / "_wasmtime.dll"
            x64_dll = root / "win32-x86_64" / "_wasmtime.dll"
            if not aarch_dll.is_file() and x64_dll.is_file():
                restore_machine = platform.machine

                def _machine_amd64() -> str:
                    return "AMD64"

                platform.machine = _machine_amd64  # type: ignore[assignment]

    try:
        import guppylang  # noqa: F401
    except ImportError as exc:
        raise ImportError("Guppy is required: pip install guppylang") from exc
    except OSError as exc:
        raise RuntimeError(
            "Guppy failed to load wasmtime's native library. "
            "On Windows ARM64: install Python's **x86-64** build from python.org (not the "
            "ARM64 installer) so the bundled win32-x86_64 wasmtime DLL can load, then "
            "`pip install --upgrade --force-reinstall wasmtime guppylang` and the MSVC "
            "x64 redistributable. "
            "Details: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
        ) from exc
    finally:
        if restore_machine is not None:
            platform.machine = restore_machine


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


def _prepare_parity_B_v(
    B: np.ndarray, v: np.ndarray
) -> tuple[int, int, list[list[int]], list[int]]:
    """Validate parity data for ``B y = v (mod 2)`` and return dense 0/1 row lists."""
    b = np.asarray(B, dtype=int)
    v_vec = np.asarray(v, dtype=int).ravel()
    if b.ndim != 2:
        raise ValueError("B must be a 2D matrix")
    m, n = int(b.shape[0]), int(b.shape[1])
    if v_vec.shape[0] != m:
        raise ValueError(f"v must have length {m} (number of rows of B), got {v_vec.shape[0]}")
    if not np.all(np.isin(b, (0, 1))):
        raise ValueError("B must contain only 0 and 1")
    if not np.all(np.isin(v_vec, (0, 1))):
        raise ValueError("v must contain only 0 and 1")
    rows: list[list[int]] = [[int(b[j, i]) for i in range(n)] for j in range(m)]
    v_list = [int(v_vec[j]) for j in range(m)]
    return m, n, rows, v_list


def by_xor_v_mod2(B: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return ``(B @ y + v) mod 2`` as 0/1 vector (syndrome before Hadamard / post-selection)."""
    b = np.asarray(B, dtype=int)
    y_vec = (np.asarray(y, dtype=int).ravel() & 1)
    v_vec = (np.asarray(v, dtype=int).ravel() & 1)
    if b.ndim != 2 or y_vec.shape[0] != b.shape[1]:
        raise ValueError("y length must match number of columns of B")
    syn = (b @ y_vec) % 2
    return (syn ^ v_vec) % 2


def _phase_coeffs_from_qubo(Q: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Diagonal linear coefficients for phase encoding (challenge doc ``c_i`` from QUBO diagonal)."""
    q = np.asarray(Q, dtype=float)
    q = (q + q.T) * 0.5
    c = np.diag(q).astype(float)
    if normalize:
        scale = float(np.max(np.abs(c))) if c.size else 1.0
        if scale < 1e-15:
            scale = 1.0
        c = c / scale
    return c


def _filter_postselected_counts(
    bitstring_counts: dict[str, int], n_vars: int, n_syn: int
) -> dict[str, int]:
    """Keep shots whose syndrome register (last ``n_syn`` bits) is all zeros."""
    if n_syn <= 0:
        return dict(bitstring_counts)
    suffix = "0" * n_syn
    out: dict[str, int] = {}
    for s, cnt in bitstring_counts.items():
        if len(s) != n_vars + n_syn:
            continue
        if s.endswith(suffix):
            out[s[:n_vars]] = out.get(s[:n_vars], 0) + int(cnt)
    return out


def _build_dqi_parity_source(
    n: int,
    m: int,
    rows: list[list[int]],
    v_list: list[int],
    c: np.ndarray,
    gammas: list[float],
) -> str:
    """Guppy kernel: uniform superposition on variables, Rz phase layers, syndrome from B,v, H on syndrome.

    Matches the Travelers ``04_dqi_pipeline.html`` structure: variables hold ``y``, ancillas hold
    syndrome ``(B y) ⊕ v`` before the final ``H`` on syndrome qubits; post-select syndrome all-0.
    """
    if len(gammas) < 1:
        raise ValueError("At least one phase layer (gamma) is required")
    c_vec = np.asarray(c, dtype=float).ravel()
    if c_vec.shape[0] != n:
        raise ValueError(f"c must have length n={n}")
    total = n + m
    lines: list[str] = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rz, x",
        "",
        "@guppy",
        "def dqi_kernel() -> None:",
    ]
    for i in range(total):
        lines.append(f"    q{i} = qubit()")
    for i in range(n):
        lines.append(f"    h(q{i})")

    for g in gammas:
        g_f = float(g)
        for i in range(n):
            ci = float(c_vec[i])
            phi_rad = g_f * ci
            if abs(phi_rad) < 1e-15:
                continue
            lit = _angle_literal(phi_rad)
            lines.append(f"    rz(q{i}, angle({lit}))")

    for j in range(m):
        for i in range(n):
            if rows[j][i]:
                lines.append(f"    cx(q{i}, q{n + j})")
        if v_list[j]:
            lines.append(f"    x(q{n + j})")

    for j in range(m):
        lines.append(f"    h(q{n + j})")

    for i in range(total):
        lines.append(f'    result("m{i}", measure(q{i}))')
    return "\n".join(lines) + "\n"


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
    B: np.ndarray | None = None,
    v: np.ndarray | None = None,
    phase_c: np.ndarray | None = None,
    normalize_phase_c: bool = True,
    legacy_ising: bool = False,
    shots: int = 512,
    seed: int = 0,
    mixer: str = "rx",
    max_qubits: int = 50,
    constant_offset: float = 0.0,
    execution: str = "nexus_selene",
    nexus_hugr_name: str = "dqi-hugr",
    nexus_job_name: str = "dqi-execute",
    nexus_helios_system: str = "Helios-1",
    nexus_timeout: float | None = 300.0,
    eval_tag: str = "",
) -> DqiSampleStats:
    """Run a DQI circuit with fixed parameters and return sampled statistics.

    **Parity / max-XORSAT mode (Travelers DQI pipeline, default):** the circuit uses a syndrome
    register for ``B y = v (mod 2)``, ``H`` on syndrome, and post-selection on all-zero syndrome
    outcomes (see ``Travelers/docs/04_dqi_pipeline.html``). Omit both ``B`` and ``v`` for no XOR
    constraints (empty ``B``). Phase kicks use ``Rz(gamma_l * c_i)`` on each variable qubit; by
    default ``c`` is the diagonal of ``Q`` (normalized), or pass ``phase_c`` explicitly.

    **Legacy variational mode:** set ``legacy_ising=True`` for the previous Ising / QAOA-style
    alternating cost (``RZZ`` + ``Rz``) and mixer layers (``betas`` required).

    **Default (Travelers parity ansatz):** when ``legacy_ising`` is false, omitting both ``B`` and
    ``v`` uses an empty parity matrix (no XOR constraints, only phase + measurement on the
    variable register). Pass ``B`` and ``v`` together for ``B y = v (mod 2)``.

    ``execution``:
        - ``local`` / ``selene`` — Guppy emulator on this machine (default).
        - ``nexus_selene`` — Nexus job with ``SeleneConfig`` (cloud Selene emulator).
        - ``nexus_helios`` — Nexus job with ``HeliosConfig`` (set ``nexus_helios_system``).

    Nexus runs require ``qnexus`` login, an active project, and unique HUGR/job names per
    submission when optimizing (use ``eval_tag``, set automatically by ``optimize_dqi``).
    """
    import_guppylang_with_workaround()

    q = np.asarray(Q, dtype=float)
    q = (q + q.T) * 0.5
    if q.shape[0] != q.shape[1]:
        raise ValueError("Q must be square")
    n = int(q.shape[0])

    use_parity = not legacy_ising
    if use_parity:
        if B is None and v is None:
            b_arr = np.zeros((0, n), dtype=int)
            v_arr = np.zeros(0, dtype=int)
        elif B is None or v is None:
            raise ValueError("Provide both B and v, or neither (for unconstrained parity DQI)")
        else:
            b_arr = B
            v_arr = v
        m_syn, n_b, rows, v_list = _prepare_parity_B_v(b_arr, v_arr)
        if n_b != n:
            raise ValueError(f"B has {n_b} columns but Q has shape {n}×{n}")
        if len(gammas) < 1:
            raise ValueError("At least one phase layer (gamma) is required")
        n_tot = n + m_syn
        if n_tot > int(max_qubits):
            raise ValueError(f"n_qubits={n_tot} exceeds max_qubits={max_qubits}")
        if phase_c is not None:
            c_use = np.asarray(phase_c, dtype=float).ravel()
        else:
            c_use = _phase_coeffs_from_qubo(q, normalize=normalize_phase_c)
        src = _build_dqi_parity_source(n, m_syn, rows, v_list, c_use, list(gammas))
    else:
        if n > int(max_qubits):
            raise ValueError(f"n_qubits={n} exceeds max_qubits={max_qubits}")
        if len(gammas) < 1 or len(betas) < 1:
            raise ValueError("At least one layer is required")
        _, c_z, zz = qubo_to_ising(q)
        src = _build_dqi_source(n, c_z, zz, list(gammas), list(betas), mixer=mixer)
        n_tot = n
        m_syn = 0

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

        backend = normalize_execution(execution)
        tag = str(eval_tag).strip()
        hugr_name = f"{nexus_hugr_name}-{tag}" if tag else nexus_hugr_name
        job_name = f"{nexus_job_name}-{tag}" if tag else nexus_job_name

        if backend == "local":
            res = run_kernel_local(kernel, n_tot, int(shots), int(seed))
            bitstring_counts_raw = bitstring_counts_from_shots(res.results, n_tot)
        elif backend == "nexus_selene":
            bitstring_counts_raw = run_kernel_nexus(
                kernel,
                n_tot,
                int(shots),
                mode="selene",
                hugr_name=hugr_name,
                job_name=job_name,
                helios_system_name=nexus_helios_system,
                timeout=nexus_timeout,
            )
        else:
            bitstring_counts_raw = run_kernel_nexus(
                kernel,
                n_tot,
                int(shots),
                mode="helios",
                hugr_name=hugr_name,
                job_name=job_name,
                helios_system_name=nexus_helios_system,
                timeout=nexus_timeout,
            )

        if use_parity and m_syn > 0:
            bitstring_counts = _filter_postselected_counts(bitstring_counts_raw, n, m_syn)
            if not bitstring_counts:
                # No syndrome-0 shots: fall back to marginalizing variable register for stability.
                c_m: Counter[str] = Counter()
                for s, c_ in bitstring_counts_raw.items():
                    if len(s) >= n:
                        c_m[s[:n]] += int(c_)
                bitstring_counts = dict(c_m)
        else:
            bitstring_counts = dict(bitstring_counts_raw)

        best_s: str | None = None
        best_val = float("inf")
        for s in bitstring_counts:
            x = np.array([float(int(ch)) for ch in s], dtype=float)
            e = qubo_energy(x, q, constant_offset=constant_offset)
            if e < best_val:
                best_val = e
                best_s = s
        assert best_s is not None

        betas_out = [0.0] * len(gammas) if use_parity else [float(v) for v in betas]

        return DqiSampleStats(
            n_qubits=n_tot,
            p=len(gammas),
            shots=int(shots),
            gammas=[float(v) for v in gammas],
            betas=betas_out,
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
