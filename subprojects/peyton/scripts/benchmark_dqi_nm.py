#!/usr/bin/env python3
"""Benchmark DQI for arbitrary problem size ``n`` (QUBO variables) and ``m`` (parity rows of ``B``).

The circuit is the Travelers parity ansatz (default): ``n`` variable qubits plus ``m`` syndrome
qubits when ``B y = v (mod 2)`` is enforced via post-selection.

Gate counts match the generated Guppy kernel in ``dqi_core._build_dqi_parity_source``: ``H`` on all
qubits before measurement, one block of ``Rz`` per layer (angles with magnitude below ``1e-15`` are
omitted), CNOTs for the parity matrix, ``X`` on ancillas where ``v_j=1``, and one measurement per
qubit. ``gates_total_shots_evals`` is ``gates_per_circuit * shots * n_evaluations``.

Examples::

    python scripts/benchmark_dqi_nm.py --n 8 --m 3 --seed 0 --execution local
    python scripts/benchmark_dqi_nm.py --n 6 --m 0 --variational --p 2 --optimizer cobyla
    python scripts/benchmark_dqi_nm.py --sweep-n 4,6,8 --sweep-m 0,2 --csv artifacts/dqi_nm_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_core import build_hamming_weight_penalty_qubo, by_xor_v_mod2, qubo_energy
from src.run_dqi import run_dqi_with_details

# Match ``dqi_core._build_dqi_parity_source`` / ``sample_dqi`` when skipping near-zero rotations.
_RZ_EPS = 1e-15


def _phase_coeffs_for_counting(
    Q: np.ndarray,
    *,
    normalize_phase_c: bool,
    phase_c: np.ndarray | None = None,
) -> np.ndarray:
    """Same diagonal phase vector as parity ``sample_dqi`` (for Rz gate counting)."""
    if phase_c is not None:
        return np.asarray(phase_c, dtype=float).ravel()
    q = np.asarray(Q, dtype=float)
    q = (q + q.T) * 0.5
    c = np.diag(q).astype(float)
    if normalize_phase_c:
        scale = float(np.max(np.abs(c))) if c.size else 1.0
        if scale < 1e-15:
            scale = 1.0
        c = c / scale
    return c


def count_dqi_parity_gates(
    n: int,
    m: int,
    B: np.ndarray | None,
    v: np.ndarray | None,
    Q: np.ndarray,
    gammas: list[float],
    *,
    normalize_phase_c: bool,
    phase_c: np.ndarray | None = None,
) -> dict[str, int]:
    """Count gates in one parity DQI kernel (Guppy source layout in ``dqi_core``).

    Returns Hadamards, parameterized Rz, CNOTs, Pauli X on ancillas, and measurements.
    Rz layers omit angles with ``|gamma * c_i| < 1e-15`` like the generated circuit.
    """
    c_vec = _phase_coeffs_for_counting(Q, normalize_phase_c=normalize_phase_c, phase_c=phase_c)
    rz = 0
    for g in gammas:
        g_f = float(g)
        for i in range(n):
            if abs(g_f * float(c_vec[i])) >= _RZ_EPS:
                rz += 1

    if m <= 0 or B is None:
        cx = 0
        x_cnt = 0
    else:
        b = np.asarray(B, dtype=int)
        cx = int(np.sum(b))
        x_cnt = int(np.sum(np.asarray(v, dtype=int).ravel())) if v is not None else 0

    h_cnt = int(n + m)
    meas = int(n + m)
    unitary = int(h_cnt + rz + cx + x_cnt)
    return {
        "gates_h": h_cnt,
        "gates_rz": rz,
        "gates_cx": cx,
        "gates_x": x_cnt,
        "gates_measure": meas,
        "gates_unitary": unitary,
        "gates_per_circuit": unitary + meas,
    }


def random_symmetric_qubo(
    n: int,
    rng: np.random.Generator,
    *,
    target_weight: int | None = None,
    penalty: float = 1.4,
) -> np.ndarray:
    """Dense symmetric QUBO with diagonal structure and random couplings (minimization)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    linear = rng.uniform(0.5, 2.0, size=n)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        Q[i, i] -= float(linear[i])
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.uniform(-0.3, 0.5))
            Q[i, j] += w
            Q[j, i] += w
    if target_weight is not None:
        Q += build_hamming_weight_penalty_qubo(
            n=n, target_weight=int(target_weight), penalty=float(penalty)
        )
    return (Q + Q.T) * 0.5


def random_parity_system(
    n: int,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(B, v)`` with ``B`` shape ``(m, n)`` binary and ``v`` length ``m`` binary."""
    if m < 0:
        raise ValueError("m must be >= 0")
    if m == 0:
        return None, None
    B = rng.integers(0, 2, size=(m, n), dtype=int)
    v = rng.integers(0, 2, size=(m,), dtype=int)
    return B, v


def is_parity_feasible(y: np.ndarray, B: np.ndarray | None, v: np.ndarray | None) -> bool:
    """True iff ``B y = v (mod 2)`` (vacuously true if there are no parity rows)."""
    if B is None or v is None or B.shape[0] == 0:
        return True
    syn = by_xor_v_mod2(B, y, v)
    return bool(np.all(syn == 0))


def brute_force_constrained_min(
    Q: np.ndarray,
    B: np.ndarray | None,
    v: np.ndarray | None,
    *,
    max_n: int = 18,
    constant_offset: float = 0.0,
) -> tuple[np.ndarray | None, float | None]:
    """Exact minimum over feasible 0/1 assignments, or ``(None, None)`` if ``n > max_n``."""
    q = np.asarray(Q, dtype=float)
    n = int(q.shape[0])
    if n > max_n:
        return None, None
    best_val = float("inf")
    best_x: np.ndarray | None = None
    for k in range(1 << n):
        x = np.array([float((k >> i) & 1) for i in range(n)], dtype=float)
        if not is_parity_feasible(x, B, v):
            continue
        ev = qubo_energy(x, q, constant_offset=constant_offset)
        if ev < best_val:
            best_val = float(ev)
            best_x = x.copy()
    if best_x is None:
        return None, None
    return best_x, float(best_val)


def run_single_benchmark(
    n: int,
    m: int,
    *,
    seed: int,
    shots: int,
    p: int,
    optimizer: str,
    variational: bool,
    fixed_gammas: list[float] | None,
    maxiter: int,
    n_samples: int,
    statistic: str,
    mixer: str,
    max_qubits: int,
    execution: str,
    normalize_phase_c: bool,
    brute_force_max_n: int,
    target_weight: int | None,
    qubo_penalty: float,
    phase_c: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build one random instance and return timing + quality metrics."""
    rng = np.random.default_rng(int(seed))
    Q = random_symmetric_qubo(
        n,
        rng,
        target_weight=target_weight,
        penalty=qubo_penalty,
    )
    B, v = random_parity_system(n, m, rng)

    n_tot = n + m
    if n_tot > max_qubits:
        raise ValueError(f"n+m={n_tot} exceeds max_qubits={max_qubits}")

    t0 = time.perf_counter()
    x_best, value, meta = run_dqi_with_details(
        Q,
        p=p,
        optimizer=optimizer,
        B=B,
        v=v,
        variational=variational,
        fixed_gammas=fixed_gammas,
        shots=shots,
        seed=seed,
        maxiter=maxiter,
        n_samples=n_samples,
        statistic=statistic,
        mixer=mixer,
        max_qubits=max_qubits,
        execution=execution,
        normalize_phase_c=normalize_phase_c,
        phase_c=phase_c,
    )
    runtime = time.perf_counter() - t0

    gc = count_dqi_parity_gates(
        n,
        m,
        B,
        v,
        Q,
        list(meta.optimizer_result.gammas),
        normalize_phase_c=normalize_phase_c,
        phase_c=phase_c,
    )
    n_eval = int(meta.optimizer_result.n_evaluations)
    gates_total_est = int(gc["gates_per_circuit"]) * int(shots) * n_eval

    bf_x, bf_val = brute_force_constrained_min(
        Q, B, v, max_n=brute_force_max_n, constant_offset=0.0
    )
    approx_ratio: float | None = None
    if bf_val is not None and abs(bf_val) > 1e-12:
        approx_ratio = float(value) / float(bf_val)

    feasible = is_parity_feasible(x_best, B, v)

    return {
        "n": n,
        "m": m,
        "n_qubits": int(meta.optimizer_result.stats_at_best.n_qubits),
        "seed": seed,
        "runtime_sec": float(runtime),
        "best_value": float(value),
        "n_evaluations": int(meta.optimizer_result.n_evaluations),
        "bitstring": meta.bitstring,
        "feasible_sample_best": bool(feasible),
        "bruteforce_best_value": bf_val,
        "approx_ratio_vs_bruteforce": approx_ratio,
        **gc,
        "gates_total_shots_evals": gates_total_est,
    }


def _parse_int_list(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if not out:
        raise ValueError("expected at least one integer in comma-separated list")
    return out


def _parse_float_list(s: str | None) -> list[float] | None:
    if s is None or s.strip() == "":
        return None
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _write_csv_row(path: Path, row: dict[str, Any], header: list[str], write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=None, help="Number of QUBO variables (if not sweeping).")
    ap.add_argument("--m", type=int, default=None, help="Number of parity rows in B (0 = unconstrained).")
    ap.add_argument("--sweep-n", type=str, default=None, help="Comma-separated n values (e.g. 4,6,8).")
    ap.add_argument("--sweep-m", type=str, default=None, help="Comma-separated m values (e.g. 0,2,4).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=512)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--optimizer", choices=["random", "cobyla", "spsa"], default="cobyla")
    ap.add_argument("--no-variational", action="store_true")
    ap.add_argument("--fixed-gammas", type=str, default=None, help="Comma-separated; length must equal --p.")
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--statistic", choices=["mean", "best"], default="mean")
    ap.add_argument("--mixer", choices=["rx", "h"], default="rx")
    ap.add_argument("--max-qubits", type=int, default=50)
    ap.add_argument(
        "--execution",
        type=str,
        default="local",
        help="local / selene / nexus_selene / nexus_helios (same as run_dqi).",
    )
    ap.add_argument("--normalize-phase-c", action="store_true", help="Normalize phase c_i by max|diag(Q)|.")
    ap.add_argument("--bruteforce-max-n", type=int, default=18, help="Skip BF if n exceeds this.")
    ap.add_argument(
        "--target-weight",
        type=int,
        default=None,
        help="If set, add Hamming-weight penalty toward this count (like example_dqi_10var).",
    )
    ap.add_argument("--qubo-penalty", type=float, default=1.4)
    ap.add_argument("--csv", type=Path, default=None, help="Append one CSV row per run.")
    ap.add_argument(
        "--phase-c-csv",
        type=Path,
        default=None,
        help="Optional path to a length-n row of comma-separated phase coefficients (overrides Q diagonal for Rz counting; must match n at runtime).",
    )
    args = ap.parse_args()

    exec_key = str(args.execution).strip().lower().replace("-", "_")

    if args.sweep_n is not None or args.sweep_m is not None:
        if args.sweep_n is None or args.sweep_m is None:
            ap.error("Both --sweep-n and --sweep-m are required for a sweep.")
        n_list = _parse_int_list(args.sweep_n)
        m_list = _parse_int_list(args.sweep_m)
        pairs = [(n, m) for n in n_list for m in m_list]
    else:
        if args.n is None or args.m is None:
            ap.error("Provide --n and --m, or use --sweep-n and --sweep-m.")
        pairs = [(int(args.n), int(args.m))]

    fixed_gammas = _parse_float_list(args.fixed_gammas)
    if fixed_gammas is not None and len(fixed_gammas) != int(args.p):
        ap.error(f"--fixed-gammas length must equal --p ({args.p}), got {len(fixed_gammas)}")

    variational = not args.no_variational

    phase_c_arg: np.ndarray | None = None
    if args.phase_c_csv is not None:
        raw = Path(args.phase_c_csv).read_text(encoding="utf-8").strip()
        phase_c_arg = np.array([float(x.strip()) for x in raw.split(",") if x.strip() != ""])

    header = [
        "n",
        "m",
        "n_qubits",
        "seed",
        "runtime_sec",
        "best_value",
        "n_evaluations",
        "gates_h",
        "gates_rz",
        "gates_cx",
        "gates_x",
        "gates_measure",
        "gates_unitary",
        "gates_per_circuit",
        "gates_total_shots_evals",
        "feasible_sample_best",
        "bruteforce_best_value",
        "approx_ratio_vs_bruteforce",
        "bitstring",
    ]

    csv_path = args.csv
    write_header = bool(csv_path and not csv_path.is_file())

    print(
        f"{'n':>4} {'m':>4} {'nq':>4} {'t(s)':>8} {'g_circ':>7} {'g_tot':>10} {'best':>12} {'evals':>6} "
        f"{'feas':>5} {'bf_opt':>12} {'ratio':>8}  bitstring"
    )
    for n_i, m_i in pairs:
        pc = phase_c_arg
        if pc is not None and pc.shape[0] != n_i:
            ap.error(f"--phase-c-csv length {pc.shape[0]} does not match n={n_i}")
        row = run_single_benchmark(
            n_i,
            m_i,
            seed=int(args.seed),
            shots=int(args.shots),
            p=int(args.p),
            optimizer=str(args.optimizer),
            variational=variational,
            fixed_gammas=fixed_gammas,
            maxiter=int(args.maxiter),
            n_samples=int(args.n_samples),
            statistic=str(args.statistic),
            mixer=str(args.mixer),
            max_qubits=int(args.max_qubits),
            execution=exec_key,
            normalize_phase_c=bool(args.normalize_phase_c),
            brute_force_max_n=int(args.bruteforce_max_n),
            target_weight=args.target_weight,
            qubo_penalty=float(args.qubo_penalty),
            phase_c=pc,
        )
        bf_s = f"{row['bruteforce_best_value']:.6f}" if row["bruteforce_best_value"] is not None else "n/a"
        ratio_s = (
            f"{row['approx_ratio_vs_bruteforce']:.4f}"
            if row["approx_ratio_vs_bruteforce"] is not None
            else "n/a"
        )
        print(
            f"{row['n']:4d} {row['m']:4d} {row['n_qubits']:4d} {row['runtime_sec']:8.3f} "
            f"{row['gates_per_circuit']:7d} {row['gates_total_shots_evals']:10d} "
            f"{row['best_value']:12.6f} {row['n_evaluations']:6d} "
            f"{str(row['feasible_sample_best']):>5} {bf_s:>12} {ratio_s:>8}  {row['bitstring']}"
        )
        if csv_path is not None:
            _write_csv_row(csv_path, row, header, write_header)
            write_header = False

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
