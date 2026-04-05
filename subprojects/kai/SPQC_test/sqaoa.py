"""Superposed QAOA advisers for package-local QUBOs.

This module lifts the core idea from ``sqpc.py`` into the Guppy-based QAOA
workflow in ``SOLUTIONS/qaoa.py``:

1. Build a small bank of reduced-problem QAOA advisers.
2. Put those advisers in address superposition.
3. Reuse the ``sqpc`` term-register idea so the selected adviser acts on the
   first ``k`` work registers coherently, where ``k`` is encoded in a pyramid
   state.
4. Post-select *classically* on shots whose repeated adviser samples are both
   low-energy and consistent across the active work registers.
5. Use the post-selected address posterior as a warm start for a final QAOA on
   the full target problem.

The quantum kernels are emitted as literal-angle Guppy source, just like the
existing QAOA workflow, because Guppy kernels used by the emulator must be
zero-argument functions.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SOLUTIONS.qaoa as base_qaoa
from SOLUTIONS.classical_baseline import solve_classical_baseline
from SOLUTIONS.qaoa_plots import save_training_loss_plot


# Top-level research controls.
# ``BRANCH_COVERAGES`` / ``BRANCH_PACKAGES`` are the reduced adviser problem
# sizes. They replace the awkward old ``n'`` / ``m'`` notation.
TARGET_COVERAGES = 4
TARGET_PACKAGES = 2
P_DEPTH = 1
TERM_QUBITS = 2
BRANCH_PACKAGES = 2
BRANCH_COVERAGES = 4
REPETITIONS = 2

# SQAOA-specific runtime controls.
ADVISER_SHOTS = 128
SUPERPOSITION_SHOTS = 512
POSTSELECT_ENERGY_QUANTILE = 0.25
POSTSELECT_VARIANCE_QUANTILE = 0.50
POSTSELECT_DEPTH_WEIGHT = 1.0
EXECUTION_TARGET = base_qaoa.EXECUTION_TARGET

OUTPUT_ROOT = Path(__file__).resolve().parent / "SQAOA"
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
HISTORIES_DIR = OUTPUT_ROOT / "histories"
SOLUTIONS_DIR = OUTPUT_ROOT / "solutions"
PLOTS_DIR = OUTPUT_ROOT / "plots"
RUN_SUMMARY_INDEX_PATH = OUTPUT_ROOT / "run_summaries.csv"

SUMMARY_FIELDS = [
    "run_id",
    "algorithm",
    "execution_target",
    "optimizer",
    "seed",
    "n",
    "m",
    "p",
    "t",
    "branch_coverages",
    "branch_packages",
    "repetitions",
    "runtime_sec",
    "best_profit",
    "classical_opt_profit",
    "num_samples_total",
    "num_samples_feasible",
    "num_samples_postselected",
    "num_objective_evals",
    "solution_path",
    "history_path",
    "plot_path",
    "notes",
]

HISTORY_FIELDS = [
    "run_id",
    "algorithm",
    "stage",
    "execution_target",
    "optimizer",
    "package_index",
    "package_name",
    "evaluation_index",
    "objective_value",
    "best_objective",
    "best_bitstring",
    "coverage_bitstring",
    "best_qubo_energy",
    "shots_total",
    "shots_feasible",
    "improved",
]


@dataclass
class SuperpositionShotRecord:
    adviser_index: int
    address_slot: int
    active_depth: int
    energies: list[float]
    mean_energy: float
    energy_variance: float


@dataclass
class AdviserPosterior:
    theta_seed: np.ndarray
    adviser_probabilities: np.ndarray
    accepted_shots: int
    total_scored_shots: int
    mean_energy_threshold: float
    variance_threshold: float
    per_adviser_mean_energy: np.ndarray
    raw_slot_counts: np.ndarray


def _ensure_output_dirs() -> None:
    for directory in (OUTPUT_ROOT, SUMMARIES_DIR, HISTORIES_DIR, SOLUTIONS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _path_for_csv(directory: Path, stem: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.csv"


def _path_to_repo_relative_string(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _append_csv_row(path: Path, fieldnames: Sequence[str], row: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _write_solution_matrix_csv(
    path: Path,
    matrix: np.ndarray,
    coverage_names: Sequence[str],
    package_names: Sequence[str],
) -> Path:
    rows: list[dict[str, Any]] = []
    for row_index, coverage_name in enumerate(coverage_names):
        row: dict[str, Any] = {"coverage_index": row_index, "coverage_name": coverage_name}
        for package_index, package_name in enumerate(package_names):
            row[package_name] = int(matrix[row_index, package_index])
        rows.append(row)
    return _write_csv_rows(path, ["coverage_index", "coverage_name", *package_names], rows)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_run_id(
    *,
    n: int,
    m: int,
    p: int,
    t: int,
    branch_coverages: int,
    branch_packages: int,
    repetitions: int,
    execution_target: str,
    optimizer: str,
) -> str:
    return (
        "sqaoa"
        f"_{execution_target}_{optimizer}"
        f"_n{n}_m{m}_p{p}_t{t}"
        f"_bc{branch_coverages}_bp{branch_packages}_r{repetitions}"
        f"_{_utc_timestamp()}"
    )


def _build_feasibility_checker(problem: base_qaoa.BundlingProblem) -> Callable[[Sequence[int]], bool]:
    coverage_lookup = {coverage.name: index for index, coverage in enumerate(problem.coverages)}
    incompatible_pairs = [
        (coverage_lookup[rule.coverage_i], coverage_lookup[rule.coverage_j])
        for rule in problem.compatibility_rules
        if not rule.compatible
    ]
    dependency_pairs = [
        (coverage_lookup[rule.requires], coverage_lookup[rule.dependent]) for rule in problem.dependency_rules
    ]

    def is_feasible(bits: Sequence[int]) -> bool:
        if len(bits) != problem.N:
            return False
        for indices in problem.mandatory_families.values():
            if sum(int(bits[index]) for index in indices) != 1:
                return False
        for indices in problem.optional_families.values():
            if len(indices) > 1 and sum(int(bits[index]) for index in indices) > 1:
                return False
        if sum(int(value) for value in bits) > problem.max_options_per_package:
            return False
        for left, right in incompatible_pairs:
            if int(bits[left]) and int(bits[right]):
                return False
        for requires_index, dependent_index in dependency_pairs:
            if int(bits[dependent_index]) and not int(bits[requires_index]):
                return False
        return True

    return is_feasible


def _count_feasible_samples(
    bitstring_counts: dict[str, int],
    n_coverage: int,
    is_feasible: Callable[[Sequence[int]], bool],
) -> int:
    feasible_count = 0
    for bitstring, count in bitstring_counts.items():
        coverage_bits = [int(char) for char in bitstring[:n_coverage]]
        if is_feasible(coverage_bits):
            feasible_count += int(count)
    return feasible_count


def _log(message: str) -> None:
    base_qaoa._log(message)


def _angle_literal_from_radians(phi: float) -> str:
    return repr(2.0 * float(phi) / math.pi)


def _decode_active_depth(term_bits: str, repetitions: int) -> int:
    if not term_bits:
        return repetitions
    depth = 0
    for bit in term_bits:
        if bit != "1":
            break
        depth += 1
    return min(depth, repetitions)


def _coerce_theta(theta: np.ndarray | None, p: int, seed: int) -> np.ndarray:
    if theta is None:
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, math.pi, size=2 * p)
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size != 2 * p:
        raise ValueError(f"Expected {2 * p} parameters, got {arr.size}")
    return base_qaoa._clip_theta(arr)


def _append_line(lines: list[str], text: str) -> None:
    lines.append(f"    {text}")


def _append_controlled_ry(lines: list[str], control: str, target: str, angle_literal: str) -> None:
    _append_line(lines, f"rx({target}, angle(-0.5))")
    _append_line(lines, f"crz({control}, {target}, angle({angle_literal}))")
    _append_line(lines, f"rx({target}, angle(0.5))")


def _append_pyramid_state(lines: list[str], term_names: list[str]) -> None:
    if not term_names:
        return
    _append_line(lines, f"x({term_names[0]})")
    for k in range(2, len(term_names) + 1):
        theta = 2.0 * math.acos(1.0 / math.sqrt(len(term_names) - k + 2))
        literal = _angle_literal_from_radians(theta)
        _append_controlled_ry(lines, term_names[k - 2], term_names[k - 1], literal)


def _append_multi_control_compute(lines: list[str], controls: list[str], ancillas: list[str]) -> str:
    if not controls:
        raise ValueError("At least one control qubit is required")
    if len(controls) == 1:
        return controls[0]
    required = len(controls) - 1
    if len(ancillas) < required:
        raise ValueError(f"Need at least {required} ancillas for {len(controls)} controls")
    _append_line(lines, f"toffoli({controls[0]}, {controls[1]}, {ancillas[0]})")
    for idx in range(2, len(controls)):
        _append_line(lines, f"toffoli({controls[idx]}, {ancillas[idx - 2]}, {ancillas[idx - 1]})")
    return ancillas[len(controls) - 2]


def _append_multi_control_uncompute(lines: list[str], controls: list[str], ancillas: list[str]) -> None:
    if len(controls) <= 1:
        return
    for idx in range(len(controls) - 1, 1, -1):
        _append_line(lines, f"toffoli({controls[idx]}, {ancillas[idx - 2]}, {ancillas[idx - 1]})")
    _append_line(lines, f"toffoli({controls[0]}, {controls[1]}, {ancillas[0]})")


def _append_qaoa_body(
    lines: list[str],
    work_names: list[str],
    c_z: np.ndarray,
    zz_terms: list[tuple[int, int, float]],
    gammas: np.ndarray,
    betas: np.ndarray,
    control: str | None = None,
) -> None:
    for qubit_name in work_names:
        if control is None:
            _append_line(lines, f"h({qubit_name})")
        else:
            _append_line(lines, f"ch({control}, {qubit_name})")

    for gamma, beta in zip(gammas, betas, strict=True):
        for i, j, weight in zz_terms:
            literal = _angle_literal_from_radians(float(gamma) * float(weight))
            _append_line(lines, f"cx({work_names[i]}, {work_names[j]})")
            if control is None:
                _append_line(lines, f"rz({work_names[j]}, angle({literal}))")
            else:
                _append_line(lines, f"crz({control}, {work_names[j]}, angle({literal}))")
            _append_line(lines, f"cx({work_names[i]}, {work_names[j]})")

        for idx, weight in enumerate(c_z):
            if abs(weight) < 1e-15:
                continue
            literal = _angle_literal_from_radians(float(gamma) * float(weight))
            if control is None:
                _append_line(lines, f"rz({work_names[idx]}, angle({literal}))")
            else:
                _append_line(lines, f"crz({control}, {work_names[idx]}, angle({literal}))")

        for qubit_name in work_names:
            literal = _angle_literal_from_radians(float(beta))
            if control is None:
                _append_line(lines, f"rx({qubit_name}, angle({literal}))")
            else:
                _append_line(lines, f"h({qubit_name})")
                _append_line(lines, f"crz({control}, {qubit_name}, angle({literal}))")
                _append_line(lines, f"h({qubit_name})")


def _build_superposed_qaoa_source(
    padded_blocks: list[base_qaoa.QuboBlock],
    padded_thetas: list[np.ndarray],
    address_qubits: int,
    term_qubits: int,
    repetitions: int,
) -> tuple[str, int]:
    work_qubits = padded_blocks[0].n_vars
    max_controls = address_qubits + min(term_qubits, repetitions)
    ancilla_count = max(0, max_controls - 1)
    total_qubits = address_qubits + term_qubits + repetitions * work_qubits + ancilla_count

    address_names = [f"addr_{idx}" for idx in range(address_qubits)]
    term_names = [f"term_{idx}" for idx in range(term_qubits)]
    work_names = [[f"w_{copy_idx}_{var_idx}" for var_idx in range(work_qubits)] for copy_idx in range(repetitions)]
    ancilla_names = [f"anc_{idx}" for idx in range(ancilla_count)]

    lines = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import ch, crz, cx, h, measure, qubit, rx, rz, toffoli, x",
        "",
        "@guppy",
        "def qaoa_kernel() -> None:",
    ]

    for name in address_names + term_names:
        _append_line(lines, f"{name} = qubit()")
    for register in work_names:
        for name in register:
            _append_line(lines, f"{name} = qubit()")
    for name in ancilla_names:
        _append_line(lines, f"{name} = qubit()")

    for name in address_names:
        _append_line(lines, f"h({name})")
    _append_pyramid_state(lines, term_names)

    block_coefficients = [base_qaoa.qubo_to_ising_pauli_coefficients(block.Q) for block in padded_blocks]
    for slot_index, (block, theta) in enumerate(zip(padded_blocks, padded_thetas, strict=True)):
        del block
        gammas, betas = base_qaoa._split_theta(theta, len(theta) // 2)
        c_z, zz_terms = block_coefficients[slot_index]
        address_bits = format(slot_index, f"0{address_qubits}b") if address_qubits else ""

        for bit_index, bit in enumerate(address_bits):
            if bit == "0":
                _append_line(lines, f"x({address_names[bit_index]})")

        for copy_idx, register in enumerate(work_names):
            term_prefix = term_names[: min(copy_idx + 1, term_qubits)]
            controls = term_prefix + address_names
            if controls:
                selector = _append_multi_control_compute(lines, controls, ancilla_names)
                _append_qaoa_body(lines, register, c_z, zz_terms, gammas, betas, control=selector)
                _append_multi_control_uncompute(lines, controls, ancilla_names)
            else:
                _append_qaoa_body(lines, register, c_z, zz_terms, gammas, betas, control=None)

        for bit_index, bit in reversed(list(enumerate(address_bits))):
            if bit == "0":
                _append_line(lines, f"x({address_names[bit_index]})")

    for idx, name in enumerate(address_names):
        _append_line(lines, f'result("addr{idx}", measure({name}))')
    for idx, name in enumerate(term_names):
        _append_line(lines, f'result("term{idx}", measure({name}))')
    for copy_idx, register in enumerate(work_names):
        for var_idx, name in enumerate(register):
            _append_line(lines, f'result("w{copy_idx}_{var_idx}", measure({name}))')
    for idx, name in enumerate(ancilla_names):
        _append_line(lines, f'result("anc{idx}", measure({name}))')

    return "\n".join(lines) + "\n", total_qubits


def _shot_value(shot: Any, key: str) -> int:
    if hasattr(shot, "as_dict"):
        data = shot.as_dict()
    elif hasattr(shot, "entries"):
        data = dict(shot.entries)
    elif isinstance(shot, dict):
        data = shot
    else:
        raise TypeError(f"Unsupported shot type: {type(shot)!r}")
    return base_qaoa._read_measurement_value(data[key])


def _score_superposition_shots(
    shots: list[Any],
    slot_to_adviser: list[int],
    branch_blocks: list[base_qaoa.QuboBlock],
    address_qubits: int,
    term_qubits: int,
    repetitions: int,
) -> list[SuperpositionShotRecord]:
    records: list[SuperpositionShotRecord] = []
    work_qubits = branch_blocks[0].n_vars
    for shot in shots:
        if address_qubits:
            address_bits = "".join(str(_shot_value(shot, f"addr{idx}")) for idx in range(address_qubits))
            address_slot = int(address_bits, 2)
        else:
            address_slot = 0
        adviser_index = slot_to_adviser[address_slot]

        if term_qubits:
            term_bits = "".join(str(_shot_value(shot, f"term{idx}")) for idx in range(term_qubits))
            active_depth = _decode_active_depth(term_bits, repetitions)
        else:
            active_depth = repetitions
        if active_depth <= 0:
            continue

        energies: list[float] = []
        for copy_idx in range(active_depth):
            bitstring = np.array(
                [_shot_value(shot, f"w{copy_idx}_{var_idx}") for var_idx in range(work_qubits)],
                dtype=float,
            )
            energies.append(branch_blocks[adviser_index].energy(bitstring))
        if not energies:
            continue
        records.append(
            SuperpositionShotRecord(
                adviser_index=adviser_index,
                address_slot=address_slot,
                active_depth=active_depth,
                energies=energies,
                mean_energy=float(np.mean(energies)),
                energy_variance=float(np.var(energies)),
            )
        )
    return records


def _posterior_from_records(
    records: list[SuperpositionShotRecord],
    adviser_thetas: list[np.ndarray],
    adviser_count: int,
    slot_count: int,
) -> AdviserPosterior:
    if not records:
        raise RuntimeError("No superposition shots produced a scoreable active depth")

    mean_energies = np.array([record.mean_energy for record in records], dtype=float)
    variances = np.array([record.energy_variance for record in records], dtype=float)
    mean_threshold = float(np.quantile(mean_energies, POSTSELECT_ENERGY_QUANTILE))
    variance_threshold = float(np.quantile(variances, POSTSELECT_VARIANCE_QUANTILE))

    adviser_weights = np.zeros(adviser_count, dtype=float)
    adviser_energy_sums = np.zeros(adviser_count, dtype=float)
    adviser_energy_counts = np.zeros(adviser_count, dtype=float)
    raw_slot_counts = np.zeros(slot_count, dtype=float)
    accepted = 0

    for record in records:
        adviser_energy_sums[record.adviser_index] += record.mean_energy
        adviser_energy_counts[record.adviser_index] += 1.0
        raw_slot_counts[record.address_slot] += 1.0
        keep = record.mean_energy <= mean_threshold and record.energy_variance <= variance_threshold
        if not keep:
            continue
        accepted += 1
        adviser_weights[record.adviser_index] += max(
            1.0,
            POSTSELECT_DEPTH_WEIGHT * float(record.active_depth),
        )

    per_adviser_mean_energy = np.divide(
        adviser_energy_sums,
        np.maximum(adviser_energy_counts, 1.0),
        out=np.full(adviser_count, float("inf"), dtype=float),
        where=adviser_energy_counts > 0,
    )

    if adviser_weights.sum() <= 0.0:
        best_index = int(np.argmin(per_adviser_mean_energy))
        adviser_weights[best_index] = 1.0

    adviser_probabilities = adviser_weights / adviser_weights.sum()
    theta_seed = np.zeros_like(adviser_thetas[0], dtype=float)
    for idx, theta in enumerate(adviser_thetas):
        theta_seed += adviser_probabilities[idx] * theta
    theta_seed = base_qaoa._clip_theta(theta_seed)

    return AdviserPosterior(
        theta_seed=theta_seed,
        adviser_probabilities=adviser_probabilities,
        accepted_shots=accepted,
        total_scored_shots=len(records),
        mean_energy_threshold=mean_threshold,
        variance_threshold=variance_threshold,
        per_adviser_mean_energy=per_adviser_mean_energy,
        raw_slot_counts=raw_slot_counts,
    )


def learn_adviser_seed(
    n: int = TARGET_COVERAGES,
    m: int = TARGET_PACKAGES,
    p: int = P_DEPTH,
    t: int = TERM_QUBITS,
    branch_packages: int = BRANCH_PACKAGES,
    branch_coverages: int = BRANCH_COVERAGES,
    repetitions: int = REPETITIONS,
    execution_target: str = EXECUTION_TARGET,
) -> AdviserPosterior:
    del n, m
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if t < 0:
        raise ValueError("TERM_QUBITS must be non-negative")
    if t > 0 and repetitions > t:
        raise ValueError(
            "This SQAOA prototype expects REPETITIONS <= TERM_QUBITS so each copy has a distinct prefix control"
        )
    if branch_packages < 1 or branch_coverages < 1:
        raise ValueError("Reduced adviser dimensions must both be positive")

    reduced_problem = base_qaoa.subsample_problem(
        base_qaoa.load_ltm_instance(base_qaoa.DATA_DIR),
        n_coverages=branch_coverages,
        m_packages=branch_packages,
    )
    branch_blocks = [
        base_qaoa.build_qubo_block_for_package(reduced_problem, package_index)
        for package_index in range(reduced_problem.M)
    ]
    adviser_thetas: list[np.ndarray] = []

    _log(
        f"[sqaoa] Optimizing {len(branch_blocks)} reduced advisers "
        f"(branch_coverages={branch_coverages}, branch_packages={branch_packages}, p={p})"
    )
    for adviser_index, block in enumerate(branch_blocks):
        result = base_qaoa.optimize_block(
            block,
            p=p,
            shots=ADVISER_SHOTS,
            seed=base_qaoa.SEED + 1000 * adviser_index,
            execution_target=execution_target,
        )
        adviser_thetas.append(result.theta.copy())

    address_qubits = 0 if len(branch_blocks) == 1 else int(math.ceil(math.log2(len(branch_blocks))))
    slot_count = 1 if address_qubits == 0 else 1 << address_qubits
    slot_to_adviser = [slot % len(branch_blocks) for slot in range(slot_count)]
    padded_blocks = [branch_blocks[idx] for idx in slot_to_adviser]
    padded_thetas = [adviser_thetas[idx] for idx in slot_to_adviser]

    source, total_qubits = _build_superposed_qaoa_source(
        padded_blocks=padded_blocks,
        padded_thetas=padded_thetas,
        address_qubits=address_qubits,
        term_qubits=t,
        repetitions=repetitions,
    )
    with base_qaoa._loaded_qaoa_kernel_from_source(source) as kernel:
        emulator = kernel.emulator(n_qubits=total_qubits).with_shots(int(SUPERPOSITION_SHOTS)).with_seed(base_qaoa.SEED)
        result = emulator.run()

    records = _score_superposition_shots(
        shots=list(result.results),
        slot_to_adviser=slot_to_adviser,
        branch_blocks=branch_blocks,
        address_qubits=address_qubits,
        term_qubits=t,
        repetitions=repetitions,
    )
    posterior = _posterior_from_records(
        records=records,
        adviser_thetas=adviser_thetas,
        adviser_count=len(branch_blocks),
        slot_count=slot_count,
    )
    adviser_summary = ", ".join(
        f"{idx}:{prob:.3f}" for idx, prob in enumerate(posterior.adviser_probabilities)
    )
    _log(
        f"[sqaoa] Postselected {posterior.accepted_shots}/{posterior.total_scored_shots} adviser shots; "
        f"posterior={adviser_summary}"
    )
    return posterior


def optimize_block_from_seed(
    block: base_qaoa.QuboBlock,
    p: int,
    shots: int,
    seed: int,
    initial_theta: np.ndarray,
    execution_target: str = EXECUTION_TARGET,
    evaluation_callback: base_qaoa.QaoaEvaluationCallback | None = None,
) -> base_qaoa.QaoaOptimizationResult:
    theta0 = _coerce_theta(initial_theta, p=p, seed=seed)

    if base_qaoa.OPTIMIZER == "cobyla":
        from scipy.optimize import minimize

        best_theta = theta0.copy()
        best_stats: base_qaoa.QaoaSampleStats | None = None
        best_objective = float("inf")
        eval_index = 0
        trace = base_qaoa.QaoaOptimizationTrace()

        _log(
            f"[package {block.package_index + 1}] Starting SQAOA-warm COBYLA on "
            f"{block.n_vars} qubits with seeded theta"
        )

        def objective(theta: np.ndarray) -> float:
            nonlocal best_theta, best_stats, best_objective, eval_index
            evaluation_number = eval_index + 1
            clipped = base_qaoa._clip_theta(theta)
            gammas, betas = base_qaoa._split_theta(clipped, p)
            evaluation_start = time.perf_counter()
            stats = base_qaoa._run_qaoa_on_block(
                block,
                gammas,
                betas,
                shots=shots,
                seed=seed + eval_index,
                execution_target=execution_target,
            )
            value = base_qaoa._mean_sample_energy(block, stats)
            elapsed = time.perf_counter() - evaluation_start
            eval_index += 1
            improved = False
            note = ""
            if value < best_objective:
                best_objective = value
                best_theta = clipped.copy()
                best_stats = stats
                improved = True
                note = "new best"
            trace.record(value, best_stats=best_stats, improved=improved)
            base_qaoa._emit_evaluation_callback(
                evaluation_callback,
                block,
                evaluation_number,
                value,
                best_objective,
                stats,
                improved,
            )
            base_qaoa._report_evaluation(
                block,
                evaluation_number,
                value,
                best_objective,
                stats,
                elapsed,
                note=note,
            )
            return float(value)

        minimize(objective, theta0, method="COBYLA", options={"maxiter": int(base_qaoa.COBYLA_MAXITER)})
        if best_stats is None:
            raise RuntimeError("Seeded COBYLA did not produce any evaluations")
        return base_qaoa.QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)

    if base_qaoa.OPTIMIZER == "spsa":
        rng = np.random.default_rng(seed)
        theta = theta0.copy()
        best_theta = theta.copy()
        best_stats: base_qaoa.QaoaSampleStats | None = None
        best_objective = float("inf")
        eval_index = 0
        trace = base_qaoa.QaoaOptimizationTrace()
        a = 0.15
        c = 0.12
        alpha = 0.602
        gamma_spsa = 0.101
        stability_A = 10.0

        _log(
            f"[package {block.package_index + 1}] Starting SQAOA-warm SPSA on "
            f"{block.n_vars} qubits with seeded theta"
        )

        for step in range(base_qaoa.SPSA_MAXITER):
            ak = a / ((step + 1 + stability_A) ** alpha)
            ck = c / ((step + 1) ** gamma_spsa)
            delta = rng.choice([-1.0, 1.0], size=2 * p)

            theta_plus = base_qaoa._clip_theta(theta + ck * delta)
            theta_minus = base_qaoa._clip_theta(theta - ck * delta)

            g_plus, b_plus = base_qaoa._split_theta(theta_plus, p)
            evaluation_number = eval_index + 1
            plus_start = time.perf_counter()
            stats_plus = base_qaoa._run_qaoa_on_block(
                block,
                g_plus,
                b_plus,
                shots=shots,
                seed=seed + eval_index,
                execution_target=execution_target,
            )
            value_plus = base_qaoa._mean_sample_energy(block, stats_plus)
            plus_elapsed = time.perf_counter() - plus_start
            eval_index += 1
            improved_plus = False
            if value_plus < best_objective:
                best_objective = value_plus
                best_theta = theta_plus.copy()
                best_stats = stats_plus
                improved_plus = True
            best_plus = trace.record(value_plus, best_stats=best_stats, improved=improved_plus)
            base_qaoa._emit_evaluation_callback(
                evaluation_callback,
                block,
                evaluation_number,
                value_plus,
                best_plus,
                stats_plus,
                improved_plus,
            )
            base_qaoa._report_evaluation(
                block,
                evaluation_number,
                value_plus,
                best_plus,
                stats_plus,
                plus_elapsed,
                note=f"SPSA step {step + 1} (+){' new best' if improved_plus else ''}",
            )

            g_minus, b_minus = base_qaoa._split_theta(theta_minus, p)
            evaluation_number_minus = eval_index + 1
            minus_start = time.perf_counter()
            stats_minus = base_qaoa._run_qaoa_on_block(
                block,
                g_minus,
                b_minus,
                shots=shots,
                seed=seed + eval_index,
                execution_target=execution_target,
            )
            value_minus = base_qaoa._mean_sample_energy(block, stats_minus)
            minus_elapsed = time.perf_counter() - minus_start
            eval_index += 1

            gradient = ((value_plus - value_minus) / (2.0 * ck)) * (1.0 / delta)
            theta = base_qaoa._clip_theta(theta - ak * gradient)

            improved_minus = False
            if value_minus < best_objective:
                best_objective = value_minus
                best_theta = theta_minus.copy()
                best_stats = stats_minus
                improved_minus = True
            best_minus = trace.record(value_minus, best_stats=best_stats, improved=improved_minus)
            base_qaoa._emit_evaluation_callback(
                evaluation_callback,
                block,
                evaluation_number_minus,
                value_minus,
                best_minus,
                stats_minus,
                improved_minus,
            )
            base_qaoa._report_evaluation(
                block,
                evaluation_number_minus,
                value_minus,
                best_minus,
                stats_minus,
                minus_elapsed,
                note=f"SPSA step {step + 1} (-){' new best' if improved_minus else ''}",
            )

        if best_stats is None:
            raise RuntimeError("Seeded SPSA did not produce any evaluations")
        return base_qaoa.QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)

    raise ValueError("OPTIMIZER must be 'cobyla' or 'spsa'")


def solve_sqaoa_matrix(
    n: int = TARGET_COVERAGES,
    m: int = TARGET_PACKAGES,
    p: int = P_DEPTH,
    t: int = TERM_QUBITS,
    branch_packages: int = BRANCH_PACKAGES,
    branch_coverages: int = BRANCH_COVERAGES,
    repetitions: int = REPETITIONS,
    execution_target: str = EXECUTION_TARGET,
) -> np.ndarray:
    _ensure_output_dirs()
    workflow_start = time.perf_counter()
    resolved_execution_target = base_qaoa._normalize_execution_target(execution_target)
    if branch_coverages > n:
        raise ValueError("BRANCH_COVERAGES must be less than or equal to TARGET_COVERAGES")
    if branch_packages > m:
        raise ValueError("BRANCH_PACKAGES must be less than or equal to TARGET_PACKAGES")
    _log(f"[sqaoa] Loading instance data from {base_qaoa.DATA_DIR}")
    problem = base_qaoa.subsample_problem(base_qaoa.load_ltm_instance(base_qaoa.DATA_DIR), n_coverages=n, m_packages=m)
    classical_opt = solve_classical_baseline(n=n, m=m)
    is_feasible = _build_feasibility_checker(problem)
    run_id = _build_run_id(
        n=n,
        m=m,
        p=p,
        t=t,
        branch_coverages=branch_coverages,
        branch_packages=branch_packages,
        repetitions=repetitions,
        execution_target=resolved_execution_target,
        optimizer=base_qaoa.OPTIMIZER,
    )

    posterior = learn_adviser_seed(
        n=n,
        m=m,
        p=p,
        t=t,
        branch_packages=branch_packages,
        branch_coverages=branch_coverages,
        repetitions=repetitions,
        execution_target=resolved_execution_target,
    )

    blocks = [base_qaoa.build_qubo_block_for_package(problem, package_index) for package_index in range(problem.M)]
    x_matrix = np.zeros((problem.N, problem.M), dtype=int)
    c_matrix = base_qaoa.make_c_matrix(problem)
    num_samples_total = 0
    num_samples_feasible = 0
    num_objective_evals = 0
    history_rows: list[dict[str, Any]] = []
    plot_series: list[tuple[str, list[float], list[float]]] = []
    _log(
        f"[sqaoa] Solving full problem with seeded theta "
        f"{np.array2string(posterior.theta_seed, precision=4, separator=', ')}"
    )

    for package_index, block in enumerate(blocks):
        block_start = time.perf_counter()
        package_name = (
            problem.package_names[package_index]
            if problem.package_names and package_index < len(problem.package_names)
            else f"package {package_index + 1}"
        )

        def on_evaluation(
            callback_block: base_qaoa.QuboBlock,
            evaluation_index: int,
            value: float,
            best_objective: float,
            stats: base_qaoa.QaoaSampleStats,
            improved: bool,
        ) -> None:
            nonlocal num_samples_total, num_samples_feasible, num_objective_evals
            shots_total = int(sum(stats.bitstring_counts.values()))
            shots_feasible = _count_feasible_samples(stats.bitstring_counts, problem.N, is_feasible)
            num_samples_total += shots_total
            num_samples_feasible += shots_feasible
            num_objective_evals += 1
            history_rows.append(
                {
                    "run_id": run_id,
                    "algorithm": "sqaoa",
                    "stage": "final_qaoa",
                    "execution_target": resolved_execution_target,
                    "optimizer": base_qaoa.OPTIMIZER,
                    "package_index": int(callback_block.package_index),
                    "package_name": package_name,
                    "evaluation_index": int(evaluation_index),
                    "objective_value": float(value),
                    "best_objective": float(best_objective),
                    "best_bitstring": stats.best_bitstring,
                    "coverage_bitstring": stats.best_bitstring[: problem.N],
                    "best_qubo_energy": float(stats.best_qubo_energy),
                    "shots_total": shots_total,
                    "shots_feasible": shots_feasible,
                    "improved": int(bool(improved)),
                }
            )

        result = optimize_block_from_seed(
            block,
            p=p,
            shots=base_qaoa.SHOTS,
            seed=base_qaoa.SEED + package_index * 1000,
            initial_theta=posterior.theta_seed,
            execution_target=resolved_execution_target,
            evaluation_callback=on_evaluation,
        )
        coverage_bits = np.array([int(char) for char in result.stats.best_bitstring[: problem.N]], dtype=int)
        x_matrix[:, package_index] = coverage_bits
        plot_series.append(
            (
                package_name,
                result.trace.objective_values,
                result.trace.best_objective_values,
            )
        )
        _log(
            f"[sqaoa] Package {package_index + 1}/{problem.M} finished in "
            f"{time.perf_counter() - block_start:.2f}s with best sample "
            f"{result.stats.best_bitstring[:problem.N]}"
        )

    runtime_sec = time.perf_counter() - workflow_start
    best_profit = float(np.sum(c_matrix * x_matrix))
    coverage_names = [coverage.name for coverage in problem.coverages]
    package_names = problem.package_names or [f"package {index + 1}" for index in range(problem.M)]
    stem = run_id

    solution_path = _write_solution_matrix_csv(
        _path_for_csv(SOLUTIONS_DIR, f"{stem}_solution"),
        matrix=x_matrix,
        coverage_names=coverage_names,
        package_names=package_names,
    )
    history_path = _write_csv_rows(
        _path_for_csv(HISTORIES_DIR, f"{stem}_history"),
        HISTORY_FIELDS,
        history_rows,
    )
    plot_path = save_training_loss_plot(
        plot_series,
        PLOTS_DIR / f"{stem}_training_loss.png",
        title="SQAOA Final-Stage Training Loss",
    )

    posterior_token = ",".join(f"{idx}:{prob:.4f}" for idx, prob in enumerate(posterior.adviser_probabilities))
    summary_row = {
        "run_id": run_id,
        "algorithm": "sqaoa",
        "execution_target": resolved_execution_target,
        "optimizer": base_qaoa.OPTIMIZER,
        "seed": int(base_qaoa.SEED),
        "n": int(n),
        "m": int(m),
        "p": int(p),
        "t": int(t),
        "branch_coverages": int(branch_coverages),
        "branch_packages": int(branch_packages),
        "repetitions": int(repetitions),
        "runtime_sec": float(runtime_sec),
        "best_profit": best_profit,
        "classical_opt_profit": float(classical_opt["profit"]),
        "num_samples_total": int(num_samples_total),
        "num_samples_feasible": int(num_samples_feasible),
        "num_samples_postselected": int(posterior.accepted_shots),
        "num_objective_evals": int(num_objective_evals),
        "solution_path": _path_to_repo_relative_string(solution_path),
        "history_path": _path_to_repo_relative_string(history_path),
        "plot_path": _path_to_repo_relative_string(plot_path),
        "notes": (
            f"total_scored_postselection_shots={posterior.total_scored_shots}; "
            f"mean_energy_threshold={posterior.mean_energy_threshold:.6f}; "
            f"variance_threshold={posterior.variance_threshold:.6f}; "
            f"adviser_posterior={posterior_token}"
        ),
    }
    summary_path = _write_csv_rows(
        _path_for_csv(SUMMARIES_DIR, f"{stem}_summary"),
        SUMMARY_FIELDS,
        [summary_row],
    )
    _append_csv_row(RUN_SUMMARY_INDEX_PATH, SUMMARY_FIELDS, summary_row)

    _log(f"[sqaoa] Saved summary to {_path_to_repo_relative_string(summary_path)}")
    _log(f"[sqaoa] Saved solution to {_path_to_repo_relative_string(solution_path)}")
    _log(f"[sqaoa] Saved history to {_path_to_repo_relative_string(history_path)}")
    _log(f"[sqaoa] Saved plot to {_path_to_repo_relative_string(plot_path)}")
    _log(f"[sqaoa] Finished workflow in {runtime_sec:.2f}s")
    return x_matrix


def main() -> np.ndarray:
    return solve_sqaoa_matrix(
        n=TARGET_COVERAGES,
        m=TARGET_PACKAGES,
        p=P_DEPTH,
        t=TERM_QUBITS,
        branch_packages=BRANCH_PACKAGES,
        branch_coverages=BRANCH_COVERAGES,
        repetitions=REPETITIONS,
        execution_target=EXECUTION_TARGET,
    )


if __name__ == "__main__":
    matrix = main()
    print(matrix)
