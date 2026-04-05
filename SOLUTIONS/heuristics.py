"""Run classical or QAOA heuristics and save structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

import classical_baseline
import qaoa
import qaoa_selene
from qaoa_plots import save_training_loss_plot

REPO_ROOT = Path(__file__).resolve().parent.parent
DQI_ADAPTER_ROOT = REPO_ROOT / "subprojects" / "kai" / "kai-dqi"
if str(DQI_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(DQI_ADAPTER_ROOT))

import travelers_dqi

OUTPUT_ROOT = Path(__file__).resolve().parent / "HEURISTICS"
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
HISTORIES_DIR = OUTPUT_ROOT / "histories"
SOLUTIONS_DIR = OUTPUT_ROOT / "solutions"
PLOTS_DIR = OUTPUT_ROOT / "plots"
RUN_SUMMARY_INDEX_PATH = OUTPUT_ROOT / "run_summaries.csv"

# Top-level run config
ALGORITHM = "dqi"  # "classical", "qaoa", or "dqi" when implemented
N_COVERAGES = 10
M_PACKAGES = 3
P_DEPTH = 1
QAOA_EXECUTION_TARGET = "local"  # "local" or "selene"
QAOA_OPTIMIZER = "cobyla"  # "cobyla" or "spsa"
SEED = 0
GENERATE_TRAINING_LOSS_PLOT = False
# On Selene, subsamples with N×M at or below this use regular SPSA/COBYLA unless --force-fast-selene.
SELENE_FAST_SELENE_MIN_NM = 41

SUMMARY_FIELDS = [
    "run_id",
    "algorithm",
    "optimizer",
    "seed",
    "N_local",
    "M_blocks",
    "n_total",
    "p",
    "lambda",
    "runtime_sec",
    "best_profit",
    "classical_opt_profit",
    "num_samples_total",
    "num_samples_feasible",
    "num_samples_postselected",
    "num_objective_evals",
    "num_qubits",
    "circuit_depth",
    "two_qubit_gate_count",
    "solution_path",
    "history_path",
    "notes",
]

QAOA_HISTORY_FIELDS = [
    "run_id",
    "algorithm",
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

CLASSICAL_HISTORY_FIELDS = [
    "run_id",
    "algorithm",
    "optimizer",
    "status",
    "runtime_sec",
    "profit",
]

DQI_HISTORY_FIELDS = [
    "run_id",
    "algorithm",
    "bp_mode",
    "package_index",
    "package_name",
    "objective_mode",
    "objective_lower_bound",
    "expanded_num_vars",
    "num_constraints",
    "B_rows",
    "B_cols",
    "ell",
    "dqi_expected_f",
    "dqi_expected_s",
    "quantum_attempted",
    "quantum_ran",
    "quantum_postselected_shots",
    "quantum_best_assignment",
    "note",
]


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
    matrix: np.ndarray | list[list[int]],
    coverage_names: Sequence[str],
    package_names: Sequence[str],
) -> Path:
    matrix_array = np.asarray(matrix, dtype=int)
    rows: list[dict[str, Any]] = []
    for row_index, coverage_name in enumerate(coverage_names):
        row: dict[str, Any] = {"coverage_index": row_index, "coverage_name": coverage_name}
        for package_index, package_name in enumerate(package_names):
            row[package_name] = int(matrix_array[row_index, package_index])
        rows.append(row)
    fieldnames = ["coverage_index", "coverage_name", *package_names]
    return _write_csv_rows(path, fieldnames, rows)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_run_id(
    algorithm: str,
    n: int,
    m: int,
    p: int | None,
    target: str | None,
    optimizer: str | None,
    lambda_token: str | None = None,
    compact_qubo: bool = False,
    fast_selene: bool = False,
) -> str:
    p_token = "na" if p is None else str(p)
    target_token = target if target is not None else "na"
    optimizer_token = optimizer if optimizer is not None else "na"
    cq_part = "_cq1" if compact_qubo else ""
    fs_part = "_fs1" if fast_selene else ""
    lambda_part = f"_{lambda_token}" if lambda_token else ""
    return f"{algorithm}_{target_token}_{optimizer_token}_n{n}_m{m}_p{p_token}{cq_part}{fs_part}{lambda_part}_{_utc_timestamp()}"


def _format_lambda_field(lambda_values: Sequence[float]) -> str:
    if not lambda_values:
        return ""
    if all(abs(value - lambda_values[0]) <= 1e-12 for value in lambda_values[1:]):
        return f"{float(lambda_values[0]):.17g}"
    return json.dumps([float(value) for value in lambda_values])


def _format_lambda_token_value(value: float) -> str:
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted.replace("-", "m").replace(".", "p")


def _build_lambda_token(lambda_values: Sequence[float]) -> str | None:
    if not lambda_values:
        return None
    compact_values = [_format_lambda_token_value(value) for value in lambda_values]
    if len(compact_values) == 1:
        return f"lam{compact_values[0]}"
    return f"lamx{len(compact_values)}_" + "-".join(compact_values)


def _relaxed_block_notes(blocks: Sequence[qaoa.QuboBlock]) -> list[str]:
    notes: list[str] = []
    for block in blocks:
        if not block.is_relaxed:
            continue
        notes.append(f"package {block.package_index + 1}: {block.constraint_budget_summary()}")
    return notes


def _build_feasibility_checker(problem: qaoa.BundlingProblem) -> Callable[[Sequence[int]], bool]:
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


def _qaoa_resource_fields(block: qaoa.QuboBlock, p: int) -> tuple[int, str, int]:
    _, zz_terms = qaoa.qubo_to_ising_pauli_coefficients(block.Q)
    return block.n_vars, "", int(p) * len(zz_terms)


def _extract_dqi_solution_matrix(
    workflow_result: travelers_dqi.DqiWorkflowResult,
    problem,
) -> tuple[np.ndarray, bool]:
    x_matrix = np.full((problem.N, problem.M), -1, dtype=int)
    all_packages_have_assignments = True
    for block in workflow_result.blocks:
        assignment = block.quantum_best_semantic_assignment
        if assignment is None:
            all_packages_have_assignments = False
            continue
        for coverage_index in range(problem.N):
            key = f"x_{coverage_index}"
            if key in assignment:
                x_matrix[coverage_index, block.package_index] = int(assignment[key])
            else:
                all_packages_have_assignments = False
    return x_matrix, all_packages_have_assignments


def _run_classical(
    n: int,
    m: int,
    seed: int,
    summaries_dir: Path,
    histories_dir: Path,
    solutions_dir: Path,
) -> dict[str, Any]:
    result = classical_baseline.solve_classical_baseline(n=n, m=m)
    run_id = _build_run_id("classical", n=n, m=m, p=None, target=None, optimizer=None)
    stem = run_id

    solution_path = _write_solution_matrix_csv(
        _path_for_csv(solutions_dir, f"{stem}_solution"),
        matrix=result["x_matrix"],
        coverage_names=result["coverage_names"],
        package_names=[package["package_name"] for package in result["packages"]],
    )
    history_path = _write_csv_rows(
        _path_for_csv(histories_dir, f"{stem}_history"),
        CLASSICAL_HISTORY_FIELDS,
        [
            {
                "run_id": run_id,
                "algorithm": "classical",
                "optimizer": "",
                "status": result["status"],
                "runtime_sec": float(result["solve_time_s"]),
                "profit": float(result["profit"]),
            }
        ],
    )

    summary_row = {
        "run_id": run_id,
        "algorithm": "classical",
        "optimizer": "",
        "seed": "",
        "N_local": int(result["n_coverages"]),
        "M_blocks": int(result["n_packages"]),
        "n_total": int(result["n_coverages"]) * int(result["n_packages"]),
        "p": "",
        "lambda": "",
        "runtime_sec": float(result["solve_time_s"]),
        "best_profit": float(result["profit"]),
        "classical_opt_profit": float(result["profit"]),
        "num_samples_total": "",
        "num_samples_feasible": "",
        "num_samples_postselected": "",
        "num_objective_evals": "",
        "num_qubits": "",
        "circuit_depth": "",
        "two_qubit_gate_count": "",
        "solution_path": _path_to_repo_relative_string(solution_path),
        "history_path": _path_to_repo_relative_string(history_path),
        "notes": f"classical ILP baseline; configured seed={seed} not used by solver",
    }

    summary_path = _write_csv_rows(
        _path_for_csv(summaries_dir, f"{stem}_summary"),
        SUMMARY_FIELDS,
        [summary_row],
    )
    _append_csv_row(RUN_SUMMARY_INDEX_PATH, SUMMARY_FIELDS, summary_row)
    print(f"Saved summary to {_path_to_repo_relative_string(summary_path)}")
    print(f"Saved solution to {_path_to_repo_relative_string(solution_path)}")
    print(f"Saved history to {_path_to_repo_relative_string(history_path)}")
    return summary_row


def _run_qaoa(
    n: int,
    m: int,
    p: int,
    seed: int,
    execution_target: str,
    optimizer: str,
    summaries_dir: Path,
    histories_dir: Path,
    solutions_dir: Path,
    generate_training_loss_plot: bool,
    compact_qubo: bool = False,
    *,
    shots: int | None = None,
    cobyla_maxiter: int | None = None,
    spsa_maxiter: int | None = None,
    fast_selene: bool = False,
    fast_selene_trials: int = 10,
    fast_selene_strategy: str = "random",
) -> dict[str, Any]:
    if optimizer not in {"cobyla", "spsa"}:
        raise ValueError("optimizer must be 'cobyla' or 'spsa'")
    if fast_selene:
        if execution_target != "selene":
            raise ValueError("--fast-selene requires --target selene")
        if fast_selene_strategy == "grid" and p != 1:
            raise ValueError("--fast-selene-strategy grid requires --p 1")
        if fast_selene_strategy not in {"random", "grid"}:
            raise ValueError("fast_selene_strategy must be 'random' or 'grid'")
        if fast_selene_strategy == "random" and fast_selene_trials < 1:
            raise ValueError("--fast-selene-trials must be at least 1")

    problem = qaoa.subsample_problem(qaoa.load_ltm_instance(qaoa.DATA_DIR), n_coverages=n, m_packages=m)
    blocks = [
        qaoa.build_qubo_block_for_package(problem, package_index, compact_penalties=compact_qubo)
        for package_index in range(problem.M)
    ]
    lambda_values = [block.penalty_weight for block in blocks]
    effective_optimizer = (
        ("grid_batch" if fast_selene_strategy == "grid" else "random_batch")
        if fast_selene
        else optimizer
    )
    run_id = _build_run_id(
        "qaoa",
        n=n,
        m=m,
        p=p,
        target=execution_target,
        optimizer=effective_optimizer,
        lambda_token=_build_lambda_token(lambda_values),
        compact_qubo=compact_qubo,
        fast_selene=fast_selene,
    )
    stem = run_id
    workflow_start = time.perf_counter()
    c_matrix = qaoa.make_c_matrix(problem)
    x_matrix = np.zeros((problem.N, problem.M), dtype=int)
    classical_opt = classical_baseline.solve_classical_baseline(n=n, m=m)
    is_feasible = _build_feasibility_checker(problem)

    num_samples_total = 0
    num_samples_feasible = 0
    num_objective_evals = 0
    history_rows: list[dict[str, Any]] = []
    plot_series: list[tuple[str, list[float], list[float]]] = []

    effective_shots_override = shots if shots is not None else (48 if fast_selene else None)

    session = None
    original_qaoa_optimizer = qaoa.OPTIMIZER
    original_selene_optimizer = qaoa_selene.OPTIMIZER
    original_qaoa_shots = qaoa.SHOTS
    original_selene_shots = qaoa_selene.SHOTS
    original_qaoa_cobyla = qaoa.COBYLA_MAXITER
    original_selene_cobyla = qaoa_selene.COBYLA_MAXITER
    original_qaoa_spsa = qaoa.SPSA_MAXITER
    original_selene_spsa = qaoa_selene.SPSA_MAXITER
    original_selene_parallel = qaoa_selene.MAX_PARALLEL_PACKAGES
    try:
        qaoa.OPTIMIZER = optimizer
        qaoa_selene.OPTIMIZER = optimizer
        if fast_selene:
            qaoa_selene.MAX_PARALLEL_PACKAGES = max(1, len(blocks))
        if effective_shots_override is not None:
            qaoa.SHOTS = int(effective_shots_override)
            qaoa_selene.SHOTS = int(effective_shots_override)
        if cobyla_maxiter is not None:
            qaoa.COBYLA_MAXITER = int(cobyla_maxiter)
            qaoa_selene.COBYLA_MAXITER = int(cobyla_maxiter)
        if spsa_maxiter is not None:
            qaoa.SPSA_MAXITER = int(spsa_maxiter)
            qaoa_selene.SPSA_MAXITER = int(spsa_maxiter)
        if execution_target == "selene":
            session = qaoa_selene.SeleneSession.create(project_name=qaoa_selene.NEXUS_PROJECT)

        use_parallel_selene = fast_selene and execution_target == "selene" and len(blocks) > 1
        hist_lock = threading.Lock()

        def on_eval_factory(package_name: str):
            def on_evaluation(
                callback_block: qaoa.QuboBlock,
                evaluation_index: int,
                value: float,
                best_objective: float,
                stats: qaoa.QaoaSampleStats,
                improved: bool,
            ) -> None:
                nonlocal num_samples_total, num_samples_feasible, num_objective_evals
                shots_total = int(sum(stats.bitstring_counts.values()))
                shots_feasible = _count_feasible_samples(stats.bitstring_counts, problem.N, is_feasible)
                row = {
                    "run_id": run_id,
                    "algorithm": "qaoa",
                    "execution_target": execution_target,
                    "optimizer": effective_optimizer,
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
                if use_parallel_selene:
                    with hist_lock:
                        num_samples_total += shots_total
                        num_samples_feasible += shots_feasible
                        num_objective_evals += 1
                        history_rows.append(row)
                else:
                    num_samples_total += shots_total
                    num_samples_feasible += shots_feasible
                    num_objective_evals += 1
                    history_rows.append(row)

            return on_evaluation

        def solve_package(
            package_index: int, block: qaoa.QuboBlock
        ) -> tuple[int, np.ndarray, tuple[str, list[float], list[float]] | None]:
            package_name = (
                problem.package_names[package_index]
                if problem.package_names and package_index < len(problem.package_names)
                else f"package {package_index + 1}"
            )
            on_evaluation = on_eval_factory(package_name)
            if execution_target == "selene":
                if session is None:
                    raise RuntimeError("Expected an active Selene session")
                if fast_selene:
                    result = qaoa_selene.optimize_block_batched_theta_search_selene(
                        session=session,
                        block=block,
                        p=p,
                        shots=qaoa_selene.SHOTS,
                        seed=seed + package_index * 1000,
                        n_trials=int(fast_selene_trials),
                        strategy=fast_selene_strategy,
                        evaluation_callback=on_evaluation,
                    )
                else:
                    result = qaoa_selene.optimize_block_selene(
                        session=session,
                        block=block,
                        p=p,
                        shots=qaoa_selene.SHOTS,
                        seed=seed + package_index * 1000,
                        evaluation_callback=on_evaluation,
                    )
            else:
                result = qaoa.optimize_block(
                    block,
                    p=p,
                    shots=qaoa.SHOTS,
                    seed=seed + package_index * 1000,
                    execution_target=execution_target,
                    evaluation_callback=on_evaluation,
                )
            coverage_bits = np.array([int(char) for char in result.stats.best_bitstring[: problem.N]], dtype=int)
            plot_entry: tuple[str, list[float], list[float]] | None = None
            if generate_training_loss_plot:
                plot_entry = (
                    package_name,
                    list(result.trace.objective_values),
                    list(result.trace.best_objective_values),
                )
            return package_index, coverage_bits, plot_entry

        if use_parallel_selene:
            outcomes: list[tuple[int, np.ndarray, tuple[str, list[float], list[float]] | None]] = []
            workers = min(qaoa_selene.MAX_PARALLEL_PACKAGES, len(blocks))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(solve_package, package_index, block)
                    for package_index, block in enumerate(blocks)
                ]
                for future in as_completed(futures):
                    outcomes.append(future.result())
            for package_index, coverage_bits, plot_entry in outcomes:
                x_matrix[:, package_index] = coverage_bits
                if plot_entry is not None:
                    plot_series.append(plot_entry)
        else:
            for package_index, block in enumerate(blocks):
                _, coverage_bits, plot_entry = solve_package(package_index, block)
                x_matrix[:, package_index] = coverage_bits
                if plot_entry is not None:
                    plot_series.append(plot_entry)
    finally:
        qaoa.OPTIMIZER = original_qaoa_optimizer
        qaoa_selene.OPTIMIZER = original_selene_optimizer
        qaoa.SHOTS = original_qaoa_shots
        qaoa_selene.SHOTS = original_selene_shots
        qaoa.COBYLA_MAXITER = original_qaoa_cobyla
        qaoa_selene.COBYLA_MAXITER = original_selene_cobyla
        qaoa.SPSA_MAXITER = original_qaoa_spsa
        qaoa_selene.SPSA_MAXITER = original_selene_spsa
        qaoa_selene.MAX_PARALLEL_PACKAGES = original_selene_parallel

    runtime_sec = time.perf_counter() - workflow_start
    best_profit = float(np.sum(c_matrix * x_matrix))
    num_qubits = ""
    circuit_depth = ""
    two_qubit_gate_count = ""
    if blocks:
        num_qubits, circuit_depth, two_qubit_gate_count = _qaoa_resource_fields(blocks[0], p)

    coverage_names = [coverage.name for coverage in problem.coverages]
    package_names = problem.package_names or [f"package {index + 1}" for index in range(problem.M)]
    solution_path = _write_solution_matrix_csv(
        _path_for_csv(solutions_dir, f"{stem}_solution"),
        matrix=x_matrix,
        coverage_names=coverage_names,
        package_names=package_names,
    )
    history_path = _write_csv_rows(
        _path_for_csv(histories_dir, f"{stem}_history"),
        QAOA_HISTORY_FIELDS,
        history_rows,
    )
    plot_path: Path | None = None
    if generate_training_loss_plot and plot_series:
        plot_path = save_training_loss_plot(
            plot_series,
            PLOTS_DIR / f"{stem}_training_loss.png",
            title="QAOA Training Loss",
        )

    notes_parts = [
        f"execution_target={execution_target}",
        f"optimizer={effective_optimizer}",
        "num_samples_feasible counts per-evaluation package-local samples against original coverage constraints",
        "two_qubit_gate_count is derived from generated QAOA zz_phase terms",
        "compiled circuit depth is unavailable from the current guppylang/hugr package API",
    ]
    if plot_path is not None:
        notes_parts.append(f"training_loss_plot={_path_to_repo_relative_string(plot_path)}")
    if len(blocks) > 1:
        notes_parts.append("lambda stores one penalty value per package-local block")
    relaxed_notes = _relaxed_block_notes(blocks)
    if relaxed_notes:
        notes_parts.append("constraint priority relaxation enabled to fit MAX_QUBITS")
        notes_parts.extend(relaxed_notes)
    if compact_qubo:
        notes_parts.append(
            "compact_qubo: optional/incompat/dependency penalties without slack (capacity still uses slack); "
            "see qaoa.build_qubo_block_for_package(compact_penalties=True)"
        )
    if fast_selene:
        notes_parts.append(
            "fast_selene: batched theta search (one Nexus execute per package) + parallel packages; "
            "not SPSA/COBYLA — for wall-clock smoke tests only"
        )

    summary_row = {
        "run_id": run_id,
        "algorithm": "qaoa",
        "optimizer": effective_optimizer,
        "seed": int(seed),
        "N_local": int(problem.N),
        "M_blocks": int(problem.M),
        "n_total": int(problem.N) * int(problem.M),
        "p": int(p),
        "lambda": _format_lambda_field(lambda_values),
        "runtime_sec": float(runtime_sec),
        "best_profit": best_profit,
        "classical_opt_profit": float(classical_opt["profit"]),
        "num_samples_total": int(num_samples_total),
        "num_samples_feasible": int(num_samples_feasible),
        "num_samples_postselected": "",
        "num_objective_evals": int(num_objective_evals),
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "two_qubit_gate_count": two_qubit_gate_count,
        "solution_path": _path_to_repo_relative_string(solution_path),
        "history_path": _path_to_repo_relative_string(history_path),
        "notes": "; ".join(notes_parts),
    }

    summary_path = _write_csv_rows(
        _path_for_csv(summaries_dir, f"{stem}_summary"),
        SUMMARY_FIELDS,
        [summary_row],
    )
    _append_csv_row(RUN_SUMMARY_INDEX_PATH, SUMMARY_FIELDS, summary_row)
    print(f"Saved summary to {_path_to_repo_relative_string(summary_path)}")
    print(f"Saved solution to {_path_to_repo_relative_string(solution_path)}")
    print(f"Saved history to {_path_to_repo_relative_string(history_path)}")
    if plot_path is not None:
        print(f"Saved plot to {_path_to_repo_relative_string(plot_path)}")
    return summary_row


def _run_dqi(
    n: int,
    m: int,
    seed: int,
    bp_mode: str,
    objective_mode: str,
    objective_scale: int,
    objective_target_max: int,
    ell: int,
    bp_iterations: int,
    try_quantum: bool,
    quantum_row_limit: int,
    summaries_dir: Path,
    histories_dir: Path,
    solutions_dir: Path,
) -> dict[str, Any]:
    if bp_mode not in {"BP1", "BP2"}:
        raise ValueError("bp_mode must be 'BP1' or 'BP2'")

    workflow_start = time.perf_counter()
    problem = travelers_dqi.subsample_problem(
        travelers_dqi.load_ltm_instance(travelers_dqi.DATA_DIR),
        n_coverages=n,
        n_packages=m,
    )
    workflow_result = travelers_dqi.solve_dqi_workflow(
        n=n,
        m=m,
        objective_mode=objective_mode,
        objective_scale=objective_scale,
        objective_target_max=objective_target_max,
        bp_mode=bp_mode,
        ell=ell,
        bp_iterations=bp_iterations,
        seed=seed,
        try_quantum=try_quantum,
        quantum_row_limit=quantum_row_limit,
    )
    runtime_sec = time.perf_counter() - workflow_start
    classical_opt = classical_baseline.solve_classical_baseline(n=n, m=m)

    run_id = _build_run_id(
        "dqi",
        n=n,
        m=m,
        p=workflow_result.ell,
        target="analytic",
        optimizer=bp_mode.lower(),
    )
    stem = run_id

    x_matrix, has_complete_solution = _extract_dqi_solution_matrix(workflow_result, problem)
    coverage_names = [coverage.name for coverage in problem.coverages]
    package_names = problem.package_names or [f"package {index + 1}" for index in range(problem.M)]
    solution_path = _write_solution_matrix_csv(
        _path_for_csv(solutions_dir, f"{stem}_solution"),
        matrix=x_matrix,
        coverage_names=coverage_names,
        package_names=package_names,
    )

    history_rows: list[dict[str, Any]] = []
    for block in workflow_result.blocks:
        history_rows.append(
            {
                "run_id": run_id,
                "algorithm": "dqi",
                "bp_mode": bp_mode,
                "package_index": int(block.package_index),
                "package_name": block.package_name,
                "objective_mode": workflow_result.objective_mode,
                "objective_lower_bound": int(block.objective_lower_bound),
                "expanded_num_vars": int(block.expanded_num_vars),
                "num_constraints": int(block.num_constraints),
                "B_rows": int(block.B_shape[0]),
                "B_cols": int(block.B_shape[1]),
                "ell": int(block.ell),
                "dqi_expected_f": float(block.dqi_expected_f) if block.dqi_expected_f is not None else "",
                "dqi_expected_s": float(block.dqi_expected_s) if block.dqi_expected_s is not None else "",
                "quantum_attempted": int(bool(block.quantum_attempted)),
                "quantum_ran": int(bool(block.quantum_ran)),
                "quantum_postselected_shots": (
                    int(block.quantum_postselected_shots)
                    if block.quantum_postselected_shots is not None
                    else ""
                ),
                "quantum_best_assignment": (
                    json.dumps(block.quantum_best_semantic_assignment, sort_keys=True)
                    if block.quantum_best_semantic_assignment is not None
                    else ""
                ),
                "note": block.note,
            }
        )
    history_path = _write_csv_rows(
        _path_for_csv(histories_dir, f"{stem}_history"),
        DQI_HISTORY_FIELDS,
        history_rows,
    )

    postselected_total = sum(
        int(block.quantum_postselected_shots)
        for block in workflow_result.blocks
        if block.quantum_postselected_shots is not None
    )
    objective_eval_count = len(workflow_result.blocks)

    best_profit: float | str = ""
    if has_complete_solution:
        c_matrix = np.array(workflow_result.c_im_float, dtype=float)
        mask = np.clip(x_matrix, 0, 1)
        best_profit = float(np.sum(c_matrix * mask))

    notes_parts = [
        f"bp_mode={bp_mode}",
        f"objective_mode={workflow_result.objective_mode}",
        f"bp_iterations={bp_iterations}",
        f"try_quantum={int(bool(try_quantum))}",
        "solution matrix uses -1 where no quantum semantic assignment was available",
        "analytic DQI estimates come from the patched package-local max-XOR workflow",
    ]
    if not has_complete_solution:
        notes_parts.append(
            "best_profit left blank because the current DQI run did not yield a complete package matrix",
        )

    summary_row = {
        "run_id": run_id,
        "algorithm": "dqi",
        "optimizer": bp_mode,
        "seed": int(seed),
        "N_local": int(problem.N),
        "M_blocks": int(problem.M),
        "n_total": int(problem.N) * int(problem.M),
        "p": int(workflow_result.ell),
        "lambda": "",
        "runtime_sec": float(runtime_sec),
        "best_profit": best_profit,
        "classical_opt_profit": float(classical_opt["profit"]),
        "num_samples_total": "",
        "num_samples_feasible": "",
        "num_samples_postselected": int(postselected_total),
        "num_objective_evals": int(objective_eval_count),
        "num_qubits": "",
        "circuit_depth": "",
        "two_qubit_gate_count": "",
        "solution_path": _path_to_repo_relative_string(solution_path),
        "history_path": _path_to_repo_relative_string(history_path),
        "notes": "; ".join(notes_parts),
    }

    summary_path = _write_csv_rows(
        _path_for_csv(summaries_dir, f"{stem}_summary"),
        SUMMARY_FIELDS,
        [summary_row],
    )
    _append_csv_row(RUN_SUMMARY_INDEX_PATH, SUMMARY_FIELDS, summary_row)
    print(f"Saved summary to {_path_to_repo_relative_string(summary_path)}")
    print(f"Saved solution to {_path_to_repo_relative_string(solution_path)}")
    print(f"Saved history to {_path_to_repo_relative_string(history_path)}")
    return summary_row


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default=ALGORITHM, choices=["classical", "qaoa", "dqi"])
    parser.add_argument("--n", type=int, default=N_COVERAGES)
    parser.add_argument("--m", type=int, default=M_PACKAGES)
    parser.add_argument("--p", type=int, default=P_DEPTH)
    parser.add_argument("--target", default=QAOA_EXECUTION_TARGET, choices=["local", "selene"])
    parser.add_argument("--optimizer", default=QAOA_OPTIMIZER, choices=["cobyla", "spsa"])
    parser.add_argument("--bp-mode", default=DQI_BP_MODE, choices=["BP1", "BP2"])
    parser.add_argument("--dqi-objective-mode", default=DQI_OBJECTIVE_MODE, choices=["compressed", "scaled"])
    parser.add_argument("--dqi-objective-scale", type=int, default=DQI_OBJECTIVE_SCALE)
    parser.add_argument("--dqi-objective-target-max", type=int, default=DQI_OBJECTIVE_TARGET_MAX)
    parser.add_argument("--dqi-ell", type=int, default=DQI_ELL)
    parser.add_argument("--dqi-bp-iterations", type=int, default=DQI_BP_ITERATIONS)
    parser.add_argument("--dqi-try-quantum", type=int, default=int(DQI_TRY_QUANTUM), choices=[0, 1])
    parser.add_argument("--dqi-quantum-row-limit", type=int, default=DQI_QUANTUM_ROW_LIMIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--compact-qubo",
        action="store_true",
        help=(
            "Fewer slack qubits (pairwise/quad penalties for optional/incompat/dependency); "
            "fits Selene for N=20. See HEURISTICS/COMPACT_QUBO.md."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override qaoa.SHOTS for this run only (default: module default). "
            "Lower values reduce Selene time at the cost of noisier objectives."
        ),
    )
    parser.add_argument(
        "--cobyla-maxiter",
        type=int,
        default=None,
        metavar="N",
        help="Override COBYLA maxiter for this run only (default: qaoa.COBYLA_MAXITER).",
    )
    parser.add_argument(
        "--spsa-maxiter",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override SPSA outer steps for this run only (default: qaoa.SPSA_MAXITER). "
            "Each step uses two circuit evaluations (theta+ and theta-)."
        ),
    )
    parser.add_argument(
        "--fast-selene",
        action="store_true",
        help=(
            "Selene only: run many random (or grid) QAOA angles in a single execute() per package, "
            "and solve packages in parallel. Ignores SPSA/COBYLA for execution; use for short wall-clock "
            "smoke tests on large N×M (see HEURISTICS/COMPACT_QUBO.md). For N×M below "
            f"{SELENE_FAST_SELENE_MIN_NM}, this flag is ignored unless --force-fast-selene. Requires --target selene."
        ),
    )
    parser.add_argument(
        "--force-fast-selene",
        action="store_true",
        help=(
            "Apply --fast-selene even when N×M is small (normally regular SPSA/COBYLA on Selene is used)."
        ),
    )
    parser.add_argument(
        "--fast-selene-trials",
        type=int,
        default=8,
        metavar="N",
        help=(
            "With --fast-selene --fast-selene-strategy random: number of random (gamma, beta) samples "
            "per package (default: 8). Ignored for grid strategy."
        ),
    )
    parser.add_argument(
        "--fast-selene-strategy",
        choices=["random", "grid"],
        default="random",
        help=(
            "With --fast-selene: random uniform angles (needs --fast-selene-trials) or a fixed 3x3 "
            "gamma/beta grid (requires --p 1)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    _ensure_output_dirs()

    if args.algorithm == "dqi":
        return _run_dqi(
            n=args.n,
            m=args.m,
            seed=args.seed,
            bp_mode=args.bp_mode,
            objective_mode=args.dqi_objective_mode,
            objective_scale=args.dqi_objective_scale,
            objective_target_max=args.dqi_objective_target_max,
            ell=args.dqi_ell,
            bp_iterations=args.dqi_bp_iterations,
            try_quantum=bool(args.dqi_try_quantum),
            quantum_row_limit=args.dqi_quantum_row_limit,
            summaries_dir=SUMMARIES_DIR,
            histories_dir=HISTORIES_DIR,
            solutions_dir=SOLUTIONS_DIR,
        )
    if args.algorithm == "classical":
        return _run_classical(
            n=args.n,
            m=args.m,
            seed=args.seed,
            summaries_dir=SUMMARIES_DIR,
            histories_dir=HISTORIES_DIR,
            solutions_dir=SOLUTIONS_DIR,
        )
    return _run_qaoa(
        n=args.n,
        m=args.m,
        p=args.p,
        seed=args.seed,
        execution_target=args.target,
        optimizer=args.optimizer,
        summaries_dir=SUMMARIES_DIR,
        histories_dir=HISTORIES_DIR,
        solutions_dir=SOLUTIONS_DIR,
        generate_training_loss_plot=GENERATE_TRAINING_LOSS_PLOT,
        compact_qubo=args.compact_qubo,
        shots=args.shots,
        cobyla_maxiter=args.cobyla_maxiter,
        spsa_maxiter=args.spsa_maxiter,
        fast_selene=bool(args.fast_selene),
        fast_selene_trials=args.fast_selene_trials,
        fast_selene_strategy=args.fast_selene_strategy,
    )


if __name__ == "__main__":
    main()
