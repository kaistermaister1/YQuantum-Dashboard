"""Run classical or QAOA heuristics and save structured CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import classical_baseline
import qaoa
import qaoa_selene

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path(__file__).resolve().parent / "HEURISTICS"
SUMMARIES_DIR = OUTPUT_ROOT / "summaries"
HISTORIES_DIR = OUTPUT_ROOT / "histories"
SOLUTIONS_DIR = OUTPUT_ROOT / "solutions"
RUN_SUMMARY_INDEX_PATH = OUTPUT_ROOT / "run_summaries.csv"

# Top-level run config
ALGORITHM = "qaoa"  # "classical", "qaoa", or "dqi" when implemented
N_COVERAGES = 10
M_PACKAGES = 3
P_DEPTH = 1
QAOA_EXECUTION_TARGET = "selene"  # "local" or "selene"
QAOA_OPTIMIZER = "spsa"  # "cobyla" or "spsa"
SEED = 0

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


def _ensure_output_dirs() -> None:
    for directory in (OUTPUT_ROOT, SUMMARIES_DIR, HISTORIES_DIR, SOLUTIONS_DIR):
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
) -> str:
    p_token = "na" if p is None else str(p)
    target_token = target if target is not None else "na"
    optimizer_token = optimizer if optimizer is not None else "na"
    lambda_part = f"_{lambda_token}" if lambda_token else ""
    return f"{algorithm}_{target_token}_{optimizer_token}_n{n}_m{m}_p{p_token}{lambda_part}_{_utc_timestamp()}"


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
) -> dict[str, Any]:
    if optimizer not in {"cobyla", "spsa"}:
        raise ValueError("optimizer must be 'cobyla' or 'spsa'")

    problem = qaoa.subsample_problem(qaoa.load_ltm_instance(qaoa.DATA_DIR), n_coverages=n, m_packages=m)
    blocks = [qaoa.build_qubo_block_for_package(problem, package_index) for package_index in range(problem.M)]
    lambda_values = [block.penalty_weight for block in blocks]
    run_id = _build_run_id(
        "qaoa",
        n=n,
        m=m,
        p=p,
        target=execution_target,
        optimizer=optimizer,
        lambda_token=_build_lambda_token(lambda_values),
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

    session = None
    original_qaoa_optimizer = qaoa.OPTIMIZER
    original_selene_optimizer = qaoa_selene.OPTIMIZER
    try:
        qaoa.OPTIMIZER = optimizer
        qaoa_selene.OPTIMIZER = optimizer
        if execution_target == "selene":
            session = qaoa_selene.SeleneSession.create(project_name=qaoa_selene.NEXUS_PROJECT)

        for package_index, block in enumerate(blocks):
            package_name = (
                problem.package_names[package_index]
                if problem.package_names and package_index < len(problem.package_names)
                else f"package {package_index + 1}"
            )

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
                num_samples_total += shots_total
                num_samples_feasible += shots_feasible
                num_objective_evals += 1
                history_rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": "qaoa",
                        "execution_target": execution_target,
                        "optimizer": optimizer,
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

            if execution_target == "selene":
                if session is None:
                    raise RuntimeError("Expected an active Selene session")
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
            x_matrix[:, package_index] = coverage_bits
    finally:
        qaoa.OPTIMIZER = original_qaoa_optimizer
        qaoa_selene.OPTIMIZER = original_selene_optimizer

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

    notes_parts = [
        f"execution_target={execution_target}",
        f"optimizer={optimizer}",
        "num_samples_feasible counts per-evaluation package-local samples against original coverage constraints",
        "two_qubit_gate_count is derived from generated QAOA zz_phase terms",
        "compiled circuit depth is unavailable from the current guppylang/hugr package API",
    ]
    if len(blocks) > 1:
        notes_parts.append("lambda stores one penalty value per package-local block")

    summary_row = {
        "run_id": run_id,
        "algorithm": "qaoa",
        "optimizer": optimizer,
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
    return summary_row


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default=ALGORITHM, choices=["classical", "qaoa", "dqi"])
    parser.add_argument("--n", type=int, default=N_COVERAGES)
    parser.add_argument("--m", type=int, default=M_PACKAGES)
    parser.add_argument("--p", type=int, default=P_DEPTH)
    parser.add_argument("--target", default=QAOA_EXECUTION_TARGET, choices=["local", "selene"])
    parser.add_argument("--optimizer", default=QAOA_OPTIMIZER, choices=["cobyla", "spsa"])
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    _ensure_output_dirs()

    if args.algorithm == "dqi":
        raise NotImplementedError("DQI logging is not implemented yet.")
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
    )


if __name__ == "__main__":
    main()
