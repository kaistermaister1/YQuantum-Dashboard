"""Export QAOA grid frames for the Kai Raylib viewer."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SOLUTIONS import qaoa

GRID_PADDING = 1


@dataclass
class MatrixFrame:
    iteration: int
    package_index: int | None
    matrix: np.ndarray
    changed_mask: np.ndarray
    best_objective: float | None
    best_qubo_energy: float | None
    current_profit: float
    improved: bool
    note: str


@dataclass
class QaoaGridHistory:
    problem: qaoa.BundlingProblem
    c_matrix: np.ndarray
    frames: list[MatrixFrame]
    final_matrix: np.ndarray


def padded_lattice_shape(n_rows: int, n_cols: int, padding: int = GRID_PADDING) -> tuple[int, int]:
    """Return the viewer lattice shape when matrix entries live on interior intersections."""
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError("matrix dimensions must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return n_rows + 2 * padding, n_cols + 2 * padding


def matrix_index_to_lattice_index(row: int, col: int, padding: int = GRID_PADDING) -> tuple[int, int]:
    """Map an x(i,m) matrix index onto the padded lattice used by the Raylib viewer."""
    if row < 0 or col < 0:
        raise ValueError("matrix indices must be non-negative")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return row + padding, col + padding


def _package_label(problem: qaoa.BundlingProblem, package_index: int) -> str:
    if problem.package_names and 0 <= package_index < len(problem.package_names):
        return problem.package_names[package_index]
    return f"package {package_index + 1}"


def _matrix_profit(c_matrix: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.sum(c_matrix * matrix))


def run_qaoa_with_matrix_history(
    n: int = qaoa.N_COVERAGES,
    m: int = qaoa.M_PACKAGES,
    p: int = qaoa.P_DEPTH,
    execution_target: str = qaoa.EXECUTION_TARGET,
) -> QaoaGridHistory:
    """Run QAOA and capture every evaluated matrix iterate."""
    problem = qaoa.subsample_problem(qaoa.load_ltm_instance(qaoa.DATA_DIR), n_coverages=n, m_packages=m)
    c_matrix = qaoa.make_c_matrix(problem)
    blocks = [qaoa.build_qubo_block_for_package(problem, package_index) for package_index in range(problem.M)]

    current_matrix = np.zeros((problem.N, problem.M), dtype=int)
    previous_matrix = current_matrix.copy()
    frames = [
        MatrixFrame(
            iteration=0,
            package_index=None,
            matrix=current_matrix.copy(),
            changed_mask=np.zeros_like(current_matrix, dtype=bool),
            best_objective=None,
            best_qubo_energy=None,
            current_profit=0.0,
            improved=False,
            note="Initial white x(i,m) grid",
        )
    ]
    global_iteration = 0

    for package_index, block in enumerate(blocks):

        def on_evaluation(
            callback_block: qaoa.QuboBlock,
            evaluation_index: int,
            value: float,
            best_objective: float,
            stats: qaoa.QaoaSampleStats,
            improved: bool,
        ) -> None:
            nonlocal current_matrix, previous_matrix, global_iteration
            coverage_bits = np.array([int(char) for char in stats.best_bitstring[: problem.N]], dtype=int)
            next_matrix = current_matrix.copy()
            next_matrix[:, callback_block.package_index] = coverage_bits
            global_iteration += 1
            frames.append(
                MatrixFrame(
                    iteration=global_iteration,
                    package_index=callback_block.package_index,
                    matrix=next_matrix.copy(),
                    changed_mask=next_matrix != previous_matrix,
                    best_objective=float(best_objective),
                    best_qubo_energy=float(stats.best_qubo_energy),
                    current_profit=_matrix_profit(c_matrix, next_matrix),
                    improved=bool(improved),
                    note=f"{_package_label(problem, callback_block.package_index)} eval {evaluation_index}",
                )
            )
            current_matrix = next_matrix
            previous_matrix = next_matrix.copy()

        qaoa.optimize_block(
            block,
            p=p,
            shots=qaoa.SHOTS,
            seed=qaoa.SEED + package_index * 1000,
            execution_target=execution_target,
            evaluation_callback=on_evaluation,
        )
        previous_matrix = current_matrix.copy()

    return QaoaGridHistory(
        problem=problem,
        c_matrix=c_matrix,
        frames=frames,
        final_matrix=current_matrix.copy(),
    )


def _format_optional_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{float(value):.17g}"


def save_grid_history_text(history: QaoaGridHistory, output_path: str | Path) -> Path:
    """Write the history to a plain-text format consumed by the C++ Raylib viewer."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lattice_rows, lattice_cols = padded_lattice_shape(history.problem.N, history.problem.M)

    lines: list[str] = [
        "# qaoa_grid_history v1",
        (
            "# matrix points live on interior intersections of a padded lattice "
            f"(padding={GRID_PADDING}, lattice_rows={lattice_rows}, lattice_cols={lattice_cols})"
        ),
        "grid_history_v1",
        f"{history.problem.N} {history.problem.M} {len(history.frames)}",
    ]
    for coverage in history.problem.coverages:
        lines.append(f"COVERAGE\t{coverage.name}")
    for package_index in range(history.problem.M):
        lines.append(f"PACKAGE\t{_package_label(history.problem, package_index)}")

    lines.append("COEFFS")
    for row in range(history.problem.N):
        lines.append(" ".join(f"{float(history.c_matrix[row, col]):.17g}" for col in range(history.problem.M)))

    for frame in history.frames:
        package_index = -1 if frame.package_index is None else int(frame.package_index)
        lines.append(
            "FRAME "
            f"{frame.iteration} {package_index} {_format_optional_float(frame.best_objective)} "
            f"{_format_optional_float(frame.best_qubo_energy)} {frame.current_profit:.17g} "
            f"{1 if frame.improved else 0}"
        )
        lines.append(f"NOTE\t{frame.note}")
        for row in range(history.problem.N):
            lines.append(" ".join(str(int(frame.matrix[row, col])) for col in range(history.problem.M)))
        lines.append("CHANGE")
        for row in range(history.problem.N):
            lines.append(" ".join(str(int(frame.changed_mask[row, col])) for col in range(history.problem.M)))

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _default_output_path(n: int, m: int, p: int) -> Path:
    return Path(__file__).resolve().parent / "plots" / f"qaoa_grid_history_n{n}_m{m}_p{p}.txt"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export QAOA grid history for the Kai Raylib viewer.")
    parser.add_argument("--n", type=int, default=qaoa.N_COVERAGES, help="Number of coverages to keep.")
    parser.add_argument("--m", type=int, default=qaoa.M_PACKAGES, help="Number of packages to keep.")
    parser.add_argument("--p", type=int, default=qaoa.P_DEPTH, help="QAOA depth.")
    parser.add_argument(
        "--execution-target",
        default=qaoa.EXECUTION_TARGET,
        choices=["local", "selene"],
        help="Where to run the QAOA evaluations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="History text destination. Defaults to subprojects/kai/plots/.",
    )
    args = parser.parse_args(argv)

    history = run_qaoa_with_matrix_history(
        n=args.n,
        m=args.m,
        p=args.p,
        execution_target=args.execution_target,
    )
    output_path = args.output if args.output is not None else _default_output_path(args.n, args.m, args.p)
    saved_path = save_grid_history_text(history, output_path)
    print(saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
