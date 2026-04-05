"""Solve the Travelers bundling problem with the classical ILP baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
TRAVELERS_CODE_EXAMPLES_DIR = ROOT_DIR / "subprojects" / "will" / "Travelers" / "code_examples"
DATA_DIR = ROOT_DIR / "subprojects" / "will" / "Travelers" / "docs" / "data" / "YQH26_data"

if str(TRAVELERS_CODE_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(TRAVELERS_CODE_EXAMPLES_DIR))

from src.insurance_model import load_ltm_instance, solve_ilp, subsample_problem


N_COVERAGES = 10
M_PACKAGES = 4


def _solution_vector_to_matrix(solution_vector: list[int], n: int, m: int) -> list[list[int]]:
    vector = [int(value) for value in solution_vector]
    return [[vector[package_index * n + coverage_index] for package_index in range(m)] for coverage_index in range(n)]


def solve_classical_baseline(n: int = N_COVERAGES, m: int = M_PACKAGES) -> dict[str, Any]:
    full_problem = load_ltm_instance(DATA_DIR)
    if n < 1 or n > full_problem.N:
        raise ValueError(f"n must be between 1 and {full_problem.N}, got {n}")
    if m < 1 or m > full_problem.M:
        raise ValueError(f"m must be between 1 and {full_problem.M}, got {m}")

    problem = subsample_problem(full_problem, n_coverages=n, n_packages=m)

    start = time.perf_counter()
    result = solve_ilp(problem)
    solve_time_s = time.perf_counter() - start
    solution_vector = result["solution_vector"].tolist()
    x_matrix = _solution_vector_to_matrix(solution_vector, n=problem.N, m=problem.M)

    packages = []
    for package_index, selected_coverages in enumerate(result["packages"]):
        package_name = problem.package_names[package_index] if problem.package_names else f"Package {package_index + 1}"
        packages.append(
            {
                "package_index": package_index,
                "package_name": package_name,
                "coverages": selected_coverages,
            }
        )

    return {
        "n_coverages": problem.N,
        "n_packages": problem.M,
        "status": result["status"],
        "profit": float(result["objective"]),
        "solve_time_s": solve_time_s,
        "packages": packages,
        "coverage_names": [coverage.name for coverage in problem.coverages],
        "x_matrix": x_matrix,
        "solution_vector": solution_vector,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_COVERAGES, help="Number of coverages to keep from the LTM instance")
    parser.add_argument("--m", type=int, default=M_PACKAGES, help="Number of packages to keep from the LTM instance")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    result = solve_classical_baseline(n=args.n, m=args.m)

    if args.json:
        print(json.dumps(result, indent=2))
        return result

    print(f"Classical baseline for n={result['n_coverages']}, m={result['n_packages']}")
    print(f"Status: {result['status']}")
    print(f"Optimal profit: {result['profit']:.4f}")
    print(f"Solve time: {result['solve_time_s'] * 1000:.1f} ms")
    print("Coverage order:")
    print(result["coverage_names"])
    print("Selection matrix x[i, m]:")
    print(result["x_matrix"])
    print("Flat solution vector:")
    print(result["solution_vector"])
    print("Optimal bundles:")
    for package in result["packages"]:
        coverages = ", ".join(package["coverages"]) if package["coverages"] else "(empty)"
        print(f"  {package['package_name']}: {coverages}")

    return result


if __name__ == "__main__":
    main()
