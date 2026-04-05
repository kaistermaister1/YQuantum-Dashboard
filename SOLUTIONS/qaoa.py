"""Build Travelers package-local QUBOs and solve them with QAOA.

Set ``N_COVERAGES``, ``M_PACKAGES``, and ``P_DEPTH`` below, then run this file.
The script returns the binary solution matrix ``M`` with entries ``x[i, m]``.
"""

from __future__ import annotations

import csv
import contextlib
import importlib.util
import math
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from SOLUTIONS.qaoa_plots import save_training_loss_plot
except ModuleNotFoundError:
    from qaoa_plots import save_training_loss_plot


# Top-level choices
N_COVERAGES = 10
M_PACKAGES = 4
P_DEPTH = 1

# QAOA settings
OPTIMIZER = "cobyla"  # "cobyla" or "spsa"
# Lower shots speed up Selene round-trips; raise for quieter energy estimates (e.g. 256).
SHOTS = 128
SEED = 0
# Package-local blocks are N coverages + slack qubits (e.g. N=20 → ~39 vars).
MAX_QUBITS = 64
# Nexus Selene statevector execute() rejects programs above this (HTTP 400 from API).
# Local statevector sim also explodes in RAM (~2^n); see HEURISTICS/QAOA_SCALE_LIMITS.md.
SELENE_STATEVECTOR_MAX_QUBITS = 26
# Iteration caps: SPSA uses two circuit evaluations per outer step (theta+ / theta-).
# Selene wall time is often dominated by queue + job latency per execute, not shot count.
COBYLA_MAXITER = 24
SPSA_MAXITER = 12
PENALTY_SCALE = 3.0
EXECUTION_TARGET = "local"  # "local" or "selene"
NEXUS_PROJECT = "YQuantum QAOA"
SELENE_SIMULATOR = "statevector"
DATA_DIR = Path(__file__).resolve().parent.parent / "subprojects" / "will" / "Travelers" / "docs" / "data" / "YQH26_data"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def default_loss_plot_path(n: int, m: int, p: int) -> Path:
    return PLOTS_DIR / f"qaoa_training_loss_n{n}_m{m}_p{p}.png"


@dataclass
class InsuranceCoverage:
    name: str
    family: str
    price: float
    take_rate: float
    contribution_margin_pct: float = 0.3
    is_mandatory_in_family: bool = False


@dataclass
class CompatibilityRule:
    coverage_i: str
    coverage_j: str
    compatible: bool = True


@dataclass
class DependencyRule:
    requires: str
    dependent: str


@dataclass
class BundlingProblem:
    coverages: list[InsuranceCoverage]
    num_packages: int
    max_options_per_package: int
    discount_factor: float
    compatibility_rules: list[CompatibilityRule] = field(default_factory=list)
    dependency_rules: list[DependencyRule] = field(default_factory=list)
    package_discounts: list[float] | None = None
    segment_affinity: np.ndarray | None = None
    price_sensitivity_beta: float = 1.2
    package_names: list[str] | None = None

    @property
    def N(self) -> int:
        return len(self.coverages)

    @property
    def M(self) -> int:
        return self.num_packages

    @property
    def families(self) -> dict[str, list[int]]:
        families: dict[str, list[int]] = {}
        for index, coverage in enumerate(self.coverages):
            families.setdefault(coverage.family, []).append(index)
        return families

    @property
    def mandatory_families(self) -> dict[str, list[int]]:
        return {
            family: indices
            for family, indices in self.families.items()
            if any(self.coverages[index].is_mandatory_in_family for index in indices)
        }

    @property
    def optional_families(self) -> dict[str, list[int]]:
        mandatory = set(self.mandatory_families)
        return {family: indices for family, indices in self.families.items() if family not in mandatory}

    def get_discount(self, package_index: int) -> float:
        if self.package_discounts is not None:
            return float(self.package_discounts[package_index])
        return float(self.discount_factor)

    def get_affinity(self, coverage_index: int, package_index: int) -> float:
        if self.segment_affinity is not None:
            return float(self.segment_affinity[coverage_index, package_index])
        return 1.0

    def coverage_index(self, name: str) -> int:
        for index, coverage in enumerate(self.coverages):
            if coverage.name == name:
                return index
        raise ValueError(f"Coverage {name!r} not found")


@dataclass
class QuboBlock:
    package_index: int
    Q: np.ndarray
    n_coverage: int
    n_slack: int
    penalty_weight: float
    constant_offset: float = 0.0
    max_qubits: int | None = None
    slack_budget: int | None = None
    optional_family_constraints_encoded: int = 0
    optional_family_constraints_dropped: int = 0
    capacity_constraint_encoded: bool = True
    incompatibility_constraints_encoded: int = 0
    incompatibility_constraints_dropped: int = 0
    dependency_constraints_encoded: int = 0
    dependency_constraints_dropped: int = 0

    @property
    def n_vars(self) -> int:
        return int(self.Q.shape[0])

    @property
    def is_relaxed(self) -> bool:
        return any(
            (
                self.optional_family_constraints_dropped,
                int(not self.capacity_constraint_encoded),
                self.incompatibility_constraints_dropped,
                self.dependency_constraints_dropped,
            )
        )

    def constraint_budget_summary(self) -> str:
        if self.slack_budget is None:
            return "uncapped slack budget"
        if not self.is_relaxed:
            return (
                f"slack budget {self.slack_budget}, kept all prioritized constraints "
                f"({self.n_slack} slack used)"
            )
        parts = [
            f"slack budget {self.slack_budget}",
            (
                f"optional families kept {self.optional_family_constraints_encoded}/"
                f"{self.optional_family_constraints_encoded + self.optional_family_constraints_dropped}"
            ),
            f"capacity {'kept' if self.capacity_constraint_encoded else 'dropped'}",
            (
                f"dependencies kept {self.dependency_constraints_encoded}/"
                f"{self.dependency_constraints_encoded + self.dependency_constraints_dropped}"
            ),
            (
                f"incompatibilities kept {self.incompatibility_constraints_encoded}/"
                f"{self.incompatibility_constraints_encoded + self.incompatibility_constraints_dropped}"
            ),
        ]
        return ", ".join(parts)

    def energy(self, x: np.ndarray) -> float:
        bits = np.asarray(x, dtype=float).ravel()
        return float(bits @ self.Q @ bits) + float(self.constant_offset)


@dataclass
class QaoaSampleStats:
    bitstring_counts: dict[str, int]
    best_bitstring: str
    best_qubo_energy: float


@dataclass
class QaoaOptimizationTrace:
    objective_values: list[float] = field(default_factory=list)
    best_objective_values: list[float] = field(default_factory=list)
    best_bitstrings: list[str] = field(default_factory=list)
    best_sample_energies: list[float] = field(default_factory=list)
    improved_flags: list[bool] = field(default_factory=list)

    def record(
        self,
        value: float,
        best_stats: "QaoaSampleStats | None" = None,
        improved: bool = False,
    ) -> float:
        objective = float(value)
        self.objective_values.append(objective)
        if not self.best_objective_values:
            best_objective = objective
        else:
            best_objective = min(self.best_objective_values[-1], objective)
        self.best_objective_values.append(best_objective)
        self.best_bitstrings.append("" if best_stats is None else best_stats.best_bitstring)
        self.best_sample_energies.append(float("nan") if best_stats is None else float(best_stats.best_qubo_energy))
        self.improved_flags.append(bool(improved))
        return best_objective


@dataclass
class QaoaOptimizationResult:
    theta: np.ndarray
    stats: QaoaSampleStats
    trace: QaoaOptimizationTrace


QaoaEvaluationCallback = Callable[
    [QuboBlock, int, float, float, QaoaSampleStats, bool],
    None,
]


def _log(message: str) -> None:
    print(message, flush=True)


def _normalize_execution_target(execution_target: str) -> str:
    target = str(execution_target).strip().lower()
    if target in {"local", "selene"}:
        return target
    raise ValueError("execution_target must be 'local' or 'selene'")


def _read_measurement_value(raw_value: Any) -> int:
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, (int, np.integer)):
        return int(raw_value)
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"1", "true"}:
            return 1
        if lowered in {"0", "false"}:
            return 0
    raise TypeError(f"Unsupported measurement value type: {raw_value!r}")


def _shot_to_str(shot: Any, n_vars: int) -> str:
    if hasattr(shot, "as_dict"):
        values = shot.as_dict()
    elif hasattr(shot, "entries"):
        values = dict(shot.entries)
    elif isinstance(shot, dict):
        values = shot
    else:
        raise TypeError(f"Unsupported shot type: {type(shot)!r}")
    return "".join(str(_read_measurement_value(values[f"m{i}"])) for i in range(n_vars))


# Guppy's compiler is not safe under concurrent dynamic loads (e.g. parallel Selene packages).
_guppy_dynamic_kernel_lock = threading.Lock()


@contextlib.contextmanager
def _loaded_qaoa_kernel_from_source(source: str):
    with _guppy_dynamic_kernel_lock:
        fd, path = tempfile.mkstemp(suffix="_qaoa_dynamic.py", text=True)
        module_name = f"qaoa_dynamic_{uuid.uuid4().hex}"
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(source)
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise RuntimeError("Failed to load generated QAOA module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            kernel = module.qaoa_kernel
            kernel.check()
            yield kernel
        finally:
            sys.modules.pop(module_name, None)
            try:
                os.unlink(path)
            except OSError:
                pass

def _compile_qaoa_hugr_from_source(source: str):
    with _loaded_qaoa_kernel_from_source(source) as kernel:
        return kernel.compile()


def _run_qaoa_locally(block: QuboBlock, source: str, shots: int, seed: int) -> QaoaSampleStats:
    with _loaded_qaoa_kernel_from_source(source) as kernel:
        emulator = kernel.emulator(n_qubits=block.n_vars).with_shots(int(shots)).with_seed(int(seed))
        result = emulator.run()
    counts = Counter(_shot_to_str(shot, block.n_vars) for shot in result.results)
    best_bitstring = min(
        counts,
        key=lambda bitstring: block.energy(np.array([int(char) for char in bitstring], dtype=float)),
    )
    best_energy = block.energy(np.array([int(char) for char in best_bitstring], dtype=float))
    return QaoaSampleStats(dict(counts), best_bitstring, float(best_energy))


def _run_qaoa_on_selene(block: QuboBlock, source: str, shots: int, seed: int) -> QaoaSampleStats:
    if block.n_vars > SELENE_STATEVECTOR_MAX_QUBITS:
        raise ValueError(
            f"Block uses {block.n_vars} qubits; Nexus Selene statevector allows at most "
            f"{SELENE_STATEVECTOR_MAX_QUBITS} per program. Use local execution or a smaller subsample."
        )
    try:
        import qnexus as qnx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Selene execution requires the 'qnexus' package. Install it with 'pip install qnexus'."
        ) from exc

    qnx.login()
    project = qnx.projects.get_or_create(name=NEXUS_PROJECT)
    qnx.context.set_active_project(project)

    hugr_ref = qnx.hugr.upload(
        hugr_package=_compile_qaoa_hugr_from_source(source),
        name=f"QAOA package {block.package_index + 1} HUGR {time.time_ns()}",
    )
    simulator_name = SELENE_SIMULATOR.strip().lower()
    if simulator_name != "statevector":
        raise ValueError("SELENE_SIMULATOR must be 'statevector' for this QAOA workflow")
    config = qnx.models.SeleneConfig(
        n_qubits=block.n_vars,
        simulator=qnx.models.StatevectorSimulator(),
    )
    job_ref = qnx.start_execute_job(
        programs=[hugr_ref],
        n_shots=[int(shots)],
        backend_config=config,
        name=f"QAOA Selene package {block.package_index + 1} seed {seed} {time.time_ns()}",
    )
    qnx.jobs.wait_for(job_ref, timeout=None)
    result = qnx.jobs.results(job_ref)[0].download_result()
    shots_result = getattr(result, "results", None)
    if shots_result is None:
        raise RuntimeError("Unexpected Nexus Selene result format: missing shot results")
    counts = Counter(_shot_to_str(shot, block.n_vars) for shot in shots_result)
    best_bitstring = min(
        counts,
        key=lambda bitstring: block.energy(np.array([int(char) for char in bitstring], dtype=float)),
    )
    best_energy = block.energy(np.array([int(char) for char in best_bitstring], dtype=float))
    return QaoaSampleStats(dict(counts), best_bitstring, float(best_energy))


def _report_evaluation(
    block: QuboBlock,
    evaluation_index: int,
    value: float,
    best_objective: float,
    stats: QaoaSampleStats,
    elapsed_seconds: float,
    note: str = "",
) -> None:
    suffix = f" ({note})" if note else ""
    _log(
        f"[package {block.package_index + 1}] eval {evaluation_index}: "
        f"mean_energy={value:.6f}, best_mean={best_objective:.6f}, "
        f"best_sample={stats.best_qubo_energy:.6f}, elapsed={elapsed_seconds:.2f}s{suffix}"
    )


def _emit_evaluation_callback(
    evaluation_callback: QaoaEvaluationCallback | None,
    block: QuboBlock,
    evaluation_index: int,
    value: float,
    best_objective: float,
    stats: QaoaSampleStats,
    improved: bool,
) -> None:
    if evaluation_callback is None:
        return
    evaluation_callback(
        block,
        int(evaluation_index),
        float(value),
        float(best_objective),
        stats,
        bool(improved),
    )


def load_ltm_instance(data_dir: str | Path, beta: float = 1.2) -> BundlingProblem:
    data_path = Path(data_dir)

    coverages: list[InsuranceCoverage] = []
    with open(data_path / "instance_coverages.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            coverages.append(
                InsuranceCoverage(
                    name=row["name"],
                    family=row["family"],
                    price=float(row["price_usd"]),
                    take_rate=float(row["take_rate"]),
                    contribution_margin_pct=float(row["margin_pct"]),
                    is_mandatory_in_family=row["mandatory"].strip() == "True",
                )
            )

    package_names: list[str] = []
    package_discounts: list[float] = []
    max_options_list: list[int] = []
    with open(data_path / "instance_packages.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            package_names.append(row["name"])
            package_discounts.append(float(row["discount"]))
            max_options_list.append(int(row["max_options"]))

    dependency_rules: list[DependencyRule] = []
    with open(data_path / "instance_dependencies.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dependency_rules.append(
                DependencyRule(
                    requires=row["required_coverage_name"],
                    dependent=row["dependent_coverage_name"],
                )
            )

    compatibility_rules: list[CompatibilityRule] = []
    with open(data_path / "instance_incompatible_pairs.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            compatibility_rules.append(
                CompatibilityRule(
                    coverage_i=row["coverage_a_name"],
                    coverage_j=row["coverage_b_name"],
                    compatible=False,
                )
            )

    coverage_lookup = {coverage.name: index for index, coverage in enumerate(coverages)}
    affinity = np.ones((len(coverages), len(package_names)), dtype=float)
    with open(data_path / "instance_segment_affinity.csv", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            i = coverage_lookup[row["coverage"]]
            for m, package_name in enumerate(package_names):
                affinity[i, m] = float(row[package_name])

    return BundlingProblem(
        coverages=coverages,
        num_packages=len(package_names),
        max_options_per_package=min(max_options_list),
        discount_factor=sum(package_discounts) / len(package_discounts),
        compatibility_rules=compatibility_rules,
        dependency_rules=dependency_rules,
        package_discounts=package_discounts,
        segment_affinity=affinity,
        price_sensitivity_beta=beta,
        package_names=package_names,
    )


def subsample_problem(problem: BundlingProblem, n_coverages: int, m_packages: int) -> BundlingProblem:
    kept_coverages = problem.coverages[:n_coverages]
    kept_names = {coverage.name for coverage in kept_coverages}
    kept_dependencies = [
        rule
        for rule in problem.dependency_rules
        if rule.requires in kept_names and rule.dependent in kept_names
    ]
    kept_compatibility = [
        rule
        for rule in problem.compatibility_rules
        if rule.coverage_i in kept_names and rule.coverage_j in kept_names
    ]
    package_discounts = problem.package_discounts[:m_packages] if problem.package_discounts else None
    package_names = problem.package_names[:m_packages] if problem.package_names else None
    affinity = problem.segment_affinity[:n_coverages, :m_packages] if problem.segment_affinity is not None else None
    return BundlingProblem(
        coverages=kept_coverages,
        num_packages=m_packages,
        max_options_per_package=problem.max_options_per_package,
        discount_factor=problem.discount_factor,
        compatibility_rules=kept_compatibility,
        dependency_rules=kept_dependencies,
        package_discounts=package_discounts,
        segment_affinity=affinity,
        price_sensitivity_beta=problem.price_sensitivity_beta,
        package_names=package_names,
    )


def make_c_matrix(problem: BundlingProblem) -> np.ndarray:
    c_matrix = np.zeros((problem.N, problem.M), dtype=float)
    beta = problem.price_sensitivity_beta
    for m in range(problem.M):
        discount = problem.get_discount(m)
        for i, coverage in enumerate(problem.coverages):
            affinity = problem.get_affinity(i, m)
            c_matrix[i, m] = (
                coverage.price
                * coverage.contribution_margin_pct
                * (1.0 - discount)
                * coverage.take_rate
                * affinity
                * (1.0 + beta * discount)
            )
    return c_matrix


def default_penalty_weight(problem: BundlingProblem, package_index: int, factor: float = PENALTY_SCALE) -> float:
    c_matrix = make_c_matrix(problem)
    return factor * max(float(np.max(np.abs(c_matrix[:, package_index]))), 1.0)


def _add_squared_linear_penalty(Q: np.ndarray, coeffs: dict[int, float], rhs: float, lam: float) -> float:
    indices = sorted(coeffs)
    for i in indices:
        ai = coeffs[i]
        Q[i, i] += lam * (ai * ai - 2.0 * rhs * ai)
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            i = indices[left]
            j = indices[right]
            weight = lam * coeffs[i] * coeffs[j]
            Q[i, j] += weight
            Q[j, i] += weight
    return lam * rhs * rhs


def _add_pairwise_at_most_one_penalty(Q: np.ndarray, family_indices: Sequence[int], lam: float) -> None:
    """Optional-family style: at most one selected — lam * sum_{i<j} x_i x_j (no slack)."""
    idx = [int(i) for i in family_indices]
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            half = lam / 2.0
            Q[i, j] += half
            Q[j, i] += half


def _add_incompatible_pair_penalty(Q: np.ndarray, i: int, j: int, lam: float) -> None:
    """Incompatible pair cannot both be 1: lam * x_i x_j."""
    half = lam / 2.0
    Q[i, j] += half
    Q[j, i] += half


def _add_dependency_penalty_no_slack(Q: np.ndarray, requires_idx: int, dependent_idx: int, lam: float) -> None:
    """Dependent j requires prerequisite i: x_j <= x_i via lam * (x_j - x_i x_j) for binary x."""
    r, d = int(requires_idx), int(dependent_idx)
    Q[d, d] += lam
    half = -lam / 2.0
    Q[r, d] += half
    Q[d, r] += half


def build_qubo_block_for_package(
    problem: BundlingProblem,
    package_index: int,
    *,
    compact_penalties: bool = False,
) -> QuboBlock:
    """Build one package-local QUBO.

    If ``compact_penalties`` is True, optional at-most-one, incompatibility, and dependency
    constraints use pairwise / quadratic penalties **without** slack bits. Capacity still
    uses binary slack (ceil(log2(K+1)) bits). This cuts total qubits (e.g. N=20 → N+3 instead
    of N+19) so Selene statevector (≤26 qubits) and local emulation can run larger subsamples.
    The formulation is equivalent at high λ for feasible optima but is not identical to the
    slack+equality bridge used in the teaching notebooks.
    """
    lam = default_penalty_weight(problem, package_index)
    N = problem.N
    K = problem.max_options_per_package
    slack_budget = MAX_QUBITS - N
    if slack_budget < 0:
        raise ValueError(
            f"Block needs {N} coverage qubits before slack variables, which exceeds MAX_QUBITS={MAX_QUBITS}"
        )
    slack_count = 0

    def alloc_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        indices = list(range(start, start + n_bits))
        slack_count += n_bits
        return indices

    cap_slacks = max(1, int(math.ceil(math.log2(K + 1))))

    encoded_optional_families: list[list[int]] = []
    dropped_optional_family_count = 0
    encoded_dependency_rules: list[DependencyRule] = []
    dropped_dependency_count = 0
    encoded_incompatibility_rules: list[CompatibilityRule] = []
    dropped_incompatibility_count = 0
    capacity_constraint_encoded = False

    if compact_penalties:
        alloc_slack(cap_slacks)
        capacity_constraint_encoded = True
    else:

        def can_allocate(n_bits: int) -> bool:
            return slack_count + n_bits <= slack_budget

        for indices in problem.optional_families.values():
            if len(indices) <= 1:
                continue
            if can_allocate(1):
                alloc_slack(1)
                encoded_optional_families.append(indices)
            else:
                dropped_optional_family_count += 1

        capacity_constraint_encoded = can_allocate(cap_slacks)
        if capacity_constraint_encoded:
            alloc_slack(cap_slacks)

        for rule in problem.dependency_rules:
            if can_allocate(1):
                alloc_slack(1)
                encoded_dependency_rules.append(rule)
            else:
                dropped_dependency_count += 1

        for rule in problem.compatibility_rules:
            if rule.compatible:
                continue
            if can_allocate(1):
                alloc_slack(1)
                encoded_incompatibility_rules.append(rule)
            else:
                dropped_incompatibility_count += 1

    n_slack = slack_count
    Q = np.zeros((N + n_slack, N + n_slack), dtype=float)
    constant_offset = 0.0
    c_matrix = make_c_matrix(problem)
    for i in range(N):
        Q[i, i] -= c_matrix[i, package_index]

    slack_count = 0

    def next_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        indices = list(range(start, start + n_bits))
        slack_count += n_bits
        return indices

    for indices in problem.mandatory_families.values():
        constant_offset += _add_squared_linear_penalty(Q, {i: 1.0 for i in indices}, 1.0, lam)
    if compact_penalties:
        for indices in problem.optional_families.values():
            if len(indices) <= 1:
                continue
            _add_pairwise_at_most_one_penalty(Q, indices, lam)
    else:
        for indices in encoded_optional_families:
            s = next_slack(1)[0]
            coeffs = {i: 1.0 for i in indices}
            coeffs[s] = 1.0
            constant_offset += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    if capacity_constraint_encoded:
        cap_coeffs: dict[int, float] = {i: 1.0 for i in range(N)}
        for bit, s in enumerate(next_slack(cap_slacks)):
            cap_coeffs[s] = float(2**bit)
        constant_offset += _add_squared_linear_penalty(Q, cap_coeffs, float(K), lam)

    if compact_penalties:
        for rule in problem.compatibility_rules:
            if rule.compatible:
                continue
            i = problem.coverage_index(rule.coverage_i)
            j = problem.coverage_index(rule.coverage_j)
            _add_incompatible_pair_penalty(Q, i, j, lam)
        for rule in problem.dependency_rules:
            i = problem.coverage_index(rule.requires)
            j = problem.coverage_index(rule.dependent)
            _add_dependency_penalty_no_slack(Q, i, j, lam)
    else:
        for rule in encoded_incompatibility_rules:
            i = problem.coverage_index(rule.coverage_i)
            j = problem.coverage_index(rule.coverage_j)
            s = next_slack(1)[0]
            constant_offset += _add_squared_linear_penalty(Q, {i: 1.0, j: 1.0, s: 1.0}, 1.0, lam)

        for rule in encoded_dependency_rules:
            i = problem.coverage_index(rule.requires)
            j = problem.coverage_index(rule.dependent)
            s = next_slack(1)[0]
            constant_offset += _add_squared_linear_penalty(Q, {j: 1.0, i: -1.0, s: 1.0}, 0.0, lam)

    if compact_penalties:
        opt_enc = sum(1 for ind in problem.optional_families.values() if len(ind) > 1)
        opt_drop = 0
        inc_enc = sum(1 for r in problem.compatibility_rules if not r.compatible)
        inc_drop = 0
        dep_enc = len(problem.dependency_rules)
        dep_drop = 0
    else:
        opt_enc = len(encoded_optional_families)
        opt_drop = dropped_optional_family_count
        inc_enc = len(encoded_incompatibility_rules)
        inc_drop = dropped_incompatibility_count
        dep_enc = len(encoded_dependency_rules)
        dep_drop = dropped_dependency_count

    return QuboBlock(
        package_index=package_index,
        Q=Q,
        n_coverage=N,
        n_slack=n_slack,
        penalty_weight=lam,
        constant_offset=constant_offset,
        max_qubits=MAX_QUBITS,
        slack_budget=slack_budget,
        optional_family_constraints_encoded=opt_enc,
        optional_family_constraints_dropped=opt_drop,
        capacity_constraint_encoded=capacity_constraint_encoded,
        incompatibility_constraints_encoded=inc_enc,
        incompatibility_constraints_dropped=inc_drop,
        dependency_constraints_encoded=dep_enc,
        dependency_constraints_dropped=dep_drop,
    )


def qubo_to_ising_pauli_coefficients(Q: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    sym_q = (np.asarray(Q, dtype=float) + np.asarray(Q, dtype=float).T) * 0.5
    n = sym_q.shape[0]
    c_z = np.zeros(n, dtype=float)
    zz_terms: list[tuple[int, int, float]] = []
    for i in range(n):
        c_z[i] += -sym_q[i, i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            q_ij = sym_q[i, j]
            if q_ij == 0.0:
                continue
            c_z[i] += -q_ij / 2.0
            c_z[j] += -q_ij / 2.0
            zz_terms.append((i, j, q_ij / 2.0))
    return c_z, zz_terms


def _angle_literal_from_radians(phi: float) -> str:
    return repr(2.0 * float(phi) / math.pi)


def _append_cost_layer(lines: list[str], c_z: np.ndarray, zz_terms: list[tuple[int, int, float]], gamma: float) -> None:
    for i, j, weight in zz_terms:
        literal = _angle_literal_from_radians(gamma * weight)
        lines.append(f"    zz_phase(q{i}, q{j}, angle({literal}))")
    for i, weight in enumerate(c_z):
        if abs(weight) < 1e-15:
            continue
        literal = _angle_literal_from_radians(gamma * float(weight))
        lines.append(f"    rz(q{i}, angle({literal}))")


def _build_qaoa_source(c_z: np.ndarray, zz_terms: list[tuple[int, int, float]], gammas: np.ndarray, betas: np.ndarray) -> str:
    n_qubits = len(c_z)
    lines = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.qsystem import rz, zz_phase",
        "from guppylang.std.quantum import h, measure, qubit, rx",
        "",
        "@guppy",
        "def qaoa_kernel() -> None:",
    ]
    for layer, beta in enumerate(betas):
        lines.append(f"    beta_{layer} = angle({_angle_literal_from_radians(float(beta))})")
    for qubit_index in range(n_qubits):
        lines.append(f"    q{qubit_index} = qubit()")
    for qubit_index in range(n_qubits):
        lines.append(f"    h(q{qubit_index})")
    for layer, gamma in enumerate(gammas):
        _append_cost_layer(lines, c_z, zz_terms, float(gamma))
        for qubit_index in range(n_qubits):
            lines.append(f"    rx(q{qubit_index}, beta_{layer})")
    for qubit_index in range(n_qubits):
        lines.append(f'    result("m{qubit_index}", measure(q{qubit_index}))')
    return "\n".join(lines) + "\n"


def _run_qaoa_on_block(
    block: QuboBlock,
    gammas: np.ndarray,
    betas: np.ndarray,
    shots: int,
    seed: int,
    execution_target: str = EXECUTION_TARGET,
) -> QaoaSampleStats:
    if block.n_vars > MAX_QUBITS:
        raise ValueError(f"Block uses {block.n_vars} qubits, which exceeds MAX_QUBITS={MAX_QUBITS}")
    c_z, zz_terms = qubo_to_ising_pauli_coefficients(block.Q)
    source = _build_qaoa_source(c_z, zz_terms, gammas, betas)
    target = _normalize_execution_target(execution_target)
    if target == "local":
        return _run_qaoa_locally(block, source, shots=shots, seed=seed)
    if block.n_vars > SELENE_STATEVECTOR_MAX_QUBITS:
        raise ValueError(
            f"Block uses {block.n_vars} qubits; Nexus Selene statevector allows at most "
            f"{SELENE_STATEVECTOR_MAX_QUBITS} per program. Use execution_target='local' for "
            f"larger blocks, or reduce n (coverages per package)."
        )
    return _run_qaoa_on_selene(block, source, shots=shots, seed=seed)


def _mean_sample_energy(block: QuboBlock, stats: QaoaSampleStats) -> float:
    total = sum(stats.bitstring_counts.values())
    energy_sum = 0.0
    for bitstring, count in stats.bitstring_counts.items():
        x = np.array([int(char) for char in bitstring], dtype=float)
        energy_sum += float(count) * block.energy(x)
    return energy_sum / float(total)


def _split_theta(theta: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    return theta[:p], theta[p:]


def _clip_theta(theta: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(theta, dtype=float), 0.0, math.pi)


def optimize_block_cobyla(
    block: QuboBlock,
    p: int,
    shots: int,
    seed: int,
    execution_target: str = EXECUTION_TARGET,
    evaluation_callback: QaoaEvaluationCallback | None = None,
) -> QaoaOptimizationResult:
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, math.pi, size=2 * p)
    best_theta = theta0.copy()
    best_stats: QaoaSampleStats | None = None
    best_objective = float("inf")
    eval_index = 0
    trace = QaoaOptimizationTrace()

    _log(
        f"[package {block.package_index + 1}] Starting COBYLA on "
        f"{block.n_vars} qubits ({block.n_coverage} coverage + {block.n_slack} slack), "
        f"shots={shots}, maxiter={COBYLA_MAXITER}"
    )

    def objective(theta: np.ndarray) -> float:
        nonlocal best_theta, best_stats, best_objective, eval_index
        evaluation_number = eval_index + 1
        clipped = _clip_theta(theta)
        gammas, betas = _split_theta(clipped, p)
        evaluation_start = time.perf_counter()
        stats = _run_qaoa_on_block(
            block,
            gammas,
            betas,
            shots=shots,
            seed=seed + eval_index,
            execution_target=execution_target,
        )
        value = _mean_sample_energy(block, stats)
        elapsed_seconds = time.perf_counter() - evaluation_start
        eval_index += 1
        note = ""
        improved = False
        if value < best_objective:
            best_objective = value
            best_theta = clipped.copy()
            best_stats = stats
            note = "new best"
            improved = True
        trace.record(value, best_stats=best_stats, improved=improved)
        _emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number,
            value,
            best_objective,
            stats,
            improved,
        )
        _report_evaluation(
            block,
            evaluation_number,
            value,
            best_objective,
            stats,
            elapsed_seconds,
            note=note,
        )
        return float(value)

    minimize(objective, theta0, method="COBYLA", options={"maxiter": int(COBYLA_MAXITER)})
    if best_stats is None:
        raise RuntimeError("COBYLA did not produce any QAOA evaluations")
    _log(
        f"[package {block.package_index + 1}] COBYLA finished after "
        f"{len(trace.objective_values)} evaluations with best_mean={best_objective:.6f}"
    )
    return QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)


def optimize_block_spsa(
    block: QuboBlock,
    p: int,
    shots: int,
    seed: int,
    execution_target: str = EXECUTION_TARGET,
    evaluation_callback: QaoaEvaluationCallback | None = None,
) -> QaoaOptimizationResult:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, math.pi, size=2 * p)
    best_theta = theta.copy()
    best_stats: QaoaSampleStats | None = None
    best_objective = float("inf")
    eval_index = 0
    trace = QaoaOptimizationTrace()
    a = 0.15
    c = 0.12
    alpha = 0.602
    gamma_spsa = 0.101
    stability_A = 10.0

    _log(
        f"[package {block.package_index + 1}] Starting SPSA on "
        f"{block.n_vars} qubits ({block.n_coverage} coverage + {block.n_slack} slack), "
        f"shots={shots}, maxiter={SPSA_MAXITER}"
    )

    for step in range(SPSA_MAXITER):
        ak = a / ((step + 1 + stability_A) ** alpha)
        ck = c / ((step + 1) ** gamma_spsa)
        delta = rng.choice([-1.0, 1.0], size=2 * p)

        theta_plus = _clip_theta(theta + ck * delta)
        theta_minus = _clip_theta(theta - ck * delta)

        g_plus, b_plus = _split_theta(theta_plus, p)
        evaluation_number = eval_index + 1
        plus_start = time.perf_counter()
        stats_plus = _run_qaoa_on_block(
            block,
            g_plus,
            b_plus,
            shots=shots,
            seed=seed + eval_index,
            execution_target=execution_target,
        )
        value_plus = _mean_sample_energy(block, stats_plus)
        plus_elapsed = time.perf_counter() - plus_start
        eval_index += 1
        improved_plus = False
        if value_plus < best_objective:
            best_objective = value_plus
            best_theta = theta_plus.copy()
            best_stats = stats_plus
            improved_plus = True
        best_plus = trace.record(value_plus, best_stats=best_stats, improved=improved_plus)
        _emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number,
            value_plus,
            best_plus,
            stats_plus,
            improved_plus,
        )
        _report_evaluation(
            block,
            evaluation_number,
            value_plus,
            best_plus,
            stats_plus,
            plus_elapsed,
            note=f"SPSA step {step + 1} (+){' new best' if improved_plus else ''}",
        )

        g_minus, b_minus = _split_theta(theta_minus, p)
        evaluation_number_minus = eval_index + 1
        minus_start = time.perf_counter()
        stats_minus = _run_qaoa_on_block(
            block,
            g_minus,
            b_minus,
            shots=shots,
            seed=seed + eval_index,
            execution_target=execution_target,
        )
        value_minus = _mean_sample_energy(block, stats_minus)
        minus_elapsed = time.perf_counter() - minus_start
        eval_index += 1

        gradient = ((value_plus - value_minus) / (2.0 * ck)) * (1.0 / delta)
        theta = _clip_theta(theta - ak * gradient)

        improved_minus = False
        if value_minus < best_objective:
            best_objective = value_minus
            best_theta = theta_minus.copy()
            best_stats = stats_minus
            improved_minus = True
        best_minus = trace.record(value_minus, best_stats=best_stats, improved=improved_minus)
        _emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number_minus,
            value_minus,
            best_minus,
            stats_minus,
            improved_minus,
        )
        _report_evaluation(
            block,
            evaluation_number_minus,
            value_minus,
            best_minus,
            stats_minus,
            minus_elapsed,
            note=f"SPSA step {step + 1} (-){' new best' if improved_minus else ''}",
        )

    if best_stats is None:
        raise RuntimeError("SPSA did not produce any QAOA evaluations")
    _log(
        f"[package {block.package_index + 1}] SPSA finished after "
        f"{len(trace.objective_values)} evaluations with best_mean={best_objective:.6f}"
    )
    return QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)


def optimize_block(
    block: QuboBlock,
    p: int,
    shots: int,
    seed: int,
    execution_target: str = EXECUTION_TARGET,
    evaluation_callback: QaoaEvaluationCallback | None = None,
) -> QaoaOptimizationResult:
    if OPTIMIZER == "cobyla":
        return optimize_block_cobyla(
            block,
            p=p,
            shots=shots,
            seed=seed,
            execution_target=execution_target,
            evaluation_callback=evaluation_callback,
        )
    if OPTIMIZER == "spsa":
        return optimize_block_spsa(
            block,
            p=p,
            shots=shots,
            seed=seed,
            execution_target=execution_target,
            evaluation_callback=evaluation_callback,
        )
    raise ValueError("OPTIMIZER must be 'cobyla' or 'spsa'")


def solve_qaoa_matrix(
    n: int = N_COVERAGES,
    m: int = M_PACKAGES,
    p: int = P_DEPTH,
    plot_output_path: str | Path | None = None,
    execution_target: str = EXECUTION_TARGET,
    compact_penalties: bool = False,
) -> np.ndarray:
    workflow_start = time.perf_counter()
    resolved_plot_output_path = (
        Path(plot_output_path) if plot_output_path is not None else default_loss_plot_path(n=n, m=m, p=p)
    )
    resolved_execution_target = _normalize_execution_target(execution_target)
    _log(f"[workflow] Loading instance data from {DATA_DIR}")
    problem = subsample_problem(load_ltm_instance(DATA_DIR), n_coverages=n, m_packages=m)
    blocks = [
        build_qubo_block_for_package(problem, package_index, compact_penalties=compact_penalties)
        for package_index in range(problem.M)
    ]
    x_matrix = np.zeros((problem.N, problem.M), dtype=int)
    plot_series: list[tuple[str, list[float], list[float]]] = []

    _log(
        f"[workflow] Solving {len(blocks)} package-local QUBO blocks "
        f"(N={problem.N} coverages, M={problem.M} packages, p={p}, optimizer={OPTIMIZER}, "
        f"execution_target={resolved_execution_target})"
    )
    if resolved_execution_target == "selene":
        _log(
            "[workflow] Selene mode submits Nexus jobs for each QAOA evaluation, "
            "so optimization can be much slower than local emulation."
        )

    for package_index, block in enumerate(blocks):
        block_start = time.perf_counter()
        block_summary = (
            f"[workflow] Package {package_index + 1}/{len(blocks)} block summary: "
            f"{block.n_vars} qubits = {block.n_coverage} coverage + {block.n_slack} slack"
        )
        if block.is_relaxed:
            block_summary += f" ({block.constraint_budget_summary()})"
        _log(block_summary)
        result = optimize_block(
            block,
            p=p,
            shots=SHOTS,
            seed=SEED + package_index * 1000,
            execution_target=resolved_execution_target,
        )
        coverage_bits = np.array([int(char) for char in result.stats.best_bitstring[:problem.N]], dtype=int)
        x_matrix[:, package_index] = coverage_bits
        plot_series.append(
            (
                f"package {package_index + 1}",
                result.trace.objective_values,
                result.trace.best_objective_values,
            )
        )
        _log(
            f"[workflow] Package {package_index + 1} complete in "
            f"{time.perf_counter() - block_start:.2f}s with best bitstring "
            f"{result.stats.best_bitstring[:problem.N]} and sampled QUBO energy "
            f"{result.stats.best_qubo_energy:.6f}"
        )

    if plot_series:
        plot_path = save_training_loss_plot(plot_series, resolved_plot_output_path)
        _log(f"[workflow] Saved QAOA training loss plot to {plot_path}")

    _log(f"[workflow] Finished all package-local subproblems in {time.perf_counter() - workflow_start:.2f}s")

    return x_matrix


def main() -> np.ndarray:
    return solve_qaoa_matrix(n=N_COVERAGES, m=M_PACKAGES, p=P_DEPTH, execution_target=EXECUTION_TARGET)


if __name__ == "__main__":
    M = main()
    print(M)
