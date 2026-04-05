"""Single-file QUBO + QAOA entrypoint for the Travelers insurance bundling problem."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Optimizer = Literal["none", "grid", "random"]


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
    num_packages: int = 2
    max_options_per_package: int = 4
    discount_factor: float = 0.15
    price_elasticity: float = -2.0
    compatibility_rules: list[CompatibilityRule] = field(default_factory=list)
    dependency_rules: list[DependencyRule] = field(default_factory=list)
    co_take_rates: dict[frozenset[str], float] = field(default_factory=dict)
    reference_price_factor: float = 2.0
    package_discounts: list[float] | None = None
    segment_affinity: np.ndarray | None = None
    price_sensitivity_beta: float = 0.0
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
        result: dict[str, list[int]] = {}
        for family_name, indices in self.families.items():
            if any(self.coverages[index].is_mandatory_in_family for index in indices):
                result[family_name] = indices
        return result

    @property
    def optional_families(self) -> dict[str, list[int]]:
        mandatory = set(self.mandatory_families)
        return {name: indices for name, indices in self.families.items() if name not in mandatory}

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
    coverage_offset: int
    penalty_weight: float
    constant_offset: float = 0.0

    @property
    def n_vars(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x: np.ndarray) -> float:
        bitvec = np.asarray(x, dtype=float).ravel()
        if bitvec.shape[0] != self.n_vars:
            raise ValueError(f"x length {bitvec.shape[0]} does not match n_vars={self.n_vars}")
        return float(bitvec @ self.Q @ bitvec) + float(self.constant_offset)


@dataclass
class QaoaSampleStats:
    n_qubits: int
    shots: int
    bitstring_counts: dict[str, int]
    best_bitstring: str
    best_qubo_energy: float
    constant_offset: float


@dataclass
class QaoaRunSummary:
    depth: int
    optimizer: Optimizer
    objective: float
    angles: dict[str, float]
    stats: QaoaSampleStats
    package_index: int
    package_name: str


def default_data_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "subprojects" / "will" / "Travelers" / "docs" / "data" / "YQH26_data"


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
    with open(
        data_path / "instance_incompatible_pairs.csv",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            compatibility_rules.append(
                CompatibilityRule(
                    coverage_i=row["coverage_a_name"],
                    coverage_j=row["coverage_b_name"],
                    compatible=False,
                )
            )

    coverage_name_to_idx = {coverage.name: index for index, coverage in enumerate(coverages)}
    affinity = np.ones((len(coverages), len(package_names)), dtype=float)
    with open(
        data_path / "instance_segment_affinity.csv",
        newline="",
        encoding="utf-8",
    ) as handle:
        for row in csv.DictReader(handle):
            coverage_name = row["coverage"]
            coverage_index = coverage_name_to_idx[coverage_name]
            for package_index, package_name in enumerate(package_names):
                affinity[coverage_index, package_index] = float(row[package_name])

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

    dependency_rules = [
        rule
        for rule in problem.dependency_rules
        if rule.requires in kept_names and rule.dependent in kept_names
    ]
    compatibility_rules = [
        rule
        for rule in problem.compatibility_rules
        if rule.coverage_i in kept_names and rule.coverage_j in kept_names
    ]

    package_discounts = (
        problem.package_discounts[:m_packages] if problem.package_discounts is not None else None
    )
    package_names = problem.package_names[:m_packages] if problem.package_names is not None else None
    affinity = (
        problem.segment_affinity[:n_coverages, :m_packages]
        if problem.segment_affinity is not None
        else None
    )

    return BundlingProblem(
        coverages=kept_coverages,
        num_packages=m_packages,
        max_options_per_package=problem.max_options_per_package,
        discount_factor=problem.discount_factor,
        price_elasticity=problem.price_elasticity,
        compatibility_rules=compatibility_rules,
        dependency_rules=dependency_rules,
        co_take_rates=problem.co_take_rates,
        reference_price_factor=problem.reference_price_factor,
        package_discounts=package_discounts,
        segment_affinity=affinity,
        price_sensitivity_beta=problem.price_sensitivity_beta,
        package_names=package_names,
    )


def make_c_matrix(problem: BundlingProblem) -> np.ndarray:
    """Peyton's objective coefficient matrix C[i, m], rewritten without PuLP."""
    c_matrix = np.zeros((problem.N, problem.M), dtype=float)
    beta = problem.price_sensitivity_beta
    for package_index in range(problem.M):
        discount = problem.get_discount(package_index)
        for coverage_index, coverage in enumerate(problem.coverages):
            affinity = problem.get_affinity(coverage_index, package_index)
            c_matrix[coverage_index, package_index] = (
                coverage.price
                * coverage.contribution_margin_pct
                * (1.0 - discount)
                * coverage.take_rate
                * affinity
                * (1.0 + beta * discount)
            )
    return c_matrix


def _max_margin_coeff(problem: BundlingProblem, package_index: int) -> float:
    return float(np.max(np.abs(make_c_matrix(problem)[:, package_index])))


def default_penalty_weight(problem: BundlingProblem, package_index: int, factor: float = 3.0) -> float:
    return factor * max(_max_margin_coeff(problem, package_index), 1.0)


def _add_squared_linear_penalty(
    Q: np.ndarray,
    coeffs: dict[int, float],
    rhs: float,
    penalty_weight: float,
) -> float:
    indices = sorted(coeffs)
    for i in indices:
        ai = coeffs[i]
        Q[i, i] += penalty_weight * (ai * ai - 2.0 * rhs * ai)
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            i = indices[left]
            j = indices[right]
            weight = penalty_weight * coeffs[i] * coeffs[j]
            Q[i, j] += weight
            Q[j, i] += weight
    return penalty_weight * rhs * rhs


def build_qubo_block_for_package(
    problem: BundlingProblem,
    package_index: int,
    penalty_weight: float | None = None,
) -> QuboBlock:
    if package_index < 0 or package_index >= problem.M:
        raise ValueError(f"package_index={package_index} is out of range for M={problem.M}")

    lam = (
        float(penalty_weight)
        if penalty_weight is not None
        else default_penalty_weight(problem, package_index)
    )

    N = problem.N
    K = problem.max_options_per_package
    slack_count = 0

    def alloc_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        indices = list(range(start, start + n_bits))
        slack_count += n_bits
        return indices

    for indices in problem.optional_families.values():
        if len(indices) > 1:
            alloc_slack(1)

    cap_slacks = max(1, int(math.ceil(math.log2(K + 1))))
    alloc_slack(cap_slacks)

    for rule in problem.compatibility_rules:
        if not rule.compatible:
            alloc_slack(1)

    for _rule in problem.dependency_rules:
        alloc_slack(1)

    n_slack = slack_count
    Q = np.zeros((N + n_slack, N + n_slack), dtype=float)
    constant_offset = 0.0

    c_matrix = make_c_matrix(problem)
    for coverage_index in range(N):
        Q[coverage_index, coverage_index] -= c_matrix[coverage_index, package_index]

    slack_count = 0

    def next_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        indices = list(range(start, start + n_bits))
        slack_count += n_bits
        return indices

    for indices in problem.mandatory_families.values():
        coeffs = {index: 1.0 for index in indices}
        constant_offset += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    for indices in problem.optional_families.values():
        if len(indices) <= 1:
            continue
        slack_index = next_slack(1)[0]
        coeffs = {index: 1.0 for index in indices}
        coeffs[slack_index] = 1.0
        constant_offset += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    cap_slack_indices = next_slack(cap_slacks)
    coeffs_capacity: dict[int, float] = {index: 1.0 for index in range(N)}
    for bit, slack_index in enumerate(cap_slack_indices):
        coeffs_capacity[slack_index] = float(2**bit)
    constant_offset += _add_squared_linear_penalty(Q, coeffs_capacity, float(K), lam)

    for rule in problem.compatibility_rules:
        if rule.compatible:
            continue
        i = problem.coverage_index(rule.coverage_i)
        j = problem.coverage_index(rule.coverage_j)
        slack_index = next_slack(1)[0]
        coeffs = {i: 1.0, j: 1.0, slack_index: 1.0}
        constant_offset += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    for rule in problem.dependency_rules:
        i = problem.coverage_index(rule.requires)
        j = problem.coverage_index(rule.dependent)
        slack_index = next_slack(1)[0]
        coeffs = {j: 1.0, i: -1.0, slack_index: 1.0}
        constant_offset += _add_squared_linear_penalty(Q, coeffs, 0.0, lam)

    if slack_count != n_slack:
        raise RuntimeError("Slack accounting mismatch while building QUBO")

    return QuboBlock(
        package_index=package_index,
        Q=Q,
        n_coverage=N,
        n_slack=n_slack,
        coverage_offset=0,
        penalty_weight=lam,
        constant_offset=constant_offset,
    )


def build_all_qubo_blocks(
    problem: BundlingProblem,
    penalty_weight: float | None = None,
) -> list[QuboBlock]:
    blocks: list[QuboBlock] = []
    for package_index in range(problem.M):
        package_penalty = (
            penalty_weight
            if penalty_weight is not None
            else default_penalty_weight(problem, package_index)
        )
        blocks.append(build_qubo_block_for_package(problem, package_index, package_penalty))
    return blocks


def qubo_to_ising_pauli_coefficients(Q: np.ndarray) -> tuple[float, np.ndarray, list[tuple[int, int, float]]]:
    Q = np.asarray(Q, dtype=float)
    Q = (Q + Q.T) * 0.5
    n = Q.shape[0]
    c_identity = 0.0
    c_z = np.zeros(n, dtype=float)
    zz_terms: list[tuple[int, int, float]] = []

    for i in range(n):
        c_identity += Q[i, i] / 2.0
        c_z[i] += -Q[i, i] / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            q_ij = Q[i, j]
            if q_ij == 0.0:
                continue
            c_identity += q_ij / 2.0
            c_z[i] += -q_ij / 2.0
            c_z[j] += -q_ij / 2.0
            zz_terms.append((i, j, q_ij / 2.0))

    return c_identity, c_z, zz_terms


def bruteforce_minimize_qubo(
    Q: np.ndarray,
    *,
    constant_offset: float = 0.0,
    max_n: int = 16,
) -> tuple[float, np.ndarray]:
    Q = np.asarray(Q, dtype=float)
    Q = (Q + Q.T) * 0.5
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(f"n={n} exceeds max_n={max_n} for brute-force minimization")

    best_energy = float("inf")
    best_x = np.zeros(n, dtype=float)
    for mask in range(1 << n):
        x = np.array([float((mask >> bit) & 1) for bit in range(n)], dtype=float)
        energy = float(x @ Q @ x) + float(constant_offset)
        if energy < best_energy - 1e-15:
            best_energy = energy
            best_x = x.copy()
    return best_energy, best_x


def mean_sample_energy(block: QuboBlock, stats: QaoaSampleStats) -> float:
    total_shots = sum(stats.bitstring_counts.values())
    if total_shots <= 0:
        return float("nan")

    total_energy = 0.0
    for bitstring, count in stats.bitstring_counts.items():
        x = np.array([float(int(char)) for char in bitstring], dtype=float)
        total_energy += float(count) * block.energy(x)
    return total_energy / float(total_shots)


def _angle_literal_from_radians(phi: float) -> str:
    return repr(2.0 * float(phi) / math.pi)


def _guppy_append_cost_layer(
    lines: list[str],
    n_qubits: int,
    c_z: np.ndarray,
    zz_terms: list[tuple[int, int, float]],
    gamma: float,
) -> None:
    for i, j, weight in zz_terms:
        if abs(weight) < 1e-15:
            continue
        literal = _angle_literal_from_radians(gamma * weight)
        lines.append(f"    cx(q{i}, q{j})")
        lines.append(f"    rz(q{j}, angle({literal}))")
        lines.append(f"    cx(q{i}, q{j})")

    for qubit in range(n_qubits):
        if abs(c_z[qubit]) < 1e-15:
            continue
        literal = _angle_literal_from_radians(gamma * float(c_z[qubit]))
        lines.append(f"    rz(q{qubit}, angle({literal}))")


def _guppy_append_mixer_layer(lines: list[str], n_qubits: int, beta_var: str) -> None:
    for qubit in range(n_qubits):
        lines.append(f"    rx(q{qubit}, {beta_var})")


def _build_guppy_p1_source(
    n_qubits: int,
    c_z: np.ndarray,
    zz_terms: list[tuple[int, int, float]],
    gamma: float,
    beta: float,
) -> str:
    beta_literal = _angle_literal_from_radians(beta)
    lines = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rx, rz",
        "",
        "@guppy",
        "def qaoa_p1_kernel() -> None:",
        f"    g_beta = angle({beta_literal})",
    ]

    for qubit in range(n_qubits):
        lines.append(f"    q{qubit} = qubit()")
    for qubit in range(n_qubits):
        lines.append(f"    h(q{qubit})")

    _guppy_append_cost_layer(lines, n_qubits, c_z, zz_terms, gamma)
    _guppy_append_mixer_layer(lines, n_qubits, "g_beta")

    for qubit in range(n_qubits):
        lines.append(f'    result("m{qubit}", measure(q{qubit}))')
    return "\n".join(lines) + "\n"


def _build_guppy_p2_source(
    n_qubits: int,
    c_z: np.ndarray,
    zz_terms: list[tuple[int, int, float]],
    gamma1: float,
    beta1: float,
    gamma2: float,
    beta2: float,
) -> str:
    beta1_literal = _angle_literal_from_radians(beta1)
    beta2_literal = _angle_literal_from_radians(beta2)
    lines = [
        "from guppylang import guppy",
        "from guppylang.std.angles import angle",
        "from guppylang.std.builtins import result",
        "from guppylang.std.quantum import cx, h, measure, qubit, rx, rz",
        "",
        "@guppy",
        "def qaoa_p2_kernel() -> None:",
        f"    g_beta1 = angle({beta1_literal})",
        f"    g_beta2 = angle({beta2_literal})",
    ]

    for qubit in range(n_qubits):
        lines.append(f"    q{qubit} = qubit()")
    for qubit in range(n_qubits):
        lines.append(f"    h(q{qubit})")

    _guppy_append_cost_layer(lines, n_qubits, c_z, zz_terms, gamma1)
    _guppy_append_mixer_layer(lines, n_qubits, "g_beta1")
    _guppy_append_cost_layer(lines, n_qubits, c_z, zz_terms, gamma2)
    _guppy_append_mixer_layer(lines, n_qubits, "g_beta2")

    for qubit in range(n_qubits):
        lines.append(f'    result("m{qubit}", measure(q{qubit}))')
    return "\n".join(lines) + "\n"


def _load_dynamic_module(path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generated Guppy module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_qaoa_sample_stats(
    block: QuboBlock,
    *,
    kernel,
    n_qubits: int,
    shots: int,
    seed: int,
) -> tuple[dict[str, int], str, float]:
    emulator = kernel.emulator(n_qubits=n_qubits).with_shots(int(shots)).with_seed(int(seed))
    result = emulator.run()

    def shot_to_str(shot) -> str:
        values = shot.as_dict()
        return "".join(str(int(values[f"m{qubit}"])) for qubit in range(n_qubits))

    counts = Counter(shot_to_str(shot) for shot in result.results)
    bitstring_counts = dict(counts.most_common())

    best_bitstring: str | None = None
    best_energy = float("inf")
    for bitstring in counts:
        x = np.array([float(int(char)) for char in bitstring], dtype=float)
        energy = block.energy(x)
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    if best_bitstring is None:
        raise RuntimeError("No QAOA samples were returned")
    return bitstring_counts, best_bitstring, best_energy


def run_qaoa_p1_on_block(
    block: QuboBlock,
    gamma: float,
    beta: float,
    *,
    shots: int = 512,
    seed: int = 0,
    max_qubits: int = 24,
) -> QaoaSampleStats:
    Q = np.asarray(block.Q, dtype=float)
    n_qubits = Q.shape[0]
    if n_qubits > max_qubits:
        raise ValueError(f"n_qubits={n_qubits} exceeds max_qubits={max_qubits}")

    _c_identity, c_z, zz_terms = qubo_to_ising_pauli_coefficients(Q)
    source = _build_guppy_p1_source(n_qubits, c_z, zz_terms, gamma, beta)

    fd, path = tempfile.mkstemp(suffix="_qaoa_p1_guppy.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(source)
        module = _load_dynamic_module(path, "qaoa_p1_dynamic")
        kernel = module.qaoa_p1_kernel
        kernel.check()
        bitstring_counts, best_bitstring, best_energy = _run_qaoa_sample_stats(
            block,
            kernel=kernel,
            n_qubits=n_qubits,
            shots=shots,
            seed=seed,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return QaoaSampleStats(
        n_qubits=n_qubits,
        shots=int(shots),
        bitstring_counts=bitstring_counts,
        best_bitstring=best_bitstring,
        best_qubo_energy=float(best_energy),
        constant_offset=float(block.constant_offset),
    )


def run_qaoa_p2_on_block(
    block: QuboBlock,
    gamma1: float,
    beta1: float,
    gamma2: float,
    beta2: float,
    *,
    shots: int = 512,
    seed: int = 0,
    max_qubits: int = 24,
) -> QaoaSampleStats:
    Q = np.asarray(block.Q, dtype=float)
    n_qubits = Q.shape[0]
    if n_qubits > max_qubits:
        raise ValueError(f"n_qubits={n_qubits} exceeds max_qubits={max_qubits}")

    _c_identity, c_z, zz_terms = qubo_to_ising_pauli_coefficients(Q)
    source = _build_guppy_p2_source(n_qubits, c_z, zz_terms, gamma1, beta1, gamma2, beta2)

    fd, path = tempfile.mkstemp(suffix="_qaoa_p2_guppy.py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(source)
        module = _load_dynamic_module(path, "qaoa_p2_dynamic")
        kernel = module.qaoa_p2_kernel
        kernel.check()
        bitstring_counts, best_bitstring, best_energy = _run_qaoa_sample_stats(
            block,
            kernel=kernel,
            n_qubits=n_qubits,
            shots=shots,
            seed=seed,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    return QaoaSampleStats(
        n_qubits=n_qubits,
        shots=int(shots),
        bitstring_counts=bitstring_counts,
        best_bitstring=best_bitstring,
        best_qubo_energy=float(best_energy),
        constant_offset=float(block.constant_offset),
    )


def _run_depth1(
    block: QuboBlock,
    *,
    gamma: float,
    beta: float,
    shots: int,
    seed: int,
    max_qubits: int,
) -> tuple[float, dict[str, float], QaoaSampleStats]:
    stats = run_qaoa_p1_on_block(
        block,
        gamma=gamma,
        beta=beta,
        shots=shots,
        seed=seed,
        max_qubits=max_qubits,
    )
    return mean_sample_energy(block, stats), {"gamma": gamma, "beta": beta}, stats


def _run_depth2(
    block: QuboBlock,
    *,
    gamma1: float,
    beta1: float,
    gamma2: float,
    beta2: float,
    shots: int,
    seed: int,
    max_qubits: int,
) -> tuple[float, dict[str, float], QaoaSampleStats]:
    stats = run_qaoa_p2_on_block(
        block,
        gamma1=gamma1,
        beta1=beta1,
        gamma2=gamma2,
        beta2=beta2,
        shots=shots,
        seed=seed,
        max_qubits=max_qubits,
    )
    return (
        mean_sample_energy(block, stats),
        {"gamma1": gamma1, "beta1": beta1, "gamma2": gamma2, "beta2": beta2},
        stats,
    )


def optimize_qaoa(
    block: QuboBlock,
    *,
    depth: int,
    optimizer: Optimizer,
    shots: int = 256,
    seed: int = 0,
    grid_points: int = 4,
    random_samples: int = 16,
    max_qubits: int = 24,
    gamma: float | None = None,
    beta: float | None = None,
    gamma1: float | None = None,
    beta1: float | None = None,
    gamma2: float | None = None,
    beta2: float | None = None,
) -> tuple[float, dict[str, float], QaoaSampleStats]:
    if depth not in {1, 2}:
        raise ValueError("depth must be 1 or 2")

    if optimizer == "none":
        if depth == 1:
            if gamma is None or beta is None:
                raise ValueError("depth=1 with optimizer=none requires gamma and beta")
            return _run_depth1(
                block,
                gamma=float(gamma),
                beta=float(beta),
                shots=shots,
                seed=seed,
                max_qubits=max_qubits,
            )
        if None in {gamma1, beta1, gamma2, beta2}:
            raise ValueError("depth=2 with optimizer=none requires gamma1, beta1, gamma2, beta2")
        return _run_depth2(
            block,
            gamma1=float(gamma1),
            beta1=float(beta1),
            gamma2=float(gamma2),
            beta2=float(beta2),
            shots=shots,
            seed=seed,
            max_qubits=max_qubits,
        )

    best_objective = float("inf")
    best_angles: dict[str, float] = {}
    best_stats: QaoaSampleStats | None = None

    if optimizer == "grid":
        if grid_points < 1:
            raise ValueError("grid_points must be at least 1")
        gamma_values = np.linspace(0.0, math.pi, grid_points, dtype=float)
        beta_values = np.linspace(0.0, math.pi, grid_points, dtype=float)
        if depth == 1:
            eval_index = 0
            for current_gamma in gamma_values:
                for current_beta in beta_values:
                    objective, angles, stats = _run_depth1(
                        block,
                        gamma=float(current_gamma),
                        beta=float(current_beta),
                        shots=shots,
                        seed=seed + eval_index,
                        max_qubits=max_qubits,
                    )
                    eval_index += 1
                    if objective < best_objective:
                        best_objective = objective
                        best_angles = angles
                        best_stats = stats
        else:
            eval_index = 0
            for current_gamma1 in gamma_values:
                for current_beta1 in beta_values:
                    for current_gamma2 in gamma_values:
                        for current_beta2 in beta_values:
                            objective, angles, stats = _run_depth2(
                                block,
                                gamma1=float(current_gamma1),
                                beta1=float(current_beta1),
                                gamma2=float(current_gamma2),
                                beta2=float(current_beta2),
                                shots=shots,
                                seed=seed + eval_index,
                                max_qubits=max_qubits,
                            )
                            eval_index += 1
                            if objective < best_objective:
                                best_objective = objective
                                best_angles = angles
                                best_stats = stats
    elif optimizer == "random":
        if random_samples < 1:
            raise ValueError("random_samples must be at least 1")
        rng = np.random.default_rng(seed)
        for eval_index in range(random_samples):
            if depth == 1:
                objective, angles, stats = _run_depth1(
                    block,
                    gamma=float(rng.uniform(0.0, math.pi)),
                    beta=float(rng.uniform(0.0, math.pi)),
                    shots=shots,
                    seed=seed + eval_index,
                    max_qubits=max_qubits,
                )
            else:
                objective, angles, stats = _run_depth2(
                    block,
                    gamma1=float(rng.uniform(0.0, math.pi)),
                    beta1=float(rng.uniform(0.0, math.pi)),
                    gamma2=float(rng.uniform(0.0, math.pi)),
                    beta2=float(rng.uniform(0.0, math.pi)),
                    shots=shots,
                    seed=seed + eval_index,
                    max_qubits=max_qubits,
                )
            if objective < best_objective:
                best_objective = objective
                best_angles = angles
                best_stats = stats
    else:
        raise ValueError(f"Unsupported optimizer {optimizer!r}")

    if best_stats is None:
        raise RuntimeError("QAOA optimizer did not produce any samples")
    return best_objective, best_angles, best_stats


def run_problem_instance(
    *,
    n: int,
    m: int,
    package_index: int = 0,
    depth: int = 1,
    optimizer: Optimizer = "grid",
    data_dir: str | Path | None = None,
    beta: float = 1.2,
    penalty_weight: float | None = None,
    shots: int = 256,
    seed: int = 0,
    grid_points: int = 4,
    random_samples: int = 16,
    max_qubits: int = 24,
    gamma: float | None = None,
    beta_angle: float | None = None,
    gamma1: float | None = None,
    beta1: float | None = None,
    gamma2: float | None = None,
    beta2: float | None = None,
) -> tuple[BundlingProblem, np.ndarray, QuboBlock, QaoaRunSummary]:
    problem = load_ltm_instance(data_dir or default_data_dir(), beta=beta)
    subproblem = subsample_problem(problem, n_coverages=n, m_packages=m)
    c_matrix = make_c_matrix(subproblem)
    block = build_qubo_block_for_package(subproblem, package_index, penalty_weight=penalty_weight)

    objective, angles, stats = optimize_qaoa(
        block,
        depth=depth,
        optimizer=optimizer,
        shots=shots,
        seed=seed,
        grid_points=grid_points,
        random_samples=random_samples,
        max_qubits=max_qubits,
        gamma=gamma,
        beta=beta_angle,
        gamma1=gamma1,
        beta1=beta1,
        gamma2=gamma2,
        beta2=beta2,
    )

    package_name = (
        subproblem.package_names[package_index]
        if subproblem.package_names is not None
        else f"package_{package_index}"
    )
    summary = QaoaRunSummary(
        depth=depth,
        optimizer=optimizer,
        objective=float(objective),
        angles=angles,
        stats=stats,
        package_index=package_index,
        package_name=package_name,
    )
    return subproblem, c_matrix, block, summary


def _require_guppy() -> None:
    try:
        import guppylang  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Running QAOA requires guppylang. Install it with: pip install guppylang numpy"
        ) from exc


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a QUBO block and run QAOA on it.")
    parser.add_argument("--n", type=int, required=True, help="Number of coverages to keep.")
    parser.add_argument("--m", type=int, required=True, help="Number of packages to keep.")
    parser.add_argument("--package-index", type=int, default=0, help="Package block to optimize.")
    parser.add_argument("--depth", type=int, choices=[1, 2], default=1, help="QAOA depth p.")
    parser.add_argument(
        "--optimizer",
        choices=["none", "grid", "random"],
        default="grid",
        help="How to choose QAOA angles.",
    )
    parser.add_argument("--grid-points", type=int, default=4, help="Points per angle for grid search.")
    parser.add_argument(
        "--random-samples",
        type=int,
        default=16,
        help="Number of random angle samples when optimizer=random.",
    )
    parser.add_argument("--shots", type=int, default=256, help="Number of Selene shots.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--beta-factor", type=float, default=1.2, help="Problem beta parameter.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Path to the LTM instance CSV directory.",
    )
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=None,
        help="Override lambda for QUBO penalties.",
    )
    parser.add_argument("--max-qubits", type=int, default=24, help="Safety limit for Guppy runs.")
    parser.add_argument("--gamma", type=float, default=None, help="Depth-1 gamma if optimizer=none.")
    parser.add_argument("--beta", type=float, default=None, help="Depth-1 beta if optimizer=none.")
    parser.add_argument("--gamma1", type=float, default=None, help="Depth-2 gamma1 if optimizer=none.")
    parser.add_argument("--beta1", type=float, default=None, help="Depth-2 beta1 if optimizer=none.")
    parser.add_argument("--gamma2", type=float, default=None, help="Depth-2 gamma2 if optimizer=none.")
    parser.add_argument("--beta2", type=float, default=None, help="Depth-2 beta2 if optimizer=none.")
    parser.add_argument(
        "--dump-qubo",
        action="store_true",
        help="Include the full QUBO matrix in the JSON output.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    _require_guppy()

    if args.n < 1 or args.m < 1:
        raise ValueError("n and m must be positive integers")

    subproblem, c_matrix, block, summary = run_problem_instance(
        n=args.n,
        m=args.m,
        package_index=args.package_index,
        depth=args.depth,
        optimizer=args.optimizer,
        data_dir=args.data_dir,
        beta=args.beta_factor,
        penalty_weight=args.penalty_weight,
        shots=args.shots,
        seed=args.seed,
        grid_points=args.grid_points,
        random_samples=args.random_samples,
        max_qubits=args.max_qubits,
        gamma=args.gamma,
        beta_angle=args.beta,
        gamma1=args.gamma1,
        beta1=args.beta1,
        gamma2=args.gamma2,
        beta2=args.beta2,
    )

    bruteforce = None
    if block.n_vars <= 16:
        energy, bitvec = bruteforce_minimize_qubo(
            block.Q,
            constant_offset=block.constant_offset,
            max_n=16,
        )
        bruteforce = {
            "energy": float(energy),
            "bitstring": "".join(str(int(bit)) for bit in bitvec.astype(int)),
        }

    output = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_coverages": subproblem.N,
        "m_packages": subproblem.M,
        "package_index": block.package_index,
        "package_name": summary.package_name,
        "coverages": [coverage.name for coverage in subproblem.coverages],
        "c_matrix": c_matrix.tolist(),
        "qubo": {
            "shape": list(block.Q.shape),
            "n_coverage": block.n_coverage,
            "n_slack": block.n_slack,
            "penalty_weight": float(block.penalty_weight),
            "constant_offset": float(block.constant_offset),
        },
        "qaoa": {
            "depth": summary.depth,
            "optimizer": summary.optimizer,
            "objective": float(summary.objective),
            "angles": summary.angles,
            "best_bitstring": summary.stats.best_bitstring,
            "best_qubo_energy": float(summary.stats.best_qubo_energy),
            "mean_qubo_energy": float(mean_sample_energy(block, summary.stats)),
            "shots": summary.stats.shots,
            "bitstring_counts": summary.stats.bitstring_counts,
        },
        "bruteforce_minimum": bruteforce,
    }
    if args.dump_qubo:
        output["qubo"]["matrix"] = block.Q.tolist()

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
