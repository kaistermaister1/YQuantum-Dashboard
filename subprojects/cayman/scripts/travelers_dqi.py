"""Travelers -> patched DQI max-XOR workflow.

This script mirrors the package-local decomposition used in `SOLUTIONS/qaoa.py`,
but swaps in a patched version of the DQI repo's ILP -> max-XOR conversion.

What it does:
1. Loads a Travelers instance and slices it to `n` coverages and `m` packages.
2. Builds one package-local ILP block per package.
3. Integerizes the package objective.
4. Lifts negative dependency rows into positive-coefficient rows with complement variables.
5. Preserves row senses (`=`, `<=`, `>=`) when reducing to max-XOR.
6. Runs the DQI repo's analytic evaluator on each max-XOR block.
7. Tries full quantum simulation only when the reduced block is genuinely tiny.

Notes:
- Exact integer objectives can make the DQI reduction explode in size very quickly.
- The default "compressed" objective mode is for a runnable DQI demo, not an exact
  reproduction of the original floating-point Travelers objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
from typing import Any
import warnings

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DQI_ROOT = REPO_ROOT / "subprojects" / "kai" / "dqi-main"
TRAVELERS_ROOT = REPO_ROOT / "subprojects" / "will" / "Travelers"
DATA_DIR = TRAVELERS_ROOT / "docs" / "data" / "YQH26_data"

for path in (str(DQI_ROOT), str(TRAVELERS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from code_examples.src.insurance_model import load_ltm_instance, subsample_problem  # noqa: E402
from pipelines.DQI_classical import expected_constrains_DQI  # noqa: E402
from pipelines.belief_propagations import (  # noqa: E402
    belief_propagation_gallager,
    belief_propagation_ldpc,
)
from pipelines.generate_B import generate_F2_instance, systematic_form_G  # noqa: E402


DEFAULT_N_COVERAGES = 4
DEFAULT_M_PACKAGES = 2
DEFAULT_OBJECTIVE_MODE = "compressed"  # "scaled" or "compressed"
DEFAULT_OBJECTIVE_SCALE = 1
DEFAULT_OBJECTIVE_TARGET_MAX = 3
DEFAULT_BP_MODE = "BP2"  # "BP1" or "BP2"
DEFAULT_ELL = 1
DEFAULT_BP_ITERATIONS = 2
DEFAULT_BETA_MODE = "half_sum"  # "half_sum" or "zero"
DEFAULT_SEED = 0
DEFAULT_QUANTUM_ROW_LIMIT = 24
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("travelers_dqi_output.json")


@dataclass
class PositiveIntegerIlp:
    package_index: int
    package_name: str
    objective_float: list[float]
    objective_int: list[int]
    semantic_var_names: list[str]
    original_var_names: list[str]
    complement_pairs: list[tuple[str, str]]
    A: np.ndarray
    b: np.ndarray
    senses: list[str]


@dataclass
class IlpStructureTemplate:
    original_var_names: list[str]
    semantic_var_names: list[str]
    complement_pairs: list[tuple[str, str]]
    A: np.ndarray
    b: np.ndarray
    senses: list[str]


@dataclass
class MaxXorInstance:
    package_index: int
    package_name: str
    objective_lower_bound: int
    B: np.ndarray
    v: np.ndarray
    ell: int
    code_distance: int
    max_constraints_satisfiable: int
    filtered_semantic_names: list[str]


@dataclass
class DqiBlockSummary:
    package_index: int
    package_name: str
    original_num_vars: int
    expanded_num_vars: int
    num_constraints: int
    num_complements: int
    objective_int: list[int]
    objective_lower_bound: int
    B_shape: tuple[int, int]
    ell: int
    code_distance: int
    max_constraints_satisfiable: int
    dqi_expected_f: float | None
    dqi_expected_s: float | None
    quantum_attempted: bool
    quantum_ran: bool
    quantum_postselected_shots: int | None
    quantum_best_semantic_assignment: dict[str, int] | None
    note: str


@dataclass
class DqiWorkflowResult:
    n_coverages: int
    m_packages: int
    objective_mode: str
    objective_scale: int
    objective_target_max: int
    bp_mode: str
    ell: int
    bp_iterations: int
    beta_mode: str
    seed: int
    c_im_float: list[list[float]]
    c_im_int: list[list[int]]
    blocks: list[DqiBlockSummary]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blocks"] = [asdict(block) for block in self.blocks]
        return result


def _log(message: str) -> None:
    print(message, flush=True)


def make_cim_matrix(problem) -> np.ndarray:
    c_matrix = np.zeros((problem.N, problem.M), dtype=float)
    beta = problem.price_sensitivity_beta
    base_contribution = np.array(
        [
            coverage.price * coverage.contribution_margin_pct * coverage.take_rate
            for coverage in problem.coverages
        ],
        dtype=float,
    )
    for package_index in range(problem.M):
        discount = float(problem.get_discount(package_index))
        demand_multiplier = (1.0 - discount) * (1.0 + beta * discount)
        affinity_column = np.array(
            [float(problem.get_affinity(coverage_index, package_index)) for coverage_index in range(problem.N)],
            dtype=float,
        )
        c_matrix[:, package_index] = base_contribution * affinity_column * demand_multiplier
    return c_matrix


def integerize_objective(
    objective_float: np.ndarray,
    mode: str = DEFAULT_OBJECTIVE_MODE,
    scale: int = DEFAULT_OBJECTIVE_SCALE,
    target_max: int = DEFAULT_OBJECTIVE_TARGET_MAX,
) -> np.ndarray:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized == "scaled":
        if scale <= 0:
            raise ValueError("scale must be positive in scaled mode")
        return np.rint(objective_float * scale).astype(int)
    if mode_normalized == "compressed":
        if target_max <= 0:
            raise ValueError("target_max must be positive in compressed mode")
        if objective_float.size == 0:
            return np.zeros_like(objective_float, dtype=int)
        max_value = float(np.max(objective_float))
        if max_value <= 0:
            return np.zeros_like(objective_float, dtype=int)
        compressed = np.rint(objective_float / max_value * target_max).astype(int)
        positive_mask = objective_float > 0
        compressed[positive_mask] = np.maximum(compressed[positive_mask], 1)
        return compressed
    raise ValueError("objective mode must be 'scaled' or 'compressed'")


def build_package_local_positive_ilp(
    problem,
    package_index: int,
    objective_column: np.ndarray | None = None,
    ilp_structure: IlpStructureTemplate | None = None,
    objective_mode: str = DEFAULT_OBJECTIVE_MODE,
    objective_scale: int = DEFAULT_OBJECTIVE_SCALE,
    objective_target_max: int = DEFAULT_OBJECTIVE_TARGET_MAX,
) -> PositiveIntegerIlp:
    if objective_column is None:
        objective_float = make_cim_matrix(problem)[:, package_index]
    else:
        objective_float = np.asarray(objective_column, dtype=float)
        if objective_float.shape != (problem.N,):
            raise ValueError(
                f"objective_column must have shape ({problem.N},), got {objective_float.shape}",
            )

    structure = ilp_structure if ilp_structure is not None else _build_ilp_structure_template(problem)

    package_name = (
        problem.package_names[package_index]
        if problem.package_names and package_index < len(problem.package_names)
        else f"package_{package_index}"
    )
    objective_int = integerize_objective(
        objective_float,
        mode=objective_mode,
        scale=objective_scale,
        target_max=objective_target_max,
    )

    objective_expanded = np.concatenate((objective_int.astype(int), np.zeros(len(structure.complement_pairs), dtype=int)))

    return PositiveIntegerIlp(
        package_index=package_index,
        package_name=package_name,
        objective_float=[float(value) for value in objective_float.tolist()],
        objective_int=[int(value) for value in objective_expanded.tolist()],
        semantic_var_names=list(structure.semantic_var_names),
        original_var_names=list(structure.original_var_names),
        complement_pairs=list(structure.complement_pairs),
        A=structure.A.copy(),
        b=structure.b.copy(),
        senses=list(structure.senses),
    )


def _build_ilp_structure_template(problem) -> IlpStructureTemplate:
    original_var_names = [f"x_{coverage_index}" for coverage_index in range(problem.N)]
    raw_rows: list[np.ndarray] = []
    raw_rhs: list[int] = []
    raw_senses: list[str] = []
    negative_columns: set[int] = set()

    for indices in problem.mandatory_families.values():
        row = np.zeros(problem.N, dtype=int)
        row[indices] = 1
        raw_rows.append(row)
        raw_rhs.append(1)
        raw_senses.append("=")

    for indices in problem.optional_families.values():
        row = np.zeros(problem.N, dtype=int)
        row[indices] = 1
        raw_rows.append(row)
        raw_rhs.append(1)
        raw_senses.append("<")

    capacity_row = np.ones(problem.N, dtype=int)
    raw_rows.append(capacity_row)
    raw_rhs.append(int(problem.max_options_per_package))
    raw_senses.append("<")

    for rule in problem.compatibility_rules:
        if rule.compatible:
            continue
        left = problem.coverage_index(rule.coverage_i)
        right = problem.coverage_index(rule.coverage_j)
        row = np.zeros(problem.N, dtype=int)
        row[left] = 1
        row[right] = 1
        raw_rows.append(row)
        raw_rhs.append(1)
        raw_senses.append("<")

    for rule in problem.dependency_rules:
        requires_index = problem.coverage_index(rule.requires)
        dependent_index = problem.coverage_index(rule.dependent)
        row = np.zeros(problem.N, dtype=int)
        row[dependent_index] = 1
        row[requires_index] = -1
        raw_rows.append(row)
        raw_rhs.append(0)
        raw_senses.append("<")
        negative_columns.add(requires_index)

    negative_list = sorted(negative_columns)
    complement_offset = problem.N
    complement_index_by_original = {
        original_index: complement_offset + offset
        for offset, original_index in enumerate(negative_list)
    }

    expanded_rows: list[np.ndarray] = []
    expanded_rhs: list[int] = []
    expanded_senses: list[str] = []

    for row, bound, sense in zip(raw_rows, raw_rhs, raw_senses):
        expanded = np.zeros(problem.N + len(negative_list), dtype=int)
        expanded[: problem.N] = np.maximum(row, 0)
        bound_shift = 0
        for original_index in negative_list:
            coefficient = int(row[original_index])
            if coefficient < 0:
                expanded[complement_index_by_original[original_index]] += -coefficient
                bound_shift += -coefficient
        expanded_rows.append(expanded)
        expanded_rhs.append(int(bound + bound_shift))
        expanded_senses.append(sense)

    complement_pairs: list[tuple[str, str]] = []
    semantic_var_names = list(original_var_names)
    for original_index in negative_list:
        complement_name = f"u_{original_index}"
        semantic_var_names.append(complement_name)
        complement_pairs.append((original_var_names[original_index], complement_name))

        equality_row = np.zeros(problem.N + len(negative_list), dtype=int)
        equality_row[original_index] = 1
        equality_row[complement_index_by_original[original_index]] = 1
        expanded_rows.append(equality_row)
        expanded_rhs.append(1)
        expanded_senses.append("=")

    return IlpStructureTemplate(
        original_var_names=original_var_names,
        semantic_var_names=semantic_var_names,
        complement_pairs=complement_pairs,
        A=np.array(expanded_rows, dtype=int),
        b=np.array(expanded_rhs, dtype=int),
        senses=list(expanded_senses),
    )


def choose_objective_lower_bound(c_vec: np.ndarray, mode: str = DEFAULT_BETA_MODE) -> int:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized == "zero":
        return 0
    if mode_normalized == "half_sum":
        return int(np.sum(np.abs(c_vec)) // 2)
    raise ValueError("beta mode must be 'zero' or 'half_sum'")


def generate_max_xor_instance(
    ilp: PositiveIntegerIlp,
    beta_mode: str = DEFAULT_BETA_MODE,
) -> MaxXorInstance:
    A = ilp.A
    b = ilp.b
    c = np.array(ilp.objective_int, dtype=int)
    beta_value = choose_objective_lower_bound(c, mode=beta_mode)

    A_aug = np.concatenate((A, c.reshape(1, -1)), axis=0)
    b_aug = np.concatenate((b, np.array([beta_value + 1], dtype=int)))
    senses_aug = list(ilp.senses) + [">="]

    common_var_name_list = [f"x_{index}_({index})" for index in range(A_aug.shape[1])]
    common_name_set = set(common_var_name_list)
    full_var_list = list(common_var_name_list)

    list_of_list_B: list[list[dict[str, int]]] = []
    list_of_list_v: list[list[int]] = []
    max_constraints_satisfiable = 0

    for row_index in range(A_aug.shape[0]):
        list_B, list_v, vm, max_per_circuit = generate_F2_instance(
            A_aug[row_index, :].tolist(),
            int(b_aug[row_index]),
            eq=senses_aug[row_index],
        )
        max_constraints_satisfiable += int(max_per_circuit)
        list_of_list_B.append(list_B)
        list_of_list_v.append(list_v)
        for variable_name in vm.vars:
            if variable_name in common_name_set:
                continue
            full_var_list.append(f"{variable_name}_{row_index}")

    name_to_index = {name: index for index, name in enumerate(full_var_list)}
    num_rows = sum(len(part) for part in list_of_list_B)
    B = np.zeros((num_rows, len(full_var_list)), dtype=int)
    v = np.concatenate([np.array(part, dtype=int) for part in list_of_list_v])

    output_row = 0
    for row_index, list_B in enumerate(list_of_list_B):
        for row in list_B:
            for variable_name in row.keys():
                if variable_name in common_name_set:
                    key = variable_name
                else:
                    key = f"{variable_name}_{row_index}"
                B[output_row, name_to_index[key]] = 1
            output_row += 1

    nonzero_columns = B.sum(axis=0) != 0
    B_filtered = B[:, nonzero_columns]
    kept_indices = np.where(nonzero_columns)[0]
    filtered_semantic_names: list[str] = []
    for kept_index in kept_indices:
        semantic_index = int(kept_index)
        if semantic_index < len(ilp.semantic_var_names):
            filtered_semantic_names.append(ilp.semantic_var_names[semantic_index])
        else:
            filtered_semantic_names.append(full_var_list[semantic_index])

    generator = systematic_form_G(B_filtered.T)
    code_distance = int(np.min(generator.sum(axis=1)))
    ell = max(1, int((code_distance - 1) / 2))

    return MaxXorInstance(
        package_index=ilp.package_index,
        package_name=ilp.package_name,
        objective_lower_bound=beta_value,
        B=B_filtered.astype(np.int8),
        v=v.astype(np.int8),
        ell=ell,
        code_distance=code_distance,
        max_constraints_satisfiable=max_constraints_satisfiable,
        filtered_semantic_names=filtered_semantic_names,
    )


def _bp_function(bp_mode: str):
    mode_normalized = str(bp_mode).strip().upper()
    if mode_normalized == "BP1":
        return belief_propagation_gallager
    if mode_normalized == "BP2":
        return belief_propagation_ldpc
    raise ValueError("bp_mode must be 'BP1' or 'BP2'")


def estimate_dqi_block(
    instance: MaxXorInstance,
    ell: int = DEFAULT_ELL,
    bp_mode: str = DEFAULT_BP_MODE,
    bp_iterations: int = DEFAULT_BP_ITERATIONS,
) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in arctanh",
            category=RuntimeWarning,
        )
        f_expected, s_expected = expected_constrains_DQI(
            instance.B,
            instance.v,
            ell,
            bp_iterations,
            _bp_function(bp_mode),
            jit_version=True,
        )
    return float(f_expected), float(s_expected)


def try_quantum_dqi_block(
    instance: MaxXorInstance,
    ell: int = DEFAULT_ELL,
    bp_iterations: int = DEFAULT_BP_ITERATIONS,
    shots: int = 256,
    seed: int = DEFAULT_SEED,
    row_limit: int = DEFAULT_QUANTUM_ROW_LIMIT,
) -> tuple[bool, int | None, dict[str, int] | None, str]:
    if instance.B.shape[0] > row_limit:
        return (
            False,
            None,
            None,
            (
                f"skipped full quantum simulation because reduced max-XOR has "
                f"{instance.B.shape[0]} rows, above row_limit={row_limit}"
            ),
        )

    try:
        from pipelines.DQI_full_circuit import dqi_max_xorsat, get_optimal_w
        from qiskit import ClassicalRegister, transpile
        from qiskit_aer import AerSimulator
    except ModuleNotFoundError as exc:
        return False, None, None, f"skipped quantum simulation because dependency import failed: {exc}"

    W_k = get_optimal_w(instance.B.shape[0], ell)
    circuit, y_reg, bt_reg = dqi_max_xorsat(
        instance.B,
        instance.v,
        W_k,
        bp_iterations,
    )

    data_creg = ClassicalRegister(instance.B.shape[0], "y_bits")
    bt_creg = ClassicalRegister(instance.B.shape[1], "bt_bits")
    circuit.add_register(bt_creg)
    circuit.add_register(data_creg)
    circuit.measure(bt_reg, bt_creg)
    circuit.measure(y_reg, data_creg)

    measured_qubits = set(bt_reg[:] + y_reg[:])
    all_qubits = [qubit for register in circuit.qregs for qubit in register]
    ancilla_qubits = [qubit for qubit in all_qubits if qubit not in measured_qubits]
    ancilla_creg = ClassicalRegister(len(ancilla_qubits), "ancilla_bits")
    circuit.add_register(ancilla_creg)
    circuit.measure(ancilla_qubits, ancilla_creg)

    simulator = AerSimulator(seed_simulator=int(seed))
    compiled = transpile(circuit, simulator, seed_transpiler=int(seed))
    result = simulator.run(compiled, shots=int(shots), seed_simulator=int(seed)).result()
    counts = result.get_counts()

    semantic_counts: dict[str, int] = {}
    postselected_shots = 0
    best_assignment: dict[str, int] | None = None
    best_count = -1

    for bitstring, count in counts.items():
        ancilla_bits, y_bits, x_bits = bitstring.split(" ")
        y_bits = y_bits[::-1]
        x_bits = x_bits[::-1]
        if any(bit != "0" for bit in ancilla_bits):
            continue
        if any(bit != "0" for bit in y_bits):
            continue
        postselected_shots += int(count)
        assignment = {
            semantic_name: int(x_bits[index])
            for index, semantic_name in enumerate(instance.filtered_semantic_names)
        }
        key = json.dumps(assignment, sort_keys=True)
        semantic_counts[key] = semantic_counts.get(key, 0) + int(count)
        if semantic_counts[key] > best_count:
            best_count = semantic_counts[key]
            best_assignment = assignment

    if postselected_shots == 0:
        return True, 0, None, "quantum simulation ran but produced zero postselected shots"
    return True, postselected_shots, best_assignment, "quantum simulation ran"


def solve_dqi_workflow(
    n: int = DEFAULT_N_COVERAGES,
    m: int = DEFAULT_M_PACKAGES,
    objective_mode: str = DEFAULT_OBJECTIVE_MODE,
    objective_scale: int = DEFAULT_OBJECTIVE_SCALE,
    objective_target_max: int = DEFAULT_OBJECTIVE_TARGET_MAX,
    bp_mode: str = DEFAULT_BP_MODE,
    ell: int = DEFAULT_ELL,
    bp_iterations: int = DEFAULT_BP_ITERATIONS,
    beta_mode: str = DEFAULT_BETA_MODE,
    seed: int = DEFAULT_SEED,
    try_quantum: bool = True,
    quantum_row_limit: int = DEFAULT_QUANTUM_ROW_LIMIT,
) -> DqiWorkflowResult:
    problem = subsample_problem(load_ltm_instance(DATA_DIR), n_coverages=n, n_packages=m)
    c_im_float = make_cim_matrix(problem)
    ilp_structure = _build_ilp_structure_template(problem)
    c_im_int = np.zeros_like(c_im_float, dtype=int)
    summaries: list[DqiBlockSummary] = []

    _log(
        f"[workflow] Travelers DQI adapter on package-local blocks "
        f"(N={problem.N}, M={problem.M}, objective_mode={objective_mode}, "
        f"bp_mode={bp_mode}, ell={ell}, beta_mode={beta_mode})",
    )

    for package_index in range(problem.M):
        ilp = build_package_local_positive_ilp(
            problem,
            package_index,
            objective_column=c_im_float[:, package_index],
            ilp_structure=ilp_structure,
            objective_mode=objective_mode,
            objective_scale=objective_scale,
            objective_target_max=objective_target_max,
        )
        c_im_int[:, package_index] = np.array(ilp.objective_int[: problem.N], dtype=int)

        _log(
            f"[package {package_index + 1}] building positive integer ILP block with "
            f"{len(ilp.original_var_names)} original vars, {len(ilp.semantic_var_names)} expanded vars, "
            f"{len(ilp.complement_pairs)} complements",
        )
        instance = generate_max_xor_instance(ilp, beta_mode=beta_mode)
        _log(
            f"[package {package_index + 1}] max-XOR shape {instance.B.shape}, "
            f"ell={instance.ell}, code_distance={instance.code_distance}, "
            f"objective_lower_bound={instance.objective_lower_bound}",
        )

        f_expected, s_expected = estimate_dqi_block(
            instance,
            ell=ell,
            bp_mode=bp_mode,
            bp_iterations=bp_iterations,
        )
        _log(
            f"[package {package_index + 1}] analytic DQI estimate: "
            f"<f>={f_expected:.4f}, <s>={s_expected:.4f}",
        )

        quantum_attempted = bool(try_quantum)
        quantum_ran = False
        quantum_postselected_shots: int | None = None
        quantum_best_assignment: dict[str, int] | None = None
        note = "analytic DQI estimate completed"

        if try_quantum:
            quantum_ran, quantum_postselected_shots, quantum_best_assignment, note = try_quantum_dqi_block(
                instance,
                ell=ell,
                bp_iterations=bp_iterations,
                seed=seed + package_index,
                row_limit=quantum_row_limit,
            )
            if quantum_ran:
                _log(
                    f"[package {package_index + 1}] quantum note: {note}; "
                    f"postselected_shots={quantum_postselected_shots}",
                )
            else:
                _log(f"[package {package_index + 1}] quantum note: {note}")

        summaries.append(
            DqiBlockSummary(
                package_index=package_index,
                package_name=ilp.package_name,
                original_num_vars=len(ilp.original_var_names),
                expanded_num_vars=len(ilp.semantic_var_names),
                num_constraints=int(ilp.A.shape[0]),
                num_complements=len(ilp.complement_pairs),
                objective_int=list(ilp.objective_int),
                objective_lower_bound=instance.objective_lower_bound,
                B_shape=tuple(int(value) for value in instance.B.shape),
                ell=int(instance.ell),
                code_distance=int(instance.code_distance),
                max_constraints_satisfiable=int(instance.max_constraints_satisfiable),
                dqi_expected_f=f_expected,
                dqi_expected_s=s_expected,
                quantum_attempted=quantum_attempted,
                quantum_ran=quantum_ran,
                quantum_postselected_shots=quantum_postselected_shots,
                quantum_best_semantic_assignment=quantum_best_assignment,
                note=note,
            ),
        )

    return DqiWorkflowResult(
        n_coverages=int(problem.N),
        m_packages=int(problem.M),
        objective_mode=str(objective_mode),
        objective_scale=int(objective_scale),
        objective_target_max=int(objective_target_max),
        bp_mode=str(bp_mode),
        ell=int(ell),
        bp_iterations=int(bp_iterations),
        beta_mode=str(beta_mode),
        seed=int(seed),
        c_im_float=[[float(value) for value in row] for row in c_im_float.tolist()],
        c_im_int=[[int(value) for value in row] for row in c_im_int.tolist()],
        blocks=summaries,
    )


def main() -> DqiWorkflowResult:
    result = solve_dqi_workflow()
    output_path = DEFAULT_OUTPUT_PATH
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    _log(f"[workflow] wrote summary to {output_path}")
    return result


if __name__ == "__main__":
    workflow_result = main()
    print(json.dumps(workflow_result.to_dict(), indent=2))
