# Cayman — DQI / QAOA insurance bundling (Y Quantum)

This subproject builds **per-package penalty QUBOs** from the P&C bundling ILP ([`src/insurance_model.py`](src/insurance_model.py), [`src/qubo_block.py`](src/qubo_block.py)), runs **QAOA** with classical angle optimization ([`src/qubo_qaoa.py`](src/qubo_qaoa.py)), and runs a **single-kernel quantum routine** exposed as DQI ([`src/run_dqi.py`](src/run_dqi.py), [`src/dqi_core.py`](src/dqi_core.py)).

## DQI modes

### 1. Baseline (default)

Hadamard (or mixer) on all **problem** qubits, alternating **cost** layers (ZZ via CX–RZ–CX and Z phases) and **mixer** layers with **fixed** angles—no classical hybrid loop. Same QUBO/Ising encoding as QAOA, one shot.

### 2. Parity + optional Dicke (Travelers-style building blocks)

Pass **`insurance_parity=(problem, package_index)`** to [`run_dqi`](src/run_dqi.py) / [`run_dqi_with_details`](src/run_dqi.py), or set **`B`** and **`parity_rhs`** explicitly (GF(2) matrix matching the QUBO variable order from [`build_insurance_parity_B_rhs`](src/dqi_insurance_parity.py)).

The kernel then:

1. Optionally prepares a **Dicke state** on the first **`n_coverage`** qubits (coverage bits) with Hamming weight **`dicke_k`** using the **Bärtschi–Eidenbenz** schedule ported from Qrisp ([`src/dqi_dicke_prep.py`](src/dqi_dicke_prep.py)); otherwise Hadamards on the full problem register (plus Hadamards on slack bits only in the Dicke branch).
2. XORs each parity row into a **syndrome** ancilla via CNOTs.
3. Runs the same **QUBO phase + mixer** on the **problem** register only (syndrome qubits idle).
4. Measures all qubits.  Shots are **post-selected** when **`B @ x ≡ rhs (mod 2)`** on the problem prefix (equivalent to syndrome matching); see **`post_selection_rate`** on [`DqiSampleStats`](src/dqi_core.py).

**Caveats:** the GF(2) rows are **necessary but not sufficient** for the full integer slack equalities (see docstring in [`dqi_insurance_parity.py`](src/dqi_insurance_parity.py)). There is **no** BP decoder stage yet—only algebraic filtering.

## Comparing to sponsor starter code (when the repo is available)

| Aspect | Cayman parity+DQI path | Typical deck narrative |
|--------|------------------------|-------------------------|
| Parity | `B` from insurance slack layout + CNOT syndromes | Parity / max-XORSAT |
| Prep | Optional Dicke on coverages (BE19) | Dicke (named explicitly) |
| Post-process | Classical filter on `B @ x` | Syndrome-0 / discard rate |
| Decoder | Not implemented | Optional belief propagation |

**Kernel builders:** [`_build_dqi_dicke_parity_source`](src/dqi_core.py) vs legacy [`_build_dqi_source`](src/dqi_core.py).

## Quick commands

- Run DQI on a block: `python scripts/run_dqi_cli.py --help` (use `--insurance-parity` with `--source ltm-block` for parity+Dicke options).
- Benchmark official sizes: `python scripts/benchmark_dqi_official_sizes.py --help`
- Tests: `python -m unittest discover -s tests -v`

Dependencies: see [`requirements.txt`](requirements.txt).
