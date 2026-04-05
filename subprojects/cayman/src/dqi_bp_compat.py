"""dqi-main compatible BP circuit path (Qiskit simulation).

This module ports the key ingredients used in ``dqi-main``:
- weighted unary encoding ``W_k``
- Unk/Dicke-style state preparation
- BP1 circuit composition
- strict post-selection on ``y=0`` and clean ancillas
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def _require_qiskit():
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
        from qiskit.circuit import Gate
        from qiskit.circuit.library import IntegerComparator, RYGate
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "BP-compatible DQI path requires qiskit and qiskit-aer. "
            "Install with: pip install qiskit qiskit-aer"
        ) from exc
    return ClassicalRegister, QuantumCircuit, QuantumRegister, transpile, Gate, IntegerComparator, RYGate, AerSimulator


def get_optimal_w(m: int, ell: int, p: int = 2, r: int = 1) -> np.ndarray:
    """Return principal eigenvector weights padded to power-of-two length."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1) * d
    off = np.sqrt(np.arange(1, ell + 1) * (m - np.arange(1, ell + 1) + 1))
    A = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")
    _, vecs = eigsh(A, k=1, which="LA")
    w = vecs.flatten()
    w /= np.linalg.norm(w)
    orig = w.size
    target = 1 << math.ceil(math.log2(orig))
    return np.pad(w, (0, target - orig))


def _weighted_unary_gate(num_bit: int, weights: np.ndarray):
    _, QuantumCircuit, _, _, _, _, RYGate, _ = _require_qiskit()
    qc = QuantumCircuit(num_bit, name="WeightedUnary")
    betas: list[float] = []
    for ell in range(len(weights)):
        denom = 1.0 - float(np.sum(weights[:ell] ** 2))
        ratio = float(weights[ell] ** 2) / denom if denom > 0 else 0.0
        val = min(math.sqrt(max(ratio, 0.0)), 1.0)
        betas.append(2.0 * math.acos(val))
    if betas:
        qc.ry(betas[0], 0)
    for i in range(1, min(num_bit, len(betas))):
        if np.isnan(betas[i]) or abs(betas[i]) < 1e-15:
            continue
        cry = RYGate(float(betas[i])).control(ctrl_state="1")
        qc.append(cry, [i - 1, i])
    return qc.to_gate()


def _unk_state_gate(n: int, k: int):
    """Port of UnkStatePreparation from dqi-main/dicke_states.py."""
    _, QuantumCircuit, _, _, Gate, _, RYGate, _ = _require_qiskit()

    class SCS2Gate(Gate):
        def __init__(self, nn: int) -> None:
            self.nn = nn
            super().__init__("SCS2", 2, [])

        def _define(self) -> None:
            qc = QuantumCircuit(2)
            qc.cx(1, 0)
            theta = 2 * np.arccos(1 / np.sqrt(self.nn))
            qc.append(RYGate(theta).control(ctrl_state="1"), [0, 1])
            qc.cx(1, 0)
            self.definition = qc

    class SCS3Gate(Gate):
        def __init__(self, ell: int, nn: int) -> None:
            self.ell = ell
            self.nn = nn
            super().__init__("SCS3", 3, [])

        def _define(self) -> None:
            qc = QuantumCircuit(3)
            qc.cx(2, 0)
            theta = 2 * np.arccos(np.sqrt(self.ell / self.nn))
            qc.append(RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11"), [0, 1, 2])
            qc.cx(2, 0)
            self.definition = qc

    class SCSnkGate(Gate):
        def __init__(self, nn: int, kk: int) -> None:
            self.nn = nn
            self.kk = kk
            super().__init__(f"SCS{nn},{kk}", kk + 1, [])

        def _define(self) -> None:
            qc = QuantumCircuit(self.kk + 1)
            qc.append(SCS2Gate(self.nn), [0, 1])
            for ell in range(2, self.kk + 1):
                qc.append(SCS3Gate(ell, self.nn), [0, ell - 1, ell])
            self.definition = qc

    qc = QuantumCircuit(n, name=f"U_{n}_{k}")
    for ell in range(n, k, -1):
        qc.append(SCSnkGate(ell, k), list(range(n - ell, n - ell + k + 1)))
    for ell in range(k, 1, -1):
        qc.append(SCSnkGate(ell, ell - 1), list(range(n - ell, n)))
    return qc.to_gate()


def _transpiled_control_plus_one_gate_efficient(n: int, *, inverse: bool = False):
    _, QuantumCircuit, _, transpile, _, _, _, _ = _require_qiskit()
    qc = QuantumCircuit(n)
    for i in list(range(n))[::-1]:
        ctrls = list(range(i))
        if i == 0:
            qc.x(i)
        else:
            qc.mcx(ctrls, i)
    gate = qc.to_gate(label="+1")
    controlled = gate.inverse().control(1) if inverse else gate.control(1)
    temp = QuantumCircuit(1 + n)
    temp.append(controlled, list(range(1 + n)))
    return transpile(temp, basis_gates=["z", "cx", "rx", "ry", "rz", "swap"])


def _explicit_inverse(circuit):
    _, QuantumCircuit, _, _, _, _, _, _ = _require_qiskit()
    inv = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr, qargs, cargs in reversed(circuit.data):
        inv.append(instr.inverse(), qargs, cargs)
    return inv


def _bp1_circuit(B: np.ndarray, num_iterations: int):
    _, QuantumCircuit, QuantumRegister, transpile, _, IntegerComparator, _, _ = _require_qiskit()

    Bt = B.T
    m, n = B.shape
    n_var = Bt.shape[1]
    n_check = Bt.shape[0]
    syndrome_reg = QuantumRegister(n_check, name="Bty")
    y_reg = QuantumRegister(n_var, name="y")

    threshold_list = [len(np.where(Bt[:, var] == 1)[0]) for var in range(n_var)]
    max_threshold = int(np.max(threshold_list)) if threshold_list else 1
    qubits_hamming = int(math.ceil(math.log2(max_threshold + 1)))
    comp_qubits = qubits_hamming

    ham_store_reg = QuantumRegister(qubits_hamming, name="ham_store")
    comparator_reg = QuantumRegister(comp_qubits, name="comp_ancilla")
    bp_qc = QuantumCircuit(syndrome_reg, y_reg, ham_store_reg, comparator_reg)

    for it in range(1, num_iterations + 1):
        if it > 1:
            for i in range(n):
                for j in range(m):
                    if Bt[i, j] == 1:
                        bp_qc.cx(flip_register[j], syndrome_reg[i])
        flip_register = QuantumRegister(n_var, name=f"flip_{it}")
        bp_qc.add_register(flip_register)

        for var in range(n_var):
            connected = np.where(Bt[:, var] == 1)[0]
            threshold = len(connected)
            if threshold <= 0:
                bp_qc.x(flip_register[var])
                continue
            needed = int(math.ceil(math.log2(threshold + 1)))
            store_ham = ham_store_reg[:needed]
            plus_one = _transpiled_control_plus_one_gate_efficient(needed, inverse=False)
            for index in connected:
                bp_qc.compose(plus_one, qubits=[syndrome_reg[index]] + list(store_ham), inplace=True)

            comp = IntegerComparator(num_state_qubits=needed, value=threshold, geq=True)
            comp_t = transpile(comp, basis_gates=["z", "cx", "rx", "ry", "rz", "swap"])
            comp_anc = comparator_reg[:needed]
            bp_qc.compose(comp_t, qubits=list(store_ham) + list(comp_anc), inplace=True)
            bp_qc.cx(comp_anc[0], flip_register[var])

            comp_i = transpile(comp.inverse(), basis_gates=["z", "cx", "rx", "ry", "rz", "swap"])
            bp_qc.compose(comp_i, qubits=list(store_ham) + list(comp_anc), inplace=True)
            minus_one = _transpiled_control_plus_one_gate_efficient(needed, inverse=True)
            for index in connected:
                bp_qc.compose(minus_one, qubits=[syndrome_reg[index]] + list(store_ham), inplace=True)

    regs = bp_qc.qregs
    full = QuantumCircuit(*regs, name="BeliefPropagation")
    flat = [q for reg in bp_qc.qregs for q in reg]
    full.compose(bp_qc, qubits=flat, inplace=True)
    for it in range(1, num_iterations + 1):
        flip = [reg for reg in full.qregs if reg.name == f"flip_{it}"][0]
        for fq, yq in zip(flip, y_reg):
            full.cx(fq, yq)
    full.barrier()
    full.compose(_explicit_inverse(bp_qc), qubits=bp_qc.qubits, inplace=True)
    return full


def run_dqi_bp_histogram(
    B: np.ndarray,
    v: np.ndarray,
    *,
    ell: int,
    num_iterations_bp: int,
    shots: int = 4096,
    seed: int = 12345,
) -> tuple[dict[str, int], float, dict[str, int]]:
    """Return ``(x_counts_postselected, keep_rate, raw_counts)`` in dqi-main style."""
    ClassicalRegister, QuantumCircuit, QuantumRegister, transpile, _, _, _, AerSimulator = _require_qiskit()

    B = (np.asarray(B, dtype=np.uint8) & 1).astype(int)
    v = (np.asarray(v, dtype=np.uint8) & 1).astype(int)
    m, n = B.shape

    W_k = get_optimal_w(m, max(1, int(ell)))

    k_reg = QuantumRegister(m, name="k")
    y_reg = QuantumRegister(m, name="y")
    qc = QuantumCircuit(y_reg)
    init = QuantumCircuit(k_reg)
    init.append(_weighted_unary_gate(m, W_k), range(m))
    temp = QuantumCircuit(m)
    max_errors = int(np.nonzero(W_k)[0][-1]) if np.any(W_k) else 0
    temp.append(_unk_state_gate(m, max_errors), range(m))
    qc.compose(init, qubits=range(m), inplace=True)
    qc.compose(temp, qubits=range(m), inplace=True)

    for qubit, vv in zip(y_reg, v):
        if vv > 0:
            qc.z(qubit)

    bt_reg = QuantumRegister(n, name="bt")
    qc.add_register(bt_reg)
    for i in range(n):
        for j in range(m):
            if int(B.T[i, j]) == 1:
                qc.cx(y_reg[j], bt_reg[i])

    bpc = _bp1_circuit(B, int(num_iterations_bp))
    bt_main = [reg for reg in qc.qregs if reg.name == "bt"][0]
    y_main = [reg for reg in qc.qregs if reg.name == "y"][0]
    bt_bpc = [reg for reg in bpc.qregs if reg.name == "Bty"][0]
    y_bpc = [reg for reg in bpc.qregs if reg.name == "y"][0]
    anc_regs_bpc = [reg for reg in bpc.qregs if reg not in [bt_bpc, y_bpc]]
    new_ancs = []
    for reg in anc_regs_bpc:
        new_reg = QuantumRegister(len(reg), name=f"{reg.name}_from_bpc")
        qc.add_register(new_reg)
        new_ancs.append(new_reg)
    qarg = [bt_main[i] for i in range(len(bt_bpc))] + [y_main[i] for i in range(len(y_bpc))]
    for orig, new in zip(anc_regs_bpc, new_ancs):
        qarg.extend(new[i] for i in range(len(orig)))
    qc.compose(bpc, qubits=qarg, inplace=True)

    for qubit in bt_main:
        qc.h(qubit)

    data_creg = ClassicalRegister(m, "y_bits")
    bt_creg = ClassicalRegister(n, "bt_bits")
    qc.add_register(bt_creg)
    qc.add_register(data_creg)
    qc.measure(bt_reg, bt_creg)
    qc.measure(y_reg, data_creg)
    measured_qubits = set(bt_reg[:] + y_reg[:])
    all_qubits = [q for qreg in qc.qregs for q in qreg]
    ancilla_qubits = [q for q in all_qubits if q not in measured_qubits]
    anc_creg = ClassicalRegister(len(ancilla_qubits), "ancilla_bits")
    qc.add_register(anc_creg)
    qc.measure(ancilla_qubits, anc_creg)

    simulator = AerSimulator(seed_simulator=int(seed))
    tqc = transpile(qc, simulator, seed_transpiler=int(seed))
    result = simulator.run(tqc, shots=int(shots), seed_simulator=int(seed)).result()
    counts = result.get_counts()

    filtered_counts: Counter[str] = Counter()
    total = int(sum(counts.values()))
    kept = 0
    for bitstring, count in counts.items():
        parts = bitstring.split(" ")
        if len(parts) < 3:
            continue
        anc_bits = parts[0]
        y_bits = parts[1][::-1]
        x_bits = parts[2][::-1]
        if all(bit == "0" for bit in anc_bits) and all(bit == "0" for bit in y_bits):
            filtered_counts[x_bits] += int(count)
            kept += int(count)

    keep_rate = float(kept) / float(total) if total else 0.0
    return dict(filtered_counts), keep_rate, {str(k): int(vv) for k, vv in counts.items()}

