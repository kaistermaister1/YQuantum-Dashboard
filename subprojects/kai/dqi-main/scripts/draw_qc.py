import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

# ── 1-qubit registers ────────────────────────────────────────────────────────────
q_y1 = QuantumRegister(1, "y_1")
q_y2 = QuantumRegister(1, "y_2")
q_y3 = QuantumRegister(1, "y_3")
q_s1 = QuantumRegister(1, "s_1")
q_s2 = QuantumRegister(1, "s_2")
q_h1 = QuantumRegister(1, "h_1")
q_h2 = QuantumRegister(1, "h_2")
q_c1 = QuantumRegister(1, "c_1")
q_c2 = QuantumRegister(1, "c_2")
q_f11 = QuantumRegister(1, "f1_1")
q_f12 = QuantumRegister(1, "f1_2")
q_f13 = QuantumRegister(1, "f1_3")
q_f21 = QuantumRegister(1, "f2_1")
q_f22 = QuantumRegister(1, "f2_2")
q_f23 = QuantumRegister(1, "f2_3")

# ── Build the circuit ───────────────────────────────────────────────────────────
qc = QuantumCircuit(
    q_y1,
    q_y2,
    q_y3,
    q_s1,
    q_s2,
    q_h1,
    q_h2,
    q_c1,
    q_c2,
    q_f11,
    q_f12,
    q_f13,
    q_f21,
    q_f22,
    q_f23,
    name="BP1",
)

# ── “Real” gates ────────────────────────────────────────────────────────────────
comp = Gate(r"$C$ ", 4, [])
hamm = Gate(r"$H$ ", 3, [])
hamm4 = Gate(r"$H$ ", 4, [])
# “Dagger” versions (just different names for drawing):
comp_dg = Gate(r"$C^\dagger$ ", 4, [])
hamm_dg = Gate(r"$H^\dagger$ ", 3, [])
hamm_dg4 = Gate(r"$H^\dagger$ ", 4, [])

inverse = Gate("Inv ", 12, [])
# 1) Comp on h1,h2,c1,c2
qc.append(hamm, [q_s1[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
# 2) CNOT from c1→f1_1
qc.cx(q_c1[0], q_f11[0])
# 3) Comp† on h1,h2,c1,c2
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
# 4) Hamm† on s1,s2,c1
qc.append(hamm_dg, [q_s1[0], q_h1[0], q_h2[0]])
# qc.barrier()
qc.append(hamm4, [q_s1[0], q_s2[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
qc.cx(q_c1[0], q_f12[0])
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
qc.append(hamm_dg4, [q_s1[0], q_s2[0], q_h1[0], q_h2[0]])
# qc.barrier()
qc.append(hamm, [q_s2[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])

# 2) CNOT from c1→f1_1
qc.cx(q_c1[0], q_f13[0])

# 3) Comp† on h1,h2,c1,c2
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])

# 4) Hamm† on s1,s2,c1
qc.append(hamm_dg, [q_s2[0], q_h1[0], q_h2[0]])

qc.barrier()

qc.cx(q_f11[0], q_s1[0])
qc.cx(q_f12[0], q_s1[0])
qc.cx(q_f12[0], q_s2[0])
qc.cx(q_f13[0], q_s2[0])


qc.barrier()

# 1) Comp on h1,h2,c1,c2
qc.append(hamm, [q_s1[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
# 2) CNOT from c1→f1_1
qc.cx(q_c1[0], q_f21[0])
# 3) Comp† on h1,h2,c1,c2
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
# 4) Hamm† on s1,s2,c1
qc.append(hamm_dg, [q_s1[0], q_h1[0], q_h2[0]])
# qc.barrier()
qc.append(hamm4, [q_s1[0], q_s2[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
qc.cx(q_c1[0], q_f22[0])
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
qc.append(hamm_dg4, [q_s1[0], q_s2[0], q_h1[0], q_h2[0]])
# qc.barrier()
qc.append(hamm, [q_s2[0], q_h1[0], q_h2[0]])
qc.append(comp, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])
# 2) CNOT from c1→f1_1
qc.cx(q_c1[0], q_f23[0])

# 3) Comp† on h1,h2,c1,c2
qc.append(comp_dg, [q_h1[0], q_h2[0], q_c1[0], q_c2[0]])

# 4) Hamm† on s1,s2,c1
qc.append(hamm_dg, [q_s2[0], q_h1[0], q_h2[0]])

# qc.barrier()
qc.barrier()

qc.cx(q_f11[0], q_y1[0])
qc.cx(q_f12[0], q_y2[0])
qc.cx(q_f13[0], q_y3[0])


qc.cx(q_f21[0], q_y1[0])
qc.cx(q_f22[0], q_y2[0])
qc.cx(q_f23[0], q_y3[0])


qc.append(
    inverse,
    [
        q_s1[0],
        q_s2[0],
        q_h1[0],
        q_h2[0],
        q_c1[0],
        q_c2[0],
        q_f11[0],
        q_f12[0],
        q_f13[0],
        q_f21[0],
        q_f22[0],
        q_f23[0],
    ],
)


# # ── Draw it ────────────────────────────────────────────────────────────────────
# # … after drawing the circuit …
# fig = qc.draw('mpl', fold=22, idle_wires=False, justify='left')         # or whatever size you need
# plt.tight_layout()
# fig.savefig('bp1_circuit.pdf', dpi=200, bbox_inches='tight')


# Extract instruction indices for barrier positions
# Extract instruction indices for barrier positions
barrier_indices = [
    i for i, instr in enumerate(qc.data) if instr.operation.name == "barrier"
]

# Skip the first barrier as a splitting point
# Define subcircuits using the remaining barrier indices only
slices = [
    (0, barrier_indices[1]),
    (barrier_indices[1] + 1, barrier_indices[2]),
    (barrier_indices[2] + 1, len(qc.data)),
]


# Helper function to slice a circuit
def extract_subcircuit(qc, start, end):
    sub_qc = QuantumCircuit(*qc.qregs, name=qc.name)
    for instr in qc.data[start:end]:
        sub_qc.append(instr.operation, instr.qubits, instr.clbits)
    return sub_qc


# Draw and save each subcircuit
for i, (start, end) in enumerate(slices, 1):
    sub_qc = extract_subcircuit(qc, start, end)
    fig = sub_qc.draw("mpl", fold=22, idle_wires=False, justify="left")
    plt.tight_layout()
    fig.savefig(f"bp1_subcircuit_{i}.pdf", dpi=200, bbox_inches="tight")
