import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import IntegerComparator
from qiskit.quantum_info import Statevector


def increment_of_one_gate(n):
    """
    Creates a quantum gate that increments an n-qubit binary register by one.

    Constructs a reversible circuit that performs modular addition of 1
    (i.e., maps |x⟩ to |(x+1) mod 2ⁿ⟩) using multi-controlled Toffoli gates
    to propagate carry bits.

    Args:
        n (int): Number of qubits representing the binary number.

    Returns:
        Gate: A quantum gate that performs modular addition of 1 on n qubits.
    """

    qc = QuantumCircuit(n)

    for i in range(n):
        controls = list(range(i))
        if i == 0:
            qc.x(i)  # Flip the LSB
        else:
            for control in controls:
                qc.x(control)
            qc.mcx(controls, i)  # Apply carry controlled on all lower bits
            for control in controls:
                qc.x(control)

    return qc.to_gate(label="+1")


def gate_efficient_increment_of_one_gate(n):
    """
    Creates a quantum gate that increments an n-qubit binary register by one.

    Constructs a reversible circuit that performs modular addition of 1
    (i.e., maps |x⟩ to |(x+1) mod 2ⁿ⟩) using multi-controlled Toffoli gates
    to propagate carry bits.

    Avoids using single qubits X gates.

    Args:
        n (int): Number of qubits representing the binary number.

    Returns:
        Gate: A quantum gate that performs modular addition of 1 on n qubits.
    """

    qc = QuantumCircuit(n)

    for i in list(range(n))[::-1]:
        controls = list(range(i))
        if i == 0:
            qc.x(i)  # Flip the LSB
        else:
            qc.mcx(controls, i)  # Apply carry controlled on all lower bits

    return qc.to_gate(label="+1")


def transpiled_control_plus_one_gate(n, inverse=False):
    """
    Creates a transpiled, controlled version of the increment-of-one gate.

    Generates a circuit where a single control qubit conditionally increments
    an n-qubit register by one. The output is decomposed into a hardware-compatible
    gate set using transpilation.

    Args:
        n (int): Number of qubits in the target register.
        inverse (bool): If True, returns the inverse (decrement) controlled gate.

    Returns:
        QuantumCircuit: A transpiled quantum circuit implementing the controlled ±1 gate.
    """

    plus_one_gate = increment_of_one_gate(n)  # create quantum gate that adds 1

    if inverse:
        plus_one_inverse = plus_one_gate.inverse()
        controlled_gate = plus_one_inverse.control(1)

    else:
        controlled_gate = plus_one_gate.control(1)  # Step 1: create controlled version

    # Step 2: Wrap in a temporary circuit to decompose/transpile
    num_ctrl = 1
    num_target = plus_one_gate.num_qubits
    total_qubits = num_ctrl + num_target

    # Create dummy circuit with the controlled gate
    temp_circuit = QuantumCircuit(total_qubits)
    temp_circuit.append(controlled_gate, list(range(total_qubits)))

    # Step 3: Transpile into desired basis
    transpiled_circuit = transpile(
        temp_circuit,
        basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
    )

    return transpiled_circuit


def transpiled_control_plus_one_gate_efficient(n, inverse=False):
    """
    Creates a transpiled, controlled version of the increment-of-one gate.

    Generates a circuit where a single control qubit conditionally increments
    an n-qubit register by one. The output is decomposed into a hardware-compatible
    gate set using transpilation.

    Avoids using single qubits X gates.

    Args:
        n (int): Number of qubits in the target register.
        inverse (bool): If True, returns the inverse (decrement) controlled gate.

    Returns:
        QuantumCircuit: A transpiled quantum circuit implementing the controlled ±1 gate.
    """

    plus_one_gate = gate_efficient_increment_of_one_gate(
        n,
    )  # create quantum gate that adds 1

    if inverse:
        plus_one_inverse = plus_one_gate.inverse()
        controlled_gate = plus_one_inverse.control(1)

    else:
        controlled_gate = plus_one_gate.control(1)  # Step 1: create controlled version

    # Step 2: Wrap in a temporary circuit to decompose/transpile
    num_ctrl = 1
    num_target = plus_one_gate.num_qubits
    total_qubits = num_ctrl + num_target

    # Create dummy circuit with the controlled gate
    temp_circuit = QuantumCircuit(total_qubits)
    temp_circuit.append(controlled_gate, list(range(total_qubits)))

    # Step 3: Transpile into desired basis
    transpiled_circuit = transpile(
        temp_circuit,
        basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
    )

    return transpiled_circuit


def explicit_inverse(circuit):
    """
    Returns the explicit inverse of a given quantum circuit.

    Constructs a new circuit that is the inverse of the input circuit by
    reversing the order of gates and inverting each operation.

    Args:
        circuit (QuantumCircuit): The quantum circuit to invert.

    Returns:
        QuantumCircuit: The inverse of the input circuit.
    """
    inverse_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs)
    for instr, qargs, cargs in reversed(circuit.data):
        inverse_circuit.append(instr.inverse(), qargs, cargs)
    return inverse_circuit


def BP1_circuit(B, num_iterations):
    """
    Constructs the BP1 quantum circuit for Belief Propagation decoding (Gallager-style).

    Given a binary parity-check matrix B and a number of iterations, this function returns a
    quantum circuit implementing iterative belief propagation. The circuit maps the initial
    state |B·y⟩|y⟩ to |B·y⟩|0⟩, where y represents the estimated error pattern after belief updates.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        num_iterations (int): Number of belief propagation iterations to perform.

    Returns:
        QuantumCircuit: The full unitary circuit implementing BP1.
    """

    Bt = B.T

    m = B.shape[0]
    n = B.shape[1]

    n_var = Bt.shape[1]  # 3
    n_check = Bt.shape[0]  # 2

    n_syndrome = n_check  # 2
    n_y = n_var  # 3

    syndrome_reg = QuantumRegister(n_syndrome, name="Bty")
    y_reg = QuantumRegister(n_y, name="y")

    threshold_list = []
    for var in range(n_var):
        connected_cnodes = np.where(Bt[:, var] == 1)[0]
        threshold = len(connected_cnodes)
        threshold_list.append(threshold)

    max_threshold = np.max(threshold_list)
    qubits_hamming = math.ceil(
        math.log2(max_threshold + 1),
    )  # necessary qubits to store result

    comparator_qubits = qubits_hamming

    ham_store_reg = QuantumRegister(qubits_hamming, name="ham_store")
    comparator_reg = QuantumRegister(comparator_qubits, name="comp_ancilla")

    bp_qc = QuantumCircuit(
        syndrome_reg,
        y_reg,
        ham_store_reg,
        comparator_reg,
    )  # ANCILLA +BT.y circuit
    for it in range(1, num_iterations + 1):
        # for each iteration we create a register flips

        if it == 1:  # if it is the first iteration
            flip_register = None
        else:
            # Add Bt flip
            for i in range(n):  # Now iterate over n (columns of B, rows of B^T)
                for j in range(m):  # Iterate over m (rows of B, columns of B^T)
                    if Bt[i, j] == 1:  # This now accesses B^T properly
                        bp_qc.cx(
                            flip_register[j],
                            syndrome_reg[i],
                        )  # Controlled-XOR from input to output

        flip_register = QuantumRegister(n_y, name="flip_" + str(it))
        bp_qc.add_register(flip_register)

        for var in range(n_var):

            connected_cnodes = np.where(Bt[:, var] == 1)[0]
            threshold = len(connected_cnodes)

            if (
                threshold > 0
            ):  # TO DO: explore case where threshold is equal to 0 (might be able to improve classical code)
                necessary_qubits = math.ceil(math.log2(threshold + 1))

                # HAMMING WEIGHT
                store_ham_qubits = ham_store_reg[:necessary_qubits]
                controlled_plus_one_gate = transpiled_control_plus_one_gate_efficient(
                    necessary_qubits,
                )
                for index in connected_cnodes:
                    control_qubit = syndrome_reg[index]
                    target_qubits = store_ham_qubits
                    qubit_mapping = [control_qubit] + target_qubits
                    # Compose the transpiled circuit into `qc` at the specified qubits
                    bp_qc.compose(
                        controlled_plus_one_gate,
                        qubits=qubit_mapping,
                        inplace=True,
                    )

                # COMPARATOR
                ancilla_comparator_qubits = comparator_reg[:necessary_qubits]
                comparator = IntegerComparator(
                    num_state_qubits=necessary_qubits,
                    value=threshold,
                    geq=True,
                )

                comparator_transpiled = transpile(
                    comparator,
                    basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
                )
                bp_qc.compose(
                    comparator_transpiled,
                    qubits=store_ham_qubits[:] + ancilla_comparator_qubits[:],
                    inplace=True,
                )

                # CNOT TO UPDATES FLIP
                bp_qc.cx(
                    control_qubit=ancilla_comparator_qubits[0],
                    target_qubit=flip_register[var],
                )

                # UNCOMPUTE HAMMING AND COMPARATOR QUBITS TO REUSE THEM FOR NEXT ITERATION

                comparator_inverse = transpile(
                    comparator.inverse(),
                    basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
                )
                bp_qc.compose(
                    comparator_inverse,
                    qubits=store_ham_qubits[:] + ancilla_comparator_qubits[:],
                    inplace=True,
                )

                controlled_plus_one_gate_inverse = (
                    transpiled_control_plus_one_gate_efficient(
                        necessary_qubits,
                        inverse=True,
                    )
                )
                for index in connected_cnodes:
                    control_qubit = syndrome_reg[index]
                    target_qubits = store_ham_qubits
                    qubit_mapping = [control_qubit] + target_qubits
                    # Compose the transpiled circuit into `qc` at the specified qubits
                    bp_qc.compose(
                        controlled_plus_one_gate_inverse,
                        qubits=qubit_mapping,
                        inplace=True,
                    )
            else:
                bp_qc.x(flip_register[var])

        # bp_qc.barrier()

    # Get all quantum registers from qc
    all_registers = bp_qc.qregs

    # Create a new circuit with all those registers
    full_circuit = QuantumCircuit(*all_registers, name="Belief Propagation")

    # Flatten qubits in bp_qc to match the order
    bp_qc_qubits = [qubit for reg in bp_qc.qregs for qubit in reg]

    # Compose bp_qc directly into full_circuit
    full_circuit.compose(bp_qc, qubits=bp_qc_qubits, inplace=True)

    # full_circuit.barrier()

    for it in range(1, num_iterations + 1):
        flip_register = [
            reg for reg in full_circuit.qregs if reg.name == "flip_" + str(it)
        ][0]

        for flip_qubit, y_qubit in zip(flip_register, y_reg):
            full_circuit.cx(flip_qubit, y_qubit)

    full_circuit.barrier()

    bp_qc_inv = explicit_inverse(bp_qc)
    full_circuit.compose(bp_qc_inv, qubits=bp_qc.qubits, inplace=True)

    return full_circuit


def check_plus_one_mapping(n, atol=1e-12, verbose=False):
    """
    Verify that for every computational basis state |x>, the circuit U_{+1} maps it to |(x+1) mod 2^n>.

    This function builds a quantum circuit that performs an increment-by-one operation
    and checks that the output matches the expected basis state for all inputs.

    Parameters
    ----------
    n : int
        Number of qubits (defines the size of the Hilbert space, 2^n).
    atol : float, optional
        Absolute tolerance for checking amplitude magnitudes (default: 1e-12).
    verbose : bool, optional
        If True, prints verbose output for debugging or small circuits.

    Returns
    -------
    bool
        True if all mappings are correct; raises AssertionError otherwise.
    """

    N = 2**n
    # Build a tiny circuit that applies U_{+1} once
    gate = increment_of_one_gate(n)
    circ = QuantumCircuit(n)
    circ.append(gate, range(n))

    for x in range(N):
        # Qiskit uses little-endian: qubit 0 is LSB, matching our construction.
        sv_in = Statevector.from_int(x, dims=[2] * n)
        sv_out = sv_in.evolve(circ)

        # Identify which basis state has amplitude ~1
        idx = int(np.argmax(np.abs(sv_out.data)))
        amp = sv_out.data[idx]
        expected = (x + 1) % N

        # Check index AND that the amplitude has unit magnitude
        if idx != expected or not np.isclose(abs(amp), 1.0, atol=atol):
            raise AssertionError(
                f"Failed for n={n}, x={x}: got |{idx}>, |amp|={abs(amp)}, expected |{expected}>",
            )

        if verbose and n <= 6:
            print(f"|{x}> -> |{idx}> ✓")

    if verbose:
        print(f"All {N} basis mappings correct for n={n}")
    return True


if __name__ == "__main__":

    for n in [2, 3, 4]:
        # check_plus_one_mapping(n, verbose=True)
        print("\n")
        print(n)
        print("\n")

        final_circuit = transpiled_control_plus_one_gate(n)
        gate_counts = final_circuit.count_ops()

        qubit_count = final_circuit.num_qubits

        print("Total number of qubits:", qubit_count)
        print("Total number of gates:", sum(gate_counts.values()))
        print("Gate breakdown:")
        for gate, count in gate_counts.items():
            print(f"  {gate}: {count}")

        print("\n")
        final_circuit = transpiled_control_plus_one_gate_efficient(n)
        gate_counts = final_circuit.count_ops()

        qubit_count = final_circuit.num_qubits

        print("Total number of qubits:", qubit_count)
        print("Total number of gates:", sum(gate_counts.values()))
        print("Gate breakdown:")
        for gate, count in gate_counts.items():
            print(f"  {gate}: {count}")
