import math
import random
import warnings

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from pipelines.belief_propagation_circuit import BP1_circuit

# Third party code from https://github.com/BankNatchapol/DQI-Circuit:
from pipelines.dicke_states import (
    UnkStatePreparation,
    WeightedUnaryEncoding,
    get_optimal_w,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def hadamard_transpile():
    """
    Returns a transpiled 1-qubit Hadamard gate using a limited basis gate set.

    Constructs a one-qubit circuit containing a Hadamard gate and transpiles it
    into the specified basis gate set: ['z', 'cx', 'rx', 'ry', 'rz', 'swap'].
    This is useful for hardware-compatible circuit generation where native Hadamard
    gates are not directly available.

    Returns:
        QuantumCircuit: A one-qubit circuit implementing the Hadamard gate using only the allowed basis gates.
    """
    # Create a dummy 1-qubit circuit with an H gate
    h_circuit = QuantumCircuit(1)
    h_circuit.h(0)

    # Transpile it using your chosen basis gates
    transpiled_h_circuit = transpile(
        h_circuit,
        basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
    )

    return transpiled_h_circuit


def apply_phases_v(qc, v, data_qubits):
    """
    Apply Z gate to qubits based on the values in `v`.

    Parameters:
    -----------
    qc : QuantumCircuit
        The quantum circuit to which the Z gates will be applied.

    v : list
        A list of phase values corresponding to each qubit. Z gate is applied where v[i] > 0.

    data_qubits : list
        The list of qubit indices in the circuit.

    Returns:
    --------
    qc : QuantumCircuit
        The modified quantum circuit with Z gates applied where needed.
    """

    # Check that v and data_qubits have the same length
    if len(v) != len(data_qubits):
        raise ValueError(
            f"Error: v (length {len(v)}) and data_qubits (length {len(data_qubits)}) must have the same length.",
        )

    # Apply Z gates where v[i] > 0
    for qubit, v_element in zip(data_qubits, v):
        if v_element > 0:
            qc.z(qubit)  # Apply Z gate

    return qc


def matrixT_vector_multiplication(qc, B, data_qubits):
    """
    Compute |B^T y> for each basis state |y> in 'data_qubits',
    where B has shape (m x n).

    Args:
        qc          : A QuantumCircuit (already has 'data_qubits').
        B           : 2D numpy array/list-of-lists, shape (m x n).
        data_qubits : The list of m qubits representing |y>.

    Returns:
        (qc, bt_qubits): The modified circuit and the list of n newly-created qubits.
    """
    m, n = B.shape  # B has dimensions (m x n)

    # Create n new qubits to store B^T y
    bt_reg = QuantumRegister(n, name="bt")
    qc.add_register(bt_reg)
    bt_qubits = bt_reg[:]

    # Compute B^T y
    for i in range(n):  # Now iterate over n (columns of B, rows of B^T)
        for j in range(m):  # Iterate over m (rows of B, columns of B^T)
            if B.T[i, j] == 1:  # This now accesses B^T properly
                qc.cx(
                    data_qubits[j],
                    bt_qubits[i],
                )  # Controlled-XOR from input to output

    return qc, bt_qubits


def dqi_max_xorsat(B, v, W_k, num_iterations_bp, print_circuit_progress=False):
    """
    Constructs the full DQI quantum circuit for solving the MAX-XORSAT problem.

    This function builds a quantum circuit designed to solve instances of the MAX-XORSAT problem using
    techniques described in https://arxiv.org/pdf/2504.18334. The circuit:
    - Initializes a weighted unary register with amplitudes W_k.
    - Prepares a Dicke state over the error register |y⟩.
    - Applies a phase oracle conditioned on the syndrome vector v.
    - Computes Bᵗy and stores it in a register.
    - Uncomputes |y⟩ using belief propagation (BP1).
    - Applies Hadamard transforms to finalize the phase encoding.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Syndrome vector of length m.
        W_k (np.ndarray): Weight vector defining the initial amplitudes of the unary register.
        num_iterations_bp (int): Number of belief propagation iterations to apply.
        print_circuit_progress (bool, optional): If True, prints progress at each circuit stage. Default is False.

    Returns:
        Tuple[QuantumCircuit, QuantumRegister, QuantumRegister]:
            - The complete quantum circuit.
            - Quantum register representing |y⟩ (should be initialized to |0⟩ⁿ).
            - Quantum register representing |Bᵗy⟩.

    Raises:
        ValueError: If `bp` is not 0 or 1.
    """

    # convert B to integers:
    B = B.astype(int)

    m, n = B.shape  # m rows, n columns

    # ----- INITIALIZE WEIGHT REGISTER -----

    # Build the initialization circuit using Weighted Unary Encoding
    init_qregs = QuantumRegister(m, name="k")
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = WeightedUnaryEncoding(m, W_k)

    # Create a temporary circuit for the gate
    temp_circuit = QuantumCircuit(m)
    temp_circuit.append(WUE_Gate, range(m))
    # Transpile the block to your desired basis gates
    transpiled_block = transpile(
        temp_circuit,
        basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
    )

    # Append the transpiled block to your initialize_circuit
    initialize_circuit.compose(transpiled_block, qubits=range(m), inplace=True)

    if print_circuit_progress:
        print("Initialisation")

    # -------------------DICKE STATE CREATION-------------------------

    # Prepare the Dicke state circuit
    y_reg = QuantumRegister(m, name="y")
    dicke_circuit = QuantumCircuit(y_reg)
    max_errors = int(np.nonzero(W_k)[0][-1]) if np.any(W_k) else 0

    # Create a temporary circuit for the gate
    temp_circuit = QuantumCircuit(m)
    temp_circuit.append(UnkStatePreparation(m, max_errors).to_gate(), range(m))

    transpiled_block = transpile(
        temp_circuit,
        basis_gates=["z", "cx", "rx", "ry", "rz", "swap"],
    )
    dicke_circuit.compose(transpiled_block, qubits=range(m), inplace=True)

    if print_circuit_progress:
        print("Dicke state creation")

    # DEFINE CIRCUIT

    qc_y = QuantumCircuit(y_reg)
    qc_y.compose(initialize_circuit, inplace=True)
    qc_y.compose(dicke_circuit, inplace=True)

    # ---- APLY PHASES CONDITIONED ON V FOR Y_REG-----------------

    qc_y = apply_phases_v(qc_y, v, y_reg)
    if print_circuit_progress:
        print("Phases applied")

    # ----- Create BT_y register with matrix vector multiplication ------------------

    qc_2_regs, bt_reg = matrixT_vector_multiplication(qc_y, B, y_reg)

    if print_circuit_progress:
        print("Bt_Y")

    # ------------------------------ BELIEF PROPAGATION TO UNCOMPUTE Y ----------------------
    bpc = BP1_circuit(B, num_iterations_bp)

    # UNITE together both circuits

    # Registers in qc
    bt_main = [reg for reg in qc_2_regs.qregs if reg.name == "bt"][0]
    y_main = [reg for reg in qc_2_regs.qregs if reg.name == "y"][0]

    # Registers in bpc
    bt_bpc = [reg for reg in bpc.qregs if reg.name == "Bty"][0]
    y_bpc = [reg for reg in bpc.qregs if reg.name == "y"][0]
    # Find ancillas used in bpc (anything not bt or y)
    ancilla_regs_bpc = [reg for reg in bpc.qregs if reg not in [bt_bpc, y_bpc]]

    # Create matching ancilla registers in qc
    new_ancillas_in_qc = []
    for reg in ancilla_regs_bpc:
        new_reg = QuantumRegister(len(reg), name=f"{reg.name}_from_bpc")
        qc_2_regs.add_register(new_reg)
        new_ancillas_in_qc.append(new_reg)

    # Reconstruct the full qarg list in the exact order bpc expects
    qarg_list = []
    qarg_list.extend(bt_main[i] for i in range(len(bt_bpc)))
    qarg_list.extend(y_main[i] for i in range(len(y_bpc)))

    for orig_reg, new_reg in zip(ancilla_regs_bpc, new_ancillas_in_qc):
        qarg_list.extend(new_reg[i] for i in range(len(orig_reg)))

    qc_2_regs.compose(bpc, qubits=qarg_list, inplace=True)

    if print_circuit_progress:
        print("BP")
    # --------------------------------- FINAL HADAMARDS --------------------------------------------

    h_transpiled = hadamard_transpile()
    for qubit in bt_main:
        # Compose the transpiled H circuit at the correct qubit
        qc_2_regs.compose(h_transpiled, qubits=[qubit], inplace=True)

    if print_circuit_progress:
        print("hadamards")

    return qc_2_regs, y_main, bt_main


def s(B, v, x):
    """
    Computes the number of satisfied parity-check constraints for a given bitstring x.

    For each row of the parity-check matrix B, checks whether the linear constraint
    Bᵢ·x ≡ vᵢ (mod 2) is satisfied. Returns the total count of satisfied constraints.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Binary vector of length m representing the syndrome or constraint targets.
        x (Union[str, List[int]]): Bitstring (as a string of '0'/'1' or list of ints) representing a candidate solution.

    Returns:
        int: Number of satisfied constraints (between 0 and m).

    Raises:
        ValueError: If the bitstring format is invalid or does not match the expected length.
    """
    # Convert bitstring to list of ints if needed
    if isinstance(x, str):
        if not all(c in "01" for c in x):
            raise ValueError("Bitstring must contain only '0' and '1'")
        x = [int(bit) for bit in x]

    num_constrains = B.shape[0]
    num_variables = B.shape[1]

    if num_variables != len(x):
        raise ValueError(
            f"Mismatch: expected {num_variables} variables, but got {len(x)} values in x.",
        )

    satisfied_constrains = 0
    for constrain in range(num_constrains):
        v_op = 0
        for var in range(num_variables):
            v_op = (v_op + B[constrain, var] * x[var]) % 2

        if v[constrain] == v_op:
            satisfied_constrains += 1

    return satisfied_constrains


def f(B, v, x):
    """
    Computes the value of the function f = Σᵢ (-1)^(Bᵢ·x + vᵢ) for a given bitstring x.

    The function evaluates how well the bitstring x satisfies the parity-check constraints
    encoded in matrix B and vector v. For each constraint, it adds +1 if satisfied,
    and -1 if violated.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Binary vector of length m representing the syndrome or parity targets.
        x (Union[str, List[int]]): Bitstring (as string of '0'/'1' or list of ints) representing a candidate solution.

    Returns:
        int: The total score indicating how well x satisfies the constraints (range: [-m, m]).

    Raises:
        ValueError: If the bitstring format is invalid or its length does not match the number of variables.
    """
    # Convert bitstring to list of ints if needed
    if isinstance(x, str):
        if not all(c in "01" for c in x):
            raise ValueError("Bitstring must contain only '0' and '1'")
        x = [int(bit) for bit in x]

    num_constrains = B.shape[0]
    num_variables = B.shape[1]

    if num_variables != len(x):
        raise ValueError(
            f"Mismatch: expected {num_variables} variables, but got {len(x)} values in x.",
        )

    f_total = 0
    for constrain in range(num_constrains):
        v_op = 0
        for var in range(num_variables):
            v_op = (v_op + B[constrain, var] * x[var]) % 2

        if v[constrain] == v_op:
            f_total += 1
        else:
            f_total += -1

    return f_total


def resource_estimation_function(
    B,
    v,
    ell,
    num_iterations_bp,
    print_info=False,
    print_circuit_progress=False,
):
    """
    Estimate quantum resource usage for solving a Max-XORSAT problem using
    the Decoding by Quantum Inference (DQI) approach.

    This function computes the optimal parameter `w`, constructs the full
    quantum circuit for the DQI-based Max-XORSAT solver, and reports
    resource usage statistics such as total qubit count and gate composition.

    Args:
        B (np.ndarray): A binary matrix (m x n) representing the XORSAT constraints.
        v (np.ndarray): A binary vector (length m) representing the right-hand side of the constraints.
        max_errors (int): The maximum number of allowed bit errors in the solution.
        num_iterations_bp (int): The number of iterations for the belief propagation phase.

    Returns:
        tuple:
            - int: Total number of qubits used in the constructed quantum circuit.
            - dict: A dictionary with gate counts keyed by gate name (from `count_ops()`).
    """

    W_k = get_optimal_w(m=B.shape[0], ell=ell)

    if print_circuit_progress:
        print("w computed")

    final_circuit, data_qubits, bt_qubits = dqi_max_xorsat(
        B,
        v,
        W_k,
        num_iterations_bp,
        print_circuit_progress=print_circuit_progress,
    )

    gate_counts = final_circuit.count_ops()

    qubit_count = final_circuit.num_qubits

    if print_info:
        print("Total number of qubits:", qubit_count)
        print("Total number of gates:", sum(gate_counts.values()))
        print("Gate breakdown:")
        for gate, count in gate_counts.items():
            print(f"  {gate}: {count}")

    m, n = B.shape

    # t = max number of variables in any constraint = max nonzeros per row
    if hasattr(B, "getnnz"):  # scipy.sparse
        row_nnz = np.asarray(B.getnnz(axis=1)).ravel()
    else:  # dense
        row_nnz = np.count_nonzero(B, axis=1)

    t = int(row_nnz.max()) if row_nnz.size else 0  # handle empty edge case

    # log term: 2 * ceil(log2(t+1))
    log_term = 2 * math.ceil(math.log2(t + 1)) if t >= 0 else 0  # t>=0 always true

    # Final formula:
    Nq = (num_iterations_bp + 1) * m + n + log_term

    if Nq != qubit_count:
        raise ValueError(
            f"Number of qubits mismatch: predicted {Nq}, but circuit has {qubit_count}.",
        )
    else:
        print("Number of qubits predicted test passed.")

    return qubit_count, gate_counts


def quantum_dqi_results(
    B,
    v,
    ell,
    num_iterations_bp,
    shots=10**5,
    print_info=False,
):
    """
    Run the DQI-based quantum algorithm for Max-XORSAT and estimate solution quality.

    Constructs and simulates the full quantum circuit for solving the Max-XORSAT problem
    using Decoding by Quantum Inference (DQI). The circuit is postselected on both
    uncomputed data qubits (y=0) and ancilla qubits (all 0), and the quality of sampled
    solutions is measured using custom scoring functions.

    Args:
        B (np.ndarray): Binary matrix representing the constraint system (shape m x n).
        v (np.ndarray): Binary vector representing the right-hand side of the constraints.
        l (int): Maximum number of bit errors allowed.
        num_iterations_bp (int): Number of belief propagation iterations.
        shots (int): Number of measurement shots to run. Default is 100,000.
        print_info (bool): Whether to print detailed results. Default is False.

    Returns:
        tuple:
            - float: Average number of satisfied constraints over postselected samples.
            - float: Average value of the custom objective function f(B, v, x) over postselected samples.
            - int: Number of postselected shots (with y=0 and ancilla=0).
    """

    W_k = get_optimal_w(m=B.shape[0], ell=ell)
    final_circuit, data_qubits, bt_qubits = dqi_max_xorsat(
        B,
        v,
        W_k,
        num_iterations_bp,
    )

    data_creg = ClassicalRegister(B.shape[0], "y_bits")
    bt_creg = ClassicalRegister(B.shape[1], "bt_bits")
    final_circuit.add_register(bt_creg)
    final_circuit.add_register(data_creg)
    # Measure qubits into their respective registers
    final_circuit.measure(bt_qubits, bt_creg)
    final_circuit.measure(data_qubits, data_creg)

    # Identify all measured qubits so far
    measured_qubits = set(bt_qubits[:] + data_qubits[:])
    # Find all qubits in the circuit
    all_qubits = [q for qreg in final_circuit.qregs for q in qreg]
    # Identify ancilla qubits (not in bt_y_reg or y_reg)
    ancilla_qubits = [q for q in all_qubits if q not in measured_qubits]
    # Create classical register for ancillas
    ancilla_creg = ClassicalRegister(len(ancilla_qubits), "ancilla_bits")
    final_circuit.add_register(ancilla_creg)
    # Measure ancilla qubits
    final_circuit.measure(ancilla_qubits, ancilla_creg)

    # Set fixed seed for reproducibility
    seed_simulator = 57
    seed_transpiler = 57
    simulator = AerSimulator(seed_simulator=seed_simulator)
    transpiled_circuit = transpile(
        final_circuit,
        simulator,
        seed_transpiler=seed_transpiler,
    )
    # Run and get counts
    result = simulator.run(
        transpiled_circuit,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()
    counts = result.get_counts()

    # Count how many shots had all ancilla bits equal to 0
    total_shots = sum(counts.values())

    postselected_shots = 0
    satisfied_constrians_sum = 0
    f_sum = 0
    for bitstring, count in counts.items():
        # Qiskit orders classical bits from last-added (leftmost) to first-added (rightmost)
        # So ancilla bits are on the left of the full bitstring
        # print(bitstring)
        ancilla_bits = bitstring.split(" ")[0]
        y_bits = bitstring.split(" ")[1][::-1]
        x_bits = bitstring.split(" ")[2][::-1]

        if not all(
            bit == "0" for bit in ancilla_bits
        ):  # if in some bitstring theres a uncomputed ancilla
            raise RuntimeError(
                f"Uncomputed ancilla detected in result: ancilla_bits = {ancilla_bits}",
            )

        if all(bit == "0" for bit in y_bits):  # POSTSELECT ON Y REG BEING UNCOMPUTED
            # print(x_bits,count)
            postselected_shots += 1 * count
            satisfied_constrians_sum += s(B, v, x_bits) * count
            f_sum += f(B, v, x_bits) * count

    average_satisfied = satisfied_constrians_sum / postselected_shots
    f_average = f_sum / postselected_shots

    if print_info:

        print("<s> = ", average_satisfied)
        print("<f> = ", f_average)
        print("Postselected y=000 shots= ", postselected_shots, "out of ", total_shots)

    return average_satisfied, f_average, postselected_shots


def average_of_f_and_s_random(B, v, shots, histogram=False):
    """
    Computes the average and standard error of f and s over randomly sampled bitstrings.
    Optionally returns a histogram of s value distributions.

    Args:
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Syndrome vector or parameter vector used by functions f and s.
        shots (int): Number of random bitstrings to sample.
        histogram (bool): Whether to return histogram of s values.

    Returns:
        If histogram == False:
            Tuple[float, float, float, float]: Mean and SEM of f and s.
        If histogram == True:
            Tuple[float, float, float, float, dict]: Mean and SEM of f and s, and histogram.
    """
    num_variables = B.shape[1]

    f_values = []
    s_values = []
    s_hist = {}

    for _ in range(shots):
        bitstring = "".join(random.choice("01") for _ in range(num_variables))
        f_val = f(B, v, bitstring)
        s_val = s(B, v, bitstring)
        f_values.append(f_val)
        s_values.append(s_val)

        if histogram:
            s_hist[s_val] = s_hist.get(s_val, 0) + 1

    f_values = np.array(f_values)
    s_values = np.array(s_values)

    avg_f = np.mean(f_values)
    avg_s = np.mean(s_values)
    sem_f = np.std(f_values, ddof=1) / np.sqrt(shots)
    sem_s = np.std(s_values, ddof=1) / np.sqrt(shots)

    if histogram:
        return avg_f, avg_s, s_hist, sem_f, sem_s
    else:
        return avg_f, avg_s, sem_f, sem_s


def quantum_dqi_histogram_results(
    B,
    v,
    ell,
    num_iterations_bp,
    shots=10**5,
    print_info=False,
    fake_backend=None,
    approximation_degree=1,
):
    """
    Run the DQI-based quantum algorithm for Max-XORSAT and return histogram of solution quality.

    Returns:
        s_hist (dict): Histogram of satisfied constraints over postselected samples.
        average_satisfied (float): Mean number of satisfied constraints.
        std_error (float): Statistical error (standard deviation / sqrt(N)).
        postselected_shots (int): Count of postselected shots with y=0.
        total_shots (int): Total number of shots performed.
    """
    W_k = get_optimal_w(B.shape[0], ell)
    # print(W_k)
    final_circuit, data_qubits, bt_qubits = dqi_max_xorsat(
        B,
        v,
        W_k,
        num_iterations_bp,
    )

    data_creg = ClassicalRegister(B.shape[0], "y_bits")
    bt_creg = ClassicalRegister(B.shape[1], "bt_bits")
    final_circuit.add_register(bt_creg)
    final_circuit.add_register(data_creg)
    final_circuit.measure(bt_qubits, bt_creg)
    final_circuit.measure(data_qubits, data_creg)

    measured_qubits = set(bt_qubits[:] + data_qubits[:])
    all_qubits = [q for qreg in final_circuit.qregs for q in qreg]
    ancilla_qubits = [q for q in all_qubits if q not in measured_qubits]
    ancilla_creg = ClassicalRegister(len(ancilla_qubits), "ancilla_bits")
    final_circuit.add_register(ancilla_creg)
    final_circuit.measure(ancilla_qubits, ancilla_creg)

    seed_simulator = 12345
    seed_transpiler = 12345

    if fake_backend is None:
        simulator = AerSimulator(seed_simulator=seed_simulator)
    else:
        simulator = AerSimulator.from_backend(fake_backend)

    transpiled_circuit = transpile(
        final_circuit,
        simulator,
        seed_transpiler=seed_transpiler,
        approximation_degree=approximation_degree,
    )

    result = simulator.run(
        transpiled_circuit,
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()
    counts = result.get_counts()

    s_hist = {}
    postselected_shots = 0
    total_shots = sum(counts.values())
    s_values = []
    filtered_counts = {}

    for bitstring, count in counts.items():
        ancilla_bits = bitstring.split(" ")[0]
        y_bits = bitstring.split(" ")[1][::-1]
        x_bits = bitstring.split(" ")[2][::-1]

        if (
            not all(bit == "0" for bit in ancilla_bits)
            and fake_backend is None
            and approximation_degree == 1
        ):
            raise RuntimeError(
                f"Uncomputed ancilla detected: ancilla_bits = {ancilla_bits}",
            )

        if all(bit == "0" for bit in y_bits):
            filtered_counts[x_bits] = filtered_counts.get(x_bits, 0) + count
            s_val = s(B, v, x_bits)
            s_hist[s_val] = s_hist.get(s_val, 0) + count
            s_values.extend([s_val] * count)
            postselected_shots += count

    if postselected_shots > 0:
        s_array = np.array(s_values)
        average_satisfied = np.mean(s_array)
        std_error = np.std(s_array, ddof=1) / np.sqrt(postselected_shots)
    else:
        average_satisfied = 0
        std_error = 0

    if print_info:
        print("<s> = ", average_satisfied)
        print("Standard Error = ", std_error)
        print("Postselected y=000 shots= ", postselected_shots, "out of ", total_shots)

    # print(filtered_counts)
    # print(sum(filtered_counts.values()))'
    return s_hist, average_satisfied, std_error, postselected_shots, total_shots
