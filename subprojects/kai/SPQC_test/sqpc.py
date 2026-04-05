import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from functools import reduce

def create_efficientsu2_ansatz(n, thetas, reps=1):
    """
    Manually create an EfficientSU2-like ansatz for gradient compatibility.
    
    Args:
        n: Number of qubits
        thetas: List of parameters
        reps: Number of repetitions (default: 1)
    
    Returns:
        QuantumCircuit: EfficientSU2-like circuit with regular Parameters
    """
    circuit = QuantumCircuit(n)
    param_idx = 0
    
    for rep in range(reps):
        # Rotation layer: RY and RZ on each qubit
        for q in range(n):
            circuit.ry(thetas[param_idx], q)
            param_idx += 1
        for q in range(n):
            circuit.rz(thetas[param_idx], q)
            param_idx += 1
            
        # Entangling layer: CNOT gates (linear connectivity)
        # Skip entangling layer on the last repetition
        if rep < reps - 1:
            for q in range(n - 1):
                circuit.cx(q, q + 1)
    
    # Final entangling layer after all reps
    for q in range(n - 1):
        circuit.cx(q, q + 1)
        
    return circuit

def create_spqc_circuit(t=0, m=2, n=2, r=1):
    """
    Create and return an SPQC circuit with specified parameters.
    
    Args:
        t: number of polynomial terms (can be 0)
        m: address register size  
        n: data register size
        r: number of data registers
        
    Returns:
        QuantumCircuit: The complete SPQC circuit
    """
    # --- hyperparameters and registers ---
    total_qubits = n*r+m+t+1    # total number of qubits (1 for helper ancilla control)
    L = 2**m                    # number of sub-models
    T = 2**t                    # total number of term addresses
    anc = 1                     # single ancilla qubit to help with multi-control
    term_register = range(0,t)
    address_register = range(t,t+m)
    data_registers = [range(t+m+i*n,t+m+(i+1)*n) for i in range(r)]
    ancilla = [total_qubits - 1]  # ancilla index

    ## --- SPQC Circuit --- 
    # Feature map
    feature_map = QuantumCircuit(n)
    input_thetas = [Parameter(f"zinput_theta{i}") for i in range(2)] # z for alphabetical order (should come last)
    feature_map.ry(input_thetas[0]*2*np.pi,0)
    feature_map.ry(input_thetas[1]*2*np.pi,1)
    fm = feature_map.to_gate(label=f"S(X)")

    # Parameterised sub-models
    sub_models = []
    for i in range(L):
        # === CUSTOM ANSATZ ===
        thetas = [Parameter(f"model{i}_theta{j}") for j in range(4)]
        sub_model = QuantumCircuit(n)
        sub_model.ry(thetas[0], 0)
        sub_model.rx(thetas[1], 0)
        sub_model.ry(thetas[2], 1)
        sub_model.rx(thetas[3], 1)  
        sub_models.append(sub_model.to_gate(label=f"model{i}"))
        
        # === EFFICIENTSU2-LIKE ANSATZ ===
        # reps=1
        # thetas = [Parameter(f"model{i}_theta{j}") for j in range(4*n*reps)]
        # sub_model = create_efficientsu2_ansatz(n, thetas, reps)
        # sub_models.append(sub_model.to_gate(label=f"model{i}"))
   

    # Create SPQC with classical bits for measurements, append Hadamards and feature maps
    num_classical_bits = n*r + m  # One classical bit per data/address qubit
    qc = QuantumCircuit(total_qubits, num_classical_bits)
    qc.h(address_register)

    # Prepare term register into "diagonal" state, eg. (|100>  + |110> + |111>) / sqrt(3)
    def pyramid_state(t: int, inverse: bool = False) -> QuantumCircuit:
        """
        Builds the t-qubit pyramid state circuit (or its inverse if inverse=True)
        • (|10…0> + |110…0> + … + |1…1>) / sqrt(t)
        • t = 0 → empty circuit
        • t = 1 → X on qubit 0
        • t ≥ 2 → X on qubit 0, then t-1 controlled-Ry's
        """
        qc = QuantumCircuit(t)
        qc.x(0)
        # cascade controlled Ry’s for k=2..t
        for k in range(2, t+1):
            θ = 2 * np.arccos(1 / np.sqrt(t - k + 2))
            qc.cry(θ, k-2, k-1) # control = qubit k-2, target = qubit k-1
        return qc.inverse() if inverse else qc
    if t > 0:
        qc.append(pyramid_state(t), term_register)

    # Apply feature maps
    for i in range(len(data_registers)):
        qc.append(fm, data_registers[i])  # Feature maps

    # Helper function to flip bits
    def flip_bits(qc, register, address):
        for j, bit in enumerate(address):
            if bit == '0':
                qc.x(register[j])

    # Append sub-models
    for i in range(L):
        m_address = format(i, f'0{m}b')

        flip_bits(qc, address_register, m_address)

        if t > 0:
            # Apply sub-models conditionally
            for k in range(len(data_registers)):
                data_register = data_registers[k]
                controlled_sub_model = sub_models[i].control(num_ctrl_qubits=1)
                control_qubits = list(term_register)[:k+1] + list(address_register)
                qc.mcx(control_qubits, ancilla) # Use helper ancilla to control sub-model
                target_qubits = list(data_register)
                qc.append(controlled_sub_model, ancilla + target_qubits)
                qc.mcx(control_qubits, ancilla) # Reset ancilla

        else:
            # Apply sub-models conditionally
            control_qubits = list(address_register)
            qc.mcx(control_qubits, ancilla) # Use helper ancilla to control sub-model
            for k in range(len(data_registers)):
                data_register = data_registers[k]
                controlled_sub_model = sub_models[i].control(num_ctrl_qubits=1)
                target_qubits = list(data_register)
                qc.append(controlled_sub_model, ancilla + target_qubits)
            qc.mcx(control_qubits, ancilla) # Reset ancilla

        flip_bits(qc, address_register, m_address)
        qc.barrier(label="---")

    # Measure data registers
    classical_bit_index = 0
    for i, data_register in enumerate(data_registers):
        for qubit in data_register:
            qc.measure(qubit, classical_bit_index)
            classical_bit_index += 1
    qc.barrier(label="---")

    # Reset term register
    if t > 0:
        qc.append(pyramid_state(t, inverse=True), term_register)

    # Create address register ansatz
    # address_ansatz = EfficientSU2(m, reps=1, parameter_prefix='address_theta')
    # qc.append(address_ansatz, address_register)
    address_ansatz = QuantumCircuit(m)
    address_ansatz.ry(Parameter('address_theta_0'), 0)
    address_ansatz.ry(Parameter('address_theta_1'), 1)
    address_ansatz.ry(Parameter('address_theta_2'), 2)
    address_ansatz.rx(Parameter('address_theta_3'), 0)
    address_ansatz.rx(Parameter('address_theta_4'), 1)
    address_ansatz.rx(Parameter('address_theta_5'), 2)
    address_ansatz.rz(Parameter('address_theta_6'), 0)
    address_ansatz.rz(Parameter('address_theta_7'), 1)
    address_ansatz.rz(Parameter('address_theta_8'), 2)
    qc.append(address_ansatz, address_register)

    # Measure address register
    for i, qubit in enumerate(address_register):
        qc.measure(qubit, n*r + i)
    
    return qc

def get_parameter_mapping(circuit):
    """
    Create a mapping from parameter names to Parameter objects.
    This helps organize parameters by type (input vs model weights).
    """
    input_params = []
    model_params = []
    address_params = []
    
    for param in circuit.parameters:
        if param.name.startswith('zinput_theta'):
            input_params.append(param)
        elif param.name.startswith('model'):
            model_params.append(param)
        elif param.name.startswith('address_theta'):
            address_params.append(param)
    
    return input_params, model_params, address_params

def create_random_weights(circuit, seed=None, transpiled=False):
    """
    Create random weights for the SPQC with disjoint weight sets for each model.
    
    Args:
        circuit: The SPQC circuit
        seed: Random seed for reproducibility (optional)
        
    Returns:
        numpy.ndarray: Combined array of random weights (model + address)
    """
    if seed is not None:
        np.random.seed(seed)
    
    input_params, model_params, address_params = get_parameter_mapping(circuit)
    
    # Group model parameters by model number
    model_groups = {}
    for param in model_params:
        # Extract model number from parameter name
        param_name = param.name
        if 'model' in param_name:
            # Extract the number after 'model'
            import re
            match = re.search(r'model(\d+)', param_name)
            if match:
                model_num = int(match.group(1))
            else:
                # Fallback: try the old method
                model_num = int(param_name.split('_')[0].replace('model', '').replace('[', '').replace(']', ''))
        else:
            continue  # Skip non-model parameters
            
        if model_num not in model_groups:
            model_groups[model_num] = []
        model_groups[model_num].append(param)
    
    # Create disjoint weight ranges for each model
    num_models = len(model_groups)
    params_per_model = len(model_groups[0]) if model_groups else 0
        
    # Define disjoint ranges for each model
    weight_ranges = []
    range_size = 2 * np.pi / num_models  # Divide [-π, π] into disjoint segments
    
    for i in range(num_models):
        start = -np.pi + i * range_size
        end = -np.pi + (i + 1) * range_size
        weight_ranges.append((start, end))
    
    # Generate random weights with disjoint ranges for models
    model_weights = []
    
    for model_num in sorted(model_groups.keys()):
        start, end = weight_ranges[model_num]
        weights = np.random.uniform(start, end, params_per_model)
        model_weights.extend(weights)
    
    # Generate random weights for address ansatz (separate range)
    address_weights = np.random.uniform(-np.pi, np.pi, len(address_params))
    
    # Combine model and address weights
    if not transpiled:
        all_weights = np.concatenate([address_weights, model_weights])
    else:
        all_weights = np.concatenate([model_weights, address_weights])
    
    return all_weights

def bind_params(circuit, input_values, random_weights):
    """
    Bind parameters to the circuit.
    
    Args:
        circuit: The SPQC circuit
        input_values: List of input feature values [x, y]
        random_weights: Combined array of model and address weights
    """
    input_params, model_params, address_params = get_parameter_mapping(circuit)
    param_binding = {}

    # Bind input parameters
    for i, param in enumerate(input_params):
        if i < len(input_values):
            param_binding[param] = input_values[i]
        else:
            param_binding[param] = 0.0
    
    # Split random_weights into model and address portions
    num_model_params = len(model_params)
    model_values = random_weights[:num_model_params]
    address_values = random_weights[num_model_params:]
    
    # Bind model parameters  
    for i, param in enumerate(model_params):
        if i < len(model_values):
            param_binding[param] = model_values[i]
        else:
            param_binding[param] = 0.0

    # Bind address parameters
    for i, param in enumerate(address_params):
        if i < len(address_values):
            param_binding[param] = address_values[i]
        else:
            param_binding[param] = 0.0

    return circuit.assign_parameters(param_binding)

def visualize_circuit(qc):
    print(f"Circuit: {qc.num_qubits} qubits, depth {qc.depth()}")
    qc.draw(output='mpl', fold=100)
    plt.show()

def post_select(counts, m, n, r):
    # reverse each bitstring so that index 0 is the leftmost char
    fixed = { bitstr[::-1]: ct for bitstr, ct in counts.items() }

    data_size = n*r
    post = {}
    for bitstr, count in fixed.items():
        # Check if data registers (first n*r bits after reversal) are all 0
        if bitstr[:data_size] == '0'*data_size:
            addr_bits = bitstr[data_size:data_size+m]  # address bits come after data bits
            post[addr_bits] = post.get(addr_bits, 0) + count

    total = sum(post.values())
    probs = np.zeros(2**m)
    if total:
        for bits, ct in post.items():
            probs[int(bits, 2)] = ct/total
    return probs

def pre_select_convert(counts, m, n, r):
    """
    Convert raw counts to probability vector over address register 
    WITHOUT post-selection (includes all measurements).
    
    Args:
        counts: Dictionary of measurement counts
        m: Number of address register qubits 
        n: Number of input features
        r: Polynomial degree
    
    Returns:
        numpy array: Probability vector of size 2^m (before post-selection)
    """
    total_shots = sum(counts.values())
    
    # Create probability vector for address register (without filtering)
    address_counts = {}
    for bitstring, count in counts.items():
        # Keep only first m bits (address register)
        address_bits = bitstring[:m]
        address_counts[address_bits] = address_counts.get(address_bits, 0) + count
    
    # Convert to probability vector of size 2^m
    prob_vector = np.zeros(2**m)
    for bitstring, count in address_counts.items():
        index = int(bitstring, 2)
        prob_vector[index] = count / total_shots
    
    return prob_vector

def model(qc, input_vals, weights, t, m, n, r, backend=None):
    """
    Binds input values and weights to circuit
    Returns post-selected address amplitude vector
    """
    # Bind input values and weights to circuit
    spqc = bind_params(qc, input_vals, weights)

    # Run circuit and extract statevector
    if backend is not None:
        # Use the provided backend (e.g., GPU-enabled AerSimulator)
        job = backend.run(spqc, shots=1)
        statevector = job.result().get_statevector().data
    else:
        # Default statevector simulation
        statevector = Statevector.from_instruction(spqc).data

    # Direct post‑selection via tensor slicing
    N = t + m + n * r + 1                   # total qubits
    tensor = statevector.reshape([2] * N)   # view, no copy

    slice_spec = [0] * t                   \
               + [slice(None)] * m         \
               + [0] * (n * r)             \
               + [0]                       # ancilla

    addr = tensor[tuple(slice_spec)].reshape(2 ** m)

    # Renormalise
    norm = np.linalg.norm(addr)
    return addr / norm if norm != 0 else addr

# --- ADAM HELPER FUNCTIONS ---

def mse_loss(y_pred, y_true):
    return float(((y_pred - y_true)**2).mean())

if __name__ == "__main__":
    exit()