import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from tqdm import tqdm

from pipelines.belief_propagations import belief_propagation_gallager
from pipelines.DQI_classical import expected_constrains_DQI
from pipelines.DQI_full_circuit import (
    dqi_max_xorsat,
    get_optimal_w,
    quantum_dqi_results,
    resource_estimation_function,
)


def generate_random_binary_test_cases(
    num_cases=5,
    m=3,
    n=3,
    seed=None,
    sparse_xor=True,
):
    """
    Generate a list of random binary test cases (B, v) for consistency testing.

    Parameters:
    ----------
    num_cases : int
        Number of test cases to generate.

    m : int
        Number of rows in the constraint matrix B (i.e., number of constraints).

    n : int
        Number of columns in B (i.e., number of variables). Vector v will also be of length m.

    seed : int or None
        Random seed for reproducibility. If None, randomness is not seeded.

    sparse_xor : bool
        If True, ensure each row of B has at least 2 and at most n//2 ones.

    Returns:
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples (B, v) where:
        - B is a binary matrix of shape (m, n)
        - v is a binary vector of shape (m,)
    """
    if seed is not None:
        np.random.seed(seed)

    test_cases = []
    for _ in range(num_cases):
        B = np.zeros((m, n), dtype=np.int8)
        for i in range(m):
            if sparse_xor:
                k = np.random.randint(2, max(3, n // 2 + 1))  # Ensure k in [2, n//2]
                ones_indices = np.random.choice(n, size=k, replace=False)
                B[i, ones_indices] = 1
            else:
                B[i] = np.random.randint(0, 2, size=n, dtype=np.int8)
        v = np.random.randint(0, 2, size=(m,), dtype=np.int8)
        test_cases.append((B, v))

    return test_cases


def run_dqi_consistency_check(
    B,
    v,
    ell=1,
    max_iterations=2,
    num_iterations_bp=2,
    bp_function=belief_propagation_gallager,
    quantum_bp=1,
    tolerance=0.02,
):
    """
    Runs a consistency check between the classical DQI (Deterministic Quantum Inference)
    expectation and the simulated quantum DQI output.

    Parameters:
    ----------
    B : np.ndarray
        Binary constraint matrix (dtype can be float or int, will be cast to int8).
        Shape (m, n) with entries in {0, 1}.

    v : np.ndarray
        Binary input vector of shape (n,), containing entries in {0, 1}.

    ell : int, optional
        Parameter controlling the level of DQI inference (default is 1).

    max_iterations : int, optional
        Number of iterations for the classical belief propagation algorithm (default is 2).

    num_iterations_bp : int, optional
        Number of iterations for the quantum simulation pipeline (default is 2).

    bp_function : callable
        Function implementing the belief propagation algorithm to use for classical DQI.

    quantum_bp : int, optional
        Flag or mode for quantum belief propagation (default is 1).

    tolerance : float, optional
        Allowed absolute tolerance for consistency between classical and quantum results.

    Raises:
    ------
    ValueError
        If the simulated quantum result deviates from the classical result beyond the given tolerance.

    Returns:
    -------
    None
        Prints a success message if the check passes; otherwise raises an error.
    """
    B = B.astype(np.int8)
    m, n = B.shape

    # Classical expected constraints via DQI
    _, s = expected_constrains_DQI(B, v, ell, max_iterations, bp_function)

    # Quantum resource estimation (optional if you just want the result)
    n_qubits, gate_dictionary = resource_estimation_function(
        B,
        v,
        ell,
        num_iterations_bp,
        quantum_bp,
        print_info=False,
        print_circuit_progress=False,
    )

    # Quantum simulated result

    if n_qubits <= 29:
        s_quantum, _, _ = quantum_dqi_results(
            B,
            v,
            ell,
            num_iterations_bp,
            quantum_bp,
            print_info=False,
        )

        # Consistency check
        if not np.isclose(s_quantum, s, atol=tolerance):
            raise ValueError(
                f"Mismatch: s_quantum={s_quantum}, s={s}, tolerance={tolerance}",
            )

        print(f"Check passed: s_quantum ≈ s = {s: .4f} within tolerance {tolerance}")

    else:
        print("QC too big to test ")


if __name__ == "__main__":

    """
    bp_version = "BP1"
    quantum_bp=1
    bp_function=belief_propagation_gallager

    #checked for
    # m=3 , n=2
    # m=3, n=3
    # m=4, n=3
    # m=4, n=4
    test_cases = generate_random_binary_test_cases(num_cases=100, m=4, n=4)

    for idx, (B, v) in enumerate(tqdm(test_cases, desc="Running DQI consistency tests")):
        print(B,v)
        print(f"\nTest case {idx + 1}")
        run_dqi_consistency_check(B, v)

    """

    test_cases = generate_random_binary_test_cases(
        num_cases=1,
        m=5,
        n=4,
        sparse_xor=True,
    )
    for idx, (B, v) in enumerate(
        tqdm(test_cases, desc="Running DQI consistency tests"),
    ):

        """
        B = np.array([
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1]
        ], dtype=int) #8x6

        v=np.ones(B.shape[0])
        """

        # B = np.array([[1,1,0], [0, 1, 1]]).T
        # v = np.array([0,1,1])

        ell = 1
        num_iterations_bp = 3
        quantum_bp = 1
        # different behaviour when print_info=False !!!!!
        n_qubits, gate_dictionary = resource_estimation_function(
            B,
            v,
            ell,
            num_iterations_bp,
            quantum_bp,
            print_info=True,
            print_circuit_progress=False,
        )

        W_k = get_optimal_w(m=B.shape[0], l=ell)
        final_circuit, data_qubits, bt_qubits = dqi_max_xorsat(
            B,
            v,
            W_k,
            num_iterations_bp,
            quantum_bp,
        )

        simulator = AerSimulator.from_backend(FakeTorino())
        transpiled_circuit = transpile(
            final_circuit,
            simulator,
            optimization_level=3,
            approximation_degree=0.9,
        )

        print(n_qubits, transpiled_circuit.depth())
