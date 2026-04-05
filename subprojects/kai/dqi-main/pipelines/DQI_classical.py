import time
from functools import partial
from itertools import combinations

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap

from pipelines.DQI_full_circuit import get_optimal_w


def objective_function_f(x, B, v):
    """
    Evaluates the objective function f(x) = Σᵢ (-1)^(Bᵢ·x + vᵢ mod 2) for a given bitstring x.

    This function computes a score indicating how well a candidate solution x satisfies
    the parity-check constraints defined by matrix B and syndrome vector v. Each satisfied
    constraint contributes +1, and each violated constraint contributes -1.

    Args:
        x (Union[List[int], np.ndarray]): Candidate solution as a binary list or array.
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Syndrome or constraint target vector (length m).

    Returns:
        int: Total score in the range [-m, m], where m is the number of constraints.
    """

    m = B.shape[0]
    x = np.array(x)
    f = 0
    for i in range(m):
        a = np.mod(np.mod(np.sum(B[i] * x), 2) + v[i], 2)
        f += (-1) ** a
    return f


def compute_A_matrix(ell, B, v, y_in_Dk):
    """
    Computes the A matrix used in the DQI  algorithm for MAX-XORSAT.

    This matrix encodes overlaps between Dicke subspaces D_k and D_k' weighted by phase
    conditions derived from the parity-check matrix B and syndrome vector v. The matrix A
    is used in classical preprocessing for optimizing the circuit structure.

    Args:
        ell (int): Maximum Hamming weight considered (defines the dimension of A as (l+1) x (l+1)).
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        v (np.ndarray): Syndrome vector of length m.
        y_in_Dk (List[List[List[int]]]): A list where y_in_Dk[k] contains bitstrings of weight k.

    Returns:
        np.ndarray: A real-valued symmetric matrix of shape (l+1, l+1) encoding inner product structure
                    between error patterns of different Hamming weights under parity constraints.
    """

    m = B.shape[0]
    A = np.zeros((ell + 1, ell + 1))
    norm = [1, m, m * (m - 1) / 2]

    one_hot_e = []
    for i in range(m):
        e = [0] * m
        e[i] = 1
        one_hot_e.append(e)

    for k in range(ell + 1):
        for kp in range(ell + 1):
            for y in y_in_Dk[k]:
                for yp in y_in_Dk[kp]:
                    a = np.mod(np.array(y) + np.array(yp), 2)
                    a = np.mod(np.sum(a * v), 2)
                    for i in range(m):
                        b = np.mod(np.array(yp) + np.array(one_hot_e[i]), 2)
                        Bt_yp_displaced = np.mod(B.T @ np.expand_dims(b, axis=1), 2)
                        Bt_y = np.mod(B.T @ np.expand_dims(np.array(y), axis=1), 2)
                        delta = np.sum(np.mod(Bt_yp_displaced + Bt_y, 2))
                        if delta == 0:
                            A[k, kp] += (
                                (1 / np.sqrt(norm[k] * norm[kp]))
                                * ((-1) ** (v[i]))
                                * ((-1) ** a)
                            )

    return A


def compute_y_in_Dk(ell, B, max_iterations, bp):
    """
    Computes the sets of error patterns y in D_k used in Decoded Quantum Interferometry (DQI).

    For each Hamming weight k from 0 to l, this function constructs the set D_k of binary vectors y
    of length m (number of checks), such that belief propagation decoding of y under H = Bᵀ
    produces no syndrome (i.e., is consistent with all parity constraints). The belief propagation
    function `bp` is used to test validity.

    Args:
        l (int): Maximum Hamming weight to consider (computes D_k for k = 0, 1, ..., l).
        B (np.ndarray): Binary parity-check matrix (shape m x n).
        max_iterations (int): Number of iterations for the belief propagation decoder.
        bp (Callable): Belief propagation function. It must accept arguments (H, y, max_iterations) and
                       return a tuple whose second element is the residual syndrome vector.

    Returns:
        List[List[List[int]]]: A list of lists, where the k-th element contains all y ∈ {0,1}^m with weight k
                               that decode to zero syndrome under belief propagation.
    """
    m = B.shape[0]
    H = B.T
    y_in_Dk = []
    for k in range(ell + 1):
        if k == 0:
            y_in_Dk.append([[0] * m])
        elif k == 1:
            all_y = []
            for i in range(m):
                y = [0] * m
                y[i] = 1
                if np.sum(bp(H, y, max_iterations=max_iterations)[1]) == 0:
                    all_y.append(y)
            y_in_Dk.append(all_y)
        elif k == 2:
            all_y = []
            for i, j in combinations(range(m), 2):
                y = [0] * m
                y[i] = 1
                y[j] = 1
                if np.sum(bp(H, y, max_iterations=max_iterations)[1]) == 0:
                    all_y.append(y)
            y_in_Dk.append(all_y)

        elif k == 3:
            all_y = []
            for i, j, z in combinations(range(m), 3):
                y = [0] * m
                y[i] = 1
                y[j] = 1
                y[z] = 1
                if np.sum(bp(H, y, max_iterations=max_iterations)[1]) == 0:
                    all_y.append(y)
            y_in_Dk.append(all_y)

    return y_in_Dk


def compute_expected_values(A, w, m, epsilon):
    """
    Computes the expected values of the objective functions f and s using a weighted superposition.

    Given the A matrix (encoding constraint structure), the weight vector w (e.g., a principal eigenvector),
    and a correction parameter epsilon, this function computes:
    - The expected value of f: ⟨w|A|w⟩ normalized by a correction factor.
    - The expected value of s: a rescaled version of f in the range [0, m].

    Args:
        A (np.ndarray): Real symmetric matrix of shape (l+1, l+1) representing the objective structure.
        w (np.ndarray): Weight vector used in the quantum superposition (typically normalized).
        m (int): Number of constraints (used to rescale f into s).
        epsilon (float): Correction factor for normalization (1 - ε applies element-wise to w²).

    Returns:
        Tuple[float, float]:
            - f_expected: Expected value of the f objective function.
            - s_expected: Corresponding expected number of satisfied constraints.
    """

    # remove padding from w
    if len(w) != len(epsilon):
        w = w[: len(epsilon)]

    norm = np.sqrt(np.sum((w**2) * (1 - epsilon)))

    # recover padding for w
    if len(w) != A.shape[0]:
        pad_len = A.shape[0] - len(w)
        w = np.pad(w, (0, pad_len), mode="constant", constant_values=0)

    w = np.expand_dims(np.array(w), axis=1)
    f_expected = ((w.T @ A) @ w) / norm**2
    s_expected = (f_expected + m) / 2
    # print(w)
    return f_expected[0][0], s_expected[0][0]


def expected_constrains_DQI(B, v, ell, max_iterations, bp_function, jit_version=False):
    """
    Estimates the expected values of the objective functions f and s
    in the Decoded Quantum Interferometry (DQI) framework.

    This function computes the expected performance of a DQI quantum state for a given MAX-XORSAT instance,
    using a combination of classical preprocessing and belief propagation. It involves:
    - Constructing valid error patterns in Dicke subspaces Dₖ using belief propagation.
    - Computing the A matrix encoding overlaps between these subspaces,
        with an option to use a JIT-compiled version for speed.
    - Calculating a weighted expectation over the principal eigenvector of an optimal operator.

    Args:
        B (np.ndarray):
            Binary parity-check matrix of shape (m, n).
        v (np.ndarray):
            Syndrome vector of length m.
        ell (int):
            Maximum Hamming weight considered (defines the subspaces Dₖ), so k ranges from 0 to ell.
        max_iterations (int):
            Maximum number of belief propagation iterations.
        bp_function (Callable):
            Belief propagation decoder.
            Must return a tuple where the second element is the residual syndrome.
        jit_version (bool, optional):
            If True, use the JIT-compiled routines
            to compute the A matrix (requires pre-packing of y_in_Dk);
            otherwise, use the pure-Python compute_A_matrix. Defaults to False.

    Returns:
        Tuple[float, float]:
            - f: Expected value of the f objective function.
            - s: Expected number of satisfied constraints.

    Prints:
        Error rate (1 - |Dₖ| / T_k) for each k ∈ [0, ..., ell], measuring decoding failure rate.
    """
    m = B.shape[0]
    Tk = [
        1,
        m,
        m * (m - 1) / 2,
        m * (m - 1) * (m - 2) / 6,
        m * (m - 1) * (m - 2) * (m - 3) / 24,
    ]  # m!/((m-l)!l!)

    w = get_optimal_w(m, ell)
    # print('w computed')

    y_in_Dk = compute_y_in_Dk(ell, B, max_iterations, bp_function)

    # print('y computed')
    # print(y_in_Dk) # works, no problem with memory

    epsilon = []
    for k, Dk in enumerate(y_in_Dk):
        epsilon.append(1 - len(Dk) / Tk[k])
    epsilon = np.array(epsilon)
    # print("Error rate: ", epsilon)

    if jit_version:
        Y, lengths = pack_y_in_Dk(y_in_Dk)
        # print(Y) #works, no porblem with memory

        A = compute_A_matrix_lax(ell, B, v, Y, lengths)
    else:
        A = compute_A_matrix(ell, B, v, y_in_Dk)

    f, s = compute_expected_values(A, w, m, epsilon)

    return f, s


# PARALELISATION FUNCTIONS


def pack_y_in_Dk(y_in_Dk):
    """
    Packs a nested list of vectors into a padded 3D JAX array.

    Each element y_in_Dk[k] is a list of D_k vectors (each of length m).
    This function zero-pads them to the same length (D_max) and returns:
      - a JAX array of shape (num_k, D_max, m)
      - a JAX array of actual lengths per k (shape: (num_k,))

    Parameters
    ----------
    y_in_Dk : list of list of list[int]
        A nested Python list of shape (num_k, D_k, m), where each inner list
        represents a binary or integer vector of length m.

    Returns
    -------
    Y : jax.numpy.ndarray
        Zero-padded array of shape (num_k, D_max, m), where D_max = max(D_k).
    lengths : jax.numpy.ndarray
        Array of shape (num_k,) storing the true number of vectors D_k for each k.
    """

    num_k = len(y_in_Dk)
    # maximum number of rows over all k
    D_max = max(len(y_in_Dk[k]) for k in range(num_k))
    # assume there is at least one y, and take its length:
    m = len(y_in_Dk[0][0])

    # allocate a zero‐buffer
    Y = np.zeros((num_k, D_max, m), dtype=int)
    lengths = np.zeros((num_k,), dtype=int)

    for k, y_list in enumerate(y_in_Dk):
        Dk = len(y_list)
        lengths[k] = Dk
        # convert each row to array and copy
        for i, y in enumerate(y_list):
            Y[k, i, :] = np.array(y, dtype=int)

    return jnp.array(Y), jnp.array(lengths)


@jit
def compute_entry(ell, B, v, Y, lengths, k, kp):
    """
    Compute a single matrix entry for a structured matrix in a pure JAX-compatible way.

    This function computes an (ℓ, k, k')-dependent inner product term involving parity checks
    and symmetries, using binary vectors from a zero-padded collection Y and their actual lengths.

    Parameters
    ----------
    ell : int
        Symmetry index (not currently used, but reserved for generalization).
    B : jax.numpy.ndarray
        Parity-check matrix of shape (m, m), assumed binary.
    v : jax.numpy.ndarray
        Binary vector of shape (m,), used in the inner product and sign computation.
    Y : jax.numpy.ndarray
        Zero-padded array of shape (num_k, D_max, m), representing sets of binary vectors.
    lengths : jax.numpy.ndarray
        Array of shape (num_k,) giving the true number of valid vectors in each Y[k].
    k : int
        Index of the first block in Y.
    kp : int
        Index of the second block in Y.

    Returns
    -------
    float
        The computed matrix entry ⟨k | M_ell | k'⟩, normalized appropriately.
    """
    m = B.shape[0]
    norm = jnp.array([1, m, m * (m - 1) / 2, m * (m - 1) * (m - 2) / 6])
    D_max = Y.shape[1]

    # 1) grab the full D_max×m blocks for k and kp (static slice size = D_max)
    y_full = Y[k]  # shape (D_max, m)
    yp_full = Y[kp]  # shape (D_max, m)

    # 2) build masks of length D_max
    idxs = jnp.arange(D_max)
    mask_y = idxs < lengths[k]  # True for rows we care about in y_full
    mask_yp = idxs < lengths[kp]  # True for rows we care about in yp_full

    one_hot = jnp.eye(m, dtype=int)

    def inner_sum(y, yp):
        a = jnp.mod(jnp.sum(jnp.mod(y + yp, 2) * v), 2)
        Bt_y = jnp.mod(B.T @ y, 2)

        def over_i(i):
            b = jnp.mod(yp + one_hot[i], 2)
            Bt_yp_disp = jnp.mod(B.T @ b, 2)
            delta = jnp.sum(jnp.mod(Bt_yp_disp + Bt_y, 2))
            return jnp.where(
                delta == 0,
                ((-1) ** v[i]) * ((-1) ** a),
                0,
            )

        return jnp.sum(vmap(over_i)(jnp.arange(m)))

    # 3) for each y in y_full, sum over all yp in yp_full, then mask both loops
    def sum_over_yp(y):
        vals = vmap(lambda yp: inner_sum(y, yp))(yp_full)  # shape (D_max,)
        return jnp.sum(vals * mask_yp)

    vals_y = vmap(sum_over_yp)(y_full)  # shape (D_max,)
    total = jnp.sum(vals_y * mask_y)

    return total / jnp.sqrt(norm[k] * norm[kp])


@partial(jit, static_argnums=(0,))
def compute_A_matrix_lax(ell, B, v, Y, lengths):
    """
    Compute the A_ell matrix using pure-JAX loops (lax.fori_loop) for full JIT compatibility.

    Constructs a symmetric matrix A of shape (ell+1, ell+1), where each entry A[k, kp] is
    computed using `compute_entry(ell, B, v, Y, lengths, k, kp)`. This matrix encodes correlations
    between different sets of vectors under parity constraints.

    Parameters
    ----------
    ell : int
        Determines the size of the output matrix: A will have shape (ell+1, ell+1).
    B : jax.numpy.ndarray
        Parity-check matrix of shape (m, m), binary.
    v : jax.numpy.ndarray
        Binary vector of shape (m,), used for sign-weighted inner product.
    Y : jax.numpy.ndarray
        Array of shape (num_k, D_max, m), holding zero-padded lists of binary vectors.
    lengths : jax.numpy.ndarray
        Array of shape (num_k,) giving the number of valid vectors in each block of Y.

    Returns
    -------
    A : jax.numpy.ndarray
        A real symmetric matrix of shape (ell+1, ell+1) computed via JAX-compatible loops.
    """

    n = ell + 1
    float_dtype = jnp.float32  # ← or jnp.float64, whichever you’re using

    def body_k(k, A):
        row = lax.fori_loop(
            0,
            n,
            lambda kp, acc: acc.at[kp].set(compute_entry(ell, B, v, Y, lengths, k, kp)),
            jnp.zeros((n,), dtype=float_dtype),
        )
        return A.at[k].set(row)

    A0 = jnp.zeros((n, n), dtype=float_dtype)
    return lax.fori_loop(0, n, body_k, A0)


if __name__ == "__main__":

    from pipelines.belief_propagations import (
        belief_propagation_ldpc,  # belief_propagation_gallager,
    )
    from pipelines.testing_BP1 import generate_random_binary_test_cases

    test_cases = generate_random_binary_test_cases(
        num_cases=5,
        m=50,
        n=30,
        seed=None,
        sparse_xor=True,
    )

    for problem in test_cases:
        B, v = problem
        ell = 2
        max_iterations = 2
        bp = belief_propagation_ldpc

        # f_jit,s_jit=expected_constrains_DQI(B,v,ell,max_iterations,bp,jit_version=True)
        # f_str,s_str=expected_constrains_DQI_streaming(B,v,ell,max_iterations,bp)

        # print(s_jit,s_str)

        # if not (np.allclose(f_jit, f_str) and np.allclose(s_jit, s_str)):
        #     raise ValueError("Mismatch between JIT and streaming versions:\n"
        #                     f"f_jit vs f_str: {f_jit} vs {f_str}\n"
        #                     f"s_jit vs s_str: {s_jit} vs {s_str}")

        # 1) Prepare the old inputs
        y_in_Dk = compute_y_in_Dk(ell, B, max_iterations, bp)
        Y, lengths = pack_y_in_Dk(y_in_Dk)

        #
        # Warm-up all three kernels so compilation overhead is excluded
        #
        _ = compute_A_matrix_lax(ell, B, v, Y, lengths).block_until_ready()
        # — new streaming path —

        #
        # 3) Time the lax-fori_loop JIT’d version
        #
        t0 = time.perf_counter()
        A_lax = compute_A_matrix_lax(ell, B, v, Y, lengths)
        try:
            A_lax.block_until_ready()
        except AttributeError:
            pass
        t_lax = time.perf_counter() - t0
        print("=== lax-fori_loop version ===")
        print(f"Time: {t_lax * 1000: .2f} ms")
