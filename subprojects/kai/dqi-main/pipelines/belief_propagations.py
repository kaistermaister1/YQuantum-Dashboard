import numpy as np


# --- BP1 (Gallager-A/B style) ---
def belief_propagation_gallager(B_transpose, y, max_iterations=5):
    """BP1: Hard-decision binary BP (Gallager-A/B style)

    Function that runs the Gallager version of belief propagation (BP1).

    Inputs:
    B_transpose: numpy.ndarray
        Parity check matrix of dimension (n_variables, n_constraints). Contains only zeros and ones.
    y: numpy.ndarray
        Message of dimension n_constraints. Contains only zeros and ones.

    Outputs:
    The function returns a Boolean based on the success or failure of BP,
    and the decoded vector, i.e. in case of success,
    a vector bits such that B_transpose @ bits = 0

    is_feasible: Boolean
        True if BP has successfully decoded, False otherwise.
    bits: numpy.ndarray
        Decoded vector.
    """

    bits = y.copy()

    num_vnodes = B_transpose.shape[1]
    # num_cnodes = B_transpose.shape[0]

    for iteration in range(max_iterations):
        syndrome = (B_transpose @ bits) % 2

        flip_candidates = np.zeros_like(bits)

        for j in range(num_vnodes):
            connected_cnodes = np.where(B_transpose[:, j] == 1)[0]
            threshold = len(connected_cnodes)
            unsatisfied = np.sum(syndrome[connected_cnodes])
            if unsatisfied >= threshold:
                flip_candidates[j] = 1

        bits = (bits + flip_candidates) % 2

    if np.all(bits == 0):
        return True, bits

    return False, bits


# --- BP2 (state of the art BP algorithm) ---
def belief_propagation_ldpc(H, y, max_iterations=5):
    """
    BP2: Soft-decision Belief Propagation

    Implements soft-decision BP for decoding

    Inputs:
    H : numpy.ndarray
        Parity-check matrix of shape (n_constraints, n_variables), i.e., H = B.T.
        Assumed to contain only 0s and 1s.
    y : numpy.ndarray
        Received noisy codeword (binary vector of shape n_variables).
    max_iterations : int
        Maximum number of BP iterations to run.

    Outputs:
    is_feasible : bool
        True if decoding succeeded (decoded vector is all zeros), False otherwise.
    decoded_bits : numpy.ndarray
        Decoded binary vector of shape n_variables.
    """

    p = 0.001  # Hard-coded BSC probability for the moment
    received_bits = np.array(y)
    received_llrs = np.log((1 - p) / p) * (1 - 2 * received_bits)

    num_vnodes = H.shape[1]
    num_cnodes = H.shape[0]

    M_v_to_c = np.tile(received_llrs, (num_cnodes, 1))

    for iteration in range(max_iterations):
        M_c_to_v = np.zeros_like(M_v_to_c)
        for i in range(num_cnodes):
            connected_vnodes = np.where(H[i] == 1)[0]
            for v in connected_vnodes:
                product = np.prod(
                    np.tanh(0.5 * M_v_to_c[i, connected_vnodes[connected_vnodes != v]]),
                )
                M_c_to_v[i, v] = 2 * np.arctanh(product)

        for j in range(num_vnodes):
            connected_cnodes = np.where(H[:, j] == 1)[0]
            for c in connected_cnodes:
                M_v_to_c[c, j] = received_llrs[j] + np.sum(
                    M_c_to_v[connected_cnodes[connected_cnodes != c], j],
                )

        final_llrs = received_llrs + np.sum(M_c_to_v, axis=0)
        decoded_bits = (final_llrs < 0).astype(int)

        if np.all(decoded_bits == 0):
            return True, decoded_bits

    return False, decoded_bits


def gauss_jordan_solve(H, y, max_iterations=None):
    """
    Gauss–Jordan Decoder over GF(2)

    Solves the linear system H * e = s over GF(2) using Gauss–Jordan elimination,
    and returns whether decoding succeeded and the decoded bits.

    Parameters
    ----------
    H : numpy.ndarray
        Parity-check matrix of shape (m, n) with binary entries (0 or 1).
    y : numpy.ndarray
        Received bitstring (length n), assumed to be binary (0 or 1).
    max_iterations : unused
        Present for compatibility, not used in this implementation.

    Returns
    -------
    success : bool
        True if decoding succeeded (decoded_bits == 0), False otherwise.
    decoded_bits : numpy.ndarray
        Decoded binary vector of length n, given by (y + e) % 2.
    """

    # Compute syndrome s = H @ y mod 2
    s = (H @ y) & 1
    # Build augmented matrix [H | s]
    H_aug = np.concatenate((H & 1, s.reshape(-1, 1)), axis=1).astype(np.uint8)

    m, n = H.shape
    row = 0
    pivot_cols = []

    # Forward elimination to RREF
    for col in range(n):
        # Find pivot row with a 1 in this column
        ones = np.nonzero(H_aug[row:, col])[0]
        if ones.size == 0:
            continue
        pivot = ones[0] + row
        # Swap pivot row into position
        if pivot != row:
            H_aug[[row, pivot]] = H_aug[[pivot, row]]
        pivot_cols.append(col)
        # Eliminate all other 1s in this column
        mask = H_aug[:, col] == 1
        mask[row] = False
        H_aug[mask] ^= H_aug[row]
        row += 1
        if row == m:
            break

    # Extract error vector e: for each pivot column, the augmented entry
    e = np.zeros(n, dtype=np.uint8)
    for i, pcol in enumerate(pivot_cols):
        e[pcol] = H_aug[i, -1]

    # Decode bits
    decoded_bits = (y + e) & 1
    success = bool(np.all(decoded_bits == 0))
    return success, decoded_bits
