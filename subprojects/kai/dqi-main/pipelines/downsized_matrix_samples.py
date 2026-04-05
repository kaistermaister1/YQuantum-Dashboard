import random

import matplotlib.pyplot as plt
import numpy as np


def checks_and_variables(B, absolute=True):
    """
    Compute normalized check and variable node degrees from a binary parity-check matrix.

    Parameters
    ----------
    B : np.ndarray
        Binary parity-check matrix of shape (m, n).
    absolute : bool, optional
        If True, normalize by 1; if False, normalize by total number of elements.

    Returns
    -------
    check : np.ndarray
        Normalized degree of each check node (column-wise sum).
    var : np.ndarray
        Normalized degree of each variable node (row-wise sum).
    """

    if absolute:
        norm = 1
    else:
        norm = B.shape[0] * B.shape[1]
    check = np.sum(B, axis=0) / norm
    var = np.sum(B, axis=1) / norm
    return check, var


def plot_histograms(B, plot=True):
    """
    Plot histograms of variable and constraint node degrees from a binary parity-check matrix.

    Parameters
    ----------
    B : np.ndarray
        Binary parity-check matrix of shape (M, N), where rows are constraints and columns are variables.

    Returns
    -------
    delta : np.ndarray
        Normalized histogram of variable node degrees.
    kappa : np.ndarray
        Normalized histogram of constraint node degrees.
    i : np.ndarray
        Degree values for constraints (x-axis values for kappa).
    j : np.ndarray
        Degree values for variables (x-axis values for delta).
    variable_degree : np.ndarray
        Array of individual variable node degrees.
    M : int
        Number of constraints (rows of B).
    N : int
        Number of variables (columns of B).
    """

    M = B.shape[0]
    N = B.shape[1]

    variable_degree, constraint_degree = checks_and_variables(B, absolute=True)
    variable_degree = variable_degree.astype(np.int64)
    constraint_degree = constraint_degree.astype(np.int64)

    delta = np.bincount(variable_degree)[
        np.min(variable_degree) : np.max(variable_degree) + 1
    ]
    delta = delta / np.sum(delta)

    kappa = np.bincount(constraint_degree)[
        np.min(constraint_degree) : np.max(constraint_degree) + 1
    ]
    kappa = kappa / np.sum(kappa)

    j = np.arange(np.min(variable_degree), np.max(variable_degree) + 1)
    i = np.arange(np.min(constraint_degree), np.max(constraint_degree) + 1)

    if plot:
        plt.bar(j, delta)
        plt.xlabel("variable degree j")
        plt.ylabel("Fraction of variables with degree j, Delta")
        plt.show()

        plt.bar(i, kappa)
        plt.xlabel("constraint degree i")
        plt.ylabel("Fraction of constraints with degree i, Kappa")
        plt.show()

    return delta, kappa, i, j, variable_degree, M, N


def is_valid_swap(B, i1, i2, j1, j2):
    """
    Check whether a 2x2 submatrix in B allows a valid edge swap.

    A valid swap occurs when:
        B[i1, j1] == 1, B[i2, j2] == 1,
        B[i1, j2] == 0, B[i2, j1] == 0

    Parameters
    ----------
    B : np.ndarray
        Binary matrix (e.g., adjacency or parity-check matrix).
    i1, i2 : int
        Row indices of the 2x2 block.
    j1, j2 : int
        Column indices of the 2x2 block.

    Returns
    -------
    bool
        True if the swap is valid, False otherwise.
    """
    # Check if positions form a 2x2 swap-eligible block
    return B[i1][j1] == 1 and B[i2][j2] == 1 and B[i1][j2] == 0 and B[i2][j1] == 0


def perform_swap(B, i1, i2, j1, j2):
    """
    Perform a 2x2 edge swap on matrix B.

    Assumes the indices form a valid swap-eligible block as checked by `is_valid_swap`.
    The function flips:
        B[i1, j1] and B[i2, j2] from 1 → 0
        B[i1, j2] and B[i2, j1] from 0 → 1

    Parameters
    ----------
    B : np.ndarray
        Binary matrix (e.g., adjacency or parity-check matrix), modified in-place.
    i1, i2 : int
        Row indices of the 2x2 block.
    j1, j2 : int
        Column indices of the 2x2 block.
    """
    B[i1][j1] = 0
    B[i2][j2] = 0
    B[i1][j2] = 1
    B[i2][j1] = 1


def mcmc_sample(B, iterations=10000):
    """
    Perform MCMC sampling on a binary matrix by repeated valid 2x2 edge swaps.

    Randomly selects 2x2 submatrices and applies valid swaps (preserving row and column degrees)
    over a specified number of iterations. Modifies the matrix in-place.

    Parameters
    ----------
    B : np.ndarray
        Binary matrix (e.g., parity-check matrix) to be shuffled in-place.
    iterations : int, optional
        Number of MCMC iterations to perform (default is 10,000).

    Returns
    -------
    B : np.ndarray
        The modified binary matrix after MCMC sampling.
    """
    m, n = B.shape
    for _ in range(iterations):
        i1, i2 = random.sample(range(m), 2)
        j1, j2 = random.sample(range(n), 2)

        if is_valid_swap(B, i1, i2, j1, j2):
            perform_swap(B, i1, i2, j1, j2)

    return B


def generate_initial_matrix(row_sums, col_sums):
    """
    Generate a binary matrix with specified row and column sums using a greedy algorithm.

    Attempts to construct a binary matrix B such that:
        - sum of each row i is row_sums[i]
        - sum of each column j is col_sums[j]

    The method is greedy and may fail if the specified marginals are not graphical
    (i.e., do not correspond to a valid 0–1 matrix).

    Parameters
    ----------
    row_sums : list of int
        Desired row sums (length m).
    col_sums : list of int
        Desired column sums (length n).

    Returns
    -------
    B : np.ndarray
        Binary matrix of shape (m, n) matching the given row and column sums.
    """
    m, n = len(row_sums), len(col_sums)
    B = np.zeros((m, n), dtype=int)

    # Sort indices for filling
    rows = sorted(list(enumerate(row_sums)), key=lambda x: -x[1])
    cols = sorted(list(enumerate(col_sums)), key=lambda x: -x[1])

    for i, r in rows:
        possible = sorted(cols, key=lambda x: -x[1])
        for j, c in possible:
            if r > 0 and c > 0 and B[i][j] == 0:
                B[i][j] = 1
                row_sums[i] -= 1
                col_sums[j] -= 1
                cols = [(jj, cc - (1 if jj == j else 0)) for jj, cc in cols]
                r -= 1
                if row_sums[i] == 0:
                    break
    return B
