import glob
import itertools
import os
import re
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

from pipelines.belief_propagations import (
    belief_propagation_gallager,
    belief_propagation_ldpc,
    gauss_jordan_solve,
)


def success_rate_single_matrix(
    bp,
    max_iterations,
    matrix_filename,
):
    """
    Compute the decoding success rate for a given matrix and BP algorithm over all
    error patterns with weight 1 (1-bit flips).

    Parameters
    ----------
    bp : int
        Belief propagation algorithm to use:
        - 1 for Gallager-A/B style (hard-decision)
        - 2 for Soft LDPC (sum-product)
    max_iterations : int
        Maximum number of BP iterations to run per word.
    matrix_filename : str
        Path to the .npy file containing the binary matrix (e.g., parity-check matrix).

    Returns
    -------
    None
        Prints progress and final success rate to stdout.
    """

    ell_list = [1]  # Only 1-bit flip errors

    # Select BP function
    if bp == 1:
        bp_function = belief_propagation_gallager
    elif bp == 2:
        bp_function = belief_propagation_ldpc
    else:
        raise ValueError("Invalid value for `bp`. Use 1 (Gallager) or 2 (LDPC).")

    print(f"Using BP function: {bp_function.__name__}")

    # Load matrix from file
    B = np.load(matrix_filename)
    print(f"Loaded matrix {matrix_filename} with shape {B.shape}")

    m, n = B.shape
    B_t = B.T
    problem_size = m * n

    for ell in ell_list:
        print(f"Running for ell = {ell} (1-bit flips), problem size = {problem_size}")

        # Create all error patterns with weight `ell`
        list_of_words = []
        for bitflips in itertools.combinations(range(m), ell):
            word = np.zeros(m, dtype=int)
            word[list(bitflips)] = 1
            list_of_words.append(word)

        local_success = 0
        total = len(list_of_words)
        print(f"Total words to test: {total}")

        for idx, word_y in enumerate(list_of_words, start=1):
            outcome, decoded_y = bp_function(B_t, word_y, max_iterations=max_iterations)
            if np.all(decoded_y == 0):
                local_success += 1

            if idx % 10 == 0 or idx == total:
                print(f"[{idx}/{total}] Successes so far: {local_success}")

        print("Decoding finished.")

        rate = (local_success / total) * 100
        print(f"Success rate: {rate}%")


def make_words(m, ell, samples):
    """
    Generate random binary words of length m with fixed Hamming weight ell.

    Parameters
    ----------
    m : int
        Length of each binary word.
    ell : int
        Number of 1s in each word (Hamming weight).
    samples : int
        Number of words to generate.

    Returns
    -------
    words : list of np.ndarray
        List of binary vectors of shape (m,), each with exactly `ell` ones.
    """

    words = []
    for _ in range(samples):
        # pick ell distinct positions just once
        flips = np.random.choice(m, ell, replace=False)
        w = np.zeros(m, dtype=int)
        w[flips] = 1
        words.append(w)
    return words


def succes_rate_from_B_matrices(
    output_file_path,
    path_B_load_matrices,
    bp,
    max_iterations,
    samples,
):
    """
    Compute decoding success rates over multiple B matrices using a specified BP algorithm.

    Loads a set of parity-check matrices stored in `.npz` files under `path_B_load_matrices`,
    tests each matrix on random binary words of increasing Hamming weight (ell = 1 to 10),
    and saves average success rates and standard errors as CSV files.

    Parameters
    ----------
    output_file_path : str
        Path to the directory where result CSV files will be saved.
    path_B_load_matrices : str
        Path to the folder containing saved B matrices (`iter_*.npz`).
    bp : int
        Belief propagation algorithm:
        - 1 = Gallager-A/B
        - 2 = Soft LDPC
        - any other = Gauss-Jordan solver
    max_iterations : int
        Maximum number of BP iterations per decoding attempt.

    Returns
    -------
    None
        Results are printed and saved to disk as CSV files for success rates and errors.
    """

    def extract_iter_number(filename):
        match = re.search(r"iter_(\d+)\.npz", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path_B_load_matrices, "iter_*.npz")),
        key=extract_iter_number,
    )[
        :
    ]  # 10 last

    ell_list = list(range(1, 11))  # only untill ell=10

    # Select BP function
    if bp == 1:
        bp_function = belief_propagation_gallager
    elif bp == 2:
        bp_function = belief_propagation_ldpc
    else:
        bp_function = gauss_jordan_solve
        # raise ValueError("Invalid value for `bp`. Use 1 (Gallager) or 2 (LDPC).")

    # Initialize result matrices
    success_matrix = pd.DataFrame(index=ell_list)
    error_matrix = pd.DataFrame(index=ell_list)

    for list_index, fn in enumerate(files):
        data = np.load(fn)
        B = data["B"]
        # v = data["v"]

        m, n = B.shape

        B_t = B.T

        problem_size = m * n

        for ell in ell_list:
            # Create
            print(
                f"Running simulation with problem size = {problem_size} and ell = {ell}",
            )

            if ell > m:
                success_matrix.at[ell, problem_size] = 0
                error_matrix.at[ell, problem_size] = 0
            else:
                list_of_words = make_words(m, ell, samples)

                # Split into 5 groups
                chunk_size = samples // 5
                group_success_rates = []

                for i in range(5):
                    start = i * chunk_size
                    end = (
                        (i + 1) * chunk_size if i < 4 else samples
                    )  # last chunk includes leftovers
                    subset = list_of_words[start:end]

                    local_success = 0
                    for word_y in subset:
                        outcome, decoded_y = bp_function(
                            B_t,
                            word_y,
                            max_iterations=max_iterations,
                        )
                        if np.all(decoded_y == 0):
                            local_success += 1

                    rate = (local_success / len(subset)) * 100
                    group_success_rates.append(rate)

                # Compute mean success and standard error
                success_rate = mean(group_success_rates)
                se_percent = stdev(group_success_rates) / np.sqrt(5)

                print(f"m = {m}, n = {n}, ℓ = {ell}, success rate = {success_rate:.3f}")

                # Store in matrix
                success_matrix.at[ell, problem_size] = success_rate
                error_matrix.at[ell, problem_size] = se_percent

    # Save results
    Path(output_file_path).mkdir(parents=True, exist_ok=True)
    success_csv = os.path.join(
        output_file_path,
        f"success_rates_bp{bp}_it{max_iterations}_shots{samples}.csv",
    )
    error_csv = os.path.join(
        output_file_path,
        f"success_errors_bp{bp}_it{max_iterations}_shots{samples}.csv",
    )

    success_matrix.to_csv(success_csv)
    error_matrix.to_csv(error_csv)

    print(f"Saved success rates to: {success_csv}")
    print(f"Saved error estimates to: {error_csv}")


if __name__ == "__main__":
    a = None
    bp = 1

    succes_rate_from_B_matrices(
        "pipelines/full_experiment_data",
        "pipelines/stored_matrices",
        bp,
        max_iterations=5,
    )

    # output_file_path = "pipelines/data/plot_data_with_random_sparse_Bs.csv"
    # run_performance_and_resource_estimation_for_random_sparse_Bs(output_file_path)
