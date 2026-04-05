import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pipelines.generate_B import generate_B_matrix_and_rhs
from pipelines.succes_rate import succes_rate_from_B_matrices


def generate_Bs_from_random_ILPS(path_B_save_matrices):

    """
    Generate and save a series of random ILP-derived binary matrices for decoder benchmarking.

    For each problem size in a fixed range, constructs a random integer linear program (ILP),
    encodes it into a binary matrix B using max-XORSAT reduction, and saves the result to disk
    as a .npz file including B, v, shape, and ell.

    Parameters
    ----------
    path_B_save_matrices : str
        Path to the directory where the generated .npz files will be saved.
    """

    random.seed(42)
    np.random.seed(42)

    min_problem_size = 2
    problem_sizes = range(min_problem_size, 12, 1)

    os.makedirs(path_B_save_matrices, exist_ok=True)

    for list_index, problem_size in zip(range(len(list(problem_sizes))), problem_sizes):
        # Generate random ILP
        while True:

            A = np.random.randint(
                low=0,
                high=3,
                size=(problem_size, 2 * (problem_size - 1) + 1),
            )
            b = np.random.randint(low=1, high=3, size=(problem_size))
            c = np.random.randint(
                low=1,
                high=151,
                size=2 * (problem_size - 1) + 1,
            )  # high=3 before

            if not np.any(np.sum(A, axis=1) < b):
                break

        beta_max = np.sum(np.abs(c))
        beta = int(beta_max / 2)

        B, v, ell, code_distance, max_num_constraints, _ = generate_B_matrix_and_rhs(
            A,
            b,
            c,
            beta=beta,
        )

        B = B.astype(np.int8)

        m, n = B.shape

        # print(m * n, m, n)

        fn = os.path.join(
            path_B_save_matrices,
            f"iter_{list_index}.npz",
        )
        # Save B, v, and B.shape
        np.savez(
            fn,
            B=B,
            v=v,
            shape_B=B.shape,
            ell=ell,
        )


def plot_2d_success_rate_benchmark(results_folder_path, bp, max_iterations, samples):
    """
    Plot 2D heatmaps of success rates and error bars for a given BP decoder.

    Parameters
    ----------
    results_folder_path : str
        Path to the folder containing the success/error CSVs.
    bp : int
        Belief propagation algorithm index used in the benchmark.
    """

    # File paths
    success_csv = os.path.join(
        results_folder_path,
        f"success_rates_bp{bp}_it{max_iterations}_shots{samples}.csv",
    )
    error_csv = os.path.join(
        results_folder_path,
        f"success_errors_bp{bp}_it{max_iterations}_shots{samples}.csv",
    )

    # Load CSVs
    success_df = pd.read_csv(success_csv, index_col=0)
    error_df = pd.read_csv(error_csv, index_col=0)

    # Ensure columns are integers (problem sizes)
    success_df.columns = success_df.columns.astype(int)
    error_df.columns = error_df.columns.astype(int)

    # Success heatmap
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        success_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 12},
    )
    ax.invert_yaxis()
    ax.set_xlabel(r"Problem Size ($m \cdot n$)", fontsize=14)
    ax.set_ylabel("Number of Bit Flips (ℓ)", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save success plot
    success_pdf = f"success_rates_bp{bp}_it{max_iterations}_shots{samples}.pdf"
    plt.savefig(success_pdf)
    print(f"Saved success rate heatmap to: {success_pdf}")
    plt.show()

    # Error heatmap
    plt.figure(figsize=(10, 6))
    ax2 = sns.heatmap(
        error_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 12},
    )
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Problem Size ($m \cdot n$)", fontsize=14)
    ax2.set_ylabel("Number of Bit Flips (ℓ)", fontsize=14)
    ax2.tick_params(axis="both", labelsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save error plot
    error_pdf = f"success_errors_bp{bp}_it{max_iterations}_shots{samples}.pdf"
    plt.savefig(error_pdf)
    print(f"Saved error heatmap to: {error_pdf}")
    plt.show()


if __name__ == "__main__":

    generate_Bs_from_random_ILPS(
        "pipelines/random_ILPS_for_success_rate_study",
    )  # Generate max-XORSAT problems to benchmark decoders
    print("Max-XORSAT problems generated")

    # Loop over all decoder types
    for bp in [1, 2, "GJ"]:
        output_folder = "pipelines/success_rate_benchmark_results"

        # Run benchmark and save results
        succes_rate_from_B_matrices(
            output_folder,
            "pipelines/random_ILPS_for_success_rate_study",
            bp=bp,
            max_iterations=5,
            samples=10**4,
        )

        # Plot results
        plot_2d_success_rate_benchmark(
            output_folder,
            bp,
            max_iterations=5,
            samples=10**4,
        )
