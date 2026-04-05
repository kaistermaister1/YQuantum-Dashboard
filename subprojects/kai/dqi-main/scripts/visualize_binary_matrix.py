#!/usr/bin/env python3
"""
Standalone script for visualizing binary matrices.
This script provides functions to visualize binary matrices (containing only 0s and 1s)
with comprehensive sparsity analysis.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def visualize_binary_matrix(B, title="Binary Matrix", figsize=(12, 8), save_path=None):
    """
    Visualize a binary matrix using matplotlib.

    Inputs:
    B: numpy.ndarray
        Binary matrix containing only zeros and ones.
    title: str
        Title for the plot.
    figsize: tuple
        Figure size (width, height).
    save_path: str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    plt.figure(figsize=figsize)

    # Create a heatmap where 1s are black and 0s are white
    plt.imshow(B, cmap="binary", aspect="auto", interpolation="nearest")

    plt.title(f"{title}\nShape: {B.shape[0]} × {B.shape[1]}")
    plt.xlabel("Variables (Columns)")
    plt.ylabel("Constraints (Rows)")

    # Add colorbar
    cbar = plt.colorbar(shrink=0.8)
    cbar.set_label("Binary Values")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    # Add grid for better readability if matrix is small enough
    if B.shape[0] <= 50 and B.shape[1] <= 50:
        plt.grid(True, which="major", color="gray", linewidth=0.5, alpha=0.3)
        plt.xticks(range(B.shape[1]))
        plt.yticks(range(B.shape[0]))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matrix visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_matrix_sparsity(
    B, title="Matrix Sparsity Pattern", figsize=(15, 10), save_path=None
):
    """
    Create multiple visualizations to analyze the sparsity pattern of the binary matrix.

    Inputs:
    B: numpy.ndarray
        Binary matrix containing only zeros and ones.
    title: str
        Title for the plots.
    figsize: tuple
        Figure size (width, height).
    save_path: str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{title}\nMatrix Shape: {B.shape[0]} × {B.shape[1]}", fontsize=16)

    # 1. Main matrix visualization
    axes[0, 0].imshow(B, cmap="binary", aspect="auto", interpolation="nearest")
    axes[0, 0].set_title("Binary Matrix Pattern")
    axes[0, 0].set_xlabel("Variables (Columns)")
    axes[0, 0].set_ylabel("Constraints (Rows)")

    # 2. Row sparsity (number of 1s per row)
    row_sums = B.sum(axis=1)
    axes[0, 1].hist(
        row_sums,
        bins=min(50, len(np.unique(row_sums))),
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    axes[0, 1].set_title(
        f"Row Sparsity Distribution\nMean: {row_sums.mean():.2f}, Std: {row_sums.std():.2f}"
    )
    axes[0, 1].set_xlabel("Number of 1s per row")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Column sparsity (number of 1s per column)
    col_sums = B.sum(axis=0)
    axes[1, 0].hist(
        col_sums,
        bins=min(50, len(np.unique(col_sums))),
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    axes[1, 0].set_title(
        f"Column Sparsity Distribution\nMean: {col_sums.mean():.2f}, Std: {col_sums.std():.2f}"
    )
    axes[1, 0].set_xlabel("Number of 1s per column")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Overall sparsity statistics
    total_elements = B.shape[0] * B.shape[1]
    total_ones = B.sum()
    sparsity_ratio = total_ones / total_elements

    stats_text = f"""Matrix Statistics:
Total elements: {total_elements:,}
Total ones: {total_ones:,}
Total zeros: {total_elements - total_ones:,}
Sparsity ratio: {sparsity_ratio:.4f}
Density: {100 * sparsity_ratio:.2f}%

Row statistics:
Min 1s per row: {row_sums.min()}
Max 1s per row: {row_sums.max()}
Mean 1s per row: {row_sums.mean():.2f}

Column statistics:
Min 1s per column: {col_sums.min()}
Max 1s per column: {col_sums.max()}
Mean 1s per column: {col_sums.mean():.2f}"""

    axes[1, 1].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title("Matrix Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sparsity analysis saved to: {save_path}")
    else:
        plt.show()


def create_sample_binary_matrix(rows=20, cols=30, density=0.3, seed=42):
    """
    Create a sample binary matrix for testing.

    Parameters:
    rows: int
        Number of rows
    cols: int
        Number of columns
    density: float
        Density of 1s in the matrix (between 0 and 1)
    seed: int
        Random seed for reproducibility

    Returns:
    numpy.ndarray: Binary matrix
    """
    np.random.seed(seed)
    return (np.random.random((rows, cols)) < density).astype(int)


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Visualize binary matrices")
    parser.add_argument("--demo", action="store_true", help="Run with a demo matrix")
    parser.add_argument(
        "--rows", type=int, default=20, help="Number of rows for demo matrix"
    )
    parser.add_argument(
        "--cols", type=int, default=30, help="Number of columns for demo matrix"
    )
    parser.add_argument(
        "--density", type=float, default=0.3, help="Density of 1s for demo matrix"
    )
    parser.add_argument(
        "--save-basic", type=str, help="Save basic visualization to this path"
    )
    parser.add_argument(
        "--save-sparsity", type=str, help="Save sparsity analysis to this path"
    )

    args = parser.parse_args()

    if args.demo:
        # Create and visualize a demo matrix
        print(
            f"Creating demo binary matrix: {args.rows}×{args.cols} with density {args.density}"
        )
        B = create_sample_binary_matrix(args.rows, args.cols, args.density)

        print(f"Demo matrix shape: {B.shape}")
        print(f"Number of 1s: {B.sum()}")
        print(f"Actual density: {B.sum() / (B.shape[0] * B.shape[1]):.3f}")

        # Basic visualization
        visualize_binary_matrix(
            B,
            title="Demo Binary Matrix",
            save_path=args.save_basic,
        )

        # Sparsity analysis
        visualize_matrix_sparsity(
            B,
            title="Demo Matrix Sparsity Analysis",
            save_path=args.save_sparsity,
        )
    else:
        print("Use --demo to create and visualize a sample binary matrix")
        print("Or import this script and call the functions with your own matrix")


if __name__ == "__main__":
    main()
