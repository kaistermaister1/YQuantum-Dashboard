#!/usr/bin/env python3
"""
Comprehensive XORSAT instance visualization script.
This script provides functions to visualize both the binary matrix B and vector v
of a max-XORSAT instance, giving a complete bird's eye view of the problem.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def visualize_binary_matrix(
    B, title="Binary Matrix B", figsize=(12, 8), save_path=None
):
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


def visualize_vector_v(
    v, title="Right-Hand Side Vector v", figsize=(10, 8), save_path=None
):
    """
    Visualize the right-hand side vector v of the XORSAT instance.

    Inputs:
    v: numpy.ndarray
        Binary vector containing only zeros and ones.
    title: str
        Title for the plot.
    figsize: tuple
        Figure size (width, height).
    save_path: str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"{title}\nVector Length: {len(v)}", fontsize=16)

    # 1. Vector visualization as vertical strip
    v_matrix = v.reshape(-1, 1)  # Convert to column matrix for visualization
    axes[0, 0].imshow(v_matrix, cmap="binary", aspect="auto", interpolation="nearest")
    axes[0, 0].set_title("Vector Pattern (Vertical)")
    axes[0, 0].set_xlabel("Vector (Width=1)")
    axes[0, 0].set_ylabel("Elements (Rows)")

    # 2. Vector values over index
    indices = np.arange(len(v))
    colors = ["white" if val == 0 else "black" for val in v]
    axes[0, 1].scatter(indices, v, c=colors, edgecolors="black", s=20, alpha=0.8)
    axes[0, 1].set_title("Vector Values vs Index")
    axes[0, 1].set_xlabel("Index")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Value distribution histogram
    unique, counts = np.unique(v, return_counts=True)
    axes[1, 0].bar(
        unique, counts, color=["lightgray", "black"], edgecolor="black", alpha=0.7
    )
    axes[1, 0].set_title("Value Distribution")
    axes[1, 0].set_xlabel("Binary Values")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Vector statistics
    total_elements = len(v)
    total_ones = v.sum()
    total_zeros = total_elements - total_ones
    density = total_ones / total_elements if total_elements > 0 else 0

    # Find patterns (consecutive runs)
    diff = np.diff(np.concatenate(([v[0]], v, [1 - v[-1]])))
    run_starts = np.where(diff != 0)[0]
    run_lengths = np.diff(run_starts)

    stats_text = f"""Vector Statistics:
Total elements: {total_elements:,}
Total ones: {total_ones:,}
Total zeros: {total_zeros:,}
Density (ones): {100 * density:.2f}%

Pattern Analysis:
Number of runs: {len(run_lengths)}
Avg run length: {run_lengths.mean():.2f}
Max run length: {run_lengths.max()}
Min run length: {run_lengths.min()}

Index ranges:
First one at: {np.where(v == 1)[0][0] if total_ones > 0 else 'N/A'}
Last one at: {np.where(v == 1)[0][-1] if total_ones > 0 else 'N/A'}"""

    axes[1, 1].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title("Vector Statistics")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Vector visualization saved to: {save_path}")
    else:
        plt.show()


def visualize_xorsat_instance(
    B,
    v,
    title="Max-XORSAT Instance: Bx = v",
    figsize=(16, 10),
    save_path=None,
    var_names=None,
):
    """
    Create a comprehensive visualization of the max-XORSAT instance showing both B and v.

    Inputs:
    B: numpy.ndarray
        Binary matrix containing only zeros and ones.
    v: numpy.ndarray
        Binary vector containing only zeros and ones.
    title: str
        Title for the plot.
    figsize: tuple
        Figure size (width, height).
    save_path: str, optional
        Path to save the plot. If None, the plot is displayed.
    var_names: list, optional
        List of variable names corresponding to the columns of B. If None, columns are numbered.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"{title}\nMatrix B: {B.shape[0]}×{B.shape[1]}, Vector v: {len(v)}", fontsize=16
    )

    # Create a grid layout: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Main XORSAT visualization: B matrix on left, v vector on right
    ax1 = fig.add_subplot(gs[0, :3])
    ax1.imshow(B, cmap="binary", aspect="auto", interpolation="nearest")
    ax1.set_title("Matrix B (Left-Hand Side)")
    ax1.set_xlabel("Variables")
    ax1.set_ylabel("Constraints")

    # Set x-axis ticks and labels for variables
    if var_names is not None and len(var_names) == B.shape[1]:
        # Set tick positions for all variables
        ax1.set_xticks(range(B.shape[1]))
        ax1.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
    else:
        # Fallback to column numbers if no variable names provided
        n_vars = B.shape[1]
        if n_vars <= 20:  # Show all labels for small number of variables
            ax1.set_xticks(range(n_vars))
            ax1.set_xticklabels(
                [f"x_{i}" for i in range(n_vars)], rotation=45, ha="right", fontsize=8
            )
        else:  # Show subset of labels for large number of variables
            step = max(1, n_vars // 10)
            tick_positions = range(0, n_vars, step)
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(
                [f"x_{i}" for i in tick_positions], rotation=45, ha="right", fontsize=8
            )

    # Add v vector as a separate plot on the right
    ax2 = fig.add_subplot(gs[0, 3])
    v_matrix = v.reshape(-1, 1)
    ax2.imshow(v_matrix, cmap="binary", aspect="auto", interpolation="nearest")
    ax2.set_title("Vector v\n(RHS)")
    ax2.set_xlabel("v")
    ax2.set_ylabel("Constraints")
    ax2.set_xticks([])

    # 2. Matrix B statistics
    ax3 = fig.add_subplot(gs[1, 0])
    row_sums = B.sum(axis=1)
    ax3.hist(
        row_sums,
        bins=min(20, len(np.unique(row_sums))),
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    ax3.set_title(f"B Row Sparsity\nMean: {row_sums.mean():.1f}")
    ax3.set_xlabel("1s per row")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    col_sums = B.sum(axis=0)
    ax4.hist(
        col_sums,
        bins=min(20, len(np.unique(col_sums))),
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax4.set_title(f"B Column Sparsity\nMean: {col_sums.mean():.1f}")
    ax4.set_xlabel("1s per column")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    # 3. Vector v statistics
    ax5 = fig.add_subplot(gs[1, 2])
    unique_v, counts_v = np.unique(v, return_counts=True)
    ax5.bar(
        unique_v, counts_v, color=["lightgray", "black"], edgecolor="black", alpha=0.7
    )
    ax5.set_title(f"Vector v Distribution\nOnes: {v.sum()}/{len(v)}")
    ax5.set_xlabel("Value")
    ax5.set_ylabel("Count")
    ax5.set_xticks([0, 1])
    ax5.grid(True, alpha=0.3)

    # 4. Vector v pattern over indices
    ax6 = fig.add_subplot(gs[1, 3])
    indices = np.arange(len(v))
    colors = ["lightgray" if val == 0 else "black" for val in v]
    ax6.scatter(
        indices[:: max(1, len(v) // 100)],
        v[:: max(1, len(v) // 100)],
        c=[colors[i] for i in range(0, len(v), max(1, len(v) // 100))],
        s=10,
        alpha=0.7,
        edgecolors="none",
    )
    ax6.set_title("Vector v Pattern")
    ax6.set_xlabel("Index")
    ax6.set_ylabel("Value")
    ax6.set_ylim(-0.1, 1.1)
    ax6.grid(True, alpha=0.3)

    # 5. Combined statistics
    ax7 = fig.add_subplot(gs[2, :])

    # Calculate statistics
    B_total_elements = B.shape[0] * B.shape[1]
    B_total_ones = B.sum()
    B_density = B_total_ones / B_total_elements

    v_total_ones = v.sum()
    v_density = v_total_ones / len(v)

    # Constraint satisfaction analysis
    satisfied_constraints = []
    for i in range(B.shape[0]):
        # This is a simplified check - in practice you'd need actual variable assignments
        row_sum = B[i, :].sum()
        satisfied_constraints.append(row_sum)

    stats_text = f"""MAX-XORSAT INSTANCE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MATRIX B (Left-Hand Side):                          VECTOR v (Right-Hand Side):
  Shape: {B.shape[0]} × {B.shape[1]}                                    Length: {len(v)}
  Total elements: {B_total_elements:,}                              Ones: {v_total_ones} ({100*v_density:.1f}%)
  Ones: {B_total_ones} ({100*B_density:.2f}%)                        Zeros: {len(v) - v_total_ones} ({100*(1-v_density):.1f}%)
  Sparsity: {"Very sparse" if B_density < 0.1 else "Moderately sparse" if B_density < 0.3 else "Dense"} ({100*(1-B_density):.1f}% zeros)

CONSTRAINT STRUCTURE:                               PROBLEM CHARACTERISTICS:
  Avg constraint degree: {row_sums.mean():.2f}                    Total constraints: {len(satisfied_constraints)}
  Variable connectivity: {col_sums.mean():.2f}                    Constraint structure: {"Uniform" if row_sums.std() < 1 else "Variable"}
  Most connected var: {col_sums.max()} constraints               Instance type: Max-XORSAT (binary linear system)"""

    ax7.text(
        0.02,
        0.95,
        stats_text,
        transform=ax7.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1)
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.set_title("Instance Overview", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"XORSAT instance visualization saved to: {save_path}")
    else:
        plt.show()


def create_sample_xorsat_instance(
    rows=20, cols=30, density=0.3, v_density=0.4, seed=42
):
    """
    Create a sample XORSAT instance for testing.

    Parameters:
    rows: int
        Number of rows (constraints)
    cols: int
        Number of columns (variables)
    density: float
        Density of 1s in the matrix B (between 0 and 1)
    v_density: float
        Density of 1s in vector v (between 0 and 1)
    seed: int
        Random seed for reproducibility

    Returns:
    tuple: (B, v, var_names) where B is binary matrix, v is binary vector, and var_names is list of variable names
    """
    np.random.seed(seed)
    B = (np.random.random((rows, cols)) < density).astype(int)
    v = (np.random.random(rows) < v_density).astype(int)
    var_names = [f"x_{j}_({j})" for j in range(cols)]
    return B, v, var_names


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(
        description="Visualize XORSAT instances (matrix B and vector v)"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run with a demo XORSAT instance"
    )
    parser.add_argument(
        "--rows", type=int, default=15, help="Number of rows for demo instance"
    )
    parser.add_argument(
        "--cols", type=int, default=25, help="Number of columns for demo instance"
    )
    parser.add_argument(
        "--b-density", type=float, default=0.3, help="Density of 1s in matrix B"
    )
    parser.add_argument(
        "--v-density", type=float, default=0.4, help="Density of 1s in vector v"
    )
    parser.add_argument(
        "--save-matrix", type=str, help="Save matrix visualization to this path"
    )
    parser.add_argument(
        "--save-vector", type=str, help="Save vector visualization to this path"
    )
    parser.add_argument(
        "--save-combined",
        type=str,
        help="Save combined XORSAT visualization to this path",
    )

    args = parser.parse_args()

    if args.demo:
        # Create and visualize a demo XORSAT instance
        print(
            f"Creating demo XORSAT instance: {args.rows}×{args.cols} matrix, vector length {args.rows}"
        )
        print(f"Matrix density: {args.b_density}, Vector density: {args.v_density}")

        B, v, var_names = create_sample_xorsat_instance(
            args.rows, args.cols, args.b_density, args.v_density
        )

        print(f"Generated instance:")
        print(f"  Matrix B shape: {B.shape}")
        print(f"  Vector v length: {len(v)}")
        print(
            f"  Matrix B ones: {B.sum()} (density: {B.sum() / (B.shape[0] * B.shape[1]):.3f})"
        )
        print(f"  Vector v ones: {v.sum()} (density: {v.sum() / len(v):.3f})")

        # Matrix visualization
        if args.save_matrix:
            visualize_binary_matrix(
                B, title="Demo Matrix B", save_path=args.save_matrix
            )
        else:
            visualize_binary_matrix(B, title="Demo Matrix B")

        # Vector visualization
        if args.save_vector:
            visualize_vector_v(v, title="Demo Vector v", save_path=args.save_vector)
        else:
            visualize_vector_v(v, title="Demo Vector v")

        # Combined XORSAT visualization
        if args.save_combined:
            visualize_xorsat_instance(
                B,
                v,
                title="Demo XORSAT Instance",
                save_path=args.save_combined,
                var_names=var_names,
            )
        else:
            visualize_xorsat_instance(
                B, v, title="Demo XORSAT Instance", var_names=var_names
            )

    else:
        print("Use --demo to create and visualize a sample XORSAT instance")
        print(
            "Or import this script and call the functions with your own B matrix and v vector"
        )
        print("\nExample usage:")
        print(
            "  python visualize_xorsat.py --demo --rows 20 --cols 30 --b-density 0.2 --v-density 0.3"
        )
        print("  python visualize_xorsat.py --demo --save-combined xorsat_overview.png")


if __name__ == "__main__":
    main()
