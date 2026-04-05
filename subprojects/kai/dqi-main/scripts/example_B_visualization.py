#!/usr/bin/env python3
"""
Example usage of binary matrix visualization for the quantum dynamic pricing project.
This script shows how to visualize the binary matrix B generated from the ILP formulation.
"""

import os
import sys

import numpy as np

# Add the project root to Python path
sys.path.append(
    "/Users/caoyudong/Library/CloudStorage/OneDrive-TheBostonConsultingGroup,Inc/Documents/quantum_dynamic_pricing"
)

from pipelines.generate_B import (
    generate_B_matrix_and_rhs,
    visualize_binary_matrix,
    visualize_matrix_sparsity,
    visualize_matrix_statistics,
    visualize_vector_statistics,
    visualize_vector_v,
    visualize_xorsat_instance,
)


def main():
    """
    Generate and visualize the binary matrix B from a sample ILP problem.
    """
    print("Generating binary matrix B from ILP formulation...")

    # Sample problem data
    A = np.array(
        [
            [1, 2, 5, 3],
            [2, 1, 8, 4],
        ]
    )
    b = np.array([[16], [5]])
    c = np.array([10, 32, 12, 13])

    print(f"Constraint matrix A shape: {A.shape}")
    print(f"Right-hand side b: {b.flatten()}")
    print(f"Objective coefficients c: {c}")

    # Generate the binary matrix B
    B, v, ell, distance, max_num, var_names = generate_B_matrix_and_rhs(A, b, c)

    print(f"\nGenerated binary matrix B:")
    print(f"Shape: {B.shape}")
    print(f"Total elements: {B.shape[0] * B.shape[1]}")
    print(f"Number of 1s: {B.sum()}")
    print(f"Sparsity density: {B.sum() / (B.shape[0] * B.shape[1]):.4f}")
    print(f"Code distance: {distance}")
    print(f"Error correction parameter ell: {ell}")
    print(f"Max satisfiable constraints: {max_num}")

    # Create visualizations
    print("\nCreating separated visualizations...")

    # 1. Matrix pattern only
    visualize_matrix_sparsity(
        B,
        title="Binary Matrix B Pattern (ILP → XORSAT)",
        figsize=(12, 8),
        save_path="B_matrix_pattern.png",
    )

    # 2. Matrix statistics only
    visualize_matrix_statistics(
        B,
        title="Binary Matrix B - Sparsity Statistics",
        figsize=(15, 6),
        save_path="B_matrix_statistics.png",
    )

    # 3. Vector pattern only
    visualize_vector_v(
        v,
        title="Right-Hand Side Vector v Pattern",
        figsize=(12, 8),
        save_path="vector_v_pattern.png",
    )

    # 4. Vector statistics only
    visualize_vector_statistics(
        v,
        title="Vector v - Statistical Analysis",
        figsize=(12, 6),
        save_path="vector_v_statistics.png",
    )

    # 5. XORSAT instance overview (bird's eye view - patterns only)
    visualize_xorsat_instance(
        B,
        v,
        title="Max-XORSAT Instance: Bird's Eye View",
        figsize=(16, 8),
        save_path="xorsat_instance_overview.png",
        var_names=var_names,
    )

    print("\nMatrix analysis complete!")
    print("Generated files:")
    print("- B_matrix_pattern.png: Binary matrix sparsity pattern only")
    print("- B_matrix_statistics.png: Matrix sparsity statistics and histograms")
    print("- vector_v_pattern.png: Vector v pattern visualization only")
    print("- vector_v_statistics.png: Vector v statistical analysis")
    print("- xorsat_instance_overview.png: Complete XORSAT instance bird's eye view")

    # Print some interesting statistics
    row_sparsity = B.sum(axis=1)
    col_sparsity = B.sum(axis=0)

    print(f"\nRow sparsity statistics:")
    print(f"  Min: {row_sparsity.min()}")
    print(f"  Max: {row_sparsity.max()}")
    print(f"  Mean: {row_sparsity.mean():.2f}")
    print(f"  Std: {row_sparsity.std():.2f}")

    print(f"\nColumn sparsity statistics:")
    print(f"  Min: {col_sparsity.min()}")
    print(f"  Max: {col_sparsity.max()}")
    print(f"  Mean: {col_sparsity.mean():.2f}")
    print(f"  Std: {col_sparsity.std():.2f}")


if __name__ == "__main__":
    main()
