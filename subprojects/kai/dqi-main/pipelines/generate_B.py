# Script for generating the matrix B for constraints of the form
#
#    a^(1) x1 + a^(2) x2 + ... + a^(m) xm (= or <=) b
#
# where integers a^(1) through a^(m) are of n bits.

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sympy import Matrix

from pipelines.encoding_F2 import (
    IntegerComparatorOneUnknown,
    IntegerEqualityOneUnknown,
    MultipleIntegerAdder,
    VariableManager,
)


def generate_F2_instance(a, b, eq="="):
    """
    Function for generating an instance of max-XORSAT (binary equations mod 2)
    from an integer equality or inequality constraint.

    Inputs:
    a: list[int]
        List of integer coefficients. Assume that a list of positive integers.
    b: int
        Number in the right hand side. Assume that b is positive.
    eq: String
        String variable taking value from "=", "<", ">=".

    Outputs:
    The outputs of the function describe in full the max-XORSAT instance Bx=y
    with matrix B and vector y generated as the following.

    list_B: list[dict]
        List of dictionaries with each dictionary listing the variables included in the row of the matrix B.
    list_y: list[int]
        List of integers representing the right hand side coefficient of the max-XORSAT instance.
    vm: VariableManager
        Contains the list of variables in `vm.vars`.
    """
    max_number_constraints_per_circuit = 0
    if sum([abs(x) for x in a]) < b:
        print("Warning: constraint infeasible because sum(|a|) < b.")
    assert eq in ["=", "<", ">="]

    m = len(a)  # total number of terms in the sum
    tree = MultipleIntegerAdder.build_tree_structure(a)
    precomputed_tree = MultipleIntegerAdder.calculate_num_bits(a, tree)
    # print(precomputed_tree)

    # Calculate the total number of various types of bits
    # Multiple Integer Adder
    total_carry_bits = 0
    total_aux_bits = 0
    total_intermediate_bits = 0
    total_output_bits = 0
    for node_name, node_data in precomputed_tree.items():
        if "n_carry_bits" in node_data:
            total_carry_bits += node_data["n_carry_bits"]
        if "n_aux_bits" in node_data:
            total_aux_bits += node_data["n_aux_bits"]
        if "n_intermediate_bits" in node_data:
            total_intermediate_bits += node_data["n_intermediate_bits"]
        if "n_output_bits" in node_data:
            total_output_bits += node_data["n_output_bits"]
    # Final step: y = b, y < b, or y >= b
    if eq in ["<", ">="]:
        # if inequality, add the necessary bits for the IntegerComparatorOneUnknown
        total_carry_bits += total_output_bits
        total_intermediate_bits += total_output_bits + 1

    # Introduce the variables
    vm = VariableManager()
    x_vars = [vm.new_variable(f"x_{i}_") for i in range(m)]
    carry_bits = [vm.new_variable(f"c_{i}_") for i in range(total_carry_bits)]
    aux_bits = [vm.new_variable(f"aux_{i}_") for i in range(total_aux_bits)]
    intermediate_bits = [
        vm.new_variable(f"int_{i}_") for i in range(total_intermediate_bits)
    ]
    y_vars = [vm.new_variable(f"y_{i}_") for i in range(total_output_bits)]

    # Build the gadget circuit
    b_int = int(b)  # Convert numpy.int64 to Python int
    b_bits = [int(bit) for bit in bin(b_int)[2:]][::-1] + [0] * (
        total_output_bits - b_int.bit_length()
    )
    if eq == "=":
        miadder = MultipleIntegerAdder(
            x_vars,
            a,
            carry_bits,
            intermediate_bits,
            y_vars,
            aux_bits,
            precomputed_tree,
        )
        for gadget in miadder.gadgets:
            max_number_constraints_per_circuit += gadget.max_num_satisfied
        equality = IntegerEqualityOneUnknown(y_vars, b_bits)
        for gadget in equality.gadgets:
            max_number_constraints_per_circuit += gadget.max_num_satisfied
        list_equations = miadder.equations + equality.equations
    else:
        miadder = MultipleIntegerAdder(
            x_vars,
            a,
            carry_bits[:-total_output_bits],
            intermediate_bits[: -(total_output_bits + 1)],
            y_vars,
            aux_bits,
            precomputed_tree,
        )
        for gadget in miadder.gadgets:
            max_number_constraints_per_circuit += gadget.max_num_satisfied

        inequality = IntegerComparatorOneUnknown(
            y_vars,
            b_bits,
            carry_bits[-total_output_bits:],
            intermediate_bits[-(total_output_bits + 1) :],
            eq,
        )
        for gadget in inequality.gadgets:
            max_number_constraints_per_circuit += gadget.max_num_satisfied

        list_equations = miadder.equations + inequality.equations

    # Generate the output matrix B and vector y
    list_B = [eq.coefficients for eq in list_equations]
    list_y = [eq.rhs for eq in list_equations]

    return list_B, list_y, vm, max_number_constraints_per_circuit


def generate_B_matrix_and_rhs(A, b_vec, c_vec, beta=0):
    """
    Function for generating an instance of max-XORSAT (binary equations mod 2) from an ILP instance.
    Takes as input an objective function and a set of integer equality or inequality constraint.
    Along the way, the function also computes the number of constraints satisfied by a Gurobi solution.

    Inputs:
    A: numpy.ndarray
        Matrix of integer coefficients of dimension (n_constraints, n_variables):
        left hand side of the linear inequality constraints
    b_vec: numpy.ndarray
        Vector of integer coeffients of dimension n_constraints: right hand side of linear inequality constraints.
    c_vec: numpy.ndarray
         Vector of integer coeffients of dimension n_variables: linear objective function.
    beta: int
        The value of the lower/upper bound for the binary search.

    Outputs:
    Returns the B matrix and the right hand side (modulo 2) of the max-XORSAT instance,
    as well as the ell which is the number of errors that need to be corrected for the dual code.

    B: np.ndarray
        Left hand side of XORSAT instance. Contains only zeros and ones.
    rhs: np.ndarray
        Right hand side of XORSAT instance. Contains only zeros and ones.
    ell: int
        Number of errors to correct in the dual code.
    max_number_of_constraints_satisfiable: int
        Maximum number of constraints that can be satisfied in the max-XORSAT instance.
    filtered_var_names: list
        List of variable names corresponding to the columns in the filtered matrix B.
    """
    full_var_list = []
    list_of_list_B = []
    list_of_list_v = []

    # Add the objective function to the constraints

    A = np.concatenate((A, c_vec.reshape(1, -1)), axis=0)
    # b_vec = np.concatenate((b_vec, np.array([beta + 1]).reshape(1, 1)), axis=0)
    b_vec = np.concatenate((b_vec, np.array([beta + 1])))

    max_number_of_constraints_satisfiable = 0

    # Encode linear inequalities
    common_var_name_list = [f"x_{j}_({j})" for j in range(A.shape[1])]

    full_var_list += common_var_name_list
    for i in range(A.shape[0]):
        a = A[i, :].tolist()
        b = int(b_vec[i])
        if i < A.shape[0] - 1:
            eq = "<"
        else:
            eq = ">="  # the last one (corresponding to the objective function) should be obj >= LB for binary search
        list_B, list_v, vm, max_num_constraints_per_circuit = generate_F2_instance(
            a,
            b,
            eq,
        )
        max_number_of_constraints_satisfiable += max_num_constraints_per_circuit
        list_of_list_B.append(list_B)
        list_of_list_v.append(list_v)

        for j in range(len(vm.vars)):
            if vm.vars[j] in common_var_name_list:
                continue
            full_var_list.append(vm.vars[j] + f"_{i}")

    # Summarize the parameters of B and v
    nb_rows_B = sum([len(list_B) for list_B in list_of_list_B])
    nb_colums_B = len(full_var_list)
    B = np.zeros((nb_rows_B, nb_colums_B))
    v = np.concatenate([np.array(list_v) for list_v in list_of_list_v])
    # print(f"B shape: {B.shape}")
    # print(f"Number of variables: {len(full_var_list)}")

    k = 0
    for list_B, i in zip(list_of_list_B, list(range(A.shape[0] + 2))):
        for row_B in list_B:
            for var in row_B.keys():
                if var in common_var_name_list:
                    var_indices = np.where(np.array(full_var_list) == var)[0]
                    # print("common")
                    # print(var_indices)
                    assert len(var_indices) == 1
                else:
                    var_indices = np.where(np.array(full_var_list) == var + f"_{i}")[0]
                    # print(var + f"_{i}")
                    # print("not common")
                    # print(var_indices)
                    assert len(var_indices) == 1
                B[k, var_indices[0]] = 1
            k += 1

    # Find column sums
    col_sums = B.sum(axis=0)

    # Mask for columns where the sum is not zero
    nonzero_cols = col_sums != 0

    # Filter the columns and corresponding variable names
    B = B[:, nonzero_cols]
    filtered_var_names = [
        full_var_list[i] for i in range(len(full_var_list)) if nonzero_cols[i]
    ]

    # print(f"Sparsity of rows of B: {B.sum(axis=1)}")
    # print(f"Sparsity of columns of B: {B.sum(axis=0)}")

    # print(f"Common var name list: {common_var_name_list}")

    # print(" B shape: ", B.shape)
    # print(
    #     "Max number of constraints satisfiable: ",
    #     max_number_of_constraints_satisfiable,
    # )

    # Find generator

    G = systematic_form_G(B.T)

    code_distance = min(G.sum(axis=1))
    ell = max(1, int(code_distance - 1) / 2)

    # print(f"code_distance: {code_distance}")
    # print(f"ell: {ell}")

    return (
        B,
        v,
        ell,
        code_distance,
        max_number_of_constraints_satisfiable,
        filtered_var_names,
    )


def systematic_form_G(H):
    """
    Function that finds, via Gaussian elimination, a generator from a parity check matrix.

    Inputs:
    H: numpy.ndarray
        Parity check matrix. Contains only zeros and ones.

    Outputs:
    Returns a generator, i.e a matrix verifying G @ H = 0.

    G: numpy.ndarray
        Generator matrix.
    """

    H = sp.csr_matrix(H)
    m, n = H.shape
    k = n - m

    # Convert to dense for pivoting
    H_dense = H.toarray().astype(int) % 2

    # Gaussian elimination over GF(2)
    H_sym = Matrix(H_dense.tolist()).rref(iszerofunc=lambda x: x % 2 == 0)[0]
    H_bin = np.array(H_sym.tolist(), dtype=int) % 2

    # Identify pivot columns (those forming the identity matrix)
    pivot_cols = []
    for row in H_bin:
        for j in range(len(row)):
            if row[j] == 1:
                pivot_cols.append(j)
                break
    pivot_cols = sorted(set(pivot_cols))
    non_pivot_cols = [j for j in range(n) if j not in pivot_cols]

    # Permute H to [P | I]
    perm = non_pivot_cols + pivot_cols
    H_perm = H_bin[:, perm]
    P = H_perm[:, :k]

    # Construct G = [I | P^T]
    I_k = np.identity(k, dtype=int)
    G = np.hstack((I_k, P.T))

    # Re-permute G columns to original order
    reverse_perm = np.argsort(perm)
    G_final = G[:, reverse_perm]
    return G_final


def visualize_binary_matrix(
    B,
    title="Binary Matrix B",
    figsize=(12, 8),
    save_path=None,
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


def visualize_matrix_sparsity(
    B,
    title="Matrix Sparsity Pattern",
    figsize=(12, 8),
    save_path=None,
):
    """
    Create a clean visualization of the binary matrix sparsity pattern only.

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
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Calculate basic statistics for title
    total_elements = B.shape[0] * B.shape[1]
    total_ones = B.sum()
    sparsity_ratio = total_ones / total_elements

    # Main matrix visualization
    im = ax.imshow(B, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_title(
        f"{title}\nShape: {B.shape[0]} × {B.shape[1]}, Ones: {total_ones,}, Density: {100 * sparsity_ratio:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Variables (Columns)", fontsize=12)
    ax.set_ylabel("Constraints (Rows)", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Value", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matrix pattern saved to: {save_path}")
    else:
        plt.show()


def visualize_matrix_statistics(
    B,
    title="Matrix Statistics",
    figsize=(15, 6),
    save_path=None,
):
    """
    Create statistical analysis visualization of binary matrix sparsity.

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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{title}\nMatrix Shape: {B.shape[0]} × {B.shape[1]}", fontsize=16)

    # Row sparsity (number of 1s per row)
    row_sums = B.sum(axis=1)
    axes[0].hist(
        row_sums,
        bins=min(50, len(np.unique(row_sums))),
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    axes[0].set_title(
        f"Row Sparsity Distribution\n(Constraint Complexity)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Number of 1s per row")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Min: {row_sums.min():.1f}\nMax: {row_sums.max():.1f}\nMean: {row_sums.mean():.2f}\nStd: {row_sums.std():.2f}"
    axes[0].text(
        0.65,
        0.95,
        stats_text,
        transform=axes[0].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Column sparsity (number of 1s per column)
    col_sums = B.sum(axis=0)
    axes[1].hist(
        col_sums,
        bins=min(50, len(np.unique(col_sums))),
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    axes[1].set_title(
        f"Column Sparsity Distribution\n(Variable Usage)",
        fontweight="bold",
    )
    axes[1].set_xlabel("Number of 1s per column")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Min: {col_sums.min():.1f}\nMax: {col_sums.max():.1f}\nMean: {col_sums.mean():.2f}\nStd: {col_sums.std():.2f}"
    axes[1].text(
        0.65,
        0.95,
        stats_text,
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Matrix statistics saved to: {save_path}")
    else:
        plt.show()


def visualize_vector_v(
    v,
    title="Right-Hand Side Vector v",
    figsize=(12, 8),
    save_path=None,
):
    """
    Create a clean visualization of the binary vector v pattern only.

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
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{title}\nVector Length: {len(v)}", fontsize=16)

    # Calculate statistics for title
    total_ones = v.sum()
    density = total_ones / len(v) if len(v) > 0 else 0

    # 1. Vector visualization as vertical strip
    v_matrix = v.reshape(-1, 1)  # Convert to column matrix for visualization
    im = axes[0].imshow(v_matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    axes[0].set_title(
        f"Vector Pattern (Vertical)\nOnes: {total_ones}, Density: {100 * density:.1f}%",
        fontweight="bold",
    )
    axes[0].set_xlabel("Vector (Width=1)")
    axes[0].set_ylabel("Elements (Rows)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.8)
    cbar.set_label("Value")

    # 2. Vector values over index as bar chart
    indices = np.arange(len(v))
    colors = ["lightgray" if val == 0 else "darkblue" for val in v]
    axes[1].bar(indices, v, color=colors, edgecolor="black", alpha=0.7, width=0.8)
    axes[1].set_title("Vector Values vs Index", fontweight="bold")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Value")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Vector pattern saved to: {save_path}")
    else:
        plt.show()


def visualize_vector_statistics(
    v,
    title="Vector Statistics",
    figsize=(12, 6),
    save_path=None,
):
    """
    Create statistical analysis visualization of binary vector v.

    Inputs:
    v: numpy.ndarray
        Binary vector containing only zeros and ones.
    title: str
        Title for the plots.
    figsize: tuple
        Figure size (width, height).
    save_path: str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f"{title}\nVector Length: {len(v)}", fontsize=16)

    # 1. Value distribution histogram
    unique, counts = np.unique(v, return_counts=True)
    axes[0].bar(
        unique,
        counts,
        color=["lightgray", "darkblue"],
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_title("Value Distribution", fontweight="bold")
    axes[0].set_xlabel("Binary Values")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks([0, 1])
    axes[0].grid(True, alpha=0.3)

    # Add percentage labels
    total = len(v)
    for i, (val, count) in enumerate(zip(unique, counts)):
        percentage = 100 * count / total
        axes[0].text(
            val,
            count + total * 0.01,
            f"{count}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 2. Vector statistics text
    total_elements = len(v)
    total_ones = v.sum()
    total_zeros = total_elements - total_ones
    density = total_ones / total_elements if total_elements > 0 else 0

    # Find patterns (consecutive runs)
    if len(v) > 1:
        diff = np.diff(np.concatenate(([v[0]], v, [1 - v[-1]])))
        run_starts = np.where(diff != 0)[0]
        run_lengths = np.diff(run_starts)
    else:
        run_lengths = np.array([1])

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

    axes[1].text(
        0.05,
        0.95,
        stats_text,
        transform=axes[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Vector Statistics Summary", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Vector statistics saved to: {save_path}")
    else:
        plt.show()


def visualize_xorsat_instance(
    B,
    v,
    title="Max-XORSAT Instance: Bx = v",
    figsize=(16, 8),
    save_path=None,
    var_names=None,
):
    """
    Create a bird's eye view visualization of the max-XORSAT instance showing both B and v patterns only.

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
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.1},
    )

    # Calculate statistics for titles
    B_total_ones = B.sum()
    B_density = B_total_ones / (B.shape[0] * B.shape[1])
    v_total_ones = v.sum()
    v_density = v_total_ones / len(v)

    # 1. Matrix B visualization on the left
    im1 = ax1.imshow(B, cmap="Blues", aspect="auto", interpolation="nearest")
    ax1.set_title(
        f"Matrix B (Left-Hand Side)\nShape: {B.shape[0]}×{B.shape[1]}, Ones: {B_total_ones:,}, Density: {100*B_density:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Variables", fontsize=11)
    ax1.set_ylabel("Constraints", fontsize=11)
    ax1.grid(True, alpha=0.3, color="gray", linewidth=0.5)

    # Set x-axis ticks and labels for variables
    if var_names is not None and len(var_names) == B.shape[1]:
        # Set tick positions for all variables
        ax1.set_xticks(range(B.shape[1]))
        ax1.set_xticklabels(var_names, rotation=45, ha="right", fontsize=8)
    else:
        # Fallback to column numbers if no variable names provided
        n_vars = B.shape[1]
        if n_vars <= 40:  # Show all labels for small number of variables
            ax1.set_xticks(range(n_vars))
            ax1.set_xticklabels(
                [f"x_{i}" for i in range(n_vars)],
                rotation=45,
                ha="right",
                fontsize=8,
            )
        else:  # Show subset of labels for large number of variables
            step = max(1, n_vars // 10)
            tick_positions = range(0, n_vars, step)
            ax1.set_xticks(tick_positions)
            ax1.set_xticklabels(
                [f"x_{i}" for i in tick_positions],
                rotation=45,
                ha="right",
                fontsize=8,
            )

    # 2. Vector v visualization on the right
    v_matrix = v.reshape(-1, 1)  # Convert to column matrix for visualization
    im2 = ax2.imshow(v_matrix, cmap="Blues", aspect="auto", interpolation="nearest")
    ax2.set_title(
        f"Vector v\n(RHS)\nLength: {len(v)}, Ones: {v_total_ones}, Density: {100*v_density:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlabel("v", fontsize=11)
    ax2.set_ylabel("Constraints", fontsize=11)
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, color="gray", linewidth=0.5)

    # Add overall title
    fig.suptitle(f"{title}", fontsize=16, fontweight="bold", y=0.95)

    # Add colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label("Value", fontsize=10)
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
    cbar2.set_label("Value", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"XORSAT instance overview saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # A = np.array(
    #     [
    #         [1, 2, 5, 3],
    #         [2, 1, 8, 4],
    #     ],
    # )
    # b = np.array([[16], [5]])

    # c = np.array([10, 32, 12, 13])
    A = np.array([[1, 1]])
    b = np.array([[2]])
    c = np.array([[1, 2]])

    B, v, ell, distance, max_num, var_names = generate_B_matrix_and_rhs(
        A,
        b,
        c,
    )

    # Generate comprehensive visualizations
    print("\nGenerating comprehensive visualizations for the max-XORSAT instance...")

    # 1. Basic matrix visualization
    visualize_binary_matrix(
        B,
        title="Binary Matrix B",
        figsize=(10, 6),
        save_path="matrix_B_visualization.png",
    )

    # 2. Matrix sparsity analysis
    visualize_matrix_sparsity(
        B,
        title="Matrix B Sparsity Analysis",
        figsize=(15, 10),
        save_path="matrix_B_sparsity.png",
    )

    # 3. Vector v visualization
    visualize_vector_v(
        v,
        title="Right-Hand Side Vector v",
        figsize=(12, 10),
        save_path="vector_v_analysis.png",
    )

    # 4. Combined XORSAT instance visualization (the main bird's eye view)
    visualize_xorsat_instance(
        B,
        v,
        title="Max-XORSAT Instance Overview",
        figsize=(18, 12),
        save_path="xorsat_instance_overview.png",
        var_names=var_names,
    )

    print("\nVisualization complete! Generated files:")
    print("- matrix_B_visualization.png: Basic binary matrix heatmap")
    print("- matrix_B_sparsity.png: Detailed matrix sparsity analysis")
    print("- vector_v_analysis.png: Comprehensive vector v analysis")
    print("- xorsat_instance_overview.png: Complete XORSAT instance bird's eye view")

    print(f"\nInstance summary:")
    print(f"- Matrix B shape: {B.shape}")
    print(f"- Vector v length: {len(v)}")
    print(f"- Matrix density: {100 * B.sum() / (B.shape[0] * B.shape[1]):.2f}%")
    print(f"- Vector density: {100 * v.sum() / len(v):.1f}%")
