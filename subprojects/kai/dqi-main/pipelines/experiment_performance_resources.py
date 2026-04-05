import csv
import glob
import json
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" Uncomment if GPUs are available
import pickle
import re
import time

import numpy as np
import pandas as pd

from pipelines.belief_propagations import (
    belief_propagation_gallager,
    belief_propagation_ldpc,
)
from pipelines.downsized_matrix_samples import (
    generate_initial_matrix,
    mcmc_sample,
    plot_histograms,
)
from pipelines.DQI_classical import expected_constrains_DQI
from pipelines.DQI_full_circuit import (
    average_of_f_and_s_random,
    resource_estimation_function,
)
from pipelines.generate_B import generate_B_matrix_and_rhs
from pipelines.gurobi_code import (
    read_gurobi_results_fun,
    solve_max_xor_sat,
    transform_xor_problems_for_gurobi,
)
from scripts.plots_performance_plus_resources import (
    plot_performance,
    plot_resources,
    plot_type_of_gates,
)

downsized_matrix_sizes = [
    (23, 12),
    (43, 22),
    (63, 32),
    (83, 42),
    (103, 52),
    (123, 62),
    (143, 72),
    (163, 82),
    (183, 92),
    (203, 102),
    (223, 112),
    (243, 122),
    (263, 132),
    (283, 142),
    (303, 152),
    (323, 162),
    (343, 172),
    (363, 182),
    (383, 192),
    (403, 202),
    (423, 212),
    (443, 222),
    (463, 232),
    (483, 242),
    (503, 252),
    (523, 262),
    (543, 272),
    (563, 282),
    (583, 292),
]


def generate_B_from_actual_ILP(path_B_save_max_xorsat, json_ILP_file_path):

    os.makedirs(path_B_save_max_xorsat, exist_ok=True)

    with open(json_ILP_file_path) as f:
        milp_formulation = json.load(f)

    # MILP instance
    # max c^T x s.t A x <= b
    c = np.array(milp_formulation["objective"])
    c = c - np.min(c) + 1
    A = np.array(milp_formulation["constraints"])
    b = np.array(milp_formulation["constraints_rhs"])

    beta_max = np.sum(np.abs(c))
    beta = beta_max / 2
    beta_int = int(beta)
    # print("beta", beta_int)

    B, v, ell, code_distance, max_num_constraints, _ = generate_B_matrix_and_rhs(
        A,
        b,
        c,
        beta=beta_int,
    )

    B = B.astype(np.int8)

    m, n = B.shape
    # print(m, n, m * n)

    fn = os.path.join(
        path_B_save_max_xorsat,
        "real_ILP_maxXORSAT.npz",
    )
    # Save B, v, and B.shape
    np.savez(
        fn,
        B=B,
        v=v,
        shape_B=B.shape,
        ell=ell,
        max_number_constrains=max_num_constraints,
        beta_value=beta_int,
        code_distance_value=code_distance,
    )


def generate_downsized_matrices(B_input, v_input, n_matrices, downsized_matrix_sizes):

    # Extract histograms of the structure of B from ILP
    delta, kappa, i, j, variable_degree, M, N = plot_histograms(B_input, plot=False)
    # checks
    print("Normalization Delta = ", np.sum(delta))
    print("Normalization Kappa = ", np.sum(kappa))

    print("Check if: ", M * np.sum(i * kappa), " = ", N * np.sum(j * delta))

    print("Matrix size = ", M, N)

    # Sampling random marginals

    # Repeat T times until compatible marginals are found.
    T = 1000000000
    marginals = []
    for k in range(n_matrices):
        m, n = downsized_matrix_sizes[k]
        partial_marginals = []
        for _ in range(T):
            p_constraints = np.arange(1, len(kappa) + 1)
            sampled_constraints = np.random.choice(p_constraints, size=m, p=kappa)

            p_variables = np.arange(1, len(delta) + 1)
            sampled_variables = np.random.choice(p_variables, size=n, p=delta)

            if np.sum(sampled_constraints) == np.sum(sampled_variables):
                partial_marginals.append([sampled_constraints, sampled_variables])
                print("Found compatible set", k)
                if len(partial_marginals) >= 1:
                    break
        marginals.append(partial_marginals)

    my_list = marginals

    with open("pipelines/max_xorsat_industrial_ILP/marginals_real_ilp.pkl", "wb") as f:
        pickle.dump(my_list, f)

    for iteration in range(5):
        # print(iteration)

        sampled_Bs = []

        for k in range(n_matrices):
            row_sums, col_sums = marginals[k][0]
            np.random.shuffle(col_sums)
            np.random.shuffle(row_sums)

            assert sum(row_sums) == sum(col_sums), "Inconsistent marginals!"

            B_init = generate_initial_matrix(row_sums.copy(), col_sums.copy())
            B_sampled = mcmc_sample(B_init, iterations=10000)

            sampled_Bs.append(B_sampled)

            col_diff = np.sum(np.abs(np.sum(B_sampled, axis=0) - col_sums))
            row_diff = np.sum(np.abs(np.sum(B_sampled, axis=1) - row_sums))
            if col_diff != 0 or row_diff != 0:
                print("Matrix with size ", B_sampled.shape, " failed")

        sampled_vs = []
        for B in sampled_Bs:
            m, n = B.shape[0], B.shape[1]
            a, b = 0.2, 0.3
            random_sparcity = np.random.uniform(a, b)
            n_ones = int(np.round(m * random_sparcity))
            v = np.zeros((m,))
            v[:n_ones] = 1
            np.random.shuffle(v)
            # print(v)
            sampled_vs.append(v)

        # save matrices

        path_B_save_matrices = (
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(iteration)
        )
        os.makedirs(path_B_save_matrices, exist_ok=True)

        for (
            list_index,
            B,
        ) in enumerate(sampled_Bs):
            fn = os.path.join(
                path_B_save_matrices,
                f"iter_{list_index}.npz",
            )
            v = sampled_vs[list_index]
            np.savez(
                fn,
                B=B,
                v=v,
                shape_B=B.shape,
            )


def gurobi_benchmark(n_matrices_max_xorsat, n_matrices):

    """Benchmark Gurobi on a set of downsized Max-XOR-SAT instances.

    Args:
        n_matrices_max_xorsat (int): Maximum number of matrices (problem instances)
            to solve per iteration. Must not exceed the total number of available
            matrices (`n_matrices`).

    Behavior:
        - Transforms downsized Max-Xorsat matrices into XOR-SAT text input for Gurobi.
        - Iterates over 5 benchmark rounds (`it = 0..4`), each with its own input/output folder.
        - Solves up to `n_matrices_max_xorsat` XOR-SAT instances per round using Gurobi.
        - Tracks runtimes for each solve.
        - Reads results, computes satisfaction ratios, and saves summaries as CSV
          in `pipelines/performance_results/`.

    Raises:
        ValueError: If `n_matrices_max_xorsat` is larger than the available
        number of matrices (`n_matrices`).

    Outputs:
        - Individual Gurobi solution files: `pipelines/gurobi_results/iteration*/iter_*.txt`
        - Performance summary CSVs:
          `pipelines/performance_results/gurobi_satisfaction_summary_it*.csv`
    """

    if n_matrices_max_xorsat > n_matrices:
        raise ValueError(
            f"Requested {n_matrices_max_xorsat} matrices, but only {n_matrices} available.",
        )

    for it in range(5):
        path_B_matrices = (
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(it)
        )
        path_XOR_problems = (
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_gurobis_input_mc_"
            + str(it)
        )

        B_shape_list, b_list = transform_xor_problems_for_gurobi(
            path_B_matrices,
            path_XOR_problems,
        )  # transform downsized matrices into an input function reasonable for gurobi

        times = []  # save times for tracking exponential time of Gurobi
        for k in range(n_matrices_max_xorsat):
            filename = path_XOR_problems + "/iter_" + str(k) + ".txt"
            filename_save = (
                "pipelines/gurobi_results/iteration"
                + str(it)
                + "/iter_"
                + str(k)
                + ".txt"
            )
            start = time.time()
            try:
                # Read the content of the file as plain text
                with open(filename, "r") as file:
                    data_lines = file.readlines()  # Read all lines from the file
                    # Strip whitespace and process non-empty lines
                    data_lines = [line.strip() for line in data_lines if line.strip()]
                    solve_max_xor_sat(data_lines, filename, filename_save)
            except FileNotFoundError:
                print(f"File not found: {filename}")
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")

            end = time.time()
            times.append(end - start)
            print("Runtime: ", end - start)

        # sizes_matrices = downsized_matrix_sizes[
        #     :n_matrices_max_xorsat
        # ]  # for plotting time vs size

        gurobi_folder = "pipelines/gurobi_results/iteration" + str(it)

        data = read_gurobi_results_fun(gurobi_folder, path_B_matrices)
        # Sort data by iteration index
        data_sorted = sorted(data, key=lambda x: x[0])

        # Define output CSV path
        csv_output_path = (
            "pipelines/performance_results/gurobi_satisfaction_summary_it"
            + str(it)
            + ".csv"
        )  # hange path as needed
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        # Write sorted data to CSV
        with open(csv_output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration_index", "satisfaction_ratio"])  # header

            for instance in data_sorted:
                idx = instance[0]
                satisfaction = instance[5].get("satisfaction_ratio", None)
                writer.writerow([idx, satisfaction])

        print(f"CSV file written to: {csv_output_path}")


def run_performance_on_existing_Bs_l1_l2(
    output_file_path,
    path_B_load_matrices,
    max_iterations=2,
    skip_l2=False,
    start_it=0,
    final_it=None,
):
    """
    Evaluate DQI constraint satisfaction using belief propagation for existing max-XORSAT instances.

    This function loads binary matrices `B` and vectors `v` from `.npz` files in the specified directory.
    For each instance, it computes the expected number of satisfied constraints using the DQI method with
    two variants of belief propagation (Gallager and LDPC) at levels `ell = 1` and `ell = 2`. The results
    are normalized and saved in a CSV file.

    Parameters
    ----------
    output_file_path : str
        Full path (including filename) where the output CSV will be saved.

    path_B_load_matrices : str
        Directory containing `.npz` files named in the format `iter_<index>.npz`, each storing:
        - 'B': the binary matrix
        - 'v': the binary right-hand-side vector
        - 'ell': the decoding parameter
        - 'shape_B': the shape of B

    max_iterations : int, optional
        Maximum number of iterations for the belief propagation algorithm.
        Default is 2.

    Output
    ------
    A CSV file at `output_file_path` containing the following columns:
        - B_shape: shape of matrix B
        - optimal_ell: the stored `ell` parameter
        - dqi_expected_s_bp1: normalized DQI performance with BP1 at ell = 1
        - dqi_expected_s_bp2: normalized DQI performance with BP2 at ell = 1
        - dqi_expected_s_bp1_l2: normalized DQI performance with BP1 at ell = 2
        - dqi_expected_s_bp2_l2: normalized DQI performance with BP2 at ell = 2
    """

    def extract_iter_number(filename):
        match = re.search(r"iter_(\d+)\.npz", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path_B_load_matrices, "iter_*.npz")),
        key=extract_iter_number,
    )
    if final_it is not None:
        files = files[start_it:final_it]
    else:
        files = files[start_it:]

    n_instances = len(files)
    B_shape = [None] * n_instances
    optimal_ell_list = [0] * n_instances

    dqi_expected_s_bp1 = [0] * n_instances
    dqi_expected_s_bp2 = [0] * n_instances

    dqi_expected_s_bp1_l2 = [0] * n_instances
    dqi_expected_s_bp2_l2 = [0] * n_instances

    for list_index, fn in enumerate(files):
        data = np.load(fn)
        B = data["B"]
        v = data["v"]
        optimal_ell = 1  # data["ell"]

        shape_B = tuple(data["shape_B"])
        m, n = B.shape

        B_shape[list_index] = shape_B
        optimal_ell_list[list_index] = optimal_ell

        print(f"Processing {fn} | Shape: {shape_B}")

        for ell in [1, 2]:
            for bp_version in ["BP1", "BP2"]:

                if bp_version == "BP1":
                    bp_function = belief_propagation_gallager
                elif bp_version == "BP2":
                    bp_function = belief_propagation_ldpc

                print(f"Running {bp_version}, ell = {ell}")
                if ell == 2 and skip_l2:
                    f = 0
                    s = 0
                else:
                    f, s = expected_constrains_DQI(
                        B,
                        v,
                        ell,
                        max_iterations,
                        bp_function,
                        jit_version=True,
                    )

                s_normalized = s / m

                if bp_version == "BP1":
                    if ell == 1:
                        dqi_expected_s_bp1[list_index] = s_normalized
                    elif ell == 2:
                        dqi_expected_s_bp1_l2[list_index] = s_normalized

                elif bp_version == "BP2":
                    if ell == 1:
                        dqi_expected_s_bp2[list_index] = s_normalized
                    elif ell == 2:
                        dqi_expected_s_bp2_l2[list_index] = s_normalized

        # Compile results
        result_dict = {
            "B_shape": B_shape,
            "optimal_ell": optimal_ell_list,
            "dqi_expected_s_bp1": dqi_expected_s_bp1,
            "dqi_expected_s_bp2": dqi_expected_s_bp2,
            "dqi_expected_s_bp1_l2": dqi_expected_s_bp1_l2,
            "dqi_expected_s_bp2_l2": dqi_expected_s_bp2_l2,
        }

        df = pd.DataFrame(result_dict)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path)
        print(f"Results saved to: {output_file_path}")


def run_performance_on_existing_Bs_l3(
    output_file_path,
    path_B_load_matrices,
    max_iterations=2,
    ell=3,
    start_it=0,
    final_it=None,
    numpy_backup=False,
):
    """
    Evaluate DQI constraint satisfaction at ell = 3 using belief propagation for max-XORSAT instances.

    This function loads binary matrices `B` and vectors `v` from `.npz` files in the specified directory.
    For each instance, it computes the expected number of satisfied constraints using the DQI method with
    two belief propagation variants (Gallager and LDPC) at level `ell = 3`. The results are normalized and
    saved as a CSV file.

    Parameters
    ----------
    output_file_path : str
        Full path (including filename) where the output CSV will be saved.

    path_B_load_matrices : str
        Directory containing `.npz` files named in the format `iter_<index>.npz`, each storing:
        - 'B': the binary matrix
        - 'v': the binary right-hand-side vector
        - 'ell': the decoding parameter
        - 'shape_B': the shape of B

    max_iterations : int, optional
        Maximum number of iterations for the belief propagation algorithm.
        Default is 2.

    Output
    ------
    A CSV file at `output_file_path` containing the following columns:
        - B_shape: shape of matrix B
        - optimal_ell: the stored `ell` parameter
        - dqi_expected_s_bp1_l3: normalized DQI performance with BP1 at ell = 3
        - dqi_expected_s_bp2_l3: normalized DQI performance with BP2 at ell = 3
    """

    def extract_iter_number(filename):
        match = re.search(r"iter_(\d+)\.npz", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path_B_load_matrices, "iter_*.npz")),
        key=extract_iter_number,
    )
    if final_it is not None:
        files = files[start_it:final_it]
    else:
        files = files[start_it:]

    n_instances = len(files)
    B_shape = [None] * n_instances
    optimal_ell_list = [0] * n_instances

    dqi_expected_s_bp1_l3 = [0] * n_instances
    dqi_expected_s_bp2_l3 = [0] * n_instances

    for list_index, fn in enumerate(files):
        data = np.load(fn)
        B = data["B"]
        v = data["v"]
        optimal_ell = 1  # data["ell"]

        shape_B = tuple(data["shape_B"])
        m, n = B.shape

        B_shape[list_index] = shape_B
        optimal_ell_list[list_index] = optimal_ell

        print(f"Processing {fn} | Shape: {shape_B}")

        # ell = 3
        for bp_version in ["BP1", "BP2"]:

            if bp_version == "BP1":
                bp_function = belief_propagation_gallager
            elif bp_version == "BP2":
                bp_function = belief_propagation_ldpc

            print(f"Running {bp_version}, ell = {ell}")

            f, s = expected_constrains_DQI(
                B,
                v,
                ell,
                max_iterations,
                bp_function,
                jit_version=True,
            )
            # f, s = expected_constrains_DQI_streaming(
            #     B,
            #     v,
            #     ell,
            #     max_iterations,
            #     bp_function,
            # )

            s_normalized = s / m

            if bp_version == "BP1":
                dqi_expected_s_bp1_l3[list_index] = s_normalized

            elif bp_version == "BP2":
                dqi_expected_s_bp2_l3[list_index] = s_normalized

        # Compile results
        result_dict = {
            "B_shape": B_shape,
            "optimal_ell": optimal_ell_list,
            "dqi_expected_s_bp1_l3": dqi_expected_s_bp1_l3,
            "dqi_expected_s_bp2_l3": dqi_expected_s_bp2_l3,
        }

        df = pd.DataFrame(result_dict)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path)
        print(f"Results saved to: {output_file_path}")

        if numpy_backup:
            # Save backup as a .txt file
            backup_txt_path = output_file_path.replace(".csv", ".txt")
            with open(backup_txt_path, "w") as f:
                for key, values in result_dict.items():
                    f.write(f"{key} \n")
                    for item in values:
                        f.write(f"  {item}\n")
                    f.write("\n")
            print(f"Saved CSV and backup TXT to: {output_file_path}, {backup_txt_path}")


def run_random_performance_on_existing_Bs(
    output_file_path,
    path_B_load_matrices,
    start_it=0,
    final_it=None,
    samples=10**4,
):
    """
    Evaluate random sampling performance on existing max-XORSAT instances.

    This function loads `.npz` files containing matrices B and vectors v from the specified
    directory, each representing a max-XORSAT instance. It computes the average number of
    satisfied constraints using random bitstring sampling, and stores the performance metrics
    in a CSV file.

    Parameters
    ----------
    output_file_path : str
        Path where the resulting CSV file will be saved.

    path_B_load_matrices : str
        Directory containing `.npz` files named in the format `iter_<index>.npz`,
        each storing a dictionary with keys 'B', 'v', 'ell', and 'shape_B'.

    Output
    ------
    A CSV file at `output_file_path` with the following columns:
        - B_shape: Shape of each matrix B (tuple)
        - optimal_ell: Target ell value for decoding
        - random_s: Average percentage of satisfied constraints under random sampling
        - error_s: Standard deviation (as percentage) of satisfied constraints
    """

    def extract_iter_number(filename):
        match = re.search(r"iter_(\d+)\.npz", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path_B_load_matrices, "iter_*.npz")),
        key=extract_iter_number,
    )

    if final_it is not None:
        files = files[start_it:final_it]
    else:
        files = files[start_it:]

    n_instances = len(files)
    B_shape = [None] * n_instances
    optimal_ell_list = [0] * n_instances

    random_s = [0] * n_instances
    std_s = [0] * n_instances

    for list_index, fn in enumerate(files):
        data = np.load(fn)
        B = data["B"]
        v = data["v"]
        optimal_ell = 1  # data["ell"]

        shape_B = tuple(data["shape_B"])
        m, n = B.shape

        B_shape[list_index] = shape_B
        optimal_ell_list[list_index] = optimal_ell

        print(f"Processing {fn} | Shape: {shape_B}")

        _, rand_s, _, error_s = average_of_f_and_s_random(
            B,
            v,
            samples,
            histogram=False,
        )

        random_s[list_index] = rand_s * 100 / m
        std_s[list_index] = error_s * 100 / m

        # Compile results
        result_dict = {
            "B_shape": B_shape,
            "optimal_ell": optimal_ell_list,
            "random_s": random_s,
            "error_s": std_s,
        }

        df = pd.DataFrame(result_dict)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        df.to_csv(output_file_path)
        print(f"Results saved to: {output_file_path}")


def run_resource_estimation_on_existing_Bs(
    output_file_path,
    types_of_gates_file_path,
    path_B_load_matrices,
):
    """
    Estimate quantum resources for precomputed max-XORSAT instances across multiple depths and decoding levels.

    This function loads (B, v) pairs from `.npz` files and computes the number of qubits and gates required
    to simulate each instance using a resource estimation function. It evaluates combinations of:
    - decoding levels ell = 1, 2, 3
    - iteration counts from 1 to 5

    For each combination, the required number of qubits and gates is recorded and stored in a CSV file.

    Parameters
    ----------
    output_file_path : str
        Path to the output CSV file that will contain the resource estimates.

    path_B_load_matrices : str
        Directory containing `.npz` files named `iter_<index>.npz`, each storing:
        - 'B': binary matrix representing constraints
        - 'v': right-hand-side binary vector

    Output
    ------
    A CSV file at `output_file_path` with the following columns:
        - B_shape: shape of each matrix B
        - n_qubits_l{1,2,3}_iter{1-5}: number of qubits needed for decoding level {ell}
          and iteration {1-5}, e.g., `n_qubits_l2_iter4` corresponds to ell = 2, iteration = 4
        - n_gates_l{1,2,3}_iter{1-5}: number of gates needed for decoding level {ell}
          and iteration {1-5}, e.g., `n_gates_l3_iter2` corresponds to ell = 3, iteration = 2
    """

    def extract_iter_number(filename):
        match = re.search(r"iter_(\d+)\.npz", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path_B_load_matrices, "iter_*.npz")),
        key=extract_iter_number,
    )
    n_instances = len(files)
    B_shape = [None] * n_instances

    # n_qubits
    n_qubits_l1_iter1 = [0] * n_instances
    n_qubits_l1_iter2 = [0] * n_instances
    n_qubits_l1_iter3 = [0] * n_instances
    n_qubits_l1_iter4 = [0] * n_instances
    n_qubits_l1_iter5 = [0] * n_instances

    n_qubits_l2_iter1 = [0] * n_instances
    n_qubits_l2_iter2 = [0] * n_instances
    n_qubits_l2_iter3 = [0] * n_instances
    n_qubits_l2_iter4 = [0] * n_instances
    n_qubits_l2_iter5 = [0] * n_instances

    n_qubits_l3_iter1 = [0] * n_instances
    n_qubits_l3_iter2 = [0] * n_instances
    n_qubits_l3_iter3 = [0] * n_instances
    n_qubits_l3_iter4 = [0] * n_instances
    n_qubits_l3_iter5 = [0] * n_instances

    # n_gates
    n_gates_l1_iter1 = [0] * n_instances
    n_gates_l1_iter2 = [0] * n_instances
    n_gates_l1_iter3 = [0] * n_instances
    n_gates_l1_iter4 = [0] * n_instances
    n_gates_l1_iter5 = [0] * n_instances

    n_gates_l2_iter1 = [0] * n_instances
    n_gates_l2_iter2 = [0] * n_instances
    n_gates_l2_iter3 = [0] * n_instances
    n_gates_l2_iter4 = [0] * n_instances
    n_gates_l2_iter5 = [0] * n_instances

    n_gates_l3_iter1 = [0] * n_instances
    n_gates_l3_iter2 = [0] * n_instances
    n_gates_l3_iter3 = [0] * n_instances
    n_gates_l3_iter4 = [0] * n_instances
    n_gates_l3_iter5 = [0] * n_instances

    all_gate_dicts = []

    for list_index, fn in enumerate(files):
        data = np.load(fn)
        B = data["B"]
        v = data["v"]
        m, n = B.shape

        # print(m, n)

        B_shape[list_index] = (m, n)

        instance_gate_dict = {
            "filename": os.path.basename(fn),
            "B_shape": (m, n),
            "gates": {},  # nested dict: gates[ell][iteration][gate_type] = count
        }

        ell_configs = {
            1: [
                (n_qubits_l1_iter1, n_gates_l1_iter1),
                (n_qubits_l1_iter2, n_gates_l1_iter2),
                (n_qubits_l1_iter3, n_gates_l1_iter3),
                (n_qubits_l1_iter4, n_gates_l1_iter4),
                (n_qubits_l1_iter5, n_gates_l1_iter5),
            ],
            2: [
                (n_qubits_l2_iter1, n_gates_l2_iter1),
                (n_qubits_l2_iter2, n_gates_l2_iter2),
                (n_qubits_l2_iter3, n_gates_l2_iter3),
                (n_qubits_l2_iter4, n_gates_l2_iter4),
                (n_qubits_l2_iter5, n_gates_l2_iter5),
            ],
            3: [
                (n_qubits_l3_iter1, n_gates_l3_iter1),
                (n_qubits_l3_iter2, n_gates_l3_iter2),
                (n_qubits_l3_iter3, n_gates_l3_iter3),
                (n_qubits_l3_iter4, n_gates_l3_iter4),
                (n_qubits_l3_iter5, n_gates_l3_iter5),
            ],
        }

        for ell, list_pairs in ell_configs.items():
            for max_iterations, (list_n_qubits, list_n_gates) in zip(
                range(1, 6),
                list_pairs,
            ):
                print("ell, T: ", ell, max_iterations)
                n_qubits, gate_dictionary = resource_estimation_function(
                    B,
                    v,
                    ell,
                    max_iterations,
                    print_info=True,
                    print_circuit_progress=False,
                )
                n_gates = gate_dictionary.get("n_gates", sum(gate_dictionary.values()))
                list_n_qubits[list_index] = n_qubits
                list_n_gates[list_index] = n_gates

                # Save to nested dictionary
                ell_str = f"ell_{ell}"
                iter_str = f"iter_{max_iterations}"
                if ell_str not in instance_gate_dict["gates"]:
                    instance_gate_dict["gates"][ell_str] = {}
                instance_gate_dict["gates"][ell_str][iter_str] = gate_dictionary

        all_gate_dicts.append(instance_gate_dict)

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        plot_data = pd.DataFrame(
            {
                "B_shape": B_shape,
                # ell = 1
                "n_qubits_l1_iter1": n_qubits_l1_iter1,
                "n_qubits_l1_iter2": n_qubits_l1_iter2,
                "n_qubits_l1_iter3": n_qubits_l1_iter3,
                "n_qubits_l1_iter4": n_qubits_l1_iter4,
                "n_qubits_l1_iter5": n_qubits_l1_iter5,
                "n_gates_l1_iter1": n_gates_l1_iter1,
                "n_gates_l1_iter2": n_gates_l1_iter2,
                "n_gates_l1_iter3": n_gates_l1_iter3,
                "n_gates_l1_iter4": n_gates_l1_iter4,
                "n_gates_l1_iter5": n_gates_l1_iter5,
                # ell = 2
                "n_qubits_l2_iter1": n_qubits_l2_iter1,
                "n_qubits_l2_iter2": n_qubits_l2_iter2,
                "n_qubits_l2_iter3": n_qubits_l2_iter3,
                "n_qubits_l2_iter4": n_qubits_l2_iter4,
                "n_qubits_l2_iter5": n_qubits_l2_iter5,
                "n_gates_l2_iter1": n_gates_l2_iter1,
                "n_gates_l2_iter2": n_gates_l2_iter2,
                "n_gates_l2_iter3": n_gates_l2_iter3,
                "n_gates_l2_iter4": n_gates_l2_iter4,
                "n_gates_l2_iter5": n_gates_l2_iter5,
                # ell = 3
                "n_qubits_l3_iter1": n_qubits_l3_iter1,
                "n_qubits_l3_iter2": n_qubits_l3_iter2,
                "n_qubits_l3_iter3": n_qubits_l3_iter3,
                "n_qubits_l3_iter4": n_qubits_l3_iter4,
                "n_qubits_l3_iter5": n_qubits_l3_iter5,
                "n_gates_l3_iter1": n_gates_l3_iter1,
                "n_gates_l3_iter2": n_gates_l3_iter2,
                "n_gates_l3_iter3": n_gates_l3_iter3,
                "n_gates_l3_iter4": n_gates_l3_iter4,
                "n_gates_l3_iter5": n_gates_l3_iter5,
            },
        )

        plot_data.to_csv(output_file_path)
        print(f"CSV data dumped at {output_file_path}")

        # Save partial gate dictionaries after each instance
        with open(types_of_gates_file_path, "w") as f_json:
            json.dump(all_gate_dicts, f_json, indent=2)


if __name__ == "__main__":

    n_matrices = 25
    n_matrices_performance = 7

    # ------------------------------------------------ PROBLEM GENERATION ---------------------------------------
    # Read readme.txt to generate a different industrial ILP from mock data

    # Transform ILP stores in "pipelines/data/milp_formulation.json" to B max_xorsat
    generate_B_from_actual_ILP(
        "pipelines/max_xorsat_industrial_ILP",
        "pipelines/data/milp_formulation.json",
    )
    print("max-XORSAT from industrial ILP generated")

    data_maxxorsat = np.load(
        "pipelines/max_xorsat_industrial_ILP/real_ILP_maxXORSAT.npz",
    )
    B_from_actual_ilp = data_maxxorsat["B"]
    v_from_actual_ilp = data_maxxorsat["v"]

    generate_downsized_matrices(
        B_from_actual_ilp,
        v_from_actual_ilp,
        n_matrices,
        downsized_matrix_sizes,
    )
    print("downsized matrices")

    # -------------------------- GUROBI BENCHMARK ------------------------------------------------------------

    gurobi_benchmark(
        n_matrices_performance,
        n_matrices,
    )  # performance results with Gurobi

    # ------------------------------- DQI PERFORMANCE -----------------------------------

    # ell=1 + ell=2
    T = 5
    for it in range(5):
        print("Dqi performance l=1, l=2, iteration ", it)
        run_performance_on_existing_Bs_l1_l2(
            "pipelines/performance_results/performance_l1_l2_T"
            + str(T)
            + "_it"
            + str(it)
            + ".csv",
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(it),
            max_iterations=T,
            start_it=0,
            final_it=n_matrices_performance,
        )

    # ell=3
    for it in range(5):
        print("Dqi performance l=3, iteration ", it)
        run_performance_on_existing_Bs_l3(
            "pipelines/performance_results/performance_l3_T"
            + str(T)
            + "_it"
            + str(it)
            + ".csv",
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(it),
            max_iterations=T,
            start_it=0,
            final_it=2,
        )

    # --------------------------------RANDOM PERFORMANCE BENCHMARK -------------------------------

    for it in range(5):
        print("RANDOM ESTIMATION iteration ", it)
        run_random_performance_on_existing_Bs(
            "pipelines/performance_results/performance_random_it" + str(it) + ".csv",
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(it),
            final_it=n_matrices_performance,
            samples=10**4,
        )

    # ------------------------------- DQI RESOURCE ESTIMATION -----------------------------------
    for it in range(5):
        print("Resource estimation iteration ", it)
        run_resource_estimation_on_existing_Bs(
            "pipelines/resource_estimation_results/resources" + str(it) + ".csv",
            "pipelines/resource_estimation_results/type_of_gates" + str(it) + ".json",
            "pipelines/max_xorsat_industrial_ILP/downsized_real_ilp_random_matrices_mc_"
            + str(it),
        )

    # --------------------------- PLOTTING  -----------------------------

    plot_performance()
    plot_resources()
    plot_type_of_gates()
