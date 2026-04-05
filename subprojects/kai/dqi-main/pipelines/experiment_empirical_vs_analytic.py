import os

import matplotlib.pyplot as plt
import numpy as np

from pipelines.belief_propagations import belief_propagation_gallager
from pipelines.DQI_classical import expected_constrains_DQI
from pipelines.DQI_full_circuit import quantum_dqi_histogram_results
from pipelines.testing_BP1 import generate_random_binary_test_cases

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" Uncomment if GPUs are available


def process_dataset(path, label):
    data = np.load(path)
    empirical_s = data["empirical_s"]
    error_s = data["error_s"]
    expected_s = data["expected_s"]

    _, unique_indices = np.unique(expected_s, return_index=True)
    expected_s_unique = expected_s[unique_indices]
    empirical_s_unique = empirical_s[unique_indices]
    error_s_unique = error_s[unique_indices]

    diff = empirical_s_unique - expected_s_unique
    ax.errorbar(
        expected_s_unique,
        diff,
        yerr=error_s_unique,
        fmt="x",
        capsize=4,
        label=label,
    )

    all_expected.append(expected_s_unique)
    all_abs_diff.append(diff)


if __name__ == "__main__":

    # -------------------- 8by6 l=2 T=1-------------------------------
    test_cases = generate_random_binary_test_cases(num_cases=50, m=8, n=6)
    ell = 2
    n_it = 1
    empirical_s = []
    error_s = []
    expected_s = []
    i = 0
    for element in test_cases:
        i = i + 1
        print(f"Case {i}/{len(test_cases)}")
        B, v = element

        f_expected, s_expected = expected_constrains_DQI(
            B,
            v,
            ell,
            n_it,
            belief_propagation_gallager,
            jit_version=True,
        )

        (
            s_hist,
            s_av,
            error_s_dqi,
            post_shots,
            total_shots,
        ) = quantum_dqi_histogram_results(
            B,
            v,
            ell,
            n_it,
            shots=10**4,
        )

        empirical_s.append(s_av)
        error_s.append(error_s_dqi)
        expected_s.append(s_expected)

    # Ensure the directory exists
    save_dir = "empirical_vs_analytic_s_results"
    os.makedirs(save_dir, exist_ok=True)

    # Save the file
    np.savez(
        os.path.join(save_dir, "dqi_results_8by6_ell2_nit1.npz"),
        empirical_s=empirical_s,
        error_s=error_s,
        expected_s=expected_s,
    )

    # -----------------------------------------6by4 ell=1 n_it=1-----------------------------------------
    test_cases = generate_random_binary_test_cases(num_cases=50, m=6, n=4)
    ell = 1
    n_it = 1

    empirical_s = []
    error_s = []
    expected_s = []
    i = 0
    for element in test_cases:
        i = i + 1
        print(f"Case {i}/{len(test_cases)}")
        B, v = element

        f_expected, s_expected = expected_constrains_DQI(
            B,
            v,
            ell,
            n_it,
            belief_propagation_gallager,
            jit_version=True,
        )

        (
            s_hist,
            s_av,
            error_s_dqi,
            post_shots,
            total_shots,
        ) = quantum_dqi_histogram_results(
            B,
            v,
            ell,
            n_it,
            shots=10**4,
        )

        empirical_s.append(s_av)
        error_s.append(error_s_dqi)
        expected_s.append(s_expected)

    # Ensure the directory exists
    save_dir = "empirical_vs_analytic_s_results"
    os.makedirs(save_dir, exist_ok=True)

    # Save the file
    np.savez(
        os.path.join(save_dir, "dqi_results_6by4_ell1_nit1.npz"),
        empirical_s=empirical_s,
        error_s=error_s,
        expected_s=expected_s,
    )

    # ----------------------------------------5by3 ell=1 n_it=3----------------------------------------------------
    test_cases = generate_random_binary_test_cases(num_cases=50, m=5, n=3)
    ell = 1
    n_it = 3
    empirical_s = []
    error_s = []
    expected_s = []
    i = 0
    for element in test_cases:
        i = i + 1
        print(f"Case {i}/{len(test_cases)}")
        B, v = element

        f_expected, s_expected = expected_constrains_DQI(
            B,
            v,
            ell,
            n_it,
            belief_propagation_gallager,
            jit_version=True,
        )

        (
            s_hist,
            s_av,
            error_s_dqi,
            post_shots,
            total_shots,
        ) = quantum_dqi_histogram_results(
            B,
            v,
            ell,
            n_it,
            shots=10**4,
        )

        empirical_s.append(s_av)
        error_s.append(error_s_dqi)
        expected_s.append(s_expected)

    # Ensure the directory exists
    save_dir = "empirical_vs_analytic_s_results"
    os.makedirs(save_dir, exist_ok=True)

    # Save the file
    np.savez(
        os.path.join(save_dir, "dqi_results_5by3_ell1_nit3.npz"),
        empirical_s=empirical_s,
        error_s=error_s,
        expected_s=expected_s,
    )

    # ----------------------- plotting of results -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize min/max trackers
    all_expected = []
    all_empirical = []

    # --- First Dataset ---
    data = np.load(
        "empirical_vs_analytic_s_results/dqi_results_5by3_ell1_nit3.npz",
    )
    empirical_s = data["empirical_s"]
    error_s = data["error_s"]
    expected_s = data["expected_s"]

    _, unique_indices = np.unique(expected_s, return_index=True)
    expected_s_unique = expected_s[unique_indices]
    empirical_s_unique = empirical_s[unique_indices]
    error_s_unique = error_s[unique_indices]

    ax.errorbar(
        expected_s_unique,
        empirical_s_unique,
        yerr=error_s_unique,
        fmt="x",
        capsize=4,
        label=r"$m=5$ $n=3$ $\ell=1$ $T=3$",
    )

    all_expected.append(expected_s_unique)
    all_empirical.append(empirical_s_unique)

    # --- Second Dataset ---
    data = np.load(
        "empirical_vs_analytic_s_results/dqi_results_6by4_ell1_nit1.npz",
    )
    empirical_s = data["empirical_s"]
    error_s = data["error_s"]
    expected_s = data["expected_s"]

    _, unique_indices = np.unique(expected_s, return_index=True)
    expected_s_unique = expected_s[unique_indices]
    empirical_s_unique = empirical_s[unique_indices]
    error_s_unique = error_s[unique_indices]

    ax.errorbar(
        expected_s_unique,
        empirical_s_unique,
        yerr=error_s_unique,
        fmt="x",
        capsize=4,
        label=r"$m=6$ $n=4$ $\ell=1$ $T=1$",
    )

    all_expected.append(expected_s_unique)
    all_empirical.append(empirical_s_unique)

    # --- Third Dataset ---
    data = np.load(
        "empirical_vs_analytic_s_results/dqi_results_8by6_ell2_nit1.npz",
    )
    empirical_s = data["empirical_s"]
    error_s = data["error_s"]
    expected_s = data["expected_s"]

    _, unique_indices = np.unique(expected_s, return_index=True)
    expected_s_unique = expected_s[unique_indices]
    empirical_s_unique = empirical_s[unique_indices]
    error_s_unique = error_s[unique_indices]

    ax.errorbar(
        expected_s_unique,
        empirical_s_unique,
        yerr=error_s_unique,
        fmt="x",
        capsize=4,
        label=r"$m=8$ $n=6$ $\ell=2$ $T=1$",
    )

    all_expected.append(expected_s_unique)
    all_empirical.append(empirical_s_unique)

    # --- Diagonal y = x line covering all data ---
    all_expected_flat = np.concatenate(all_expected)
    all_empirical_flat = np.concatenate(all_empirical)

    min_val = min(np.min(all_expected_flat), np.min(all_empirical_flat))
    max_val = max(np.max(all_expected_flat), np.max(all_empirical_flat))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        label=r"$\langle S \rangle_{\mathrm{emp}} = \langle S \rangle$",
    )

    # --- Labels and formatting ---
    ax.set_xlabel(r"$\langle S \rangle$", fontsize=14)
    ax.set_ylabel(r"$\langle S \rangle_{\mathrm{emp}}$", fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.set_aspect("equal", "box")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("emp_vs_analytic.pdf")
    plt.close()

    # ----------------------------------DIFFERENCE PLOT---------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Initialize min/max trackers
    all_expected = []
    all_abs_diff = []

    # --- Datasets ---
    process_dataset(
        "empirical_vs_analytic_s_results/dqi_results_5by3_ell1_nit3.npz",
        r"$m=5$ $n=3$ $\ell=1$ $T=3$",
    )

    process_dataset(
        "empirical_vs_analytic_s_results/dqi_results_6by4_ell1_nit1.npz",
        r"$m=6$ $n=4$ $\ell=1$ $T=1$",
    )

    process_dataset(
        "empirical_vs_analytic_s_results/dqi_results_8by6_ell2_nit1.npz",
        r"$m=8$ $n=6$ $\ell=2$ $T=1$",
    )

    # --- Formatting ---
    ax.set_xlabel(r"$\langle S \rangle$", fontsize=14)
    ax.set_ylabel(
        r"$\langle S_{\mathrm{emp}} \rangle - \langle S \rangle$",
        fontsize=14,
    )
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("diff_emp_vs_analytic.pdf")
    plt.close()
