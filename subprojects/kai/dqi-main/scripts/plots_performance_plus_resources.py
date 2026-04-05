import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def plot_performance(T=5):
    # --- 1) Load and aggregate the 5 runs for L1/L2, L3, Random, and Gurobi ---
    all_bp1 = []
    all_bp2 = []
    all_bp1_l2 = []
    all_bp2_l2 = []
    all_bp1_l3 = []
    all_bp2_l3 = []
    all_rand = []
    all_rand_err = []
    all_gurobi = []

    problem_size = None
    problem_sizel3 = None

    for it in range(5):
        # L1/L2
        df = pd.read_csv(
            "pipelines/performance_results/performance_l1_l2_T"
            + str(T)
            + "_it"
            + str(it)
            + ".csv",
        ).head(18)
        df["problem_size"] = df["B_shape"].apply(lambda x: eval(x)[0] * eval(x)[1])
        if problem_size is None:
            problem_size = df["problem_size"].values
        all_bp1.append(df["dqi_expected_s_bp1"].values)
        all_bp2.append(df["dqi_expected_s_bp2"].values)
        all_bp1_l2.append(df["dqi_expected_s_bp1_l2"].values)
        all_bp2_l2.append(df["dqi_expected_s_bp2_l2"].values)

        # L3
        df_l3 = pd.read_csv(
            "pipelines/performance_results/performance_l3_T"
            + str(T)
            + "_it"
            + str(it)
            + ".csv",
        ).head(18)
        df_l3["problem_size"] = df_l3["B_shape"].apply(
            lambda x: eval(x)[0] * eval(x)[1],
        )
        if problem_sizel3 is None:
            problem_sizel3 = df_l3["problem_size"].values
        all_bp1_l3.append(df_l3["dqi_expected_s_bp1_l3"].values)
        all_bp2_l3.append(df_l3["dqi_expected_s_bp2_l3"].values)

        # Random baseline
        df_rand = pd.read_csv(
            "pipelines/performance_results/performance_random_it" + str(it) + ".csv",
        ).head(18)
        df_rand["problem_size"] = df_rand["B_shape"].apply(
            lambda x: eval(x)[0] * eval(x)[1],
        )
        all_rand.append(df_rand["random_s"].values / 100)
        all_rand_err.append(df_rand["error_s"].values / 100)

        # Gurobi
        df_g = pd.read_csv(
            "pipelines/performance_results/gurobi_satisfaction_summary_it"
            + str(it)
            + ".csv",
        )
        df_g_sorted = df_g.sort_values("iteration_index").reset_index(drop=True)
        # normalize to [0,1] and take first len(problem_size) entries
        all_gurobi.append((df_g_sorted["satisfaction_ratio"].values / 100))

    # stack into arrays
    all_bp1 = np.vstack(all_bp1)
    all_bp2 = np.vstack(all_bp2)
    all_bp1_l2 = np.vstack(all_bp1_l2)
    all_bp2_l2 = np.vstack(all_bp2_l2)
    all_bp1_l3 = np.vstack(all_bp1_l3)
    all_bp2_l3 = np.vstack(all_bp2_l3)
    all_rand = np.vstack(all_rand)
    all_rand_err = np.vstack(all_rand_err)
    all_gurobi = np.vstack(all_gurobi)

    # compute means & stds
    mean_bp1 = all_bp1.mean(axis=0)
    std_bp1 = all_bp1.std(axis=0)
    mean_bp2 = all_bp2.mean(axis=0)
    std_bp2 = all_bp2.std(axis=0)
    mean_bp1_l2 = all_bp1_l2.mean(axis=0)
    std_bp1_l2 = all_bp1_l2.std(axis=0)
    mean_bp2_l2 = all_bp2_l2.mean(axis=0)
    std_bp2_l2 = all_bp2_l2.std(axis=0)
    mean_bp1_l3 = all_bp1_l3.mean(axis=0)
    std_bp1_l3 = all_bp1_l3.std(axis=0)
    mean_bp2_l3 = all_bp2_l3.mean(axis=0)
    std_bp2_l3 = all_bp2_l3.std(axis=0)
    mean_rand = all_rand.mean(axis=0)
    std_rand = all_rand_err.std(axis=0)
    mean_gurobi = all_gurobi.mean(axis=0)
    std_gurobi = all_gurobi.std(axis=0)

    # --- 2) Plot with a broken y-axis ---
    plt.rcParams.update({"font.size": 14})
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(9, 6.5),
        gridspec_kw={"height_ratios": [1, 4.5], "hspace": 0.1},
    )

    def plot_everything(ax):
        # BP curves
        ax.errorbar(
            problem_size,
            mean_bp1,
            yerr=std_bp1,
            fmt="-x",
            label="BP1 ℓ=1",
            capsize=4,
        )
        ax.errorbar(
            problem_size,
            mean_bp2,
            yerr=std_bp2,
            fmt="--x",
            label="BP2 ℓ=1",
            capsize=4,
        )
        ax.errorbar(
            problem_size,
            mean_bp1_l2,
            yerr=std_bp1_l2,
            fmt="-o",
            label="BP1 ℓ=2",
            capsize=4,
        )
        ax.errorbar(
            problem_size,
            mean_bp2_l2,
            yerr=std_bp2_l2,
            fmt="--o",
            label="BP2 ℓ=2",
            capsize=4,
        )
        ax.errorbar(
            problem_sizel3,
            mean_bp1_l3,
            yerr=std_bp1_l3,
            fmt="-s",
            label="BP1 ℓ=3",
            capsize=4,
        )
        ax.errorbar(
            problem_sizel3,
            mean_bp2_l3,
            yerr=std_bp2_l3,
            fmt="--s",
            label="BP2 ℓ=3",
            capsize=4,
        )

        # Gurobi baseline with error bars
        ax.errorbar(
            problem_size,
            mean_gurobi[: len(problem_size)],
            yerr=std_gurobi[: len(problem_size)],
            fmt=":",
            label="Gurobi",
            color="grey",
            capsize=4,
        )

        # Random baseline (unchanged)
        ax.errorbar(
            problem_size,
            mean_rand[: len(problem_size)],
            yerr=std_rand[: len(problem_size)],
            fmt=":",
            label="Random",
            color="black",
            capsize=4,
        )

    # draw on both panels
    plot_everything(ax_top)
    plot_everything(ax_bot)

    # set y-limits
    ax_top.set_ylim(0.8, 0.90)
    ax_bot.set_ylim(0.495, 0.67)

    # hide middle spines
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)

    # adjust ticks
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # draw diagonal “break” marks
    trans = fig.transFigure
    b_top, b_bot = ax_top.get_position(), ax_bot.get_position()
    dx = 0.005
    for x in (b_top.x0, b_top.x1):
        fig.add_artist(
            Line2D(
                [x - dx, x + dx],
                [b_top.y0] * 2,
                transform=trans,
                color="k",
                linewidth=1,
            ),
        )
    y_cut = b_bot.y0 + b_bot.height
    for x in (b_bot.x0, b_bot.x1):
        fig.add_artist(
            Line2D(
                [x - dx, x + dx],
                [y_cut] * 2,
                transform=trans,
                color="k",
                linewidth=1,
            ),
        )

    # legend & labels
    ax_bot.legend(loc="upper right", fontsize=12)
    ax_bot.set_xlabel(r"$n \cdot m$")
    ax_bot.set_ylabel(r"$\langle S\rangle /m$")
    ax_bot.tick_params(labelbottom=True)

    plt.tight_layout()
    plt.savefig(
        "performance_T" + str(T) + ".pdf",
    )
    plt.close()


def plot_resources():

    # --- 1) Load & aggregate the 5 resource files ---
    # We'll assume they're named resources0.csv ... resources4.csv
    n_runs = 5

    # Prepare containers
    ps = None  # problem_size (same for all runs)
    # qubits: T=1..5 for ell=1
    qubit_data = {T: [] for T in range(1, 6)}
    # gates: ell=1 at T=1,3,5 and ell=3 at T=1,3,5
    gate_data = {("l1", T): [] for T in (1, 3, 5)}
    gate_data.update({("l3", T): [] for T in (1, 3, 5)})

    for it in range(n_runs):
        df = pd.read_csv(
            "pipelines/resource_estimation_results/resources" + str(it) + ".csv",
        ).head(30)
        # compute problem size
        df["problem_size"] = df["B_shape"].apply(lambda x: eval(x)[0] * eval(x)[1])
        if ps is None:
            ps = df["problem_size"].values

        # collect qubits for ℓ=1, T=1..5
        for T in range(1, 6):
            qubit_data[T].append(df[f"n_qubits_l1_iter{T}"].values)

        # collect gates
        for T in (1, 3, 5):
            gate_data[("l1", T)].append(df[f"n_gates_l1_iter{T}"].values)
            gate_data[("l3", T)].append(df[f"n_gates_l3_iter{T}"].values)

    # Convert to NumPy stacks & compute means/stds
    # qubits:
    mean_qubits = {}
    std_qubits = {}
    for T, runs in qubit_data.items():
        arr = np.vstack(runs)  # shape = [5 × 18]
        mean_qubits[T] = arr.mean(axis=0)
        std_qubits[T] = arr.std(axis=0)

    # gates:
    mean_gates = {}
    std_gates = {}
    for key, runs in gate_data.items():
        arr = np.vstack(runs)
        mean_gates[key] = arr.mean(axis=0)
        std_gates[key] = arr.std(axis=0)

    # --- 2) Plot qubits w/ error bars ---
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(9, 6))

    markers = {1: "^", 2: "v", 3: "s", 4: "D", 5: "x"}
    for T in range(1, 6):
        plt.errorbar(
            ps,
            mean_qubits[T],
            yerr=std_qubits[T],
            fmt=markers[T] + "-",
            capsize=3,
            label=rf"$T={T}$",
        )

    plt.xlabel(r"$n\cdot m$")
    plt.ylabel(r"$N_q$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qubits_vs_size.pdf")
    plt.close()

    # --- 3) Plot gates w/ error bars ---
    plt.figure(figsize=(9, 6))

    # ℓ=1 curves
    for T in (1, 3, 5):
        plt.errorbar(
            ps,
            mean_gates[("l1", T)],
            yerr=std_gates[("l1", T)],
            fmt="o-",
            capsize=3,
            mfc="none",
            label=rf"$\ell=1, \ T={T}$",
        )
    # ℓ=3 curves
    for T in (1, 3, 5):
        plt.errorbar(
            ps,
            mean_gates[("l3", T)],
            yerr=std_gates[("l3", T)],
            fmt="x-",
            capsize=3,
            label=rf"$\ell=3, \ T={T}$",
        )

    plt.xlabel(r"$n\cdot m$")
    plt.ylabel("Number of Gates")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gates_vs_size.pdf")
    plt.close()


def plot_type_of_gates(ell=3, T=5):
    n_runs = 5  # number of itX runs

    # ==== AGGREGATE ACROSS RUNS ====
    all_total = []  # list of arrays (length = n_instances) per run
    all_gate_types = {}  # gate_type -> list of arrays (one per run)

    ps = None

    for it in range(n_runs):
        # load CSV
        csv_path = "pipelines/resource_estimation_results/resources" + str(it) + ".csv"
        df = pd.read_csv(csv_path)
        df["problem_size"] = df["B_shape"].apply(lambda x: eval(x)[0] * eval(x)[1])
        if ps is None:
            ps = df["problem_size"].values

        # load JSON
        json_path = (
            "pipelines/resource_estimation_results/type_of_gates" + str(it) + ".json"
        )
        with open(json_path) as f:
            gate_data = json.load(f)

        # accumulate per-instance totals & per-type counts
        total = np.zeros(len(df), dtype=int)
        per_type = {}

        for idx, inst in enumerate(gate_data):
            counts = inst["gates"].get(f"ell_{ell}", {}).get(f"iter_{T}", {})
            for gt, c in counts.items():
                if gt == "barrier":
                    continue
                per_type.setdefault(gt, np.zeros(len(df), dtype=int))
                per_type[gt][idx] += c
                total[idx] += c

        all_total.append(total)
        for gt, arr in per_type.items():
            all_gate_types.setdefault(gt, []).append(arr)

    # stack into arrays and compute mean/std
    all_total = np.vstack(all_total)  # shape = [n_runs × n_instances]
    mean_total = all_total.mean(axis=0)
    std_total = all_total.std(axis=0)

    mean_types = {}
    std_types = {}
    for gt, runs in all_gate_types.items():
        arr = np.vstack(runs)
        mean_types[gt] = arr.mean(axis=0)
        std_types[gt] = arr.std(axis=0)

    # ==== PLOT ====
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 14})

    # total gates
    plt.errorbar(
        ps,
        mean_total,
        yerr=std_total,
        fmt="o-",
        capsize=3,
        label="Total gates",
        markerfacecolor="none",
        color="black",
        linewidth=2,
        markersize=4,
    )

    # individual types
    markers = ["s", "D", "^", "v", "<", ">", "P", "*", "X"]
    colors = plt.cm.tab10.colors
    for idx, (gt, mean_arr) in enumerate(mean_types.items()):
        plt.errorbar(
            ps,
            mean_arr,
            yerr=std_types[gt],
            fmt=markers[idx % len(markers)] + "-",
            capsize=3,
            label=gt,
            color=colors[idx % len(colors)],
            markersize=4,
        )

    plt.xlabel(r"$n \cdot m$")
    plt.ylabel("Number of Gates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "type_of_gates_ell" + str(ell) + "_T" + str(T) + ".pdf",
    )
    plt.close()
