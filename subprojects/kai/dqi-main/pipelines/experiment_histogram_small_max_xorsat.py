import matplotlib.pyplot as plt
import numpy as np

from pipelines.DQI_full_circuit import (
    average_of_f_and_s_random,
    quantum_dqi_histogram_results,
)


def plot_histogram_comparison(
    histograms,
    labels,
    averages,
    filename,
    var_names=None,
    errors=None,
    normalize=True,
    xlabel=r"$S$",
    ylabel="Frequency",
    title="Histogram Comparison",
    bars=False,
    s_max=None,
):
    """
    Plots and saves a comparative histogram with optional average markers and error bands.

    Args:
        histograms (list of dict): List of histograms (dicts mapping values to counts).
        labels (list of str): List of labels corresponding to each histogram.
        averages (list of float): List of average values (one per histogram).
        filename (str): File name to save the plot (e.g., "output.pdf").
        var_names (list of str, optional): Names of the variables to display in the legend (one per histogram);
            defaults to ['S'] * len(histograms).
        errors (list of float, optional): Statistical errors (same length as histograms).
        normalize (bool): If True, normalize counts to probabilities.
        xlabel (str): Label for the x-axis; defaults to "$S$."
        ylabel (str): Label for the y-axis; defaults to "Frequency".
        title (str): Title of the plot.
        bars (bool): If True, plots bars instead of lines with markers.
        s_max (int, optional): Maximum x value shown on x-axis.
    """

    # Validate inputs
    n = len(histograms)
    if len(labels) != n:
        raise ValueError("The number of histograms must match the number of labels.")
    if averages is not None and len(averages) != n:
        raise ValueError("The number of averages must match the number of histograms.")
    if errors is not None and len(errors) != n:
        raise ValueError("The number of errors must match the number of histograms.")

    # Prepare var_names for legend only
    if var_names is None:
        var_names = ["S"] * n
    if len(var_names) != n:
        raise ValueError("The number of var_names must match the number of histograms.")

    # Determine x-axis values
    if s_max is not None:
        all_keys = list(range(0, s_max + 1))
    else:
        all_keys = sorted(set().union(*[h.keys() for h in histograms]))

    fig, ax = plt.subplots()
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Bar settings
    width = 0.7 / n if bars else None
    group_width = width * n if bars else None

    for idx, (hist, label, var_name) in enumerate(zip(histograms, labels, var_names)):
        y_values = [hist.get(k, 0) for k in all_keys]
        if normalize:
            total = sum(y_values)
            if total > 0:
                y_values = [y / total for y in y_values]

        color = color_cycle[idx % len(color_cycle)]
        # Plot bars or lines
        if bars:
            offsets = [k - group_width / 2 + (idx + 0.5) * width for k in all_keys]
            ax.bar(offsets, y_values, width=width, label=label, color=color, alpha=0.8)
        else:
            ax.plot(all_keys, y_values, marker="o", label=label, color=color)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r"$p$" if normalize else ylabel, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)

    if bars:
        ax.set_xticks(all_keys)
        ax.set_xticklabels(all_keys, fontsize=14)

    ax.legend(fontsize=14)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":

    B = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
        ],
        dtype=int,
    )
    v = np.array([0, 0, 1, 1, 0, 0, 1, 1])

    ell = 1
    num_iterations_bp = 1
    shots_dqi = 10**4

    s_hist, s_av, error_s_dqi, post_shots, total_shots = quantum_dqi_histogram_results(
        B,
        v,
        ell,
        num_iterations_bp,
        shots=shots_dqi,
    )

    ell = 2
    (
        s_hist2,
        s_av2,
        error_s_dqi2,
        post_shots2,
        total_shots2,
    ) = quantum_dqi_histogram_results(
        B,
        v,
        ell,
        num_iterations_bp,
        shots=shots_dqi,
    )

    ell = 3
    (
        s_hist3,
        s_av3,
        error_s_dqi3,
        post_shots3,
        total_shots3,
    ) = quantum_dqi_histogram_results(
        B,
        v,
        ell,
        num_iterations_bp,
        shots=shots_dqi,
    )

    (
        random_f,
        random_s,
        random_s_hist,
        error_f,
        error_s_random,
    ) = average_of_f_and_s_random(B, v, 10**4, histogram=True)

    filename = "histogram_8_constrains_6_variables.pdf"

    hist_lists = [random_s_hist, s_hist, s_hist2, s_hist3]
    labels_lists = ["Random", r"$\ell=1$", r"$\ell=2$", r"$\ell=3$"]
    avg_lists = [random_s, s_av, s_av2, s_av3]
    error_lists = [error_s_random, error_s_dqi, error_s_dqi2, error_s_dqi3]

    plot_histogram_comparison(
        hist_lists,
        labels_lists,
        avg_lists,
        filename,
        errors=error_lists,
        bars=True,
        s_max=8,
    )
