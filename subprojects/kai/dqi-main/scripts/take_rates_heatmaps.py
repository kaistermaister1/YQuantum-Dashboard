import matplotlib.pyplot as plt
import numpy as np

# Constants
epsilon = -2
# d = 0.1
T_A = 0.5
T_B = 0.5
# T_empty = 0.3


def capped_elasticity_model(d):

    # Define f_epsilon as per the piecewise function
    def f_epsilon(x, epsilon, T):
        x_eps = np.where(
            x > 0,
            x**epsilon,
            1e12,
        )  # Avoid division by zero or negative base
        result = np.ones_like(x)

        # condition1 = x_eps <= 1
        # condition2 = (x_eps > 1) & (x_eps <= 1 / T)
        # condition3 = x_eps > 1 / T
        condition = x_eps >= 1
        notcondition = x_eps < 1

        # result[condition1] = 1
        # result[condition2] = x_eps[condition2]
        # result[condition3] = 1 / T
        result[condition] = 1
        result[notcondition] = x_eps[notcondition]

        return result

    # Generate C_A and P_B grid
    P_A_vals = np.linspace(1, 1000, 100)
    P_B_vals = np.linspace(1, 1000, 100)
    P_A, P_B = np.meshgrid(P_A_vals, P_B_vals)

    # ΔT_{A, not B → AB}
    price_jump_for_buying_B = ((1 - d) * (P_A + P_B)) / P_A
    f_eps_buying_B = f_epsilon(price_jump_for_buying_B, epsilon, T_A)
    delta_T_A_to_AB = f_eps_buying_B

    # ΔT_{B, not A → A+B}
    price_jump_for_buying_A = ((1 - d) * (P_A + P_B)) / P_B
    f_eps_buying_A = f_epsilon(price_jump_for_buying_A, epsilon, T_B)
    delta_T_B_to_AB = f_eps_buying_A

    # FRONTIER LINES for A, not B → A + B
    PA_line = np.linspace(1, 1000, 1000)
    frontier_1_A = PA_line * d / (1 - d)
    frontier_2_A = PA_line * (0.01 ** (1 / epsilon) + d - 1) / (1 - d)

    # FRONTIER LINES for B, not A → A + B
    PB_line = np.linspace(1, 1000, 1000)
    frontier_1_B = PB_line * d / (1 - d)
    frontier_2_B = PB_line * (0.01 ** (1 / epsilon) + d - 1) / (1 - d)

    return (
        delta_T_A_to_AB,
        delta_T_B_to_AB,
        frontier_1_A,
        frontier_1_B,
        frontier_2_A,
        frontier_2_B,
        PA_line,
        PB_line,
    )


def continuous_elasticty_model():

    P_A_vals = np.linspace(1, 1000, 100)
    P_B_vals = np.linspace(1, 1000, 100)
    pA, pB = np.meshgrid(P_A_vals, P_B_vals)

    def conversion_fraction_A(pA, pB, d, ela):
        # Bundle price:
        pAB = (1 - d) * (pA + pB)
        # Utility difference for A-only customers (with U_A normalized to 0):
        # U_AB - U_A = ela * ln(pAB / pA)
        ratio = (pAB / pA) ** ela  # equals exp(ela * ln(pAB/pA))
        # MNL conversion fraction: P(AB | A) = exp(U_AB)/(exp(U_A)+exp(U_AB)+exp(U_0))
        # With U_A = 0, U_0 = 0, so:
        conv_frac = ratio / (2 + ratio)
        return conv_frac

    def conversion_fraction_B(pA, pB, d, ela):
        # Bundle price:
        pAB = (1 - d) * (pA + pB)
        # For B-only customers: U_AB - U_B = ela * ln(pAB / pB)
        ratio = (pAB / pB) ** ela
        conv_frac = ratio / (2 + ratio)
        return conv_frac

    def conversion_fraction_empty(d, ela):
        # For non-buyers, assume baseline utility for nothing is normalized to 0.
        # Let U_AB_empty = -ela * ln(1-d) so that higher discount is better (more positive U_AB_empty)
        # Then conversion fraction: exp(U_AB_empty)/(exp(U_AB_empty)+exp(0))
        ratio = (1 - d) ** (ela)
        conv_frac = ratio / (1 + ratio)
        return conv_frac

    # Example parameters
    d = 0.2  # 10% discount
    ela = -2  # elasticity
    T_A = 0.4  # Original A-only take rate (15%)
    T_B = 0.3  # Original B-only take rate (35%)
    # T_empty = 0.15   # Original non-buyers take rate (15%)

    conv_A = conversion_fraction_A(pA, pB, d, ela)
    conv_B = conversion_fraction_B(pA, pB, d, ela)
    # conv_empty = conversion_fraction_empty(d, ela)

    # Absolute conversions from each segment:
    delta_T_A = T_A * conv_A
    delta_T_B = T_B * conv_B
    # delta_T_empty = T_empty * conv_empty

    return delta_T_A, delta_T_B


def plot_heatmaps(delta_T_A_to_AB_list, delta_T_B_to_AB_list):

    # Plotting heatmaps

    for i in range(9):

        heatmap1 = axs[i, 0].imshow(
            delta_T_A_to_AB_list[i] * 100,
            extent=[1, 1000, 1, 1000],
            origin="lower",
            aspect="auto",
            cmap="Blues",
            vmin=0,
        )

        axs[i, 0].set_title(
            r"$\frac{T_{A, \text{not } B -> A + B}}{T_A}$ Heatmap;  Elasticity: -2, Discount : "
            + f"{(i + 1) * (10)}%",
        )
        axs[i, 0].set_xlabel(r"$P_A$")
        axs[i, 0].set_ylabel(r"$P_B$")
        fig.colorbar(heatmap1, ax=axs[i, 0])

        heatmap2 = axs[i, 1].imshow(
            delta_T_B_to_AB_list[i] * 100,
            extent=[1, 1000, 1, 1000],
            origin="lower",
            aspect="auto",
            cmap="Blues",
            vmin=0,
        )
        axs[i, 1].set_title(
            r"$\frac{T_{B, \text{not } A -> A + B}}{T_B}$ Heatmap; Elasticity: -2, Discount : "
            + f"{(i + 1) * (10)}%",
        )
        axs[i, 1].set_xlabel(r"$P_A$")
        axs[i, 1].set_ylabel(r"$P_B$")
        fig.colorbar(heatmap2, ax=axs[i, 1])

        axs[i, 0].set_xlim(0, 1000)
        axs[i, 0].set_ylim(0, 1000)
        axs[i, 1].set_xlim(0, 1000)
        axs[i, 1].set_ylim(0, 1000)


def add_frontier_lines(
    frontier_1_A_list,
    frontier_1_B_list,
    frontier_2_A_list,
    frontier_2_B_list,
    PA_line_list,
    PB_line_list,
):
    for i in range(9):
        axs[i, 0].plot(
            PA_line_list[i],
            frontier_1_A_list[i],
            "r--",
            label=r"$x_\epsilon = 1$",
        )
        axs[i, 0].plot(
            PA_line_list[i],
            frontier_2_A_list[i],
            "g--",
            label=r"$x_\epsilon = 1/T$",
        )
        axs[i, 1].plot(
            frontier_1_B_list[i],
            PB_line_list[i],
            "r--",
            label=r"$x_\epsilon = 1$",
        )
        axs[i, 1].plot(
            frontier_2_B_list[i],
            PB_line_list[i],
            "g--",
            label=r"$x_\epsilon = 1/T$",
        )


if __name__ == "__main__":
    # fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    fig, axs = plt.subplots(9, 2, figsize=(16, 34))
    delta_T_A_to_AB_list = []
    delta_T_B_to_AB_list = []
    frontier_1_A_list = []
    frontier_1_B_list = []
    frontier_2_A_list = []
    frontier_2_B_list = []
    PA_line_list = []
    PB_line_list = []
    for i in range(1, 10):
        # epsilon = -2 * i
        d = 0.1 * i
        (
            delta_T_A_to_AB,
            delta_T_B_to_AB,
            frontier_1_A,
            frontier_1_B,
            frontier_2_A,
            frontier_2_B,
            PA_line,
            PB_line,
        ) = capped_elasticity_model(d)
        delta_T_A_to_AB_list.append(delta_T_A_to_AB)
        delta_T_B_to_AB_list.append(delta_T_B_to_AB)
        frontier_1_A_list.append(frontier_1_A)
        frontier_1_B_list.append(frontier_1_B)
        frontier_2_A_list.append(frontier_2_A)
        frontier_2_B_list.append(frontier_2_B)
        PA_line_list.append(PA_line)
        PB_line_list.append(PB_line)

    # delta_T_A_to_AB, delta_T_B_to_AB = continuous_elasticty_model()
    plot_heatmaps(delta_T_A_to_AB_list, delta_T_B_to_AB_list)

    add_frontier_lines(
        frontier_1_A_list,
        frontier_1_B_list,
        frontier_2_A_list,
        frontier_2_B_list,
        PA_line_list,
        PB_line_list,
    )

    plt.tight_layout()
    plt.savefig("./deltaTs_plots.png", dpi=300)
    plt.show()
