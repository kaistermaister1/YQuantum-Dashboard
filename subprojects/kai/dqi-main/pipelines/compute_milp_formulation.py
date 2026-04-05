import json
from itertools import combinations
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm


def compute_milp_formulation(
    options: pd.DataFrame,
    pkg_opts: pd.DataFrame,
    take_rates: pd.DataFrame,
    take_rates_packages: pd.DataFrame,
    number_of_packages_to_recommend: int,
    discount_factor: float,
    max_allowed_overlap: int,
    min_allowed_margin: int,
) -> Tuple[List[int], List[List], List[int]]:

    """
    Function that take as input the raw synthetic data and constructs the corresponding MILP instance.

    Inputs:
    options: pd.DataFrame
        DataFrame containing information about options.
    pkg_options: pd.DataFrame
        DataFrame containing information about packages, including which options are in which package.
    take_rates: pd.DataFrame
        DataFrame containing take rates at option level.
    take_rates_packages: pd.DataFrame
        DataFrame containing take rates at package level.
    number_of_packages_to_recommend: int
        The number of packages that the MILP will recommend.
    discount_factor: float
        Floating point number between 0 and 1 determining the discount applied to packages.
    max_allowe_overlap: int
        Maximum allowed number of options in common between two distinct packages.
    max_allowed_margin: int
        Minimal margin allowed for a package.

    Outputs:
    The function returns 3 lists: one for the objective function,
    one for the constraints left-hand-side and one for the constraints right-hand-side.
    objective_function: list[int]
        List of length n_packages. Each element is the coefficient c_i in the objective function sum c_i x_i.
    constraints: list[list]
        List of lists of length n_constraints. Each list is a list of length n_packages containing
        the coefficients a_{i,j} in the constraint sum_i a_{i,j} x_i <= b_j.
    constraints_rhs: list[int]
        List of length n_constraints. Each element is the coefficients b_j in the constraint sum_i a_{i,j} x_i <= b_j.
    """
    take_rates_options = options.merge(
        take_rates,
        how="left",
        left_on="opt_abbrev",
        right_on="item",
    )

    take_rates_options["discounted_margin"] = (
        take_rates_options["price_eur"] * (1 - discount_factor)
        - take_rates_options["cost_eur"]
    )

    # assert (take_rates_options["discounted_margin"] >= 0).all()

    nb_of_proposed_packages = len(take_rates_packages.index)

    objective = []
    for m in range(nb_of_proposed_packages):
        options_in_package_m = take_rates_packages.loc[m, "options_list"]
        discounted_margins_package_m = take_rates_options.loc[
            take_rates_options["opt_abbrev"].isin(options_in_package_m),
            "discounted_margin",
        ]
        objective.append(
            take_rates_packages.loc[m, "take_rate"]
            * discounted_margins_package_m.sum(),
        )

    constraints = []
    constraints_rhs = []

    # Max number of packages to recommend
    print("Adding constraints for maximum number of packages to recommend...")
    constraints.append([1] * nb_of_proposed_packages)
    constraints_rhs.append(number_of_packages_to_recommend)

    # Min margin per package
    print("Adding constraints for minimum margin per package...")
    for m in range(nb_of_proposed_packages):
        options_in_package_m = take_rates_packages.loc[m, "options_list"]
        package_margin = take_rates_options.loc[
            take_rates_options["opt_abbrev"].isin(options_in_package_m),
            "discounted_margin",
        ].sum()
        if package_margin <= min_allowed_margin:
            constraints.append([0] * nb_of_proposed_packages)
            constraints[-1][m] = 1
            constraints_rhs.append(0)

    # Overlap with existing packages
    print("Adding constraints for overlap existing packages...")
    for package, options_df in pkg_opts.groupby("pkg_abbrev"):
        options_in_existing_package = options_df["opt_abbrev"].to_list()
        for m in range(nb_of_proposed_packages):
            options_in_package_m = take_rates_packages.loc[m, "options_list"]
            overlap_len = (
                len(set(options_in_existing_package))
                + len(set(options_in_package_m))
                - len(set(options_in_existing_package + options_in_package_m))
            )
            if overlap_len >= max_allowed_overlap:
                constraints.append([0] * nb_of_proposed_packages)
                constraints[-1][m] = 1
                constraints_rhs.append(0)

    # Overlap between packages

    # Precompute sets for all packages
    option_sets = take_rates_packages["options_list"].apply(set).tolist()
    total_combinations = nb_of_proposed_packages * (nb_of_proposed_packages - 1) // 2
    print("Adding constraints for overlap between packages...")
    for m, m_prime in tqdm(
        combinations(range(nb_of_proposed_packages), 2),
        total=total_combinations,
    ):
        set_m = option_sets[m]
        set_m_prime = option_sets[m_prime]

        # Calculate overlap length using set operations
        overlap_len = len(set_m) + len(set_m_prime) - len(set_m.union(set_m_prime))

        if overlap_len >= max_allowed_overlap:
            constraint = [0] * nb_of_proposed_packages
            constraint[m] = 1
            constraint[m_prime] = 1
            constraints.append(constraint)
            constraints_rhs.append(1)

    return objective, constraints, constraints_rhs


def calculate_take_rate_deltas(
    options: pd.DataFrame,
    take_rates: pd.DataFrame,
    take_rates_packages: pd.DataFrame,
    discount_factor: int,
    elasticity: int,
    kappa: int,
    n_vehicles: int,
) -> pd.DataFrame:

    """
    Function that take as input the raw synthetic data and computes the take rates deltas from subsets to packages.

    Inputs:
    options: pd.DataFrame
        DataFrame containing information about options.
    take_rates: pd.DataFrame
        DataFrame containing take rates at option level.
    take_rates_packages: pd.DataFrame
        DataFrame containing take rates at subset/package level.
    discount_factor: float
        Floating point number between 0 and 1 determining the discount applied to packages.
    kappa: int
        Threshold value corresponding to the baseline (theoretical) price incurred
        by a customer that did not buy any options from a subset.
        Needed for the micro-economic model.
    n_vehicles: int
        Number of vehicles considered in the synthetic dataset.

    Outputs:
    The function returns a pd.DataFrame containing the updated take rate for each package,
    after applying the micro-econmic model to offering a discount on the package,
    starting from all different subsets of options.

    take_rate_packages: pd.DataFrame
        Dataframe with 3 columns: package_key, options_lsit (list of options in each package),
        take_rate (updated take rate for each package).
    """

    take_rates_options = options.merge(
        take_rates,
        how="left",
        left_on="opt_abbrev",
        right_on="item",
    )

    take_rates_options["discounted_price"] = take_rates_options["price_eur"] * (
        1 - discount_factor
    )

    take_rates_packages["options_list"] = take_rates_packages["package"].apply(
        lambda x: x.tolist(),
    )

    take_rates_packages["options_subset_list"] = take_rates_packages["subset"].apply(
        lambda x: x.tolist(),
    )

    options_price_dict = dict(
        zip(take_rates_options["opt_abbrev"], take_rates_options["discounted_price"]),
    )

    take_rates_packages["total_price_package"] = take_rates_packages[
        "options_list"
    ].apply(lambda x: sum(options_price_dict.get(opt, 0.0) for opt in x))

    take_rates_packages["total_price_subset"] = take_rates_packages[
        "options_subset_list"
    ].apply(lambda x: sum(options_price_dict.get(opt, 0.0) for opt in x))

    take_rates_packages["price_jump_ratio"] = take_rates_packages[
        "total_price_package"
    ] / take_rates_packages["total_price_subset"].apply(lambda x: max(kappa, x))

    take_rates_packages["take_rate_jump_ratio"] = take_rates_packages[
        "price_jump_ratio"
    ].apply(lambda x: min(1, x**elasticity))

    take_rates_packages["nb_customers_jumping"] = (
        take_rates_packages["count"] * take_rates_packages["take_rate_jump_ratio"]
    ).apply(lambda x: int(x))

    take_rates_packages["nb_customers_per_package"] = take_rates_packages.groupby(
        "package_key",
    )["nb_customers_jumping"].transform("sum")

    assert (take_rates_packages["nb_customers_per_package"] <= n_vehicles).all()

    take_rates_packages["take_rate"] = (
        take_rates_packages["nb_customers_per_package"] / n_vehicles
    )

    take_rates_packages = (
        take_rates_packages[["package_key", "options_list", "take_rate"]]
        .drop_duplicates(subset=["package_key", "take_rate"])
        .reset_index(drop=True)
    )

    return take_rates_packages


if __name__ == "__main__":

    discount_factor = 0.2
    elasticity = -2
    kappa = 500
    n_vehicles = 1000

    number_of_packages_to_recommend = 5
    max_allowed_overlap = 2
    min_allowed_margin = 0

    print("Loading raw data...")

    options = pd.read_csv("synthetic_data_generation/data/options.csv")
    pkg_opts = pd.read_csv("synthetic_data_generation/data/package_options.csv")
    take_rates = pd.read_csv(
        "synthetic_data_generation/data/test_vehicles_take_rates.csv",
    )
    take_rates_packages = pd.read_parquet("pipelines/data/take_rates.parquet")

    print("Applying the micro-economic model: Calculating take rates deltas...")

    take_rates_packages = calculate_take_rate_deltas(
        options,
        take_rates,
        take_rates_packages,
        discount_factor,
        elasticity,
        kappa,
        n_vehicles,
    )

    print("Constructing MILP instance...")

    objective, constraints, constraints_rhs = compute_milp_formulation(
        options,
        pkg_opts,
        take_rates,
        take_rates_packages,
        number_of_packages_to_recommend,
        discount_factor,
        max_allowed_overlap,
        min_allowed_margin,
    )

    nb_of_proposed_packages = len(take_rates_packages.index)
    print(f"Nb_of_proposed_packages: {nb_of_proposed_packages}")
    assert len(objective) == nb_of_proposed_packages

    assert len(constraints) == len(constraints_rhs)

    for constraint in constraints:
        assert len(constraint) == nb_of_proposed_packages

    # Combine MILP instance in a dictionary
    milp_formulation = {
        "objective": [round(x) for x in objective],
        "constraints": constraints,
        "constraints_rhs": constraints_rhs,
    }

    # Write to JSON file
    print("Writing MILP to JSON...")
    with open("pipelines/data/milp_formulation4.json", "w") as f:
        json.dump(milp_formulation, f, indent=4)
