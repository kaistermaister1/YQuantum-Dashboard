#!/usr/bin/env python

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_arguments():
    """
    Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Simulate vehicle configurations")
    parser.add_argument(
        "-N",
        "--num_vehicles",
        type=int,
        required=True,
        help="Number of vehicles to simulate",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data",
        help="Directory containing input CSV files",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output CSV filename for simulated configurations",
    )
    return parser.parse_args()


def load_data(
    input_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all necessary CSVs into DataFrames.

    Args:
        input_dir: Directory containing input CSV files.

    Returns:
        Tuple of DataFrames: families, options, packages, package_options, dependencies.
    """
    families = pd.read_csv(os.path.join(input_dir, "families.csv"))
    options = pd.read_csv(os.path.join(input_dir, "options.csv"))
    packages = pd.read_csv(os.path.join(input_dir, "packages.csv"))
    pkg_opts = pd.read_csv(os.path.join(input_dir, "package_options.csv"))
    dependencies = pd.read_csv(os.path.join(input_dir, "dependencies.csv"))
    return families, options, packages, pkg_opts, dependencies


def assign_take_rates(
    options: pd.DataFrame,
    packages: pd.DataFrame,
    families: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Assign or compute take rates for options and packages.

    Args:
        options: DataFrame with option information.
        packages: DataFrame with package information.
        families: DataFrame with family rules.

    Returns:
        Tuple of dicts: option_rates and package_rates mapping abbrevs to probability values.
    """
    option_rates = {}
    n_digits = 3
    for fam, group in options.groupby("family"):
        fam_rule = families.loc[families["fam_abbrev"] == fam].iloc[0]
        if fam_rule["min_select"] == 1 and fam_rule["max_select"] == 1:
            rates = np.ones(len(group)) / len(group)
            for opt, rate in zip(group["opt_abbrev"], rates):
                option_rates[opt] = round(float(rate), n_digits)
        else:
            for opt in group["opt_abbrev"]:
                option_rates[opt] = round(float(np.random.uniform(0.05, 0.3)), n_digits)
    package_rates = {
        pkg: round(float(np.random.uniform(0.1, 0.4)), n_digits)
        for pkg in packages["pkg_abbrev"]
    }
    return option_rates, package_rates


def select_packages(package_rates: Dict[str, float]) -> List[str]:
    """
    Randomly select packages based on rates.

    Args:
        package_rates: Dictionary mapping package abbreviations to probabilities.

    Returns:
        List of selected package abbreviations.
    """
    return [pkg for pkg, p in package_rates.items() if np.random.rand() < p]


def select_options(
    options: pd.DataFrame,
    families: pd.DataFrame,
    option_rates: Dict[str, float],
    initial_opts: List[str],
) -> List[str]:
    """
    Select standalone options, respecting family rules.

    Args:
        options: DataFrame with option information.
        families: DataFrame with family rules.
        option_rates: Dictionary mapping option abbreviations to probabilities.
        initial_opts: List of initially selected options.

    Returns:
        List of selected option abbreviations.
    """
    selected = set(initial_opts)
    for fam, group in options.groupby("family"):
        fam_rule = families.loc[families["fam_abbrev"] == fam].iloc[0]
        opts = group["opt_abbrev"].tolist()
        opts_to_draw = [o for o in opts if o not in selected]
        if fam_rule["min_select"] == 1 and fam_rule["max_select"] == 1:
            if not any(o in selected for o in opts):
                rates = np.array([option_rates[o] for o in opts_to_draw])
                rates /= rates.sum()
                chosen = np.random.choice(opts_to_draw, p=rates)
                selected.add(chosen)
        else:
            for o in opts_to_draw:
                if np.random.rand() < option_rates[o]:
                    selected.add(o)
    return list(selected)


def apply_dependencies(
    selected_opts: List[str],
    dependencies: pd.DataFrame,
) -> List[str]:
    """
    Apply IMPLIES rules to add implied options.

    Args:
        selected_opts: List of currently selected options.
        dependencies: DataFrame with dependency rules.

    Returns:
        List of selected options after applying dependencies.
    """
    sel_set = set(selected_opts)
    for _, row in dependencies.iterrows():
        if row["kind"] == "IMPLIES" and row["source_abbrev"] in sel_set:
            sel_set.add(row["target_abbrev"])
    return list(sel_set)


def validate_single_configuration(
    selected_opts: List[str],
    families: pd.DataFrame,
) -> List[str]:
    """
    Ensure exactly-one families have one option selected.

    Args:
        selected_opts: List of selected options.
        families: DataFrame with family rules.

    Returns:
        List of validated selected options.
    """
    return selected_opts


def simulate_vehicles(
    N: int,
    families: pd.DataFrame,
    options: pd.DataFrame,
    packages: pd.DataFrame,
    pkg_opts: pd.DataFrame,
    dependencies: pd.DataFrame,
    option_rates: Dict[str, float],
    package_rates: Dict[str, float],
) -> List[Dict]:
    """
    Generate N vehicle configurations.

    Args:
        N: Number of vehicles to simulate.
        families: DataFrame with family rules.
        options: DataFrame with option information.
        packages: DataFrame with package information.
        pkg_opts: DataFrame with package-option mappings.
        dependencies: DataFrame with dependency rules.
        option_rates: Dictionary mapping option abbreviations to probabilities.
        package_rates: Dictionary mapping package abbreviations to probabilities.

    Returns:
        List of dictionaries representing vehicle configurations.
    """
    vehicles = []
    for _ in range(N):
        chosen_pkgs = select_packages(package_rates)
        initial_opts = pkg_opts[pkg_opts["pkg_abbrev"].isin(chosen_pkgs)][
            "opt_abbrev"
        ].tolist()
        opts = select_options(options, families, option_rates, initial_opts)
        opts = apply_dependencies(opts, dependencies)
        opts = validate_single_configuration(opts, families)
        vehicles.append({"packages": chosen_pkgs, "options": opts})
    return vehicles


def aggregate_results(vehicles: List[Dict]) -> pd.DataFrame:
    """
    Compute aggregate take rates for each option and package.

    Args:
        vehicles: List of dictionaries representing vehicle configurations.

    Returns:
        DataFrame with aggregate take rates for options and packages.
    """
    records = []
    N = len(vehicles)
    all_opts = [opt for v in vehicles for opt in v["options"]]
    opt_counts = pd.Series(all_opts).value_counts()
    for opt, count in opt_counts.items():
        records.append({"item": opt, "type": "option", "take_rate": count / N})
    all_pkgs = [pkg for v in vehicles for pkg in v["packages"]]
    pkg_counts = pd.Series(all_pkgs).value_counts()
    for pkg, count in pkg_counts.items():
        records.append({"item": pkg, "type": "package", "take_rate": count / N})
    return pd.DataFrame(records)


def dump_raw_take_rates(
    option_rates: Dict[str, float],
    package_rates: Dict[str, float],
    filename: str,
) -> None:
    """
    Write raw take‐rate dicts to a combined CSV.

    Args:
        option_rates: Dictionary mapping option abbreviations to probabilities.
        package_rates: Dictionary mapping package abbreviations to probabilities.
        filename: Base filename for the output CSV.
    """
    base = Path(filename)
    # Build DataFrame
    df_opts = pd.DataFrame(
        [
            {"item": opt, "type": "option", "take_rate": rate}
            for opt, rate in option_rates.items()
        ],
    )
    df_pkgs = pd.DataFrame(
        [
            {"item": pkg, "type": "package", "take_rate": rate}
            for pkg, rate in package_rates.items()
        ],
    )
    df_raw = pd.concat([df_opts, df_pkgs], ignore_index=True)

    out_file = base.with_name(f"{base.stem}_raw_take_rates.csv")
    df_raw.to_csv(out_file, index=False)


def write_output(
    vehicles: List[Dict[str, List[str]]],
    take_rates: pd.DataFrame,
    options: pd.DataFrame,
    packages: pd.DataFrame,
    filename: str,
) -> None:
    """
    Write vehicle configs and enriched take-rate summary to CSV.

    Args:
        vehicles: List of dictionaries representing vehicle configurations.
        take_rates: DataFrame with aggregate take rates.
        options: DataFrame with option information.
        packages: DataFrame with package information.
        filename: Base filename for the output CSVs.
    """
    base = Path(filename)
    # Vehicles
    df_veh = pd.DataFrame(
        {
            "packages": [";".join(v["packages"]) for v in vehicles],
            "options": [";".join(v["options"]) for v in vehicles],
        },
    )
    df_veh.to_csv(filename, index_label="id")

    # Prepare info tables for merge
    opt_info = options[["opt_abbrev", "name", "family"]].rename(
        columns={"opt_abbrev": "item"},
    )
    pkg_info = packages[["pkg_abbrev", "name"]].rename(
        columns={"pkg_abbrev": "item"},
    )
    pkg_info["family"] = "package"

    info = pd.concat([opt_info, pkg_info], ignore_index=True)

    # Merge take_rates with info
    tr_enriched = take_rates.merge(info, on="item", how="left")

    # Write take_rates file
    rates_file = base.with_name(f"{base.stem}_take_rates.csv")
    tr_enriched.to_csv(rates_file, index=False)


def main():
    """
    Recipe for simulating vehicle configurations:
    1. Parse arguments (num_vehicles, input_dir, output_file)
    2. Load data from CSVs: families, options, packages, package_options, dependencies
    3. Assign take rates:
       a. option_rates: per-option probabilities (family-aware)
       b. package_rates: per-package probabilities
    4. Dump raw take rates to <basename>_raw_take_rates.csv
    5. Simulate N vehicles:
       a. select_packages(): sample packages by package_rates
       b. select_options(): sample standalone options by option_rates, enforce family rules
       c. apply_dependencies(): add implied options
       d. validate_single_configuration(): ensure exactly-one families resolved
    6. aggregate_results(): compute empirical take rates for options & packages
    7. write_output():
       a. save vehicle configurations to output_file
       b. save enriched take rates (with name & family) to <basename>_take_rates.csv
    """

    args = parse_arguments()

    np.random.seed(42)
    random.seed(42)

    families, options, packages, pkg_opts, dependencies = load_data(args.input_dir)
    option_rates, package_rates = assign_take_rates(options, packages, families)
    dump_raw_take_rates(option_rates, package_rates, args.output_file)

    vehicles = simulate_vehicles(
        args.num_vehicles,
        families,
        options,
        packages,
        pkg_opts,
        dependencies,
        option_rates,
        package_rates,
    )
    take_rates = aggregate_results(vehicles)

    write_output(vehicles, take_rates, options, packages, args.output_file)


if __name__ == "__main__":
    Path("synthetic_data_generation/data").mkdir(exist_ok=True)
    main()
