#!/usr/bin/env python3

import argparse
import itertools
import json
import logging
import time
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

# Import utilities from auxiliary modules
try:
    from parallel_utils import (
        classify_subset_per_package_parallel,
        get_unique_package_subsets_parallel,
    )

    has_parallel_utils = True
except ImportError:
    has_parallel_utils = False
    logging.warning(
        "parallel_utils module not found, parallel processing will not be available",
    )

try:
    from export_utils import export_flattened_data

    has_export_utils = True
except ImportError:
    has_export_utils = False
    logging.warning(
        "export_utils module not found, export functionality will be limited",
    )

try:
    from stream_utils import (
        classify_subset_per_package_streaming,
        get_unique_package_subsets_streaming,
        stream_flattened_data,
    )

    has_stream_utils = True
except ImportError:
    has_stream_utils = False
    logging.warning(
        "stream_utils module not found, streaming options will not be available",
    )

try:
    from stream_parquet_utils import classify_subset_per_package_streaming_parquet
    from stream_parquet_utils import (
        get_unique_package_subsets_streaming as get_unique_package_subsets_streaming_parquet,
    )
    from stream_parquet_utils import stream_flattened_data_parquet

    has_stream_parquet_utils = True
except ImportError:
    has_stream_parquet_utils = False
    logging.warning(
        "stream_parquet_utils module not found, Parquet streaming will not be available",
    )

# Define the format string
log_format = "[%(asctime)s][%(levelname)s][%(funcName)s:%(lineno)d] %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Basic configuration for the root logger
logging.basicConfig(
    level=logging.INFO,  # or INFO, WARNING, etc.
    format=log_format,
    datefmt=date_format,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load package templates from YAML
# ──────────────────────────────────────────────────────────────────────────────


def load_package_templates(yaml_file: str) -> Tuple[Dict[str, Any]]:
    """
    Load family definitions and package templates from a YAML file.
    """
    data = yaml.safe_load(Path(yaml_file).read_text())
    logging.info(
        f"Loaded {len(data['package_templates'])} package templates from {yaml_file}",
    )
    return data["package_templates"]


def load_families_options(
    families_file: str,
    options_file: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Load families.csv and options.csv, and build a dict mapping fam_abbrev to
    its min_select, max_select, and list of option codes.
    """
    df_fam = pd.read_csv(families_file)
    df_opts = pd.read_csv(options_file)
    price_map = dict(zip(df_opts["opt_abbrev"], df_opts["price_eur"]))
    options_by_family: Dict[str, List[str]] = {}
    for fam in df_fam["fam_abbrev"]:
        opts = df_opts.loc[df_opts["family"] == fam, "opt_abbrev"].tolist()
        options_by_family[fam] = opts
    family_rules: Dict[str, Dict[str, Any]] = {}
    for _, row in df_fam.iterrows():
        fam = row["fam_abbrev"]
        family_rules[fam] = {
            "min_select": int(row["min_select"]),
            "max_select": int(row["max_select"])
            if not pd.isna(row["max_select"])
            else len(options_by_family[fam]),
            "options": options_by_family[fam],
        }
    return df_fam, df_opts, family_rules, price_map


def load_vehicles(filename: str) -> List[Dict[str, List[str]]]:
    """
    Load vehicles from CSV; expects semicolon-delimited 'packages' and 'options'.
    """
    df = pd.read_csv(filename)
    vehicles: List[Dict[str, List[str]]] = []
    for _, row in df.iterrows():
        pkgs = row["packages"].split(";") if pd.notna(row["packages"]) else []
        opts = row["options"].split(";") if pd.notna(row["options"]) else []
        vehicles.append({"packages": pkgs, "options": opts})
    logging.info(f"Loaded {len(vehicles)} vehicles from {filename}")
    return vehicles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute historical 2×2 take-rate tables for packages.",
    )
    parser.add_argument(
        "--families_file",
        required=True,
        help="CSV with columns 'fam_abbrev,min_select,max_select'.",
    )
    parser.add_argument(
        "--options_file",
        required=True,
        help="CSV with columns 'opt_abbrev,family'.",
    )
    parser.add_argument(
        "--vehicles_file",
        required=True,
        help="CSV with columns 'packages' & 'options'.",
    )
    parser.add_argument(
        "--template_file",
        required=True,
        help="YAML defining families and package_templates.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output filename for tables.",
    )

    # Performance options
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing for computation-intensive operations.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: CPU count).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for processing and streaming operations.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream vehicles from CSV to reduce memory usage.",
    )

    # Output options
    parser.add_argument(
        "--output_format",
        choices=["csv", "json", "parquet", "excel"],
        default="csv",
        help="Format for output data.",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include processing metadata in the output.",
    )
    parser.add_argument(
        "--stream_output",
        action="store_true",
        help="Stream output to file instead of keeping in memory.",
    )
    parser.add_argument(
        "--use_parquet",
        action="store_true",
        help="Use Parquet format for streaming (better performance and memory usage).",
    )
    parser.add_argument(
        "--row_group_size",
        type=int,
        default=10000,
        help="Row group size for Parquet files (only used with --use_parquet).",
    )

    return parser.parse_args()


def enumerate_packages(
    family_rules: Dict[str, Dict[str, Any]],
    package_templates: Dict[str, Dict[str, Any]],
) -> Dict[str, List[List[str]]]:
    """
    For each package template, generate all feasible packages via per-family Cartesian product.

    package_templates maps template_name -> dict with key 'families' listing fam_abbrev.
    """
    result: Dict[str, List[List[str]]] = {}
    for tmpl, cfg in package_templates.items():
        families_list = cfg.get("families", [])
        if not isinstance(families_list, list):
            raise ValueError(
                f"Template '{tmpl}' has invalid 'families' entry: {families_list}",
            )
        per_family_choices: List[List[List[str]]] = []
        for fam in families_list:
            if fam not in family_rules:
                raise KeyError(
                    f"Unknown family abbreviation '{fam}' in template '{tmpl}'",
                )
            rules = family_rules[fam]
            opts = rules["options"]
            mn, mx = rules["min_select"], rules["max_select"]

            expected_N = sum(comb(len(opts), k) for k in range(mn, mx + 1))
            logging.info(
                f"Enumerating {tmpl} for family '{fam}' with options: {opts}, min: {mn}, max: {mx}",
            )
            logging.info(f"Expected number of combinations: {expected_N}")

            # Generate all combinations of options for this family
            combos = [
                list(c)
                for k in range(mn, mx + 1)
                for c in itertools.combinations(opts, k)
            ]

            assert (
                len(combos) == expected_N
            ), f"Unexpected number of combinations for {tmpl} family '{fam}'"

            per_family_choices.append(combos)
        # Cartesian product across the per-family choice lists
        result[tmpl] = [
            sum(prod, []) for prod in itertools.product(*per_family_choices)
        ]

        logging.info(f"Template '{tmpl}' has {len(result[tmpl])} valid packages.")
    return result


def get_all_options_for_packages(
    family_rules: Dict[str, Dict[str, Any]],
    package_templates: Dict[str, Dict[str, Any]],
) -> Dict[str, frozenset]:
    """
    For each package template, return a frozenset of all options that are allowed
    to go into the families associated with that template.

    Returns:
        Dict mapping template_name -> frozenset of all possible options across all families
    """
    result: Dict[str, frozenset] = {}
    for tmpl, cfg in package_templates.items():
        families_list = cfg.get("families", [])
        if not isinstance(families_list, list):
            raise ValueError(
                f"Template '{tmpl}' has invalid 'families' entry: {families_list}",
            )

        # Collect all options from all families for this template
        all_options = set()
        for fam in families_list:
            if fam not in family_rules:
                raise KeyError(
                    f"Unknown family abbreviation '{fam}' in template '{tmpl}'",
                )
            rules = family_rules[fam]
            opts = rules["options"]
            all_options.update(opts)

        # Store as frozenset for hashability
        result[tmpl] = frozenset(all_options)
        logging.info(
            f"Template '{tmpl}' has {len(result[tmpl])} possible options across all families.",
        )

    return result


def get_unique_package_subsets(
    vehicles: List[Dict[str, List[str]]],
    all_options: Dict[str, frozenset],
) -> Dict[str, List[Tuple[frozenset, int]]]:
    """
    For each template, count unique subsets of vehicle options that belong to that template.

    Args:
        vehicles: List of vehicle dictionaries, each with "options" key
        all_options: Dict mapping template_name -> frozenset of allowed options

    Returns:
        Dict mapping template_name -> list of (option_subset, count) tuples
    """
    result: Dict[str, List[Tuple[frozenset, int]]] = {}

    # Create counters for each template
    counters = {tmpl: Counter() for tmpl in all_options}

    # Process each vehicle in a single pass
    start_time = time.time()
    total_vehicles = len(vehicles)

    for idx, vehicle in enumerate(vehicles):
        # Progress logging every 100k vehicles
        if idx > 0 and idx % 100000 == 0:
            elapsed = time.time() - start_time
            vehicles_per_sec = idx / elapsed
            estimated_remaining = (
                (total_vehicles - idx) / vehicles_per_sec
                if vehicles_per_sec > 0
                else "unknown"
            )
            logging.info(
                f"Processed {idx}/{total_vehicles} vehicles "
                f"({vehicles_per_sec: .1f} vehicles/sec, "
                f"est. remaining: {estimated_remaining: .1f} sec)",
            )

        # Get the vehicle's options
        vehicle_opts = set(vehicle.get("options", []))

        # For each template
        for tmpl, allowed_opts in all_options.items():
            # Filter options for current template (fast set intersection)
            relevant_opts = vehicle_opts.intersection(allowed_opts)

            # Only count if there are relevant options
            if relevant_opts:
                # Use frozenset for hashability
                counters[tmpl][frozenset(relevant_opts)] += 1

    # Convert counters to sorted lists for output
    for tmpl, counter in counters.items():
        # Sort by count in descending order for potential optimization later
        result[tmpl] = [
            (option_set, count) for option_set, count in counter.most_common()
        ]
        logging.info(f"Template '{tmpl}' has {len(result[tmpl])} unique option subsets")

    return result


def classify_subset_per_package(
    pkg_candidates: Dict[str, List[List[str]]],
    subsets: Dict[str, List[Tuple[frozenset, int]]],
    total_vehicles: int,
) -> Dict[str, Dict[str, Dict]]:
    """
    For each package candidate, track detailed relationships with observed option subsets.

    Args:
        pkg_candidates: Dict mapping template_name -> list of package option lists
        subsets: Dict mapping template_name -> list of (option_subset, count) tuples
        total_vehicles: Total number of vehicles processed

    Returns:
        Dict mapping template_name -> dict mapping package_key -> package info
        where package info includes:
        - exact: count of exact matches
        - some: count of vehicles with some elements
        - none: count of vehicles with no elements
        - subsets: detailed info about which subsets relate to this package
    """
    result = {}

    # Process each template
    for tmpl, packages in pkg_candidates.items():
        if tmpl not in subsets:
            logging.warning(
                f"Template '{tmpl}' not found in subsets data, skipping",  # noqa: E713
            )
            continue

        template_subsets = subsets[tmpl]

        # Create result structure for this template
        result[tmpl] = {}

        # Initialize progress tracking
        start_time = time.time()
        total_packages = len(packages)
        logging.info(
            f"Classifying {total_packages} package candidates for template '{tmpl}'",
        )

        # Process each package for this template
        for pkg_idx, pkg_options in enumerate(packages):
            # Convert package options to frozenset for faster set operations
            pkg_set = frozenset(pkg_options)
            # Create a unique key for this package
            pkg_key = ",".join(sorted(pkg_options))

            # Track counts for this package
            exact_match_count = 0
            some_elements_count = 0
            none_elements_count = 0  # Initialize none count

            # Track detailed subset relationships
            exact_subsets = []  # Subsets that exactly match the package
            intersecting_subsets = []  # Subsets that have some overlap with the package

            # Check each observed subset against this package
            for subset, count in template_subsets:
                if subset == pkg_set:
                    # Exact match: has exactly this package
                    exact_match_count += count
                    exact_subsets.append((subset, count))
                elif subset.intersection(pkg_set):
                    # Some match: has at least one element from this package (but not exact match)
                    some_elements_count += count
                    intersecting_subsets.append((subset, count))

            # None: vehicles with no elements from this package
            none_elements_count = (
                total_vehicles - exact_match_count - some_elements_count
            )

            # Store the classification for this package
            result[tmpl][pkg_key] = {
                "package": pkg_options,
                "exact": exact_match_count,
                "some": some_elements_count,
                "none": none_elements_count,
                "total": total_vehicles,
                "subsets": {
                    "exact": exact_subsets,
                    "intersecting": intersecting_subsets,
                },
            }

            # Log progress periodically
            if pkg_idx > 0 and pkg_idx % 10000 == 0:
                elapsed = time.time() - start_time
                packages_per_sec = pkg_idx / elapsed
                remaining = (
                    (total_packages - pkg_idx) / packages_per_sec
                    if packages_per_sec > 0
                    else "unknown"
                )
                logging.info(
                    f"Processed {pkg_idx}/{total_packages} packages for '{tmpl}' "
                    f"({packages_per_sec: .1f} pkg/sec, est. remaining: {remaining: .1f} sec)",
                )

        logging.info(
            f"Completed classification for {total_packages} packages in template '{tmpl}'",
        )

    return result


def flatten_classification_data(
    classifications: Dict[str, Dict[str, Dict]],
    total_vehicles: int,
) -> List[Dict[str, Any]]:
    """
    Transform the hierarchical classification structure into a flat table with:
    - package: The package options as a tuple
    - subset: The subset of options as a tuple (empty tuple for vehicles with no options)
    - count: The number of vehicles with this subset

    Args:
        classifications: The output of classify_subset_per_package
        total_vehicles: The total number of vehicles

    Returns:
        A list of dictionaries, each with 'package', 'subset', and 'count' keys
    """
    result = []
    start_time = time.time()
    total_packages = sum(len(packages) for packages in classifications.values())
    packages_processed = 0

    # Process each template
    for tmpl, packages in classifications.items():
        logging.info(
            f"Flattening data for template '{tmpl}' with {len(packages)} packages",
        )

        for pkg_key, pkg_info in packages.items():
            pkg_options = pkg_info["package"]
            pkg_set = frozenset(pkg_options)
            n_processed = 0
            rows_added = 0  # Keep track of rows added for this package

            # Process exact match subsets
            for subset, count in pkg_info["subsets"]["exact"]:
                subset_list = sorted(subset)
                result.append(
                    {
                        "template": tmpl,
                        "package": tuple(pkg_options),
                        "package_key": pkg_key,
                        "subset": tuple(subset_list),
                        "count": count,
                        "relationship": "exact",
                    },
                )
                n_processed += count
                rows_added += 1

            # Process intersecting subsets
            for subset, count in pkg_info["subsets"]["intersecting"]:
                subset_list = sorted(subset.intersection(pkg_set))
                result.append(
                    {
                        "template": tmpl,
                        "package": tuple(pkg_options),
                        "package_key": pkg_key,
                        "subset": tuple(subset_list),
                        "count": count,
                        "relationship": "some",
                    },
                )
                n_processed += count
                rows_added += 1

            # Add entry for vehicles with none of the options in this package
            if pkg_info["none"] > 0:
                result.append(
                    {
                        "template": tmpl,
                        "package": tuple(pkg_options),
                        "package_key": pkg_key,
                        "subset": tuple(),  # Empty tuple for no options
                        "count": pkg_info["none"],
                        "relationship": "none",
                    },
                )
                n_processed += pkg_info["none"]
                rows_added += 1

            # Assert that we've processed all vehicles
            assert (
                n_processed == total_vehicles
            ), f"Processed {n_processed} vehicles, expected {total_vehicles} for package {pkg_key}"

            # Count unique subsets that we've actually processed for this package
            unique_subsets = len(pkg_info["subsets"]["exact"]) + len(
                pkg_info["subsets"]["intersecting"],
            )
            if pkg_info["none"] > 0:
                unique_subsets += 1  # Count the empty subset if present

            # Verify we have the expected number of rows based on actual observed data
            assert (
                rows_added == unique_subsets
            ), f"Package {pkg_key}: Added {rows_added} rows but found {unique_subsets} unique observed subsets"

            # Log progress periodically
            packages_processed += 1
            if packages_processed % 100 == 0:
                elapsed = time.time() - start_time
                packages_per_sec = packages_processed / elapsed
                remaining = (
                    (total_packages - packages_processed) / packages_per_sec
                    if packages_per_sec > 0
                    else "unknown"
                )
                logging.info(
                    f"Flattened {packages_processed}/{total_packages} packages "
                    f"({packages_per_sec: .1f} pkg/sec, est. remaining: {remaining: .1f} sec)",
                )

    logging.info(f"Flattened data contains {len(result)} rows")
    return result


def main():
    args = parse_args()
    _, _, family_rules, price_map = load_families_options(
        args.families_file,
        args.options_file,
    )
    package_templates = load_package_templates(args.template_file)
    vehicles = load_vehicles(args.vehicles_file)

    pkg_candidates = enumerate_packages(family_rules, package_templates)
    all_options = get_all_options_for_packages(family_rules, package_templates)

    # Get unique subsets of options for each package template
    if args.parallel and has_parallel_utils:
        subsets = get_unique_package_subsets_parallel(
            vehicles,
            all_options,
            num_processes=args.num_processes,
            batch_size=args.batch_size,
        )
    elif args.stream and args.use_parquet and has_stream_parquet_utils:
        # Use the Parquet-optimized version of streaming
        logging.info("Using Parquet-optimized streaming for subset calculation")
        subsets = get_unique_package_subsets_streaming_parquet(
            args.vehicles_file,
            all_options,
            batch_size=args.batch_size,
        )
    elif args.stream and has_stream_utils:
        # Use the regular streaming implementation
        subsets = get_unique_package_subsets_streaming(
            args.vehicles_file,
            all_options,
            batch_size=args.batch_size,
        )
    else:
        # Default to serial implementation
        subsets = get_unique_package_subsets(vehicles, all_options)

    for x in subsets:
        assert sum(subsets[x][i][1] for i in range(len(subsets[x]))) == len(
            vehicles,
        ), f"Mismatch in counts for {x}"

    # Classify relationships between package candidates and observed option subsets
    total_vehicles = len(vehicles)

    # Determine if we should use end-to-end streaming for classification and output
    use_full_streaming = args.stream and has_stream_utils and args.stream_output

    if use_full_streaming:
        # Skip creating the classifications object entirely when using full streaming
        logging.info(
            "Using full end-to-end streaming workflow to minimize memory usage",
        )
    else:
        # Only calculate classifications if we're not using full streaming
        # Check if workload is large enough to benefit from parallel processing
        use_parallel = False
        if args.parallel and has_parallel_utils:
            # Get an estimate of the total workload size
            total_packages = sum(len(packages) for packages in pkg_candidates.values())
            total_subsets = sum(len(subset_list) for subset_list in subsets.values())
            workload_size = total_packages * total_subsets

            # Only use parallel for larger workloads
            # For small workloads (< 100,000 comparisons), parallel overhead isn't worth it
            if workload_size > 500000:  # Threshold determined empirically
                use_parallel = True
                logging.info(
                    "Using parallel implementation for large workload: "
                    f"{total_packages} packages × {total_subsets} subsets",
                )
            else:
                logging.info(
                    "Workload too small for parallel processing "
                    f"({workload_size} comparisons), using serial implementation",
                )

        # Use parallel or serial based on the decision
        if use_parallel:
            classifications = classify_subset_per_package_parallel(
                pkg_candidates,
                subsets,
                total_vehicles,
                num_processes=args.num_processes,
            )
        else:
            # Default to serial implementation
            classifications = classify_subset_per_package(
                pkg_candidates,
                subsets,
                total_vehicles,
            )

    use_parquet_streaming = (
        use_full_streaming and args.use_parquet and has_stream_parquet_utils
    )

    if use_parquet_streaming:
        # Use the optimized Parquet streaming implementation
        logging.info(
            "Using end-to-end Parquet streaming workflow for optimal memory usage",
        )
        classify_subset_per_package_streaming_parquet(
            pkg_candidates,
            subsets,
            total_vehicles,
            args.output_file,
            batch_size=args.batch_size,
            row_group_size=args.row_group_size,
        )
        # No need for flatten_classification_data step - results written directly to Parquet
        logging.info(
            f"Parquet streaming classification complete - results written to {args.output_file}",
        )
    elif use_full_streaming:
        # Regular CSV streaming approach
        logging.info("Using end-to-end CSV streaming workflow to reduce memory usage")
        classify_subset_per_package_streaming(
            pkg_candidates,
            subsets,
            total_vehicles,
            args.output_file,
            batch_size=args.batch_size,
        )
        # No need for flatten_classification_data step - results written directly
        logging.info(
            f"Streaming classification complete - results written to {args.output_file}",
        )
    else:
        # Mixed or traditional approach
        # Flatten the classification data into a table structure
        if args.stream_output and args.use_parquet and has_stream_parquet_utils:
            # Stream to Parquet format
            logging.info("Using Parquet streaming for output")
            stream_flattened_data_parquet(
                classifications,
                total_vehicles,
                args.output_file,
                batch_size=args.batch_size,
                row_group_size=args.row_group_size,
            )
        elif args.stream_output and has_stream_utils:
            # Stream to CSV format
            stream_flattened_data(
                classifications,
                total_vehicles,
                args.output_file,
                batch_size=args.batch_size,
            )
        else:
            # Default to in-memory flattening
            flattened_data = flatten_classification_data(
                classifications,
                total_vehicles,
            )

            # Use export_utils if available and requested format is supported
            if has_export_utils:
                logging.info(
                    f"Exporting {len(flattened_data)} rows to {args.output_format} format...",
                )
                export_flattened_data(
                    flattened_data,
                    args.output_file,
                    format=args.output_format,
                    include_metadata=args.include_metadata,
                )
            else:
                # Fallback to simple JSON output
                with open(args.output_file, "w") as f:
                    json.dump(flattened_data, f)

    logging.info(f"Found unique option subsets for {len(subsets)} templates")

    if use_full_streaming or use_parquet_streaming:
        logging.info(
            f"Classified packages via streaming for {len(pkg_candidates)} templates",
        )
    else:
        logging.info(f"Classified packages for {len(classifications)} templates")

    logging.info(f"Output written to {args.output_file}")


if __name__ == "__main__":
    main()
