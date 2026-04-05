#!/usr/bin/env python3

import csv
import gc
import logging
import time
from collections import Counter
from typing import Dict, Iterator, List, Tuple

import pandas as pd


def stream_vehicles_from_csv(
    filename: str,
    batch_size: int = 10000,
) -> Iterator[List[Dict[str, List[str]]]]:
    """
    Stream vehicles from a CSV file in batches to minimize memory usage.

    Args:
        filename: Path to the CSV file
        batch_size: Number of vehicles to yield in each batch

    Yields:
        List of vehicle dictionaries, each with 'packages' and 'options' keys
    """
    # Use pandas to get a basic sense of the file size
    file_info = pd.read_csv(filename, nrows=1)
    has_packages_col = "packages" in file_info.columns
    has_options_col = "options" in file_info.columns

    if not (has_packages_col and has_options_col):
        raise ValueError(
            f"CSV file {filename} must contain 'packages' and 'options' columns",
        )

    batch = []

    # Open the file and read in batches
    with open(filename, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pkgs = (
                row["packages"].split(";")
                if row.get("packages") and row["packages"].strip()
                else []
            )
            opts = (
                row["options"].split(";")
                if row.get("options") and row["options"].strip()
                else []
            )
            batch.append({"packages": pkgs, "options": opts})

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining vehicles in the last batch
        if batch:
            yield batch


def get_unique_package_subsets_streaming(
    vehicles_file: str,
    all_options: Dict[str, frozenset],
    batch_size: int = 10000,
) -> Dict[str, List[Tuple[frozenset, int]]]:
    """
    Memory-efficient version of get_unique_package_subsets that processes vehicles in batches.

    Args:
        vehicles_file: Path to the CSV file with vehicle data
        all_options: Dict mapping template_name -> frozenset of allowed options
        batch_size: Number of vehicles to process in each batch

    Returns:
        Dict mapping template_name -> list of (option_subset, count) tuples
    """
    # Create counters for each template
    counters = {tmpl: Counter() for tmpl in all_options}

    # Process vehicles in batches
    start_time = time.time()
    total_vehicles = 0
    batch_number = 0

    for batch in stream_vehicles_from_csv(vehicles_file, batch_size):
        batch_number += 1
        batch_start_time = time.time()

        # Process each vehicle in the batch
        for vehicle in batch:
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

        # Update total and log progress
        total_vehicles += len(batch)
        batch_time = time.time() - batch_start_time
        total_time = time.time() - start_time
        vehicles_per_sec = total_vehicles / total_time

        logging.info(
            f"Batch {batch_number}: Processed {len(batch)} vehicles "
            f"({len(batch) / batch_time: .1f} vehicles/sec in this batch, "
            f"{vehicles_per_sec: .1f} vehicles/sec overall, "
            f"total: {total_vehicles} vehicles)",
        )

    # Convert counters to sorted lists for output
    result = {}
    for tmpl, counter in counters.items():
        # Sort by count in descending order for potential optimization
        result[tmpl] = [
            (option_set, count) for option_set, count in counter.most_common()
        ]
        logging.info(f"Template '{tmpl}' has {len(result[tmpl])} unique option subsets")

    return result


def stream_flattened_data(
    classifications: Dict[str, Dict[str, Dict]],
    total_vehicles: int,
    output_file: str,
    batch_size: int = 1000,
) -> int:
    """
    Stream flattened classification data directly to a file without
    keeping everything in memory.

    Args:
        classifications: The output of classify_subset_per_package
        total_vehicles: The total number of vehicles
        output_file: Path to the output CSV file
        batch_size: Maximum number of rows to process per template/package

    Returns:
        Total number of rows written
    """
    # Initialize CSV file
    fieldnames = [
        "template",
        "package_key",
        "package",
        "subset",
        "count",
        "relationship",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        start_time = time.time()
        total_packages = sum(len(packages) for packages in classifications.values())
        packages_processed = 0

        # Process each template
        for tmpl, packages in classifications.items():
            logging.info(
                f"Streaming data for template '{tmpl}' with {len(packages)} packages",
            )

            for pkg_key, pkg_info in packages.items():
                pkg_options = pkg_info["package"]
                pkg_set = frozenset(pkg_options)
                pkg_tuple = tuple(pkg_options)
                n_processed = 0
                rows_written = 0

                # Process exact match subsets
                for subset, count in pkg_info["subsets"]["exact"]:
                    subset_list = sorted(subset)
                    row = {
                        "template": tmpl,
                        "package_key": pkg_key,
                        "package": str(pkg_tuple),  # Convert to string for CSV
                        "subset": str(tuple(subset_list)),  # Convert to string for CSV
                        "count": count,
                        "relationship": "exact",
                    }
                    writer.writerow(row)

                    n_processed += count
                    rows_written += 1
                    total_rows += 1

                # Process intersecting subsets
                for subset, count in pkg_info["subsets"]["intersecting"]:
                    subset_list = sorted(subset.intersection(pkg_set))
                    row = {
                        "template": tmpl,
                        "package_key": pkg_key,
                        "package": str(pkg_tuple),
                        "subset": str(tuple(subset_list)),
                        "count": count,
                        "relationship": "some",
                    }
                    writer.writerow(row)

                    n_processed += count
                    rows_written += 1
                    total_rows += 1

                # Add entry for vehicles with none of the options in this package
                if pkg_info["none"] > 0:
                    row = {
                        "template": tmpl,
                        "package_key": pkg_key,
                        "package": str(pkg_tuple),
                        "subset": "()",  # Empty tuple as string
                        "count": pkg_info["none"],
                        "relationship": "none",
                    }
                    writer.writerow(row)

                    n_processed += pkg_info["none"]
                    rows_written += 1
                    total_rows += 1

                # Assert that we've processed all vehicles
                assert (
                    n_processed == total_vehicles
                ), f"Processed {n_processed} vehicles, expected {total_vehicles} for package {pkg_key}"

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
                        f"Streamed {packages_processed}/{total_packages} packages "
                        f"({packages_per_sec: .1f} pkg/sec, est. remaining: {remaining: .1f} sec, "
                        f"{total_rows} rows written)",
                    )

    logging.info(f"Streamed data contains {total_rows} rows")
    return total_rows


def classify_subset_per_package_streaming(
    pkg_candidates: Dict[str, List[List[str]]],
    subsets: Dict[str, List[Tuple[frozenset, int]]],
    total_vehicles: int,
    output_file: str,
    batch_size: int = 5000,
) -> None:
    """
    Memory-efficient version of classify_subset_per_package that processes packages
    in batches and streams results directly to a file.

    Args:
        pkg_candidates: Dict mapping template_name -> list of package option lists
        subsets: Dict mapping template_name -> list of (option_subset, count) tuples
        total_vehicles: Total number of vehicles processed
        output_file: Path to the output CSV file
        batch_size: Number of packages to process in each batch

    Returns:
        None - results are written directly to the output file
    """
    # Initialize CSV file
    fieldnames = [
        "template",
        "package_key",
        "package",
        "subset",
        "count",
        "relationship",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = 0
        total_packages = sum(len(packages) for packages in pkg_candidates.values())
        packages_processed = 0
        start_time = time.time()

        # Process each template
        for tmpl, packages in pkg_candidates.items():
            if tmpl not in subsets:
                logging.warning(
                    f"Template '{tmpl}' not found in subsets data, skipping",  # noqa: E713
                )
                continue

            template_subsets = subsets[tmpl]
            template_start_time = time.time()
            template_total = len(packages)

            logging.info(
                f"Streaming classification for {template_total} package candidates for template '{tmpl}'",
            )

            # Process packages in batches to control memory usage
            for batch_start in range(0, len(packages), batch_size):
                batch_end = min(batch_start + batch_size, len(packages))
                pkg_batch = packages[batch_start:batch_end]
                batch_size_actual = len(pkg_batch)

                batch_start_time = time.time()
                batch_rows = 0

                # Process each package in this batch
                for pkg_options in pkg_batch:
                    # Convert package options to frozenset for faster set operations
                    pkg_set = frozenset(pkg_options)
                    # Create a unique key for this package
                    pkg_key = ",".join(sorted(pkg_options))
                    pkg_tuple = tuple(pkg_options)

                    # Track counts for this package
                    exact_match_count = 0
                    some_elements_count = 0

                    # Lists to collect subsets that need to be written
                    exact_matches = []
                    some_matches = []

                    # Check each observed subset against this package
                    for subset, count in template_subsets:
                        if subset == pkg_set:
                            # Exact match: has exactly this package
                            exact_match_count += count
                            exact_matches.append((subset, count))
                        elif subset.intersection(pkg_set):
                            # Some match: has at least one element from this package
                            some_elements_count += count
                            some_matches.append((subset, count))

                    # None: vehicles with no elements from this package
                    none_elements_count = (
                        total_vehicles - exact_match_count - some_elements_count
                    )

                    # Write exact matches directly to file
                    for subset, count in exact_matches:
                        subset_list = sorted(subset)
                        row = {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(pkg_tuple),
                            "subset": str(tuple(subset_list)),
                            "count": count,
                            "relationship": "exact",
                        }
                        writer.writerow(row)
                        batch_rows += 1
                        total_rows += 1

                    # Write partial matches directly to file
                    for subset, count in some_matches:
                        subset_list = sorted(subset.intersection(pkg_set))
                        row = {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(pkg_tuple),
                            "subset": str(tuple(subset_list)),
                            "count": count,
                            "relationship": "some",
                        }
                        writer.writerow(row)
                        batch_rows += 1
                        total_rows += 1

                    # Write none case if it exists
                    if none_elements_count > 0:
                        row = {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(pkg_tuple),
                            "subset": "()",
                            "count": none_elements_count,
                            "relationship": "none",
                        }
                        writer.writerow(row)
                        batch_rows += 1
                        total_rows += 1

                    # Free matched lists to reduce memory
                    exact_matches = None
                    some_matches = None

                # Update progress counters
                packages_processed += batch_size_actual

                # Force garbage collection after each batch
                gc.collect()

                # Log progress for this batch
                batch_time = time.time() - batch_start_time
                elapsed = time.time() - start_time
                packages_per_sec = packages_processed / elapsed if elapsed > 0 else 0
                batch_speed = batch_size_actual / batch_time if batch_time > 0 else 0

                remaining = (
                    (total_packages - packages_processed) / packages_per_sec
                    if packages_per_sec > 0
                    else float("inf")
                )

                logging.info(
                    f"Processed batch {batch_start // batch_size + 1} with {batch_size_actual} packages "
                    f"({batch_speed: .1f} pkg/sec in batch, {packages_per_sec: .1f} pkg/sec overall, "
                    f"{packages_processed}/{total_packages} total, est. remaining: {remaining: .1f} sec)",
                )

            # Log completion of template processing
            template_time = time.time() - template_start_time
            template_speed = template_total / template_time if template_time > 0 else 0
            logging.info(
                f"Completed classification for template '{tmpl}' "
                f"processed {template_total} packages in {template_time: .1f} sec "
                f"({template_speed: .1f} pkg/sec for this template)",
            )

    # Log total statistics
    total_time = time.time() - start_time
    overall_speed = packages_processed / total_time if total_time > 0 else 0
    logging.info(
        f"Streaming classification complete: processed {packages_processed} packages "
        f"in {total_time: .1f} sec ({overall_speed: .1f} pkg/sec overall), "
        f"wrote {total_rows} total rows",
    )
