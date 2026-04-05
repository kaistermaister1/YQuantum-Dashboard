#!/usr/bin/env python3

import gc
import logging
import math
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple


def _process_vehicle_batch(
    batch: List[Dict[str, List[str]]],
    all_options_dict: Dict[str, frozenset],
) -> Dict[str, Counter]:
    """
    Process a batch of vehicles to find option subsets.
    This function is designed to be called by parallel workers.

    Args:
        batch: List of vehicle dictionaries for this batch
        all_options_dict: Dict mapping template_name -> frozenset of allowed options

    Returns:
        Dict mapping template_name -> Counter of (frozenset, count) for this batch
    """
    # Create counters for each template
    counters = {tmpl: Counter() for tmpl in all_options_dict}

    # Process each vehicle in the batch
    for vehicle in batch:
        # Get the vehicle's options
        vehicle_opts = set(vehicle.get("options", []))

        # For each template
        for tmpl, allowed_opts in all_options_dict.items():
            # Filter options for current template (fast set intersection)
            relevant_opts = vehicle_opts.intersection(allowed_opts)

            # Only count if there are relevant options
            if relevant_opts:
                # Use frozenset for hashability
                counters[tmpl][frozenset(relevant_opts)] += 1

    return counters


def get_unique_package_subsets_parallel(
    vehicles: List[Dict[str, List[str]]],
    all_options: Dict[str, frozenset],
    num_processes: int = None,
    batch_size: int = 10000,
) -> Dict[str, List[Tuple[frozenset, int]]]:
    """
    Parallelized version of get_unique_package_subsets.
    For each template, count unique subsets of vehicle options that belong to that template.

    Args:
        vehicles: List of vehicle dictionaries, each with "options" key
        all_options: Dict mapping template_name -> frozenset of allowed options
        num_processes: Number of parallel processes to use (default: CPU count)
        batch_size: Number of vehicles to process in each batch

    Returns:
        Dict mapping template_name -> list of (option_subset, count) tuples
    """
    if num_processes is None:
        num_processes = min(os.cpu_count() or 4, 8)  # Limit to 8 processes max

    logging.info(
        f"Starting parallel processing with {num_processes} processes, batch size: {batch_size}",
    )

    # Split vehicles into batches
    total_vehicles = len(vehicles)
    num_batches = math.ceil(total_vehicles / batch_size)
    batches = [
        vehicles[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)
    ]

    start_time = time.time()
    logging.info(f"Processing {total_vehicles} vehicles in {num_batches} batches")

    # Process batches in parallel
    combined_counters = {tmpl: Counter() for tmpl in all_options}

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(_process_vehicle_batch, batch, all_options): i
            for i, batch in enumerate(batches)
        }

        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_counters = future.result()

                # Combine the counters from this batch with the overall counters
                for tmpl, counter in batch_counters.items():
                    combined_counters[tmpl].update(counter)

                elapsed = time.time() - start_time
                processed = min((batch_idx + 1) * batch_size, total_vehicles)
                vehicles_per_sec = processed / elapsed
                remaining = (
                    (total_vehicles - processed) / vehicles_per_sec
                    if vehicles_per_sec > 0
                    else "unknown"
                )

                logging.info(
                    f"Completed batch {batch_idx + 1}/{num_batches}, "
                    f"processed {processed}/{total_vehicles} vehicles "
                    f"({vehicles_per_sec: .1f} vehicles/sec, est. remaining: {remaining: .1f} sec)",
                )

            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                raise

    # Convert combined counters to sorted lists for output
    result = {}
    for tmpl, counter in combined_counters.items():
        # Sort by count in descending order for potential optimization later
        result[tmpl] = [
            (option_set, count) for option_set, count in counter.most_common()
        ]
        logging.info(f"Template '{tmpl}' has {len(result[tmpl])} unique option subsets")

    return result


def process_package_for_classification(
    tmpl: str,
    pkg_options: List[str],
    template_subsets: List[Tuple[frozenset, int]],
    total_vehicles: int,
) -> Tuple[str, Dict[str, Any]]:
    """Process a single package for classification

    This is a helper function designed for parallel processing.
    It must be defined at the module level (not nested inside another function)
    to ensure proper serialization/deserialization by multiprocessing.

    Args:
        tmpl: The template name
        pkg_options: List of options in this package
        template_subsets: List of (subset, count) tuples for this template
        total_vehicles: Total number of vehicles processed

    Returns:
        Tuple of (package_key, package_info_dict)
    """
    # Convert package options to frozenset for faster set operations
    pkg_set = frozenset(pkg_options)
    # Create a unique key for this package
    pkg_key = ",".join(sorted(pkg_options))

    # Track counts for this package
    exact_match_count = 0
    some_elements_count = 0

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
    none_elements_count = total_vehicles - exact_match_count - some_elements_count

    # Return the classification for this package
    return (
        pkg_key,
        {
            "package": pkg_options,
            "exact": exact_match_count,
            "some": some_elements_count,
            "none": none_elements_count,
            "total": total_vehicles,
            "subsets": {
                "exact": exact_subsets,
                "intersecting": intersecting_subsets,
            },
        },
    )


def process_package_batch(
    tmpl: str,
    pkg_batch: List[List[str]],
    template_subsets: List[Tuple[frozenset, int]],
    total_vehicles: int,
) -> Dict[str, Dict[str, Any]]:
    """Process a batch of packages for classification

    This is a helper function designed for parallel processing.
    It must be defined at the module level (not nested inside another function)
    to ensure proper serialization/deserialization by multiprocessing.

    Args:
        tmpl: The template name
        pkg_batch: List of package option lists to process in this batch
        template_subsets: List of (subset, count) tuples for this template
        total_vehicles: Total number of vehicles processed

    Returns:
        Dict mapping package_key -> package_info_dict for all packages in the batch
    """
    batch_results = {}

    # Pre-compute template subsets as sets for faster lookup
    # This optimization helps when we have many subsets
    template_subset_sets = [s for s, _ in template_subsets]
    template_subset_counts = [c for _, c in template_subsets]

    # Process each package in the batch
    for pkg_idx, pkg_options in enumerate(pkg_batch):
        try:
            # Create package set and key
            pkg_set = frozenset(pkg_options)
            pkg_key = ",".join(sorted(pkg_options))

            # Initialize counts and containers
            exact_match_count = 0
            some_elements_count = 0
            exact_subsets = []
            intersecting_subsets = []

            # Process each subset for this package
            for subset_idx, subset in enumerate(template_subset_sets):
                count = template_subset_counts[subset_idx]

                if subset == pkg_set:
                    # Exact match
                    exact_match_count += count
                    exact_subsets.append((subset, count))
                elif subset.intersection(pkg_set):
                    # Some match
                    some_elements_count += count
                    intersecting_subsets.append((subset, count))

            # Calculate vehicles with none of the options
            none_elements_count = (
                total_vehicles - exact_match_count - some_elements_count
            )

            # Store the results
            batch_results[pkg_key] = {
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

            # Free memory as we go
            if pkg_idx % 500 == 0 and pkg_idx > 0:
                # Aggressive garbage collection for memory-intensive workloads
                gc.collect()

        except Exception as e:
            logging.error(f"Error processing package in batch: {e}")
            # Continue with the rest of the batch instead of failing completely
            continue

    return batch_results


def classify_subset_per_package_parallel(
    pkg_candidates: Dict[str, List[List[str]]],
    subsets: Dict[str, List[Tuple[frozenset, int]]],
    total_vehicles: int,
    num_processes: int = None,
    max_batch_size: int = 5000,
    min_batch_size: int = 1500,
) -> Dict[str, Dict[str, Dict]]:
    """
    Parallelized version of classify_subset_per_package.
    For each package candidate, track detailed relationships with observed option subsets.

    Args:
        pkg_candidates: Dict mapping template_name -> list of package option lists
        subsets: Dict mapping template_name -> list of (option_subset, count) tuples
        total_vehicles: Total number of vehicles processed
        num_processes: Number of parallel processes to use (default: CPU count)
        max_batch_size: Maximum number of packages to process in a single batch
        min_batch_size: Minimum number of packages to process in a single batch

    Returns:
        Dict mapping template_name -> dict mapping package_key -> package info
    """
    # Force process spawning to ensure clean memory state for each process
    multiprocessing.set_start_method("spawn", force=True)

    if num_processes is None:
        num_processes = min(os.cpu_count() or 4, 8)  # Limit to 8 processes max

    result = {}

    # Calculate the total workload for all templates
    total_workload = sum(
        len(pkgs) * len(subsets.get(tmpl, [])) for tmpl, pkgs in pkg_candidates.items()
    )
    logging.info(
        f"Total classification workload: {total_workload} package-subset comparisons",
    )

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

        # Threshold for serial vs. parallel based on workload
        # For small numbers of packages, just use the serial implementation
        serial_threshold = 5000  # packages
        if total_packages < serial_threshold:
            logging.info(
                f"Small number of packages ({total_packages}), using serial implementation for '{tmpl}'",
            )
            # Use inline implementation for speed
            for pkg_options in packages:
                pkg_set = frozenset(pkg_options)
                pkg_key = ",".join(sorted(pkg_options))

                exact_match_count = 0
                some_elements_count = 0

                exact_subsets = []
                intersecting_subsets = []

                for subset, count in template_subsets:
                    if subset == pkg_set:
                        exact_match_count += count
                        exact_subsets.append((subset, count))
                    elif subset.intersection(pkg_set):
                        some_elements_count += count
                        intersecting_subsets.append((subset, count))

                none_elements_count = (
                    total_vehicles - exact_match_count - some_elements_count
                )

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

            logging.info(
                f"Completed serial classification for {total_packages} packages in template '{tmpl}'",
            )
            continue

        # For larger workloads, use parallel with adaptive batching
        logging.info(
            f"Classifying {total_packages} package candidates for "
            f"template '{tmpl}' in parallel with {num_processes} processes",
        )

        # Calculate optimal batch size based on workload complexity and available processes
        # Dynamic batch sizing: smaller batches for larger subset complexity
        subset_complexity = len(template_subsets)
        # Adjust batch size inversely to subset complexity
        dynamic_batch_size = max(
            min_batch_size,
            min(max_batch_size, int(10000 / (subset_complexity + 1) / num_processes)),
        )

        # Create batches with the calculated optimal size
        batches = [
            packages[i : i + dynamic_batch_size]
            for i in range(0, len(packages), dynamic_batch_size)
        ]
        num_batches = len(batches)

        logging.info(
            f"Using {num_batches} batches with ~{dynamic_batch_size} packages per batch",
        )
        logging.info(f"Subset complexity: {subset_complexity} unique subsets")

        # Store progress metrics for adaptive batch sizing
        batch_timings = []
        packages_processed = 0
        last_batch_completion = time.time()

        try:
            # Process packages in parallel with batching
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                # Submit initial batch jobs in manageable chunks to avoid overwhelming memory
                batch_chunks = [
                    batches[i : i + num_processes]
                    for i in range(0, len(batches), num_processes)
                ]

                for chunk_idx, chunk in enumerate(batch_chunks):
                    if chunk_idx > 0:
                        # Force garbage collection between large chunks
                        gc.collect()

                    # Submit batch jobs for this chunk
                    futures = [
                        executor.submit(
                            process_package_batch,
                            tmpl,
                            pkg_batch,
                            template_subsets,
                            total_vehicles,
                        )
                        for pkg_batch in chunk
                    ]

                    # Process results as they complete
                    for batch_idx, future in enumerate(as_completed(futures)):
                        try:
                            current_time = time.time()
                            batch_results = future.result()

                            # Update results
                            result[tmpl].update(batch_results)

                            # Count packages processed in this batch
                            batch_packages = len(batch_results)
                            packages_processed += batch_packages

                            # Track timing for this batch
                            batch_time = current_time - last_batch_completion
                            batch_timings.append((batch_packages, batch_time))
                            last_batch_completion = current_time

                            # Calculate current speed and estimated time remaining
                            elapsed = time.time() - start_time
                            packages_per_sec = (
                                packages_processed / elapsed if elapsed > 0 else 0
                            )
                            remaining = (
                                (total_packages - packages_processed) / packages_per_sec
                                if packages_per_sec > 0
                                else float("inf")
                            )

                            # Log progress more frequently to track performance
                            current_batch = chunk_idx * num_processes + batch_idx + 1
                            if (
                                current_batch % max(1, num_batches // 20) == 0
                                or current_batch == num_batches
                            ):
                                logging.info(
                                    f"Processed {packages_processed}/{total_packages} packages for '{tmpl}' "
                                    f"({packages_per_sec: .1f} pkg/sec, est. remaining: {remaining: .1f} sec)",
                                )

                        except Exception as e:
                            logging.error(
                                f"Error processing batch {chunk_idx * num_processes + batch_idx} "
                                f"for template '{tmpl}': {e}",
                            )
                            # Continue with other batches instead of failing completely
                            logging.exception("Batch processing exception details:")

        except Exception as e:
            logging.error(f"Error in parallel processing for template '{tmpl}': {e}")
            logging.exception("Exception details:")

        finally:
            # Force garbage collection after template processing
            gc.collect()

        # Report completion and performance stats
        end_time = time.time()
        total_elapsed = end_time - start_time
        avg_speed = packages_processed / total_elapsed if total_elapsed > 0 else 0
        logging.info(
            f"Completed parallel classification for {packages_processed}/{total_packages} packages "
            f"in template '{tmpl}' in {total_elapsed: .1f} seconds "
            f"(average: {avg_speed: .1f} pkg/sec)",
        )

    return result
