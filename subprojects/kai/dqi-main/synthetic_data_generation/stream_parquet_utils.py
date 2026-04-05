#!/usr/bin/env python3

import csv
import gc
import logging
import multiprocessing
import os
import time
from collections import Counter
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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
                # Explicitly clear memory
                gc.collect()

        # Yield remaining vehicles in the last batch
        if batch:
            yield batch


# Create an alias for get_unique_package_subsets_streaming to avoid circular imports
get_unique_package_subsets_streaming_parquet = None

# First import the function from stream_utils
try:
    from stream_utils import get_unique_package_subsets_streaming

    get_unique_package_subsets_streaming_parquet = get_unique_package_subsets_streaming
except ImportError:

    def get_unique_package_subsets_streaming_parquet(
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

            # Force garbage collection after each batch
            gc.collect()

        # Convert counters to sorted lists for output
        result = {}
        for tmpl, counter in counters.items():
            # Sort by count in descending order for potential optimization
            result[tmpl] = [
                (option_set, count) for option_set, count in counter.most_common()
            ]
            logging.info(
                f"Template '{tmpl}' has {len(result[tmpl])} unique option subsets",
            )

        # Final garbage collection
        gc.collect()

        return result


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

        # Force garbage collection after each batch
        gc.collect()

    # Convert counters to sorted lists for output
    result = {}
    for tmpl, counter in counters.items():
        # Sort by count in descending order for potential optimization
        result[tmpl] = [
            (option_set, count) for option_set, count in counter.most_common()
        ]
        logging.info(f"Template '{tmpl}' has {len(result[tmpl])} unique option subsets")

    # Final garbage collection
    gc.collect()

    return result


def _batch_to_parquet(
    rows: List[Dict],
    output_file: str,
    batch_index: int = 0,
    schema: Optional[pa.Schema] = None,
) -> pa.Schema:
    """
    Write a batch of rows to a Parquet file.

    Args:
        rows: List of dictionaries to write
        output_file: Path to the output Parquet file
        batch_index: Index of this batch (0 for first batch, used to determine if appending)
        schema: PyArrow schema for the data (will be inferred from first batch if None)

    Returns:
        The PyArrow schema used for writing
    """
    if not rows:
        return schema if schema else pa.schema([])

    try:
        # Convert data to pandas DataFrame for easier schema handling
        df = pd.DataFrame(rows)

        # Convert tuple columns to string for proper parquet serialization
        for col in df.columns:
            if df[col].dtype == "object":
                if all(isinstance(x, tuple) for x in df[col] if x is not None):
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df, schema=schema)

        # Get schema before any potential cleanup
        result_schema = table.schema

        # Write to Parquet (append if not the first batch)
        if batch_index == 0:
            pq.write_table(
                table,
                output_file,
                compression="snappy",  # Good balance of speed and compression
                use_dictionary=True,  # Enable dictionary encoding
                write_statistics=True,  # Include statistics for better query performance
            )
        else:
            # Append to existing file
            pq.write_to_dataset(
                table,
                root_path=os.path.dirname(output_file),
                partition_cols=[],  # No partitioning for simplicity
                basename_template=os.path.basename(output_file).split(".")[0]
                + "_{i}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
            )

        # Clean up to free memory
        del df
        del table

        # Force garbage collection
        gc.collect()

        return result_schema

    except Exception as e:
        logging.error(f"Error writing batch to Parquet: {e}")
        import traceback

        logging.error(traceback.format_exc())

        # Clean up in case of error too
        gc.collect()

        # Return schema or empty schema
        return schema if schema else pa.schema([])


def classify_subset_per_package_streaming_parquet(
    pkg_candidates: Dict[str, List[List[str]]],
    subsets: Dict[str, List[Tuple[frozenset, int]]],
    total_vehicles: int,
    output_file: str,
    batch_size: int = 1000,  # Smaller default batch size for better memory management
    row_group_size: int = 10000,
) -> None:
    """
    Memory-efficient version of classify_subset_per_package that processes packages
    in batches and streams results directly to a Parquet file.

    Args:
        pkg_candidates: Dict mapping template_name -> list of package option lists
        subsets: Dict mapping template_name -> list of (option_subset, count) tuples
        total_vehicles: Total number of vehicles processed
        output_file: Path to the output Parquet file
        batch_size: Number of packages to process in each batch
        row_group_size: Size of Parquet row groups for efficient reading later

    Returns:
        None - results are written directly to the output file
    """
    # Force process spawning mode for better memory isolation
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, just continue
        pass

    # Total expected packages and subset comparisons for tracking progress
    total_packages = sum(len(packages) for packages in pkg_candidates.values())
    total_subsets = sum(len(subset_list) for subset_list in subsets.values())
    packages_processed = 0
    start_time = time.time()
    schema = None
    batch_index = 0
    total_rows = 0

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear any existing file to avoid append issues
    if os.path.exists(output_file):
        os.remove(output_file)

    # Force garbage collection before starting
    gc.collect()

    logging.info(
        f"Starting streaming classification to Parquet with {total_packages} packages "
        f"and {total_subsets} unique option subsets",
    )

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
            batch_rows_list = []  # Store rows for this batch

            # Track performance metrics for this batch
            batch_subset_comparisons = 0

            # Process each package in this batch
            try:
                for pkg_options in pkg_batch:
                    # Convert package options to frozenset for faster set operations
                    pkg_set = frozenset(pkg_options)
                    # Create a unique key for this package
                    pkg_key = ",".join(sorted(pkg_options))

                    # Track counts for this package
                    exact_match_count = 0
                    some_elements_count = 0

                    # Process batch in chunks to manage memory
                    exact_matches = []
                    some_matches = []

                    # Check each observed subset against this package
                    for subset, count in template_subsets:
                        batch_subset_comparisons += 1

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

                    # Add all rows for this package to our batch
                    # Exact matches
                    for subset, count in exact_matches:
                        subset_list = sorted(subset)
                        batch_rows_list.append(
                            {
                                "template": tmpl,
                                "package_key": pkg_key,
                                "package": str(
                                    tuple(pkg_options),
                                ),  # Convert to string for Parquet
                                "subset": str(
                                    tuple(subset_list),
                                ),  # Convert to string for Parquet
                                "count": count,
                                "relationship": "exact",
                            },
                        )
                        total_rows += 1

                    # Partial matches
                    for subset, count in some_matches:
                        subset_list = sorted(subset.intersection(pkg_set))
                        batch_rows_list.append(
                            {
                                "template": tmpl,
                                "package_key": pkg_key,
                                "package": str(tuple(pkg_options)),
                                "subset": str(tuple(subset_list)),
                                "count": count,
                                "relationship": "some",
                            },
                        )
                        total_rows += 1

                    # None case
                    if none_elements_count > 0:
                        batch_rows_list.append(
                            {
                                "template": tmpl,
                                "package_key": pkg_key,
                                "package": str(tuple(pkg_options)),
                                "subset": "()",  # Empty tuple as string
                                "count": none_elements_count,
                                "relationship": "none",
                            },
                        )
                        total_rows += 1

                    # Clear memory for this package
                    exact_matches = None
                    some_matches = None

                    # Write to parquet if we've reached the row group size
                    if len(batch_rows_list) >= row_group_size:
                        schema = _batch_to_parquet(
                            batch_rows_list,
                            output_file,
                            batch_index,
                            schema,
                        )
                        batch_index += 1
                        batch_rows_list = []
                        # Force garbage collection
                        gc.collect()

                # Write any remaining rows in this batch
                if batch_rows_list:
                    schema = _batch_to_parquet(
                        batch_rows_list,
                        output_file,
                        batch_index,
                        schema,
                    )
                    batch_index += 1
                    batch_rows_list = []

                # Update progress counters
                packages_processed += batch_size_actual

                # Force garbage collection after each batch
                gc.collect()

                # Log progress for this batch
                batch_time = time.time() - batch_start_time
                elapsed = time.time() - start_time
                packages_per_sec = packages_processed / elapsed if elapsed > 0 else 0
                batch_speed = batch_size_actual / batch_time if batch_time > 0 else 0
                comparisons_per_sec = (
                    batch_subset_comparisons / batch_time if batch_time > 0 else 0
                )

                remaining = (
                    (total_packages - packages_processed) / packages_per_sec
                    if packages_per_sec > 0
                    else float("inf")
                )

                logging.info(
                    f"Processed batch {batch_start // batch_size + 1} with {batch_size_actual} packages "
                    f"({batch_speed: .1f} pkg/sec in batch, {packages_per_sec: .1f} pkg/sec overall, "
                    f"{comparisons_per_sec: .1f} subset comparisons/sec, "
                    f"{packages_processed}/{total_packages} total, est. remaining: {remaining: .1f} sec)",
                )

            except Exception as e:
                logging.error(
                    f"Error processing batch {batch_start // batch_size + 1}: {e}",
                )
                import traceback

                logging.error(traceback.format_exc())
                # Continue with next batch to avoid total failure

            # Clean up memory
            batch_rows_list = None

        # Log completion of template processing
        template_time = time.time() - template_start_time
        template_speed = template_total / template_time if template_time > 0 else 0
        logging.info(
            f"Completed classification for template '{tmpl}' "
            f"processed {template_total} packages in {template_time: .1f} sec "
            f"({template_speed: .1f} pkg/sec for this template)",
        )

    # Finalize the Parquet file
    logging.info("Finalizing Parquet file...")

    # Log total statistics
    total_time = time.time() - start_time
    overall_speed = packages_processed / total_time if total_time > 0 else 0
    logging.info(
        f"Streaming classification complete: processed {packages_processed} packages "
        f"in {total_time: .1f} sec ({overall_speed: .1f} pkg/sec overall), "
        f"wrote {total_rows} total rows to {output_file}",
    )

    # Final garbage collection
    gc.collect()


def stream_flattened_data_parquet(
    classifications: Dict[str, Dict[str, Dict]],
    total_vehicles: int,
    output_file: str,
    batch_size: int = 1000,
    row_group_size: int = 10000,
) -> int:
    """
    Stream flattened classification data directly to a Parquet file without
    keeping everything in memory.

    Args:
        classifications: The output of classify_subset_per_package
        total_vehicles: The total number of vehicles
        output_file: Path to the output Parquet file
        batch_size: Maximum number of packages to process per batch
        row_group_size: Size of Parquet row groups for efficient reading later

    Returns:
        Total number of rows written
    """
    total_rows = 0
    start_time = time.time()
    total_packages = sum(len(packages) for packages in classifications.values())
    packages_processed = 0
    batch_index = 0
    schema = None
    batch_rows = []

    # Process each template
    for tmpl, packages in classifications.items():
        logging.info(
            f"Streaming data for template '{tmpl}' with {len(packages)} packages",
        )

        # Process packages in batches
        pkg_items = list(packages.items())
        for batch_start in range(0, len(pkg_items), batch_size):
            batch_end = min(batch_start + batch_size, len(pkg_items))
            batch = pkg_items[batch_start:batch_end]

            for pkg_key, pkg_info in batch:
                pkg_options = pkg_info["package"]
                pkg_set = frozenset(pkg_options)
                n_processed = 0

                # Process exact match subsets
                for subset, count in pkg_info["subsets"]["exact"]:
                    subset_list = sorted(subset)
                    batch_rows.append(
                        {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(tuple(pkg_options)),
                            "subset": str(tuple(subset_list)),
                            "count": count,
                            "relationship": "exact",
                        },
                    )

                    n_processed += count
                    total_rows += 1

                # Process intersecting subsets
                for subset, count in pkg_info["subsets"]["intersecting"]:
                    subset_list = sorted(subset.intersection(pkg_set))
                    batch_rows.append(
                        {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(tuple(pkg_options)),
                            "subset": str(tuple(subset_list)),
                            "count": count,
                            "relationship": "some",
                        },
                    )

                    n_processed += count
                    total_rows += 1

                # Add entry for vehicles with none of the options in this package
                if pkg_info["none"] > 0:
                    batch_rows.append(
                        {
                            "template": tmpl,
                            "package_key": pkg_key,
                            "package": str(tuple(pkg_options)),
                            "subset": "()",
                            "count": pkg_info["none"],
                            "relationship": "none",
                        },
                    )

                    n_processed += pkg_info["none"]
                    total_rows += 1

                # Assert that we've processed all vehicles
                assert (
                    n_processed == total_vehicles
                ), f"Processed {n_processed} vehicles, expected {total_vehicles} for package {pkg_key}"

                # Write to parquet if we've reached the row group size
                if len(batch_rows) >= row_group_size:
                    schema = _batch_to_parquet(
                        batch_rows,
                        output_file,
                        batch_index,
                        schema,
                    )
                    batch_index += 1
                    batch_rows = []
                    # Force garbage collection
                    gc.collect()

            # Update progress counters
            packages_processed += len(batch)

            # Log progress periodically
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

            # Force garbage collection after each batch
            gc.collect()

    # Write any remaining rows
    if batch_rows:
        _batch_to_parquet(batch_rows, output_file, batch_index, schema)

    # Final garbage collection
    gc.collect()

    logging.info(f"Streamed data contains {total_rows} rows")
    return total_rows
