#!/usr/bin/env python3

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def export_flattened_data(
    flattened_data: List[Dict[str, Any]],
    output_path: str,
    format: str = "csv",
    include_metadata: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Export the flattened data to various formats.

    Args:
        flattened_data: List of dictionaries representing the flattened data
        output_path: Path where to save the exported data
        format: Output format ('csv', 'json', 'parquet', 'excel')
        include_metadata: Whether to include metadata in the export
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata if requested
    if include_metadata and metadata:
        if format == "json":
            data_to_export = {
                "metadata": metadata,
                "data": flattened_data,
            }
        else:
            # For other formats, we might need to add metadata rows or separate files
            logging.info(
                f"Metadata will be written to a separate file for {format} format",
            )
            metadata_path = output_path.with_suffix(f".metadata{output_path.suffix}")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            data_to_export = flattened_data
    else:
        data_to_export = flattened_data

    # Convert tuples to lists for JSON serialization
    if format == "json":

        def clean_data(item):
            if isinstance(item, tuple):
                return list(item)
            elif isinstance(item, dict):
                return {k: clean_data(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [clean_data(i) for i in item]
            else:
                return item

        if isinstance(data_to_export, list):
            data_to_export = [clean_data(item) for item in data_to_export]
        elif isinstance(data_to_export, dict):
            data_to_export = {k: clean_data(v) for k, v in data_to_export.items()}

    # Export based on the requested format
    if format.lower() == "csv":
        # Convert tuples to strings for CSV
        for row in flattened_data:
            for key, value in row.items():
                if isinstance(value, tuple):
                    row[key] = ",".join(value) if value else ""

        # Write to CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=flattened_data[0].keys() if flattened_data else [],
            )
            writer.writeheader()
            writer.writerows(flattened_data)

        logging.info(f"Exported {len(flattened_data)} rows to CSV: {output_path}")

    elif format.lower() == "json":
        with open(output_path, "w") as f:
            json.dump(data_to_export, f, indent=2)

        logging.info(f"Exported data to JSON: {output_path}")

    elif format.lower() == "parquet":
        try:
            pass  # Imports moved to module level
        except ImportError:
            logging.error(
                "PyArrow is required for Parquet export. Please install via 'pip install pyarrow'",
            )
            raise

        # Convert to DataFrame for Parquet export
        df = pd.DataFrame(flattened_data)

        # Handle tuple conversions
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, tuple)).any():
                df[col] = df[col].apply(
                    lambda x: list(x) if isinstance(x, tuple) else x,
                )

        # Write to Parquet
        df.to_parquet(output_path, compression="snappy")
        logging.info(f"Exported {len(df)} rows to Parquet: {output_path}")

    elif format.lower() == "excel":
        # Convert to DataFrame for Excel export
        df = pd.DataFrame(flattened_data)

        # Handle tuple conversions for Excel
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, tuple)).any():
                df[col] = df[col].apply(
                    lambda x: ",".join(x) if isinstance(x, tuple) and x else "",
                )

        # Write to Excel
        df.to_excel(output_path, index=False)
        logging.info(f"Exported {len(df)} rows to Excel: {output_path}")

    else:
        raise ValueError(f"Unsupported export format: {format}")

    return output_path
