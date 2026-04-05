#!/usr/bin/env python3
"""Export one package QUBO block to a text file for the C++ Raylib visualizer.

The viewer lives under ``visualizations/qubo_vis/``; data and Python modules live
under ``Travelers/``. From ``Travelers/code_examples`` with PYTHONPATH including
``src``::

    cd Travelers/code_examples
    PYTHONPATH=src ./.venv/bin/python ../../visualizations/qubo_vis/scripts/export_qubo_surface.py \\
        --data-dir ../docs/data/YQH26_data \\
        --package 0 --subsample-coverages 10 --subsample-packages 3 \\
        -o ../../visualizations/qubo_vis/qubo_surface.txt

Or rely on defaults (resolves ``Travelers`` as a sibling of ``visualizations``
under ``subprojects/will/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _will_root() -> Path:
    """``subprojects/will`` — parent of ``Travelers/`` and ``visualizations/``."""
    return Path(__file__).resolve().parents[3]


def _travelers_bundle() -> Path:
    return _will_root() / "Travelers"


def _qubo_vis_root() -> Path:
    """Directory containing ``scripts/`` and default ``qubo_surface.txt``."""
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Export QUBO Q matrix for qubo_vis")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=_travelers_bundle() / "docs" / "data" / "YQH26_data",
        help="Directory with instance_*.csv",
    )
    ap.add_argument("--package", type=int, default=0, help="Package index m for the block")
    ap.add_argument(
        "--subsample-coverages",
        type=int,
        default=0,
        metavar="N",
        help="If >0, keep first N coverages (smaller mesh for demos)",
    )
    ap.add_argument(
        "--subsample-packages",
        type=int,
        default=0,
        metavar="M",
        help="If >0, keep first M packages when subsampling",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_qubo_vis_root() / "qubo_surface.txt",
        help="Output path for the surface file",
    )
    args = ap.parse_args()

    ce_src = _travelers_bundle() / "code_examples" / "src"
    sys.path.insert(0, str(ce_src))

    from insurance_model import load_ltm_instance, subsample_problem
    from qubo_block import build_qubo_block_for_package

    data_dir = args.data_dir.resolve()
    if not (data_dir / "instance_coverages.csv").is_file():
        print(f"Missing CSVs under {data_dir}", file=sys.stderr)
        return 1

    problem = load_ltm_instance(data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(
            problem,
            args.subsample_coverages,
            args.subsample_packages,
        )
    elif args.subsample_coverages > 0 or args.subsample_packages > 0:
        print("Set both --subsample-coverages and --subsample-packages, or neither.", file=sys.stderr)
        return 1

    m = args.package
    if m < 0 or m >= problem.M:
        print(f"package {m} out of range M={problem.M}", file=sys.stderr)
        return 1

    block = build_qubo_block_for_package(problem, m)
    Q = block.Q
    n = block.n_vars

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# qubo_surface v1 — consumed by visualizations/qubo_vis (Raylib)",
        f"# QUBO BLOCK for package {m} only (column m); instance has M={problem.M} packages — not the full (N*M)x(N*M) Q.",
        f"{n} {block.n_coverage} {block.n_slack} {block.package_index} {block.constant_offset:.17g}",
    ]
    for i in range(n):
        lines.append(" ".join(f"{Q[i, j]:.17g}" for j in range(n)))
    lines.append("0")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}  (n={n}, package={m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
