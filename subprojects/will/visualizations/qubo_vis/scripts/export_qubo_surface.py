#!/usr/bin/env python3
"""Export one package QUBO block to a text file for the C++ Raylib visualizer.

The viewer lives under ``visualizations/qubo_vis/``; data and Python modules live
under ``Travelers/``. From ``Travelers/code_examples`` with PYTHONPATH including
``src``::

    cd Travelers/code_examples
    # Full instance (20 coverages in YQH26_data): omit subsample flags (both default 0).
    PYTHONPATH=src ./.venv/bin/python ../../visualizations/qubo_vis/scripts/export_qubo_surface.py \\
        --data-dir ../docs/data/YQH26_data --package 0 \\
        -o ../../visualizations/qubo_vis/qubo_surface.txt

    # Smaller demo mesh only:
    #   --subsample-coverages 10 --subsample-packages 3

Or rely on defaults (resolves ``Travelers`` as a sibling of ``visualizations``
under ``subprojects/will/``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


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
    ap.add_argument(
        "--format",
        choices=("v1", "v2"),
        default="v2",
        help="v1 = single Q matrix; v2 = margin diagonal + Q_pen so the viewer can retune λ with [ / ]",
    )
    ap.add_argument(
        "--penalty-lambda",
        type=float,
        default=None,
        metavar="LAM",
        help="Reference λ for v2 (default: same as qubo_block.default_penalty_weight for this package)",
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

    block_margin = build_qubo_block_for_package(problem, m, penalty_weight=0.0)
    n = block_margin.n_vars

    if args.format == "v1":
        block = build_qubo_block_for_package(
            problem,
            m,
            penalty_weight=args.penalty_lambda,
        )
        Q = block.Q
        n = block.n_vars
        lines_v1: list[str] = [
            "# qubo_surface v1 — consumed by visualizations/qubo_vis (Raylib)",
            f"# QUBO BLOCK for package {m} only (column m); instance has M={problem.M} packages.",
            f"{n} {block.n_coverage} {block.n_slack} {block.package_index} {block.constant_offset:.17g}",
        ]
        for i in range(n):
            lines_v1.append(" ".join(f"{Q[i, j]:.17g}" for j in range(n)))
        lines_v1.append("0")
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines_v1) + "\n", encoding="utf-8")
        print(f"Wrote {out}  (n={n}, package={m}, format=v1)")
        return 0

    lam_ref = float(args.penalty_lambda) if args.penalty_lambda is not None else None
    if lam_ref is None:
        block = build_qubo_block_for_package(problem, m)
        lam_ref = float(block.penalty_weight)
    else:
        block = build_qubo_block_for_package(problem, m, penalty_weight=lam_ref)

    if lam_ref <= 0.0:
        print("penalty lambda must be positive for v2 export", file=sys.stderr)
        return 1

    qm = block_margin.Q
    qf = block.Q
    q_pen = (qf - qm) / lam_ref
    const_per_lam = float(block.constant_offset) / lam_ref
    margin_diag = np.diag(qm)

    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# qubo_surface v2 — parametric penalty λ (Q_ij = margin_ij + λ * Q_pen_ij)",
        f"# Package {m}; M={problem.M}. Reference λ={lam_ref:g}; use [ / ] in qubo_vis to rescale penalties.",
        f"{n} {block.n_coverage} {block.n_slack} {block.package_index} {const_per_lam:.17g} {lam_ref:.17g}",
        " ".join(f"{float(margin_diag[i]):.17g}" for i in range(n)),
    ]
    for i in range(n):
        lines.append(" ".join(f"{q_pen[i, j]:.17g}" for j in range(n)))
    lines.append("0")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {out}  (n={n}, package={m}, format=v2, λ_ref={lam_ref:g}, |λ const|/λ={const_per_lam:.6g})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
