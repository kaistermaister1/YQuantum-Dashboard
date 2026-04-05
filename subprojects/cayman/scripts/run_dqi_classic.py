#!/usr/bin/env python3
"""Standalone runner for parity-native classic DQI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_classic_pipeline import run_classic_dqi


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = np.loadtxt(path, delimiter=",")
    return np.asarray(arr)


def _load_B(path: Path) -> np.ndarray:
    B = _load_array(path)
    if B.ndim != 2:
        raise ValueError(f"B must be a 2D matrix, got shape={B.shape}")
    return B


def _load_v(path: Path) -> np.ndarray:
    v = _load_array(path).reshape(-1)
    if v.ndim != 1:
        raise ValueError(f"v must be a vector, got shape={v.shape}")
    return v


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--b-path", type=Path, default=None, help="Path to B matrix (.npy or .csv)")
    src.add_argument(
        "--insurance-data-dir",
        type=Path,
        default=None,
        help="LTM CSV directory (builds parity B,v for one package)",
    )
    ap.add_argument("--v-path", type=Path, default=None, help="Path to v vector (.npy or .csv)")
    ap.add_argument("--package", type=int, default=0, help="Package index when using insurance parity input")
    ap.add_argument("--subsample-coverages", type=int, default=0, help="Optional subsample coverage count")
    ap.add_argument("--subsample-packages", type=int, default=0, help="Optional subsample package count")
    ap.add_argument("--ell", type=int, default=1)
    ap.add_argument("--bp-iterations", type=int, default=1)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--allow-dirty-ancilla",
        action="store_true",
        help="Do not fail on non-zero ancilla measurements (strict mode is default).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "artifacts" / "classic_dqi_result.json",
        help="Output JSON path.",
    )
    return ap.parse_args()


def _load_insurance_parity(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    from src.dqi_insurance_parity import build_insurance_parity_B_rhs
    from src.insurance_model import load_ltm_instance, subsample_problem

    problem = load_ltm_instance(args.insurance_data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(problem, args.subsample_coverages, args.subsample_packages)
    B, v = build_insurance_parity_B_rhs(problem, int(args.package))
    return B, v


def main() -> int:
    args = _parse_args()
    if args.b_path is not None:
        if args.v_path is None:
            raise ValueError("--v-path is required when using --b-path")
        B = _load_B(args.b_path)
        v = _load_v(args.v_path)
    else:
        if args.v_path is not None:
            raise ValueError("--v-path cannot be used with --insurance-data-dir mode")
        B, v = _load_insurance_parity(args)

    result = run_classic_dqi(
        B,
        v,
        ell=int(args.ell),
        bp_iterations=int(args.bp_iterations),
        shots=int(args.shots),
        seed=int(args.seed),
        strict_ancilla=not bool(args.allow_dirty_ancilla),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print("=== Classic DQI Result ===")
    print("B shape:", tuple(int(vv) for vv in np.asarray(B).shape))
    print("v length:", int(np.asarray(v).reshape(-1).shape[0]))
    print("postselected shots:", result.postselected_shots, "/", result.total_shots)
    print("keep rate:", result.keep_rate)
    print("expected <s>:", result.expected_s)
    print("expected <f>:", result.expected_f)
    print("num qubits:", result.num_qubits)
    print("wrote:", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

