#!/usr/bin/env python3
"""Plot classic DQI quality on parity data (histogram + landscape).

Subcommands:
- histogram: one run at fixed (ell, bp_iterations), histogram of sampled score metric.
- landscape: grid over (ell, bp_iterations), heatmaps for expected <f> and keep-rate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_data_dir() -> Path:
    return _ROOT.parent / "will" / "Travelers" / "docs" / "data" / "YQH26_data"


def _load_parity(args: argparse.Namespace):
    from src.dqi_insurance_parity import build_insurance_parity_B_rhs
    from src.insurance_model import load_ltm_instance, subsample_problem

    data_dir = args.data_dir.resolve()
    if not (data_dir / "instance_coverages.csv").is_file():
        raise FileNotFoundError(f"Missing CSVs under {data_dir}")
    problem = load_ltm_instance(data_dir)
    if args.subsample_coverages > 0 and args.subsample_packages > 0:
        problem = subsample_problem(problem, args.subsample_coverages, args.subsample_packages)
    elif args.subsample_coverages > 0 or args.subsample_packages > 0:
        raise ValueError("Set both subsample N and M, or neither")
    if args.package < 0 or args.package >= problem.M:
        raise ValueError(f"package {args.package} out of range M={problem.M}")
    B, v = build_insurance_parity_B_rhs(problem, int(args.package))
    return problem, B, v


def _cmd_histogram(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("Install matplotlib: python -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from exc

    from src.dqi_classic_pipeline import run_classic_dqi, score_f, score_s

    _, B, v = _load_parity(args)
    result = run_classic_dqi(
        B,
        v,
        ell=int(args.ell),
        bp_iterations=int(args.bp_iterations),
        shots=int(args.shots),
        seed=int(args.seed),
        strict_ancilla=not bool(args.allow_dirty_ancilla),
    )

    if not result.postselected_counts:
        raise RuntimeError("No postselected shots available to build histogram")

    metric = str(args.metric).lower()
    score_fn = score_f if metric == "f" else score_s
    score_vals: list[float] = []
    weights: list[float] = []
    best_score = -float("inf")
    for bitstring, count in result.postselected_counts.items():
        val = float(score_fn(B, v, bitstring))
        score_vals.append(val)
        weights.append(float(count))
        best_score = max(best_score, val)

    s_arr = np.asarray(score_vals, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    w_arr /= float(np.sum(w_arr))
    mean_score = float(np.sum(s_arr * w_arr))

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    unique_vals = np.unique(s_arr)
    if unique_vals.size <= 24:
        centers = unique_vals
        probs = np.array([float(np.sum(w_arr[s_arr == val])) for val in centers], dtype=float)
        ax.bar(centers.astype(int), probs, color="#0066CC", edgecolor="white", linewidth=0.7, width=0.8)
    else:
        bins = int(args.bins)
        ax.hist(s_arr, bins=bins, weights=w_arr, color="#0066CC", edgecolor="white", linewidth=0.6, alpha=0.88)
    ax.axvline(mean_score, color="#2D2926", linestyle="-", linewidth=2, label=f"mean {metric} = {mean_score:.4g}")
    ax.axvline(best_score, color="#2D8C3C", linestyle="--", linewidth=2, label=f"best observed {metric} = {best_score:.4g}")
    ax.set_xlabel(f"Post-selected score ({metric})")
    ax.set_ylabel("Fraction of post-selected shots")
    ax.set_title(
        f"Classic DQI histogram | metric={metric} | ell={args.ell} bp={args.bp_iterations} | "
        f"keep={result.keep_rate:.3f} ({result.postselected_shots}/{result.total_shots})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=int(args.dpi))
        print(f"Wrote {args.out}")
    else:
        plt.show()
    plt.close(fig)
    return 0


def _cmd_landscape(args: argparse.Namespace) -> int:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("Install matplotlib: python -m pip install matplotlib", file=sys.stderr)
        raise SystemExit(1) from exc

    from src.dqi_classic_pipeline import run_classic_dqi

    _, B, v = _load_parity(args)

    ell_values = [int(x) for x in args.ell_values if int(x) >= 1]
    bp_values = [int(x) for x in args.bp_values if int(x) >= 0]
    if not ell_values or not bp_values:
        raise ValueError("ell-values and bp-values must be non-empty")

    n_bp = len(bp_values)
    n_ell = len(ell_values)
    z_f = np.full((n_bp, n_ell), np.nan, dtype=float)
    z_keep = np.full((n_bp, n_ell), np.nan, dtype=float)
    grid_rows: list[dict] = []
    k = 0
    total = n_bp * n_ell
    for j, bp in enumerate(bp_values):
        for i, ell in enumerate(ell_values):
            res = run_classic_dqi(
                B,
                v,
                ell=int(ell),
                bp_iterations=int(bp),
                shots=int(args.shots),
                seed=int(args.seed) + k,
                strict_ancilla=not bool(args.allow_dirty_ancilla),
            )
            z_f[j, i] = float(res.expected_f) if res.expected_f is not None else np.nan
            z_keep[j, i] = float(res.keep_rate)
            grid_rows.append(
                {
                    "ell": int(ell),
                    "bp_iterations": int(bp),
                    "expected_f": None if res.expected_f is None else float(res.expected_f),
                    "expected_s": None if res.expected_s is None else float(res.expected_s),
                    "keep_rate": float(res.keep_rate),
                    "postselected_shots": int(res.postselected_shots),
                    "total_shots": int(res.total_shots),
                }
            )
            k += 1
            if k % max(1, total // 10) == 0:
                print(f"  landscape {k}/{total} ...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    im0 = axes[0].imshow(z_f, origin="lower", aspect="auto", cmap="viridis")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.82)
    cb0.set_label("Expected <f>")
    axes[0].set_xticks(np.arange(n_ell), labels=[str(x) for x in ell_values])
    axes[0].set_yticks(np.arange(n_bp), labels=[str(x) for x in bp_values])
    axes[0].set_xlabel("ell")
    axes[0].set_ylabel("bp_iterations")
    axes[0].set_title("Classic DQI mean parity objective")

    im1 = axes[1].imshow(z_keep, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.82)
    cb1.set_label("Keep rate")
    axes[1].set_xticks(np.arange(n_ell), labels=[str(x) for x in ell_values])
    axes[1].set_yticks(np.arange(n_bp), labels=[str(x) for x in bp_values])
    axes[1].set_xlabel("ell")
    axes[1].set_ylabel("bp_iterations")
    axes[1].set_title("Classic DQI post-selection rate")

    fig.suptitle(
        f"Classic DQI landscape | shots={args.shots} | grid={n_ell}x{n_bp}",
        fontsize=10,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=int(args.dpi))
        print(f"Wrote {args.out}")
    else:
        plt.show()
    plt.close(fig)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ell_values": ell_values,
            "bp_values": bp_values,
            "shots": int(args.shots),
            "seed": int(args.seed),
            "rows": grid_rows,
        }
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}")
    return 0


def main() -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=_default_data_dir())
    common.add_argument("--package", type=int, default=0)
    common.add_argument("--subsample-coverages", type=int, default=10, metavar="N")
    common.add_argument("--subsample-packages", type=int, default=3, metavar="M")
    common.add_argument("--shots", type=int, default=256)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--allow-dirty-ancilla", action="store_true")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    ph = sub.add_parser("histogram", parents=[common], help="Score histogram for one classic DQI configuration")
    ph.add_argument("--ell", type=int, default=1)
    ph.add_argument("--bp-iterations", type=int, default=1)
    ph.add_argument("--metric", choices=("f", "s"), default="f")
    ph.add_argument("--bins", type=int, default=24)
    ph.add_argument("--out", type=Path, default=None, help="Output PNG path (omit to show GUI)")
    ph.add_argument("--dpi", type=int, default=150)
    ph.set_defaults(_run=_cmd_histogram)

    pl = sub.add_parser("landscape", parents=[common], help="Heatmap over ell × bp_iterations")
    pl.add_argument("--ell-values", type=int, nargs="*", default=[1, 2, 3, 4])
    pl.add_argument("--bp-values", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    pl.add_argument("--out", type=Path, default=None, help="Output PNG path (omit to show GUI)")
    pl.add_argument("--out-json", type=Path, default=None, help="Optional JSON dump of the landscape grid")
    pl.add_argument("--dpi", type=int, default=150)
    pl.set_defaults(_run=_cmd_landscape)

    args = ap.parse_args()
    return int(args._run(args))


if __name__ == "__main__":
    raise SystemExit(main())

