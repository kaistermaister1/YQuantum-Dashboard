#!/usr/bin/env python3
"""Benchmark classic DQI hyperparameter search strategies (QAOA-style 2x2 figure).

Compares search methods over classic DQI settings:
- grid search over (ell, bp_iterations)
- random search over (ell, bp_iterations)
- local hill-climb in the same discrete space
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _default_data_dir() -> Path:
    return _ROOT.parent / "will" / "Travelers" / "docs" / "data" / "YQH26_data"


@dataclass
class BenchmarkRow:
    method: str
    objective: float
    expected_f: float
    expected_s: float
    keep_rate: float
    n_evaluations: int
    seconds: float
    ell: int
    bp_iterations: int


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


def _eval_pair(
    B,
    v,
    *,
    ell: int,
    bp_iterations: int,
    shots: int,
    seed: int,
    strict_ancilla: bool,
):
    from src.dqi_classic_pipeline import run_classic_dqi

    result = run_classic_dqi(
        B,
        v,
        ell=int(ell),
        bp_iterations=int(bp_iterations),
        shots=int(shots),
        seed=int(seed),
        strict_ancilla=bool(strict_ancilla),
    )
    expected_f = float(result.expected_f) if result.expected_f is not None else float("-inf")
    expected_s = float(result.expected_s) if result.expected_s is not None else 0.0
    objective = expected_f
    return {
        "objective": objective,
        "expected_f": expected_f,
        "expected_s": expected_s,
        "keep_rate": float(result.keep_rate),
    }


def _grid_search(B, v, *, ell_values: list[int], bp_values: list[int], shots: int, seed: int, strict_ancilla: bool):
    best = None
    n_eval = 0
    for ell in ell_values:
        for bp in bp_values:
            out = _eval_pair(
                B,
                v,
                ell=ell,
                bp_iterations=bp,
                shots=shots,
                seed=seed + 100 * ell + bp,
                strict_ancilla=strict_ancilla,
            )
            n_eval += 1
            cand = (out["objective"], out["expected_f"], out["expected_s"], out["keep_rate"], ell, bp)
            if best is None or cand[0] > best[0]:
                best = cand
    assert best is not None
    return {
        "objective": float(best[0]),
        "expected_f": float(best[1]),
        "expected_s": float(best[2]),
        "keep_rate": float(best[3]),
        "ell": int(best[4]),
        "bp_iterations": int(best[5]),
        "n_evaluations": int(n_eval),
    }


def _random_search(
    B,
    v,
    *,
    ell_values: list[int],
    bp_values: list[int],
    n_samples: int,
    shots: int,
    seed: int,
    strict_ancilla: bool,
):
    import numpy as np

    rng = np.random.default_rng(int(seed))
    best = None
    for _ in range(int(n_samples)):
        ell = int(rng.choice(ell_values))
        bp = int(rng.choice(bp_values))
        out = _eval_pair(
            B,
            v,
            ell=ell,
            bp_iterations=bp,
            shots=shots,
            seed=seed + 10_000 + ell * 131 + bp,
            strict_ancilla=strict_ancilla,
        )
        cand = (out["objective"], out["expected_f"], out["expected_s"], out["keep_rate"], ell, bp)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None
    return {
        "objective": float(best[0]),
        "expected_f": float(best[1]),
        "expected_s": float(best[2]),
        "keep_rate": float(best[3]),
        "ell": int(best[4]),
        "bp_iterations": int(best[5]),
        "n_evaluations": int(n_samples),
    }


def _hillclimb_search(
    B,
    v,
    *,
    ell_values: list[int],
    bp_values: list[int],
    shots: int,
    seed: int,
    strict_ancilla: bool,
):
    import numpy as np

    ell_values = sorted(set(int(x) for x in ell_values))
    bp_values = sorted(set(int(x) for x in bp_values))
    rng = np.random.default_rng(int(seed))
    cur_ell = int(rng.choice(ell_values))
    cur_bp = int(rng.choice(bp_values))

    cache: dict[tuple[int, int], dict[str, float]] = {}

    def evaluate(ell: int, bp: int):
        key = (int(ell), int(bp))
        if key not in cache:
            cache[key] = _eval_pair(
                B,
                v,
                ell=ell,
                bp_iterations=bp,
                shots=shots,
                seed=seed + 20_000 + ell * 149 + bp,
                strict_ancilla=strict_ancilla,
            )
        return cache[key]

    improved = True
    while improved:
        improved = False
        cur = evaluate(cur_ell, cur_bp)
        best_neighbor = (cur["objective"], cur_ell, cur_bp, cur)
        ell_i = ell_values.index(cur_ell)
        bp_i = bp_values.index(cur_bp)
        neighbors: list[tuple[int, int]] = []
        if ell_i > 0:
            neighbors.append((ell_values[ell_i - 1], cur_bp))
        if ell_i < len(ell_values) - 1:
            neighbors.append((ell_values[ell_i + 1], cur_bp))
        if bp_i > 0:
            neighbors.append((cur_ell, bp_values[bp_i - 1]))
        if bp_i < len(bp_values) - 1:
            neighbors.append((cur_ell, bp_values[bp_i + 1]))

        for ne_ell, ne_bp in neighbors:
            out = evaluate(ne_ell, ne_bp)
            if out["objective"] > best_neighbor[0]:
                best_neighbor = (out["objective"], ne_ell, ne_bp, out)
        if best_neighbor[1] != cur_ell or best_neighbor[2] != cur_bp:
            cur_ell, cur_bp = int(best_neighbor[1]), int(best_neighbor[2])
            improved = True

    final = evaluate(cur_ell, cur_bp)
    return {
        "objective": float(final["objective"]),
        "expected_f": float(final["expected_f"]),
        "expected_s": float(final["expected_s"]),
        "keep_rate": float(final["keep_rate"]),
        "ell": int(cur_ell),
        "bp_iterations": int(cur_bp),
        "n_evaluations": int(len(cache)),
    }


def _plot(rows: list[BenchmarkRow], *, out: Path, title_suffix: str = "", dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    names = [r.method for r in rows]
    colors = {"grid": "#0066CC", "random": "#5BA3D6", "hillclimb": "#2D8C3C"}
    bar_colors = [colors.get(r.method, "#455A64") for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), layout="constrained")

    ax = axes[0, 0]
    ax.bar(names, [r.expected_f for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Best expected <f>")
    ax.set_title("Quality on parity objective")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    ax.bar(names, [r.expected_s for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Best expected <s>")
    ax.set_title("Satisfied checks in post-selected set")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    ax.bar(names, [r.seconds for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Runtime")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.25, which="both")

    ax = axes[1, 1]
    ax.bar(names, [r.n_evaluations for r in rows], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Classic DQI evaluations")
    ax.set_title("Search cost")
    ax.grid(True, axis="y", alpha=0.25)

    suffix = f" | {title_suffix}" if title_suffix else ""
    fig.suptitle(f"Classic DQI hyperparameter benchmark{suffix}", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_default_data_dir())
    ap.add_argument("--package", type=int, default=0)
    ap.add_argument("--subsample-coverages", type=int, default=10, metavar="N")
    ap.add_argument("--subsample-packages", type=int, default=3, metavar="M")
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty-ancilla", action="store_true")
    ap.add_argument("--ell-values", type=int, nargs="*", default=[1, 2, 3], help="Grid/search values for ell")
    ap.add_argument("--bp-values", type=int, nargs="*", default=[1, 2, 3], help="Grid/search values for bp iterations")
    ap.add_argument("--random-samples", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True, help="Output PNG")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title-suffix", type=str, default="")
    args = ap.parse_args()

    _, B, v = _load_parity(args)
    strict_ancilla = not bool(args.allow_dirty_ancilla)
    ell_values = sorted(set(int(x) for x in args.ell_values if int(x) >= 1))
    bp_values = sorted(set(int(x) for x in args.bp_values if int(x) >= 0))
    if not ell_values or not bp_values:
        raise ValueError("ell-values and bp-values must be non-empty")

    rows: list[BenchmarkRow] = []

    def run(name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        rows.append(
            BenchmarkRow(
                method=name,
                objective=float(out["objective"]),
                expected_f=float(out["expected_f"]),
                expected_s=float(out["expected_s"]),
                keep_rate=float(out["keep_rate"]),
                n_evaluations=int(out["n_evaluations"]),
                seconds=float(dt),
                ell=int(out["ell"]),
                bp_iterations=int(out["bp_iterations"]),
            )
        )
        print(
            f"{name:10s}  obj={out['objective']:.6f}  <f>={out['expected_f']:.6f}  <s>={out['expected_s']:.6f}  "
            f"keep={out['keep_rate']:.4f}  eval={out['n_evaluations']}  time={dt:.2f}s  "
            f"ell={out['ell']} bp={out['bp_iterations']}"
        )

    run(
        "grid",
        lambda: _grid_search(
            B,
            v,
            ell_values=ell_values,
            bp_values=bp_values,
            shots=int(args.shots),
            seed=int(args.seed),
            strict_ancilla=strict_ancilla,
        ),
    )
    run(
        "random",
        lambda: _random_search(
            B,
            v,
            ell_values=ell_values,
            bp_values=bp_values,
            n_samples=int(args.random_samples),
            shots=int(args.shots),
            seed=int(args.seed) + 1000,
            strict_ancilla=strict_ancilla,
        ),
    )
    run(
        "hillclimb",
        lambda: _hillclimb_search(
            B,
            v,
            ell_values=ell_values,
            bp_values=bp_values,
            shots=int(args.shots),
            seed=int(args.seed) + 2000,
            strict_ancilla=strict_ancilla,
        ),
    )

    _plot(rows, out=args.out, title_suffix=str(args.title_suffix), dpi=int(args.dpi))
    print(f"Wrote {args.out}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "method",
                    "objective",
                    "expected_f",
                    "expected_s",
                    "keep_rate",
                    "n_evaluations",
                    "seconds",
                    "ell",
                    "bp_iterations",
                ]
            )
            for r in rows:
                w.writerow(
                    [
                        r.method,
                        f"{r.objective:.17g}",
                        f"{r.expected_f:.17g}",
                        f"{r.expected_s:.17g}",
                        f"{r.keep_rate:.17g}",
                        r.n_evaluations,
                        f"{r.seconds:.6f}",
                        r.ell,
                        r.bp_iterations,
                    ]
                )
        print(f"Wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

