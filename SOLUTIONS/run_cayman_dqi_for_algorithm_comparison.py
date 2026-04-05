#!/usr/bin/env python3
"""Run Cayman DQI (Qiskit BP-decoder path) for a Travelers slice and append ``run_summaries.csv``.

For each package block (m = 0..M-1), builds ``build_qubo_block_for_package`` with the same
per-package λ as the QAOA comparison rows, runs :func:`run_dqi_with_details` with
``use_bp_decoder=True`` and ``execution='local'`` (Qiskit Aer), then stitches coverage
bits into an N×M matrix and evaluates total contribution margin with the LTM formula.

After appending one ``algorithm=dqi`` row (with λ matching ``plot_visualizations.py algorithm-comparison``
presets), regenerate ``algorithm_comparison_n5_m3_p1.png`` unless ``--no-plot``.

Requires: ``pip install qiskit qiskit-aer`` (plus cayman deps: numpy, scipy, pulp).

Example::

    python SOLUTIONS/run_cayman_dqi_for_algorithm_comparison.py --n 5 --m 3 --p 1
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CAYMAN_ROOT = REPO_ROOT / "subprojects" / "cayman"
TRAVELERS_DATA = REPO_ROOT / "subprojects" / "will" / "Travelers" / "docs" / "data" / "YQH26_data"
SUMMARIES_CSV = REPO_ROOT / "SOLUTIONS" / "HEURISTICS" / "run_summaries.csv"
SOLUTIONS_DIR = REPO_ROOT / "SOLUTIONS" / "HEURISTICS" / "solutions"
HISTORIES_DIR = REPO_ROOT / "SOLUTIONS" / "HEURISTICS" / "histories"
PLOTS_DIR = REPO_ROOT / "SOLUTIONS" / "HEURISTICS" / "plots"

DEFAULT_LAMBDA_BY_N: dict[int, tuple[float, ...]] = {
    5: (515.3601024000001, 669.5972399999999, 921.6943125),
    7: (515.3601024000001, 669.5972399999999, 921.6943125),
    10: (515.3601024000001, 669.5972399999999, 921.6943125),
}


def _ensure_cayman_path() -> None:
    s = str(CAYMAN_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def total_margin_usd(problem, x_nm: np.ndarray) -> float:
    N, M = problem.N, problem.M
    beta = problem.price_sensitivity_beta
    total = 0.0
    for m in range(M):
        delta_m = problem.get_discount(m)
        for i in range(N):
            if float(x_nm[i, m]) > 0.5:
                cov = problem.coverages[i]
                alpha_im = problem.get_affinity(i, m)
                total += (
                    cov.price
                    * cov.contribution_margin_pct
                    * (1 - delta_m)
                    * cov.take_rate
                    * alpha_im
                    * (1 + beta * delta_m)
                )
    return float(total)


def _lambda_csv_tuple(t: tuple[float, ...]) -> str:
    inner = ", ".join(repr(x) for x in t)
    return f"[{inner}]"


def _strip_prior_qiskit_dqi_rows(rows: list[dict[str, str]], *, n: int, m: int, p: int) -> list[dict[str, str]]:
    """Remove earlier cayman Qiskit BP rows for the same (n,m,p) so re-runs stay idempotent."""
    out: list[dict[str, str]] = []
    for r in rows:
        if (r.get("algorithm") or "").strip() != "dqi":
            out.append(r)
            continue
        if (r.get("optimizer") or "").strip().lower() != "qiskit-bp":
            out.append(r)
            continue
        try:
            ni = int(float(r.get("N_local") or ""))
            mi = int(float(r.get("M_blocks") or ""))
            pi = int(float(r.get("p") or ""))
        except ValueError:
            out.append(r)
            continue
        if ni == n and mi == m and pi == p:
            continue
        out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--p", type=int, default=1)
    ap.add_argument("--shots", type=int, default=512, help="Shots per package block.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bp-iterations", type=int, default=1)
    ap.add_argument("--data-dir", type=Path, default=TRAVELERS_DATA)
    ap.add_argument("--summaries", type=Path, default=SUMMARIES_CSV)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--lambda-str",
        default=None,
        help="Override λ list string for CSV (default: preset for --n).",
    )
    args = ap.parse_args()

    if args.n not in DEFAULT_LAMBDA_BY_N and args.lambda_str is None:
        raise SystemExit(f"No default λ for n={args.n}; pass --lambda-str.")

    lam_t = (
        tuple(float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", args.lambda_str))
        if args.lambda_str
        else DEFAULT_LAMBDA_BY_N[args.n]
    )
    if len(lam_t) != args.m:
        raise SystemExit(f"Expected {args.m} λ values for m={args.m}, got {len(lam_t)}.")
    lam_csv = _lambda_csv_tuple(lam_t)

    _ensure_cayman_path()
    from src.insurance_model import load_ltm_instance, subsample_problem, solve_ilp
    from src.qubo_block import build_qubo_block_for_package
    from src.run_dqi import run_dqi_with_details

    if not args.data_dir.is_dir():
        raise SystemExit(f"Travelers data dir not found: {args.data_dir}")

    problem = subsample_problem(load_ltm_instance(args.data_dir), args.n, args.m)
    ilp = solve_ilp(problem)
    classical_opt = float(ilp["objective"])

    x_mat = np.zeros((problem.N, problem.M), dtype=float)
    total_rt = 0.0
    total_shots = 0
    total_feas = 0
    max_q = 0

    for pkg in range(problem.M):
        block = build_qubo_block_for_package(problem, pkg, penalty_weight=lam_t[pkg])
        max_q = max(max_q, block.n_vars)
        t0 = time.perf_counter()
        best_x, _value, meta = run_dqi_with_details(
            block,
            args.p,
            shots=args.shots,
            seed=args.seed + pkg,
            statistic="best",
            mixer="h",
            execution="local",
            use_bp_decoder=True,
            bp_iterations=int(args.bp_iterations),
            insurance_parity=(problem, pkg),
            max_qubits=256,
        )
        total_rt += time.perf_counter() - t0
        st = meta.run_result.stats_at_best
        total_shots += int(st.shots)
        if st.post_selection_rate is not None:
            total_feas += int(round(float(st.shots) * float(st.post_selection_rate)))
        cov = int(block.n_coverage)
        for i in range(cov):
            x_mat[i, pkg] = float(best_x[i])

    best_profit = total_margin_usd(problem, x_mat)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"dqi_qiskit_bp_n{args.n}_m{args.m}_p{args.p}_{ts}"

    SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)
    HISTORIES_DIR.mkdir(parents=True, exist_ok=True)
    sol_path = SOLUTIONS_DIR / f"{run_id}_solution.csv"
    hist_path = HISTORIES_DIR / f"{run_id}_history.csv"

    with sol_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["coverage_index", "package_index", "x"])
        for m in range(problem.M):
            for i in range(problem.N):
                w.writerow([i, m, int(x_mat[i, m] > 0.5)])
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "note"])
        w.writerow([run_id, "cayman DQI Qiskit BP path; one stitched matrix; see run_summaries notes"])

    notes = (
        "cayman=subprojects/cayman; execution=Qiskit Aer BP-decoder (use_bp_decoder); "
        f"per-package λ match QAOA row; shots_per_block={args.shots}; "
        f"classical_opt_profit={classical_opt:.12g} from solve_ilp on same subsample"
    )
    new_row = {
        "run_id": run_id,
        "algorithm": "dqi",
        "optimizer": "qiskit-bp",
        "seed": str(args.seed),
        "N_local": str(args.n),
        "M_blocks": str(args.m),
        "n_total": str(args.n * args.m),
        "p": str(args.p),
        "lambda": lam_csv,
        "runtime_sec": str(total_rt),
        "best_profit": str(best_profit),
        "classical_opt_profit": str(classical_opt),
        "num_samples_total": str(total_shots),
        "num_samples_feasible": str(total_feas),
        "num_samples_postselected": "",
        "num_objective_evals": str(args.m),
        "num_qubits": str(max_q),
        "circuit_depth": "",
        "two_qubit_gate_count": "",
        "solution_path": str(sol_path.relative_to(REPO_ROOT)),
        "history_path": str(hist_path.relative_to(REPO_ROOT)),
        "notes": notes,
    }

    with args.summaries.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise SystemExit("run_summaries.csv missing header")
        rows = list(reader)

    rows = _strip_prior_qiskit_dqi_rows(rows, n=args.n, m=args.m, p=args.p)
    rows.append(new_row)

    with args.summaries.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"Appended {run_id} to {args.summaries}")
    print(f"  best_profit={best_profit:.4f}  classical_opt={classical_opt:.4f}  runtime={total_rt:.3f}s  shots={total_shots}")

    if not args.no_plot:
        import subprocess

        plot_script = REPO_ROOT / "SOLUTIONS" / "plot_visualizations.py"
        out_name = f"algorithm_comparison_n{args.n}_m{args.m}_p{args.p}.png"
        subprocess.run(
            [
                sys.executable,
                str(plot_script),
                "algorithm-comparison",
                "--n",
                str(args.n),
                "--m",
                str(args.m),
                "--p",
                str(args.p),
                "--summaries",
                str(args.summaries),
                "--out-dir",
                str(PLOTS_DIR),
                "--output-name",
                out_name,
                "--csv-out",
                str(PLOTS_DIR / out_name.replace(".png", ".csv")),
            ],
            check=True,
        )
        print(f"Regenerated {PLOTS_DIR / out_name}")


if __name__ == "__main__":
    main()
