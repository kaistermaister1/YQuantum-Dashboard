#!/usr/bin/env python3
"""Build a Classical / QAOA / DQI comparison table from HEURISTICS/run_summaries.csv.

QAOA is split into two columns (**SPSA** and **COBYLA**) for the same (n, m, p, λ).

Filters to fixed m, p, and a chosen λ vector (defaults: m=3, p=1, and a reference λ
per n from recorded runs). Rows:

- Optimal price / margin — ``best_profit`` (Travelers contribution margin).
- Runtime — ``runtime_sec``.
- Cost — for QAOA columns, ``num_samples_total`` (shots). Classical and DQI show an em
  dash or N/A when missing.

DQI is not written by ``heuristics.py`` yet; those cells show ``N/A`` until
``algorithm=dqi`` rows exist.

Requires: matplotlib.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_SOLUTIONS = Path(__file__).resolve().parent

# Reference λ vectors for m=3 QAOA rows in run_summaries.csv (one λ per package block).
# n=10 uses the 515/669/921 triple from recent local runs; an older COBYLA p=1 row used
# a different λ — pass --lambda '[687.14, …]' to target that row explicitly.
DEFAULT_LAMBDA_BY_N: dict[int, tuple[float, ...]] = {
    5: (515.3601024000001, 669.5972399999999, 921.6943125),
    7: (515.3601024000001, 669.5972399999999, 921.6943125),
    10: (515.3601024000001, 669.5972399999999, 921.6943125),
}

CLASSICAL_GREEN = "#2D8C3C"
QAOA_BLUE = "#0066CC"
DQI_BLUE = "#00356B"
HEADER_BG = "#E5E7EB"
ROW_ALT = "#FAFAFA"


def _i(row: dict[str, str], key: str) -> int | None:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return int(float(v))


def _f(row: dict[str, str], key: str) -> float | None:
    v = (row.get(key) or "").strip()
    if v == "":
        return None
    return float(v)


def parse_lambda_tuple(raw: str) -> tuple[float, ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        v = ast.literal_eval(raw)
        if isinstance(v, (list, tuple)):
            return tuple(float(x) for x in v)
        if isinstance(v, (int, float)):
            return (float(v),)
    except (SyntaxError, ValueError, TypeError):
        return None
    return None


def lambdas_close(a: tuple[float, ...], b: tuple[float, ...], *, rtol: float = 1e-9, atol: float = 1e-5) -> bool:
    if len(a) != len(b):
        return False
    return all(math.isclose(x, y, rel_tol=rtol, abs_tol=atol) for x, y in zip(a, b, strict=True))


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_classical(rows: list[dict[str, str]], *, n: int, m: int) -> dict[str, str] | None:
    candidates = [
        r
        for r in rows
        if (r.get("algorithm") or "").strip() == "classical"
        and _i(r, "N_local") == n
        and _i(r, "M_blocks") == m
    ]
    return candidates[0] if candidates else None


def pick_qaoa(
    rows: list[dict[str, str]],
    *,
    n: int,
    m: int,
    p: int,
    lambda_target: tuple[float, ...],
    optimizer: str,
) -> dict[str, str] | None:
    opt_norm = optimizer.strip().lower()
    for r in rows:
        if (r.get("algorithm") or "").strip() != "qaoa":
            continue
        if _i(r, "N_local") != n or _i(r, "M_blocks") != m or _i(r, "p") != p:
            continue
        if (r.get("optimizer") or "").strip().lower() != opt_norm:
            continue
        lam = parse_lambda_tuple(r.get("lambda") or "")
        if lam is None or not lambdas_close(lam, lambda_target):
            continue
        return r
    return None


def pick_dqi(
    rows: list[dict[str, str]],
    *,
    n: int,
    m: int,
    p: int,
    lambda_target: tuple[float, ...],
) -> dict[str, str] | None:
    for r in rows:
        if (r.get("algorithm") or "").strip() != "dqi":
            continue
        if _i(r, "N_local") != n or _i(r, "M_blocks") != m or _i(r, "p") != p:
            continue
        lam = parse_lambda_tuple(r.get("lambda") or "")
        if lam is None or not lambdas_close(lam, lambda_target):
            continue
        return r
    return None


def fmt_profit(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}"


def fmt_runtime_sec(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 100:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def fmt_shots(v: int | None) -> str:
    if v is None:
        return "—"
    return f"{v:,}"


def row_values(r: dict[str, str] | None) -> tuple[str, str, str]:
    if r is None:
        return ("N/A", "N/A", "N/A")
    profit = _f(r, "best_profit")
    rt = _f(r, "runtime_sec")
    shots = _i(r, "num_samples_total")
    algo = (r.get("algorithm") or "").strip()
    cost = fmt_shots(shots) if algo in ("qaoa", "dqi") and shots is not None else "—"
    return (fmt_profit(profit), fmt_runtime_sec(rt), cost)


def build_table_matrix(
    rows: list[dict[str, str]],
    *,
    n: int,
    m: int,
    p: int,
    lambda_target: tuple[float, ...],
) -> tuple[list[list[str]], dict[str, Any]]:
    classical = pick_classical(rows, n=n, m=m)
    qaoa_spsa = pick_qaoa(rows, n=n, m=m, p=p, lambda_target=lambda_target, optimizer="spsa")
    qaoa_cobyla = pick_qaoa(rows, n=n, m=m, p=p, lambda_target=lambda_target, optimizer="cobyla")
    dqi = pick_dqi(rows, n=n, m=m, p=p, lambda_target=lambda_target)

    c_vals = row_values(classical)
    qs_vals = row_values(qaoa_spsa)
    qc_vals = row_values(qaoa_cobyla)
    d_vals = row_values(dqi)

    body = [
        [
            "Optimal price / margin\n(CSV: best_profit)",
            c_vals[0],
            qs_vals[0],
            qc_vals[0],
            d_vals[0],
        ],
        ["Runtime (s)", c_vals[1], qs_vals[1], qc_vals[1], d_vals[1]],
        [
            "Cost (shots)\n(CSV: num_samples_total)",
            c_vals[2],
            qs_vals[2],
            qc_vals[2],
            d_vals[2],
        ],
    ]
    meta = {
        "classical_run_id": (classical or {}).get("run_id", ""),
        "qaoa_spsa_run_id": (qaoa_spsa or {}).get("run_id", ""),
        "qaoa_cobyla_run_id": (qaoa_cobyla or {}).get("run_id", ""),
        "dqi_run_id": (dqi or {}).get("run_id", ""),
    }
    return body, meta


def lambda_display(t: tuple[float, ...]) -> str:
    inner = ", ".join(f"{x:.6g}" for x in t)
    return f"[{inner}]"


def plot_table(
    body: list[list[str]],
    *,
    title_lines: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    col_labels = [
        "Classical\n(ILP baseline)",
        "QAOA\nSPSA",
        "QAOA\nCOBYLA",
        "DQI",
    ]
    headers = ["Metric"] + col_labels
    cell_text = [[row[0], row[1], row[2], row[3], row[4]] for row in body]

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=8)
            if col == 0:
                cell.set_facecolor(HEADER_BG)
                cell.get_text().set_color("black")
            elif col == 1:
                cell.set_facecolor(CLASSICAL_GREEN)
                cell.get_text().set_color("white")
            elif col in (2, 3):
                cell.set_facecolor(QAOA_BLUE)
                cell.get_text().set_color("white")
            elif col == 4:
                cell.set_facecolor(DQI_BLUE)
                cell.get_text().set_color("white")
        else:
            if col == 0:
                cell.set_facecolor("#F9FAFB")
                cell.set_text_props(fontsize=8)
            else:
                cell.set_facecolor(ROW_ALT if (row - 1) % 2 == 0 else "white")

    fig.text(0.5, 0.92, title_lines[0], ha="center", fontsize=11, weight="bold")
    y0 = 0.86
    for i, line in enumerate(title_lines[1:]):
        fig.text(0.5, y0 - i * 0.045, line, ha="center", fontsize=8, color="#374151")

    plt.subplots_adjust(top=0.78, bottom=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summaries",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv",
    )
    parser.add_argument("--n", type=int, default=5, help="Subsample size N_local (coverages).")
    parser.add_argument("--m", type=int, default=3, help="M_blocks (packages).")
    parser.add_argument("--p", type=int, default=1, help="QAOA depth p (QAOA rows only).")
    parser.add_argument(
        "--lambda",
        dest="lambda_str",
        default=None,
        help='λ list as in CSV, e.g. \'[515.36, 669.6, 921.69]\'. Default: built-in preset for --n if available.',
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_SOLUTIONS / "HEURISTICS" / "plots",
    )
    parser.add_argument(
        "--output-name",
        default="algorithm_comparison_m3_p1.png",
        help="Output PNG filename under --out-dir.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write the same table as CSV (Metric + 4 columns).",
    )
    args = parser.parse_args()

    if args.lambda_str:
        lam = parse_lambda_tuple(args.lambda_str)
        if lam is None:
            raise SystemExit("Could not parse --lambda as a number or list of numbers.")
        lambda_target = lam
    else:
        if args.n not in DEFAULT_LAMBDA_BY_N:
            raise SystemExit(
                f"No default λ for n={args.n}; pass --lambda explicitly. "
                f"Presets exist for n in {sorted(DEFAULT_LAMBDA_BY_N)}."
            )
        lambda_target = DEFAULT_LAMBDA_BY_N[args.n]

    rows = load_rows(args.summaries)
    body, meta = build_table_matrix(
        rows,
        n=args.n,
        m=args.m,
        p=args.p,
        lambda_target=lambda_target,
    )

    title_lines = [
        f"Algorithm comparison (n={args.n}, m={args.m}, p={args.p})",
        f"Fixed λ = {lambda_display(lambda_target)}  ·  QAOA columns: SPSA and COBYLA (same λ, local)",
        "Optimal price / margin = best_profit (contribution margin, USD). Cost = total measurement shots (num_samples_total) for QAOA and DQI when recorded.",
        f"classical: {meta['classical_run_id'] or '—'}  ·  qaoa_spsa: {meta['qaoa_spsa_run_id'] or '—'}  ·  qaoa_cobyla: {meta['qaoa_cobyla_run_id'] or '—'}  ·  dqi: {meta['dqi_run_id'] or '—'}",
    ]

    out_path = args.out_dir / args.output_name
    plot_table(body, title_lines=title_lines, out_path=out_path)
    print(f"Wrote {out_path}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Classical", "QAOA_SPSA", "QAOA_COBYLA", "DQI"])
            for row in body:
                w.writerow([row[0].replace("\n", " "), row[1], row[2], row[3], row[4]])
        print(f"Wrote {args.csv_out}")


if __name__ == "__main__":
    main()
