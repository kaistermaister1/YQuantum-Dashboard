#!/usr/bin/env python3
"""Unified plotting CLI for SOLUTIONS visualizations."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

REPO_SOLUTIONS = Path(__file__).resolve().parent

CLASSICAL_GREEN = "#2D8C3C"
QAOA_BLUE = "#0066CC"
QAOA_COBYLA = "#0066CC"
QAOA_SPSA = "#00356B"
DQI_BLUE = "#00356B"
TRAVELERS_RED = "#C41230"
YALE_BLUE = "#00356B"
HEADER_BG = "#E5E7EB"
ROW_ALT = "#FAFAFA"
SHELL_BG = "#F3F4F6"
SERIES_COLORS = [QAOA_BLUE, YALE_BLUE, CLASSICAL_GREEN, TRAVELERS_RED]
ERR_KW = {"elinewidth": 1.0, "capthick": 1.0, "ecolor": "#374151"}
RUN_TS = re.compile(r"_(\d{8}T\d{6}Z)$")

DEFAULT_LAMBDA_BY_N: dict[int, tuple[float, ...]] = {
    5: (515.3601024000001, 669.5972399999999, 921.6943125),
    7: (515.3601024000001, 669.5972399999999, 921.6943125),
    10: (515.3601024000001, 669.5972399999999, 921.6943125),
}


def _i(row: dict[str, str], key: str) -> int | None:
    value = (row.get(key) or "").strip()
    if value == "":
        return None
    return int(float(value))


def _f(row: dict[str, str], key: str) -> float | None:
    value = (row.get(key) or "").strip()
    if value == "":
        return None
    return float(value)


def _s(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_lambda_tuple(raw: str) -> tuple[float, ...] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, (list, tuple)):
            return tuple(float(item) for item in value)
        if isinstance(value, (int, float)):
            return (float(value),)
    except (SyntaxError, TypeError, ValueError):
        return None
    return None


def lambdas_close(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-5,
) -> bool:
    if len(left) != len(right):
        return False
    return all(math.isclose(a, b, rel_tol=rtol, abs_tol=atol) for a, b in zip(left, right, strict=True))


def lambda_display(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def is_local_qaoa(row: dict[str, str]) -> bool:
    if _s(row, "algorithm") != "qaoa":
        return False
    if _s(row, "optimizer").lower() not in {"cobyla", "spsa"}:
        return False
    return "execution_target=local" in (row.get("notes") or "")


def run_timestamp(run_id: str) -> str:
    match = RUN_TS.search(run_id or "")
    return match.group(1) if match else ""


def run_id_sort_key(run_id: str) -> str:
    match = re.search(r"_(\d{8}T\d{6}Z)$", run_id.strip())
    return match.group(1) if match else run_id


def latest_per_m(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    chosen: dict[int, dict[str, str]] = {}
    for row in rows:
        m = _i(row, "M_blocks")
        if m is None:
            continue
        current = chosen.get(m)
        if current is None or run_timestamp(row.get("run_id", "")) > run_timestamp(current.get("run_id", "")):
            chosen[m] = row
    return chosen


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit requires positive x and y")
    beta, log_a = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(log_a)), float(beta)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def crossing_power_vs_constant(A: float, beta: float, constant: float, x_lo: float, x_hi: float) -> float | None:
    def fn(x_value: float) -> float:
        return A * (x_value**beta) - constant

    xs = np.linspace(x_lo, x_hi, 512)
    fs = np.array([fn(float(value)) for value in xs])
    for index in range(len(xs) - 1):
        if fs[index] == 0.0:
            return float(xs[index])
        if fs[index] * fs[index + 1] < 0:
            lo, hi = float(xs[index]), float(xs[index + 1])
            flo, fhi = fs[index], fs[index + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fmid = fn(mid)
                if abs(fmid) < 1e-9 * (abs(flo) + abs(fhi) + 1e-12):
                    return mid
                if flo * fmid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            return 0.5 * (lo + hi)
    return None


def crossing_power_vs_linear(
    A: float, beta: float, intercept: float, slope: float, x_lo: float, x_hi: float
) -> float | None:
    def fn(x_value: float) -> float:
        return A * (x_value**beta) - (intercept + slope * x_value)

    xs = np.linspace(x_lo, x_hi, 512)
    fs = np.array([fn(float(value)) for value in xs])
    for index in range(len(xs) - 1):
        if fs[index] == 0.0:
            return float(xs[index])
        if fs[index] * fs[index + 1] < 0:
            lo, hi = float(xs[index]), float(xs[index + 1])
            flo, fhi = fs[index], fs[index + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fmid = fn(mid)
                if abs(fmid) < 1e-9 * (abs(flo) + abs(fhi) + 1e-12):
                    return mid
                if flo * fmid <= 0:
                    hi, fhi = mid, fmid
                else:
                    lo, flo = mid, fmid
            return 0.5 * (lo + hi)
    return None


def add_common_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--summaries", type=Path, default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_SOLUTIONS / "HEURISTICS" / "plots")


def pick_classical(rows: list[dict[str, str]], *, n: int, m: int) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if _s(row, "algorithm") == "classical" and _i(row, "N_local") == n and _i(row, "M_blocks") == m
    ]
    return matches[0] if matches else None


def pick_qaoa(
    rows: list[dict[str, str]],
    *,
    n: int,
    m: int,
    p: int,
    lambda_target: tuple[float, ...],
    optimizer: str,
) -> dict[str, str] | None:
    optimizer = optimizer.strip().lower()
    for row in rows:
        if _s(row, "algorithm") != "qaoa":
            continue
        if _i(row, "N_local") != n or _i(row, "M_blocks") != m or _i(row, "p") != p:
            continue
        if _s(row, "optimizer").lower() != optimizer:
            continue
        lam = parse_lambda_tuple(row.get("lambda") or "")
        if lam is None or not lambdas_close(lam, lambda_target):
            continue
        return row
    return None


def pick_dqi(
    rows: list[dict[str, str]],
    *,
    n: int,
    m: int,
    p: int,
    lambda_target: tuple[float, ...],
) -> dict[str, str] | None:
    for row in rows:
        if _s(row, "algorithm") != "dqi":
            continue
        if _i(row, "N_local") != n or _i(row, "M_blocks") != m or _i(row, "p") != p:
            continue
        lam = parse_lambda_tuple(row.get("lambda") or "")
        if lam is None or not lambdas_close(lam, lambda_target):
            continue
        return row
    return None


def fmt_profit(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"


def fmt_runtime_sec(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def fmt_shots(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def row_values(row: dict[str, str] | None) -> tuple[str, str, str]:
    if row is None:
        return ("N/A", "N/A", "N/A")
    profit = _f(row, "best_profit")
    runtime = _f(row, "runtime_sec")
    shots = _i(row, "num_samples_total")
    cost = fmt_shots(shots) if _s(row, "algorithm") in {"qaoa", "dqi"} and shots is not None else "-"
    return (fmt_profit(profit), fmt_runtime_sec(runtime), cost)


def build_algorithm_comparison_table(
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
    s_vals = row_values(qaoa_spsa)
    cby_vals = row_values(qaoa_cobyla)
    d_vals = row_values(dqi)

    body = [
        ["Optimal price / margin\n(CSV: best_profit)", c_vals[0], s_vals[0], cby_vals[0], d_vals[0]],
        ["Runtime (s)", c_vals[1], s_vals[1], cby_vals[1], d_vals[1]],
        ["Cost (shots)\n(CSV: num_samples_total)", c_vals[2], s_vals[2], cby_vals[2], d_vals[2]],
    ]
    meta = {
        "classical_run_id": (classical or {}).get("run_id", ""),
        "qaoa_spsa_run_id": (qaoa_spsa or {}).get("run_id", ""),
        "qaoa_cobyla_run_id": (qaoa_cobyla or {}).get("run_id", ""),
        "dqi_run_id": (dqi or {}).get("run_id", ""),
    }
    return body, meta


def plot_algorithm_comparison_table(body: list[list[str]], *, title_lines: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    headers = ["Metric", "Classical\n(ILP baseline)", "QAOA\nSPSA", "QAOA\nCOBYLA", "DQI"]
    table = ax.table(
        cellText=[[row[0], row[1], row[2], row[3], row[4]] for row in body],
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
    base_y = 0.86
    for index, line in enumerate(title_lines[1:]):
        fig.text(0.5, base_y - index * 0.045, line, ha="center", fontsize=8, color="#374151")

    plt.subplots_adjust(top=0.78, bottom=0.08)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cmd_algorithm_comparison(args: argparse.Namespace) -> None:
    if args.lambda_str:
        lambda_target = parse_lambda_tuple(args.lambda_str)
        if lambda_target is None:
            raise SystemExit("Could not parse --lambda as a number or list of numbers.")
    else:
        if args.n not in DEFAULT_LAMBDA_BY_N:
            raise SystemExit(f"No default lambda for n={args.n}; pass --lambda explicitly.")
        lambda_target = DEFAULT_LAMBDA_BY_N[args.n]

    rows = load_rows(args.summaries)
    body, meta = build_algorithm_comparison_table(rows, n=args.n, m=args.m, p=args.p, lambda_target=lambda_target)
    title_lines = [
        f"Algorithm comparison (n={args.n}, m={args.m}, p={args.p})",
        f"Fixed lambda = {lambda_display(lambda_target)}  ·  QAOA columns: SPSA and COBYLA (same lambda, local)",
        "Optimal price / margin = best_profit (contribution margin, USD). Cost = total measurement shots for QAOA and DQI when recorded.",
        f"classical: {meta['classical_run_id'] or '-'}  ·  qaoa_spsa: {meta['qaoa_spsa_run_id'] or '-'}  ·  qaoa_cobyla: {meta['qaoa_cobyla_run_id'] or '-'}  ·  dqi: {meta['dqi_run_id'] or '-'}",
    ]
    out_path = args.out_dir / args.output_name
    plot_algorithm_comparison_table(body, title_lines=title_lines, out_path=out_path)
    print(f"Wrote {out_path}")

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Metric", "Classical", "QAOA_SPSA", "QAOA_COBYLA", "DQI"])
            for row in body:
                writer.writerow([row[0].replace("\n", " "), row[1], row[2], row[3], row[4]])
        print(f"Wrote {args.csv_out}")


def load_qaoa_rows(path: Path, *, exclude_optimizers: frozenset[str] | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    skip = exclude_optimizers or frozenset()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _s(row, "algorithm") != "qaoa":
                continue
            if _s(row, "optimizer").lower() in skip:
                continue
            rows.append(row)
    return rows


def heuristics_group_key(row: dict[str, str]) -> tuple[str, int, int, int]:
    optimizer = _s(row, "optimizer") or "unknown"
    n = _i(row, "N_local")
    m = _i(row, "M_blocks")
    p = _i(row, "p")
    if n is None or m is None or p is None:
        raise ValueError(f"Missing N_local/M_blocks/p in row {row.get('run_id')}")
    return (optimizer, n, m, p)


def format_lambda_display(raw: str, max_len: int = 96) -> str:
    raw = raw.strip()
    if not raw:
        return "-"
    try:
        values = ast.literal_eval(raw)
        if isinstance(values, (list, tuple)) and values:
            text = "[" + ", ".join(f"{float(item):.6g}" for item in values) + "]"
            return text if len(text) <= max_len else text[: max_len - 3] + "..."
    except (SyntaxError, TypeError, ValueError):
        pass
    return raw if len(raw) <= max_len else raw[: max_len - 3] + "..."


def collect_settings_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    seeds: set[str] = set()
    lambdas: set[str] = set()
    run_ids: list[str] = []
    for row in rows:
        run_id = _s(row, "run_id")
        if run_id:
            run_ids.append(run_id)
        seed = _s(row, "seed")
        if seed:
            seeds.add(seed)
        lam = _s(row, "lambda")
        if lam:
            lambdas.add(lam)
    if len(lambdas) == 1:
        lambda_text = format_lambda_display(next(iter(lambdas)))
    elif not lambdas:
        lambda_text = "-"
    else:
        lambda_text = " | ".join(format_lambda_display(raw, max_len=72) for raw in sorted(lambdas))
    return {
        "seeds_display": ", ".join(sorted(seeds, key=lambda item: (len(item), item))) if seeds else "-",
        "lambda_display": lambda_text,
        "run_ids": run_ids,
    }


def aggregate_heuristics_groups(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[heuristics_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda item: (item[1], item[2], item[3], item[0])):
        optimizer, n, m, p = key
        group = buckets[key]
        best_profit = np.array([_f(row, "best_profit") for row in group], dtype=float)
        classical_profit = np.array([_f(row, "classical_opt_profit") for row in group], dtype=float)
        runtime = np.array([_f(row, "runtime_sec") for row in group], dtype=float)
        shots = np.array([_i(row, "num_samples_total") or 0 for row in group], dtype=float)
        feasible = np.array([_i(row, "num_samples_feasible") or 0 for row in group], dtype=float)
        gates = np.array([_i(row, "two_qubit_gate_count") or 0 for row in group], dtype=float)

        ratio = best_profit / classical_profit
        ratio_std = float(np.std(ratio, ddof=1)) if len(group) > 1 else 0.0
        runtime_std = float(np.std(runtime, ddof=1)) if len(group) > 1 else 0.0
        if math.isnan(ratio_std):
            ratio_std = 0.0
        if math.isnan(runtime_std):
            runtime_std = 0.0

        total_feasible = float(np.sum(feasible))
        total_shots = float(np.sum(shots))
        if total_shots > 0:
            pooled = total_feasible / total_shots
            feasible_se = math.sqrt(max(pooled * (1.0 - pooled) / total_shots, 0.0))
        else:
            pooled = float("nan")
            feasible_se = 0.0

        settings = collect_settings_from_rows(group)
        n_total_values = sorted({_i(row, "n_total") for row in group if _i(row, "n_total") is not None})
        output.append(
            {
                "optimizer": optimizer,
                "N_local": n,
                "M_blocks": m,
                "p": p,
                "n_runs": len(group),
                "ratio_mean": float(np.mean(ratio)),
                "ratio_std": ratio_std,
                "rt_mean": float(np.mean(runtime)),
                "rt_std": runtime_std,
                "feas_pooled": pooled,
                "feas_se": feasible_se,
                "gates_mean": float(np.mean(gates)) if len(gates) else 0.0,
                "seeds_display": settings["seeds_display"],
                "lambda_display": settings["lambda_display"],
                "run_ids": settings["run_ids"],
                "n_total_display": ", ".join(str(value) for value in n_total_values) if n_total_values else "-",
            }
        )
    return output


def heuristics_label(group: dict[str, Any]) -> str:
    return f"{group['optimizer'].upper()}\nn={group['N_local']} m={group['M_blocks']} p={group['p']}"


def plot_heuristics_summary(groups: list[dict[str, Any]], out_path: Path) -> None:
    count = len(groups)
    x = np.arange(count)
    colors = [SERIES_COLORS[index % len(SERIES_COLORS)] for index in range(count)]
    labels = [heuristics_label(group) for group in groups]
    label_size = 7 if count > 5 else 8

    fig = plt.figure(figsize=(12, 9.5 + 0.35 * count))
    fig.patch.set_facecolor("white")
    grid = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.0, 1.0, max(0.55, min(1.35, 0.22 * (count + 1)))],
        hspace=0.38, wspace=0.28, top=0.93, bottom=0.06, left=0.07, right=0.98,
    )
    fig.suptitle("QAOA heuristics summary (from run_summaries.csv)", fontsize=12, y=0.97)

    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[1, 0] = fig.add_subplot(grid[1, 0])
    axes[1, 1] = fig.add_subplot(grid[1, 1])

    ax = axes[0, 0]
    ax.set_facecolor(SHELL_BG)
    means = [group["ratio_mean"] for group in groups]
    errs = [group["ratio_std"] for group in groups]
    ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="#1F2937", linewidth=0.5, error_kw=ERR_KW)
    ax.axhline(1.0, color=CLASSICAL_GREEN, linestyle="--", linewidth=1.2, label="ratio = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_size)
    ax.set_ylabel("best / classical optimum")
    ax.set_title("Profit recovery (error bars: SD across runs in group)", fontsize=9)
    ax.set_ylim(0, max(1.05, max((mean + (err or 0.0) for mean, err in zip(means, errs, strict=True)), default=1.0) * 1.12))
    ax.legend(loc="lower right", fontsize=7, framealpha=0.95)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)

    ax = axes[0, 1]
    ax.set_facecolor(SHELL_BG)
    means = [group["feas_pooled"] for group in groups]
    errs = [group["feas_se"] for group in groups]
    ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="#1F2937", linewidth=0.5, error_kw=ERR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_size)
    ax.set_ylabel("feasible / total (pooled)")
    ax.set_title("Feasible shot rate (error bars: pooled binomial SE)", fontsize=9)
    ax.set_ylim(0, max(0.25, max((mean + err for mean, err in zip(means, errs, strict=True) if not math.isnan(mean)), default=0.2) * 1.15))
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)

    ax = axes[1, 0]
    ax.set_facecolor(SHELL_BG)
    means = [group["rt_mean"] / 60.0 for group in groups]
    errs = [group["rt_std"] / 60.0 for group in groups]
    ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor="#1F2937", linewidth=0.5, error_kw=ERR_KW)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_size)
    ax.set_ylabel("wall time (minutes)")
    ax.set_title("Runtime (error bars: SD across runs in group)", fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)

    ax = axes[1, 1]
    ax.set_facecolor(SHELL_BG)
    means = [group["gates_mean"] for group in groups]
    ax.bar(x, means, color=colors, edgecolor="#1F2937", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_size)
    ax.set_ylabel("two-qubit gates (template)")
    ax.set_title("Circuit template (no error bar)", fontsize=9)
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)

    ax_table = fig.add_subplot(grid[2, :])
    ax_table.axis("off")
    ax_table.set_title("Run settings (seed, lambda per block, n_total, run_id)", fontsize=9, loc="left", pad=6)
    headers = ["Configuration", "runs", "n_total", "seed(s)", "lambda", "run_id"]
    rows_data: list[list[str]] = []
    for group in groups:
        config = f"{group['optimizer'].upper()} n={group['N_local']} m={group['M_blocks']} p={group['p']}"
        run_ids = group.get("run_ids") or []
        rows_data.append([
            config,
            str(group["n_runs"]),
            str(group["n_total_display"]),
            str(group["seeds_display"]),
            str(group["lambda_display"]),
            run_ids[0] if len(run_ids) == 1 else ("\n".join(run_ids) if run_ids else "-"),
        ])
    table = ax_table.table(cellText=rows_data, colLabels=headers, loc="upper center", cellLoc="left", colColours=["#E5E7EB"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 2.0)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#E5E7EB" if row == 0 else ("#FAFAFA" if (row - 1) % 2 == 0 else "white"))
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=6.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cmd_heuristics_summary(args: argparse.Namespace) -> None:
    exclude = frozenset(item.strip().lower() for item in args.exclude_optimizer if item.strip())
    rows = load_qaoa_rows(args.summaries, exclude_optimizers=exclude if exclude else None)
    if not rows:
        raise SystemExit(f"No qaoa rows found in {args.summaries}")
    out_path = args.out_dir / args.output_name
    plot_heuristics_summary(aggregate_heuristics_groups(rows), out_path)
    print(f"Wrote {out_path}")


def ratio_vs_classical(qaoa_row: dict[str, str] | None, classical_row: dict[str, str] | None) -> float | None:
    if qaoa_row is None or classical_row is None:
        return None
    q_profit = _f(qaoa_row, "best_profit")
    c_profit = _f(classical_row, "best_profit")
    if q_profit is None or c_profit is None or c_profit == 0:
        return None
    return q_profit / c_profit


def cmd_profit_ratio_vs_n(args: argparse.Namespace) -> None:
    ns = [int(value.strip()) for value in args.n_list.split(",") if value.strip()]
    if not ns:
        raise SystemExit("Empty --n-list")

    if args.lambda_str:
        lambda_target = parse_lambda_tuple(args.lambda_str)
        if lambda_target is None:
            raise SystemExit("Could not parse --lambda")
    else:
        if ns[0] not in DEFAULT_LAMBDA_BY_N:
            raise SystemExit(f"No default lambda for n={ns[0]}; pass --lambda explicitly.")
        lambda_target = DEFAULT_LAMBDA_BY_N[ns[0]]

    rows = load_rows(args.summaries)
    cobyla_ratios: list[float | None] = []
    spsa_ratios: list[float | None] = []
    for n in ns:
        classical = pick_classical(rows, n=n, m=args.m)
        cobyla = pick_qaoa(rows, n=n, m=args.m, p=args.p, lambda_target=lambda_target, optimizer="cobyla")
        spsa = pick_qaoa(rows, n=n, m=args.m, p=args.p, lambda_target=lambda_target, optimizer="spsa")
        cobyla_ratios.append(ratio_vs_classical(cobyla, classical))
        spsa_ratios.append(ratio_vs_classical(spsa, classical))

    if all(value is None for value in cobyla_ratios + spsa_ratios):
        raise SystemExit("No ratios computed (missing classical or QAOA rows for this filter).")

    x = np.arange(len(ns))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SHELL_BG)

    def to_series(values: list[float | None]) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([0.0 if value is None else value for value in values], dtype=float),
            np.array([value is not None for value in values], dtype=bool),
        )

    cobyla_array, cobyla_mask = to_series(cobyla_ratios)
    spsa_array, spsa_mask = to_series(spsa_ratios)
    ax.bar(x - width / 2, np.where(cobyla_mask, cobyla_array, np.nan), width, label="QAOA COBYLA", color=QAOA_COBYLA, edgecolor="#1F2937", linewidth=0.5)
    ax.bar(x + width / 2, np.where(spsa_mask, spsa_array, np.nan), width, label="QAOA SPSA", color=QAOA_SPSA, edgecolor="#1F2937", linewidth=0.5)
    ax.axhline(1.0, color=CLASSICAL_GREEN, linestyle="--", linewidth=1.2, label="classical optimum")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N_local (coverages)")
    ax.set_ylabel("best_profit / classical optimum")
    ax.set_title(f"Profit recovery vs subsample size (m={args.m}, p={args.p}, lambda = {lambda_display(lambda_target)})", fontsize=10)
    chunks = [chunk for chunk in [cobyla_array[cobyla_mask], spsa_array[spsa_mask]] if chunk.size > 0]
    ymax = float(np.nanmax(np.concatenate(chunks))) if chunks else 1.0
    ax.set_ylim(0, max(1.05, ymax * 1.12))
    ax.grid(axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    out_path = args.out_dir / args.output_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def row_key_shots_per_m(row: dict[str, str]) -> tuple[int, int, str, int, int] | None:
    m = _i(row, "M_blocks")
    shots = _i(row, "num_samples_total")
    n = _i(row, "N_local")
    p = _i(row, "p")
    optimizer = _s(row, "optimizer").lower()
    if m is None or shots is None or n is None or p is None or m <= 0 or shots <= 0:
        return None
    if shots % m != 0:
        return None
    return (n, p, optimizer, shots // m, m)


def find_best_auto_runtime_vs_m(rows: list[dict[str, str]]) -> tuple[int, int, str, int, dict[int, dict[str, str]]]:
    groups: dict[tuple[int, int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not is_local_qaoa(row):
            continue
        key = row_key_shots_per_m(row)
        if key is None:
            continue
        n, p, optimizer, shots_per_m, _m = key
        groups[(n, p, optimizer, shots_per_m)].append(row)

    best_rank: tuple[int, int, int, int, int] | None = None
    best_payload: tuple[int, int, str, int, dict[int, dict[str, str]]] | None = None
    for (n, p, optimizer, shots_per_m), grouped_rows in groups.items():
        by_m = latest_per_m(grouped_rows)
        if len(by_m) < 3:
            continue
        rank = (len(by_m), p, 1 if optimizer == "cobyla" else 0, n, shots_per_m)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_payload = (n, p, optimizer, shots_per_m, by_m)
    if best_payload is None:
        raise SystemExit("No proportional-budget group with at least 3 distinct M_blocks.")
    return best_payload


def cmd_runtime_vs_m_blocks(args: argparse.Namespace) -> None:
    rows = load_rows(args.summaries)
    if args.auto:
        n, p, optimizer, shots_per_m, by_m = find_best_auto_runtime_vs_m(rows)
    else:
        if args.n is None or args.p is None or args.optimizer is None or args.shots_per_m is None:
            raise SystemExit("Pass --auto or all of --n --p --optimizer --shots-per-m.")
        n, p, optimizer, shots_per_m = args.n, args.p, args.optimizer.lower(), args.shots_per_m
        candidates = [
            row
            for row in rows
            if is_local_qaoa(row)
            and _i(row, "N_local") == n
            and _i(row, "p") == p
            and _s(row, "optimizer").lower() == optimizer
            and _i(row, "num_samples_total") == shots_per_m * (_i(row, "M_blocks") or -1)
        ]
        by_m = latest_per_m(candidates)
        if len(by_m) < 2:
            raise SystemExit(f"No matching rows for n={n} p={p} {optimizer} with shots = {shots_per_m} * M_blocks.")

    ms = np.array(sorted(by_m.keys()), dtype=float)
    runtimes = np.array([_f(by_m[int(value)], "runtime_sec") or 0.0 for value in ms], dtype=float)
    evals = [_i(by_m[int(value)], "num_objective_evals") for value in ms]
    slope, intercept, r2 = linear_fit_r2(ms, runtimes)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor=SHELL_BG)
    ax.set_facecolor("#FFFFFF")
    color = QAOA_COBYLA if optimizer == "cobyla" else QAOA_SPSA
    ax.scatter(ms, runtimes, s=64, color=color, zorder=3, label=f"QAOA ({optimizer.upper()}) measured")
    fit_x = np.linspace(ms.min(), ms.max(), 100)
    ax.plot(fit_x, slope * fit_x + intercept, "--", color=color, linewidth=1.8, alpha=0.85, label="Linear fit")

    note = (
        f"N_local={n}, p={p}, {optimizer.upper()}, local\n"
        f"Fixed shots/package k={shots_per_m} (num_samples_total = k * M_blocks)\n"
        f"Objective evals at points: {evals}"
    )

    if args.second_optimizer and args.second_optimizer != optimizer:
        opt2 = args.second_optimizer
        second_rows = [
            row
            for row in rows
            if is_local_qaoa(row)
            and _i(row, "N_local") == n
            and _i(row, "p") == p
            and _s(row, "optimizer").lower() == opt2
            and _i(row, "num_samples_total") == shots_per_m * (_i(row, "M_blocks") or -1)
        ]
        by_m_2 = latest_per_m(second_rows)
        if len(by_m_2) >= 2:
            ms_2 = np.array(sorted(by_m_2.keys()), dtype=float)
            runtimes_2 = np.array([_f(by_m_2[int(value)], "runtime_sec") or 0.0 for value in ms_2], dtype=float)
            color_2 = QAOA_SPSA if opt2 == "spsa" else QAOA_COBYLA
            slope_2, intercept_2, _r2_2 = linear_fit_r2(ms_2, runtimes_2)
            fit_x_2 = np.linspace(ms_2.min(), ms_2.max(), 100)
            ax.scatter(ms_2, runtimes_2, s=56, color=color_2, marker="s", zorder=3, label=f"QAOA ({opt2.upper()}) measured")
            ax.plot(fit_x_2, slope_2 * fit_x_2 + intercept_2, ":", color=color_2, linewidth=1.6, alpha=0.85, label=f"{opt2.upper()} linear fit")
            note += f"\nSecond series evals ({opt2}): {[_i(by_m_2[int(value)], 'num_objective_evals') for value in ms_2]}"

    ax.set_xlabel("Packages M_blocks")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(f"Runtime vs M_blocks with proportional shot budget\nLinear fit: R^2 = {r2:.4f}, slope ~= {slope:.3f} s / package", fontsize=10)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.5, 0.02, note, ha="center", fontsize=7, color="#374151")
    fig.subplots_adjust(bottom=0.22)

    out_name = args.output_name or f"runtime_vs_m_blocks_n{n}_p{p}_{optimizer}_k{shots_per_m}.png"
    out_path = args.out_dir / out_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"  m={list(map(int, ms))}  runtime_sec={[float(value) for value in runtimes]}  slope={slope:.6g}  R2={r2:.6g}")


def cmd_runtime_vs_n(args: argparse.Namespace) -> None:
    target_lambda: tuple[float, ...] | None = None
    if args.lambda_str:
        target_lambda = parse_lambda_tuple(args.lambda_str)
        if target_lambda is None:
            raise SystemExit("Could not parse --lambda")

    rows = load_rows(args.summaries)
    classical_rows = [
        row
        for row in rows
        if _s(row, "algorithm") == "classical" and _i(row, "N_local") is not None and _i(row, "M_blocks") == args.m
    ]
    q_rows = [row for row in rows if is_local_qaoa(row) and _i(row, "M_blocks") == args.m and _i(row, "p") == args.p]

    classical_by_n: dict[int, dict[str, str]] = {}
    for row in classical_rows:
        n = _i(row, "N_local")
        if n is None:
            continue
        prev = classical_by_n.get(n)
        if prev is None or run_id_sort_key(row["run_id"]) > run_id_sort_key(prev["run_id"]):
            classical_by_n[n] = row

    cobyla_by_n: dict[int, list[dict[str, str]]] = {}
    spsa_by_n: dict[int, list[dict[str, str]]] = {}
    for row in q_rows:
        optimizer = _s(row, "optimizer").lower()
        n = _i(row, "N_local")
        if n is None:
            continue
        if target_lambda is not None:
            lam = parse_lambda_tuple(row.get("lambda") or "")
            if lam is None or not lambdas_close(lam, target_lambda):
                continue
        if optimizer == "cobyla":
            cobyla_by_n.setdefault(n, []).append(row)
        elif optimizer == "spsa":
            spsa_by_n.setdefault(n, []).append(row)

    ns_sorted = sorted(set(classical_by_n) & set(cobyla_by_n) & set(spsa_by_n), key=int)
    if not ns_sorted:
        raise SystemExit("No n values with classical + local COBYLA + local SPSA for this filter.")

    ns_plot: list[int] = []
    classical_sec: list[float] = []
    cobyla_sec: list[float] = []
    spsa_sec: list[float] = []
    lambda_matched: list[bool] = []
    for n in ns_sorted:
        classical = classical_by_n[n]
        cobyla_options = cobyla_by_n[n]
        spsa_options = spsa_by_n[n]

        cobyla_lambdas = {parse_lambda_tuple(row.get("lambda") or "") for row in cobyla_options}
        spsa_lambdas = {parse_lambda_tuple(row.get("lambda") or "") for row in spsa_options}
        cobyla_lambdas.discard(None)
        spsa_lambdas.discard(None)
        common = cobyla_lambdas & spsa_lambdas

        matched = bool(common)
        if common:
            chosen = sorted(common, key=lambda item: tuple(item))[0]
            cobyla = max((row for row in cobyla_options if parse_lambda_tuple(row.get("lambda") or "") == chosen), key=lambda row: run_id_sort_key(row["run_id"]))
            spsa = max((row for row in spsa_options if parse_lambda_tuple(row.get("lambda") or "") == chosen), key=lambda row: run_id_sort_key(row["run_id"]))
        else:
            cobyla = max(cobyla_options, key=lambda row: run_id_sort_key(row["run_id"]))
            spsa = max(spsa_options, key=lambda row: run_id_sort_key(row["run_id"]))

        rt_cl = _f(classical, "runtime_sec")
        rt_co = _f(cobyla, "runtime_sec")
        rt_sp = _f(spsa, "runtime_sec")
        if rt_cl is None or rt_co is None or rt_sp is None:
            continue
        ns_plot.append(n)
        classical_sec.append(rt_cl)
        cobyla_sec.append(rt_co)
        spsa_sec.append(rt_sp)
        lambda_matched.append(matched)

    if not ns_plot:
        raise SystemExit("No complete runtime triples after filtering.")

    x = np.array(ns_plot, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(SHELL_BG)
    ax.plot(x, np.array(classical_sec, dtype=float), color=CLASSICAL_GREEN, marker="o", linewidth=2.0, markersize=7, label="Classical (ILP)")
    ax.plot(x, np.array(cobyla_sec, dtype=float), color=QAOA_COBYLA, marker="s", linewidth=2.0, markersize=7, label="QAOA COBYLA (local)")
    ax.plot(x, np.array(spsa_sec, dtype=float), color=QAOA_SPSA, marker="^", linewidth=2.0, markersize=7, label="QAOA SPSA (local)")
    if not args.linear_y:
        ax.set_yscale("log")

    subtitle = f"m={args.m}, p={args.p} · local QAOA only"
    if target_lambda is not None:
        subtitle += " · lambda filter applied"
    elif not all(lambda_matched):
        subtitle += " · at some n, COBYLA/SPSA used different lambda (latest run each)"
    ax.set_xlabel("N_local (coverages)")
    ax.set_ylabel("Wall time (seconds)" + ("" if args.linear_y else ", log scale"))
    ax.set_title("Runtime vs subsample size: classical vs QAOA\n" + subtitle, fontsize=10)
    ax.grid(True, which="both", axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.grid(True, which="major", axis="x", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.set_xticks(x)

    out_path = args.out_dir / args.output_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path} (n = {list(map(int, x))})")


def lambda_scale(row: dict[str, str], how: str) -> float:
    values = parse_lambda_tuple(row.get("lambda") or "")
    if values is None or len(values) == 0:
        return 1.0
    vector = np.abs(np.asarray(values, dtype=float))
    if how == "mean":
        scale = float(np.mean(vector))
    elif how == "sum":
        scale = float(np.sum(vector))
    elif how == "l2":
        scale = float(np.linalg.norm(vector))
    else:
        return 1.0
    return max(scale, 1e-12)


def find_best_auto_relaxed(rows: list[dict[str, str]]) -> tuple[int, int, list[int]]:
    classical_pairs = {
        (_i(row, "N_local"), _i(row, "M_blocks"))
        for row in rows
        if _s(row, "algorithm") == "classical" and _i(row, "N_local") is not None and _i(row, "M_blocks") is not None
    }
    q_rows = [row for row in rows if is_local_qaoa(row)]
    mp_values = {
        (_i(row, "M_blocks"), _i(row, "p"))
        for row in q_rows
        if _i(row, "M_blocks") is not None and _i(row, "p") is not None
    }

    best: tuple[int, int, int, int, list[int]] = (0, 0, 0, 0, [])
    for m, p in sorted(mp_values):
        assert m is not None and p is not None
        n_values = sorted({
            _i(row, "N_local")
            for row in q_rows
            if _i(row, "M_blocks") == m and _i(row, "p") == p and _i(row, "N_local") is not None
        })
        ok: list[int] = []
        for n in n_values:
            assert n is not None
            if (n, m) not in classical_pairs:
                continue
            co = [
                row for row in q_rows
                if _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and _s(row, "optimizer").lower() == "cobyla"
            ]
            sp = [
                row for row in q_rows
                if _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and _s(row, "optimizer").lower() == "spsa"
            ]
            if co and sp:
                ok.append(int(n))
        if len(ok) < 2:
            continue
        candidate = (len(ok), m, p, max(ok) - min(ok), ok)
        if candidate > best:
            best = candidate

    if best[0] < 2:
        raise SystemExit("Could not find at least two n values with classical + local COBYLA + local SPSA.")
    return best[1], best[2], best[4]


def find_best_auto_runtime_vs_n_total(rows: list[dict[str, str]]) -> tuple[int, int, int, str, list[int]]:
    classical_pairs = {
        (_i(row, "N_local"), _i(row, "M_blocks"))
        for row in rows
        if _s(row, "algorithm") == "classical" and _i(row, "N_local") is not None and _i(row, "M_blocks") is not None
    }
    q_rows = [row for row in rows if is_local_qaoa(row)]
    groups: dict[tuple[int, int, int, str], dict[str, set[int]]] = defaultdict(lambda: {"cobyla": set(), "spsa": set()})
    for row in q_rows:
        n = _i(row, "N_local")
        m = _i(row, "M_blocks")
        p = _i(row, "p")
        shots = _i(row, "num_samples_total")
        if n is None or m is None or p is None or shots is None or shots <= 0:
            continue
        groups[(m, p, shots, _s(row, "lambda"))][_s(row, "optimizer").lower()].add(n)

    best: tuple[int, int, int, int, list[int]] = (0, 0, 0, 0, [])
    best_key: tuple[int, int, int, str] | None = None
    for key, grouped in groups.items():
        m, p, _shots, _lambda_raw = key
        common = sorted(grouped["cobyla"] & grouped["spsa"])
        ok = [n for n in common if (n, m) in classical_pairs]
        if not ok:
            continue
        candidate = (len(ok), m, p, max(ok) - min(ok), ok)
        if candidate > best:
            best = candidate
            best_key = key

    if best_key is None or best[0] < 2:
        raise SystemExit("Could not find at least two n values with classical + local COBYLA + local SPSA at the same (m, p, shots, lambda).")
    m, p, shots, lambda_raw = best_key
    return m, p, shots, lambda_raw, best[4]


def cmd_runtime_vs_n_total(args: argparse.Namespace) -> None:
    rows = load_rows(args.summaries)
    qaoa_norm_how = args.qaoa_lambda_norm if args.qaoa_lambda_norm is not None else ("mean" if args.relax_shots else "none")

    if args.auto:
        if args.relax_shots:
            m, p, n_list = find_best_auto_relaxed(rows)
            shots: int | None = None
            lambda_raw = "(mixed lambda; shots may differ per n)"
            lambda_target = None
        else:
            m, p, shots, lambda_raw, n_list = find_best_auto_runtime_vs_n_total(rows)
            lambda_target = parse_lambda_tuple(lambda_raw) if lambda_raw else None
    else:
        if args.m is None or args.p is None:
            raise SystemExit("Pass --m and --p, or use --auto.")
        m, p = args.m, args.p
        if args.relax_shots:
            shots = None
            lambda_raw = "(mixed lambda; shots may differ per n)"
            lambda_target = None
            n_list = []
            candidates = sorted({_i(row, "N_local") for row in rows if _i(row, "N_local") is not None and _i(row, "M_blocks") == m})
            for n in candidates:
                if n is None:
                    continue
                classical = [
                    row for row in rows
                    if _s(row, "algorithm") == "classical" and _i(row, "N_local") == n and _i(row, "M_blocks") == m
                ]
                co = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "cobyla" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p]
                sp = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "spsa" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p]
                if classical and co and sp:
                    n_list.append(n)
            n_list.sort()
            if len(n_list) < 2:
                raise SystemExit(f"No n>=2 with classical + COBYLA + SPSA for m={m} p={p} (relaxed).")
        else:
            if args.shots is None or args.lambda_str is None:
                raise SystemExit("Strict mode needs --shots and --lambda, or use --relax-shots / --auto.")
            shots = int(args.shots)
            lambda_raw = args.lambda_str.strip()
            lambda_target = parse_lambda_tuple(lambda_raw)
            if lambda_target is None:
                raise SystemExit("Could not parse --lambda.")
            n_list = []
            candidates = sorted({_i(row, "N_local") for row in rows if _i(row, "N_local") is not None and _i(row, "M_blocks") == m})
            for n in candidates:
                if n is None:
                    continue
                classical = [row for row in rows if _s(row, "algorithm") == "classical" and _i(row, "N_local") == n and _i(row, "M_blocks") == m]
                co = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "cobyla" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and _i(row, "num_samples_total") == shots]
                sp = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "spsa" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and _i(row, "num_samples_total") == shots]
                co = [row for row in co if lambdas_close(parse_lambda_tuple(row.get("lambda") or "") or (), lambda_target)]
                sp = [row for row in sp if lambdas_close(parse_lambda_tuple(row.get("lambda") or "") or (), lambda_target)]
                if classical and co and sp:
                    n_list.append(n)
            n_list.sort()
            if len(n_list) < 2:
                raise SystemExit(f"No n>=2 with classical + COBYLA + SPSA for m={m} p={p} shots={shots} and that lambda.")

    def latest(selected: list[dict[str, str]]) -> dict[str, str]:
        return max(selected, key=lambda row: row.get("run_id", ""))

    n_totals: list[int] = []
    y_classical: list[float] = []
    y_cobyla: list[float] = []
    y_spsa: list[float] = []
    eval_cobyla: list[int] = []
    eval_spsa: list[int] = []
    shots_cobyla: list[int] = []
    shots_spsa: list[int] = []

    for n in n_list:
        classical_rows = [row for row in rows if _s(row, "algorithm") == "classical" and _i(row, "N_local") == n and _i(row, "M_blocks") == m]
        co_rows = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "cobyla" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and (shots is None or _i(row, "num_samples_total") == shots)]
        sp_rows = [row for row in rows if is_local_qaoa(row) and _s(row, "optimizer").lower() == "spsa" and _i(row, "N_local") == n and _i(row, "M_blocks") == m and _i(row, "p") == p and (shots is None or _i(row, "num_samples_total") == shots)]
        if lambda_target is not None:
            co_rows = [row for row in co_rows if lambdas_close(parse_lambda_tuple(row.get("lambda") or "") or (), lambda_target)]
            sp_rows = [row for row in sp_rows if lambdas_close(parse_lambda_tuple(row.get("lambda") or "") or (), lambda_target)]
        if not classical_rows or not co_rows or not sp_rows:
            continue

        classical = latest(classical_rows)
        cobyla = latest(co_rows)
        spsa = latest(sp_rows)
        rt_cl = _f(classical, "runtime_sec")
        rt_co = _f(cobyla, "runtime_sec")
        rt_sp = _f(spsa, "runtime_sec")
        if rt_cl is None or rt_co is None or rt_sp is None:
            continue

        n_totals.append(n * m)
        y_classical.append(rt_cl)
        y_cobyla.append(rt_co / (lambda_scale(cobyla, qaoa_norm_how) if qaoa_norm_how != "none" else 1.0))
        y_spsa.append(rt_sp / (lambda_scale(spsa, qaoa_norm_how) if qaoa_norm_how != "none" else 1.0))
        eval_cobyla.append(_i(cobyla, "num_objective_evals") or 0)
        eval_spsa.append(_i(spsa, "num_objective_evals") or 0)
        shots_cobyla.append(_i(cobyla, "num_samples_total") or 0)
        shots_spsa.append(_i(spsa, "num_samples_total") or 0)

    if len(n_totals) < 2:
        raise SystemExit("Fewer than two points after row selection.")

    x = np.array(n_totals, dtype=float)
    y_cl = np.array(y_classical, dtype=float)
    y_co = np.array(y_cobyla, dtype=float)
    y_sp = np.array(y_spsa, dtype=float)
    use_twin = qaoa_norm_how != "none"

    fig, ax_q = plt.subplots(figsize=(8.5, 5.4))
    fig.patch.set_facecolor("white")
    ax_q.set_facecolor(SHELL_BG)
    ax_c = ax_q.twinx() if use_twin else ax_q
    if use_twin:
        ax_c.spines["right"].set_edgecolor(CLASSICAL_GREEN)
        ax_c.tick_params(axis="y", labelcolor=CLASSICAL_GREEN)
        ax_c.grid(False)

    ax_c.plot(x, y_cl, color=CLASSICAL_GREEN, marker="o", linewidth=2.0, markersize=7, label="Classical (ILP)")
    ax_q.plot(x, y_co, color=QAOA_COBYLA, marker="s", linewidth=2.0, markersize=7, label="QAOA COBYLA (local)")
    ax_q.plot(x, y_sp, color=QAOA_SPSA, marker="^", linewidth=2.0, markersize=7, label="QAOA SPSA (local)")

    cross_notes: list[str] = []
    x_min = max(1.0, float(args.extrapolate_x_min))
    x_max = float(x.max()) * float(args.extrapolate_x_max_factor)
    x_fit = np.linspace(x_min, x_max, 400)
    if not args.no_extrapolate:
        A_co, beta_co = fit_power_law(x, y_co)
        A_sp, beta_sp = fit_power_law(x, y_sp)
        if args.classical_extrap == "constant":
            intercept = float(np.median(y_cl))
            slope = 0.0
            y_cl_fit = np.full_like(x_fit, intercept, dtype=float)
        else:
            intercept, slope = fit_linear(x, y_cl)
            y_cl_fit = intercept + slope * x_fit
        y_co_fit = A_co * np.power(x_fit, beta_co)
        y_sp_fit = A_sp * np.power(x_fit, beta_sp)
        if not args.linear_y:
            y_cl_fit = np.maximum(y_cl_fit, 1e-9)
            y_co_fit = np.maximum(y_co_fit, 1e-9)
            y_sp_fit = np.maximum(y_sp_fit, 1e-9)
        ax_c.plot(x_fit, y_cl_fit, ":", color=CLASSICAL_GREEN, linewidth=1.6, alpha=0.9, label="Classical extrap.")
        ax_q.plot(x_fit, y_co_fit, ":", color=QAOA_COBYLA, linewidth=1.6, alpha=0.9, label="COBYLA extrap.")
        ax_q.plot(x_fit, y_sp_fit, ":", color=QAOA_SPSA, linewidth=1.6, alpha=0.9, label="SPSA extrap.")

        if use_twin:
            cross_notes.append("Crossover markers skipped: QAOA y is lambda-normalized; classical stays in raw seconds.")
        else:
            for index, (name, A_val, beta_val, color) in enumerate((("COBYLA", A_co, beta_co, QAOA_COBYLA), ("SPSA", A_sp, beta_sp, QAOA_SPSA))):
                if args.classical_extrap == "constant":
                    x_cross = crossing_power_vs_constant(A_val, beta_val, intercept, x_min, x_max)
                    y_cross = float(intercept) if x_cross is not None else float("nan")
                else:
                    x_cross = crossing_power_vs_linear(A_val, beta_val, intercept, slope, x_min, x_max)
                    y_cross = float(intercept + slope * x_cross) if x_cross is not None else float("nan")
                if x_cross is not None and np.isfinite(x_cross) and x_min <= x_cross <= x_max:
                    ax_q.axvline(x_cross, color=TRAVELERS_RED, linestyle="--", linewidth=1.0, alpha=0.85)
                    ax_q.scatter([x_cross], [y_cross], color=color, s=36, zorder=5, edgecolors="#1F2937", linewidths=0.5)
                    ax_q.annotate(
                        f"{name}: extrap. QAOA = classical\nn_total ~= {x_cross:.2f}",
                        xy=(x_cross, y_cross),
                        xytext=(12, 18 - 28 * index),
                        textcoords="offset points",
                        fontsize=7,
                        color="#374151",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
                    )
                    cross_notes.append(f"{name}: extrapolated equal wall time at n_total ~= {x_cross:.3f}")
                else:
                    cross_notes.append(f"{name}: no crossover in [{x_min:.2g}, {x_max:.2g}]")

    if not args.linear_y:
        ax_q.set_yscale("log")
        if use_twin:
            ax_c.set_yscale("log")

    ax_q.set_xlabel("Problem size n_total = N_local * M_blocks")
    if qaoa_norm_how == "none":
        ax_q.set_ylabel("Wall time (seconds)" + ("" if args.linear_y else ", log scale"))
    else:
        ax_q.set_ylabel(f"QAOA wall time / lambda ({qaoa_norm_how}) (s)" + ("" if args.linear_y else ", log scale"))
        ax_c.set_ylabel("Classical wall time (s)" + ("" if args.linear_y else ", log scale"), color=CLASSICAL_GREEN)

    fixed_line = (
        f"Relaxed: M={m}, p={p}; shots COBYLA {shots_cobyla} · SPSA {shots_spsa} (may differ)."
        if shots is None else f"Fixed: M={m}, p={p}, total shots={shots}, local QAOA, matching lambda"
    )
    subtitle = fixed_line + "\n" + f"COBYLA evals: {eval_cobyla} · SPSA evals: {eval_spsa}"
    if qaoa_norm_how != "none":
        subtitle += "\n" + f"lambda-normalize QAOA: {qaoa_norm_how}"
    ax_q.set_title("Runtime vs n_total with matched QAOA budget\n" + subtitle, fontsize=9)
    ax_q.grid(True, which="both", axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax_q.grid(True, which="major", axis="x", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax_q.set_xticks(x)
    if not args.no_extrapolate:
        ax_q.set_xlim(left=max(0.5, x_min * 0.88), right=x_max * 1.02)

    if use_twin:
        hq, lq = ax_q.get_legend_handles_labels()
        hc, lc = ax_c.get_legend_handles_labels()
        ax_q.legend(hq + hc, lq + lc, loc="upper left", fontsize=7, framealpha=0.95, ncol=2)
    else:
        ax_q.legend(loc="upper left", fontsize=7, framealpha=0.95, ncol=2)

    lambda_note = lambda_raw[:77] + "..." if len(lambda_raw) > 80 else lambda_raw
    fig.text(0.5, 0.02, f"lambda note (truncated): {lambda_note}", ha="center", fontsize=7, color="#374151")
    if not args.no_extrapolate:
        fig.text(0.5, 0.055, "Dotted fits are illustrative only.", ha="center", fontsize=6.5, color="#6B7280")

    if args.output_name:
        out_name = args.output_name
    else:
        suffix = ""
        if args.relax_shots:
            suffix += "_relaxed"
        if qaoa_norm_how != "none":
            suffix += f"_lamnorm_{qaoa_norm_how}"
        out_name = f"runtime_vs_n_total_m{m}_p{p}{suffix}.png" if shots is None else f"runtime_vs_n_total_m{m}_p{p}_shots{shots}{suffix}.png"

    out_path = args.out_dir / out_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.18 if not args.no_extrapolate else 0.14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"  m={m} p={p} shots={shots if shots is not None else 'relaxed'}  n_local list={n_list}  n_total={list(map(int, x))}  qaoa_lambda_norm={qaoa_norm_how}")
    for note in cross_notes:
        print(f"  {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    algorithm = subparsers.add_parser("algorithm-comparison")
    add_common_path_arguments(algorithm)
    algorithm.add_argument("--n", type=int, default=5)
    algorithm.add_argument("--m", type=int, default=3)
    algorithm.add_argument("--p", type=int, default=1)
    algorithm.add_argument("--lambda", dest="lambda_str", default=None)
    algorithm.add_argument("--output-name", default="algorithm_comparison_m3_p1.png")
    algorithm.add_argument("--csv-out", type=Path, default=None)
    algorithm.set_defaults(func=cmd_algorithm_comparison)

    heuristics = subparsers.add_parser("heuristics-summary")
    add_common_path_arguments(heuristics)
    heuristics.add_argument("--output-name", default="heuristics_summary_v1.png")
    heuristics.add_argument("--exclude-optimizer", action="append", default=[], metavar="NAME")
    heuristics.set_defaults(func=cmd_heuristics_summary)

    profit_ratio = subparsers.add_parser("profit-ratio-vs-n")
    add_common_path_arguments(profit_ratio)
    profit_ratio.add_argument("--m", type=int, default=3)
    profit_ratio.add_argument("--p", type=int, default=1)
    profit_ratio.add_argument("--n-list", type=str, default="5,7,10")
    profit_ratio.add_argument("--lambda", dest="lambda_str", default=None)
    profit_ratio.add_argument("--output-name", default="profit_ratio_vs_n_m3_p1.png")
    profit_ratio.set_defaults(func=cmd_profit_ratio_vs_n)

    runtime_m = subparsers.add_parser("runtime-vs-m-blocks")
    add_common_path_arguments(runtime_m)
    runtime_m.add_argument("--output-name", default=None)
    runtime_m.add_argument("--auto", action="store_true")
    runtime_m.add_argument("--n", type=int, default=None)
    runtime_m.add_argument("--p", type=int, default=None)
    runtime_m.add_argument("--optimizer", type=str, default=None, choices=["cobyla", "spsa"])
    runtime_m.add_argument("--shots-per-m", type=int, default=None)
    runtime_m.add_argument("--second-optimizer", type=str, default=None, choices=["cobyla", "spsa"])
    runtime_m.set_defaults(func=cmd_runtime_vs_m_blocks)

    runtime_n = subparsers.add_parser("runtime-vs-n")
    add_common_path_arguments(runtime_n)
    runtime_n.add_argument("--m", type=int, default=3)
    runtime_n.add_argument("--p", type=int, default=1)
    runtime_n.add_argument("--lambda", dest="lambda_str", default=None)
    runtime_n.add_argument("--linear-y", action="store_true")
    runtime_n.add_argument("--output-name", default="runtime_vs_n_classical_qaoa_m3_p1.png")
    runtime_n.set_defaults(func=cmd_runtime_vs_n)

    runtime_total = subparsers.add_parser("runtime-vs-n-total")
    add_common_path_arguments(runtime_total)
    runtime_total.add_argument("--auto", action="store_true")
    runtime_total.add_argument("--relax-shots", action="store_true")
    runtime_total.add_argument("--m", type=int, default=None)
    runtime_total.add_argument("--p", type=int, default=None)
    runtime_total.add_argument("--shots", type=int, default=None)
    runtime_total.add_argument("--lambda", dest="lambda_str", default=None)
    runtime_total.add_argument("--qaoa-lambda-norm", choices=["none", "mean", "sum", "l2"], default=None)
    runtime_total.add_argument("--linear-y", action="store_true")
    runtime_total.add_argument("--no-extrapolate", action="store_true")
    runtime_total.add_argument("--extrapolate-x-min", type=float, default=1.0)
    runtime_total.add_argument("--extrapolate-x-max-factor", type=float, default=12.0)
    runtime_total.add_argument("--classical-extrap", choices=["constant", "linear"], default="constant")
    runtime_total.add_argument("--output-name", default=None)
    runtime_total.set_defaults(func=cmd_runtime_vs_n_total)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
