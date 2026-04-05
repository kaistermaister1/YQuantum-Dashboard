import os
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_RUN_ID_TS = re.compile(r"_(\d{8}T\d{6}Z)$")

# Design System Colors
COLOR_QAOA = '#0066CC'
COLOR_DQI = '#00356B'
COLOR_CLASSICAL = '#2D8C3C'
COLOR_TRAVELERS_RED = '#E31837'

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = '#F8F9FA' # light gray app shell
plt.rcParams['figure.facecolor'] = '#FFFFFF'


def _run_id_timestamp(run_id: str) -> str:
    m = _RUN_ID_TS.search(run_id or "")
    return m.group(1) if m else ""


def _norm_method_row(r: dict) -> str:
    a = str(r.get("algorithm", "") or "").strip().lower()
    notes = str(r.get("notes", "") or "")
    if a == "classical":
        return "classical (ILP)"
    if a == "qaoa":
        opt = str(r.get("optimizer", "") or "").strip().lower() or "?"
        loc = "local" if "execution_target=local" in notes else "non-local"
        return f"QAOA + {opt} ({loc})"
    if a == "dqi":
        return "DQI"
    return a or "unknown"


def _wrap_methods_cell(text: str, max_len: int = 54) -> str:
    parts = text.split(", ")
    lines, cur = [], ""
    for p in parts:
        sep = ", " if cur else ""
        if len(cur) + len(sep) + len(p) <= max_len:
            cur = cur + sep + p
        else:
            if cur:
                lines.append(cur)
            cur = p
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def generate_simulation_coverage_by_nmp_png(
    csv_path: str = "SOLUTIONS/HEURISTICS/run_summaries.csv",
    out_public: str | None = None,
    out_solutions: str | None = None,
) -> None:
    """Grouped (N_local, M_blocks, p) × methods table; matches dashboard styling."""
    out_public = out_public or os.path.join(
        "public", "plots", "heuristics", "simulation_coverage_by_nmp.png"
    )
    out_solutions = out_solutions or os.path.join(
        "SOLUTIONS", "HEURISTICS", "plots", "simulation_coverage_by_nmp.png"
    )
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    def key_rec(r):
        n = str(r.get("N_local", "") or "").strip()
        m = str(r.get("M_blocks", "") or "").strip()
        pp = str(r.get("p", "") or "").strip()
        if not n or not m:
            return None
        try:
            ni, mi = int(float(n)), int(float(m))
        except ValueError:
            return None
        if pp == "":
            pi = None
        else:
            try:
                pi = int(float(pp))
            except ValueError:
                pi = None
        return (ni, mi, pi, _norm_method_row(r))

    seen: dict[tuple, str] = {}
    for r in rows:
        k = key_rec(r)
        if k is None:
            continue
        rid = str(r.get("run_id", "") or "")
        if k not in seen or _run_id_timestamp(rid) > _run_id_timestamp(seen[k]):
            seen[k] = rid

    by_nmp: dict[tuple, set] = defaultdict(set)
    for ni, mi, pi, meth in seen.keys():
        by_nmp[(ni, mi, pi)].add(meth)

    def p_key(t):
        pi = t[2]
        return (pi is None, pi if pi is not None else -1)

    sorted_nmp = sorted(by_nmp.keys(), key=lambda x: (x[0], x[1], p_key(x)))
    cell_text = []
    for ni, mi, pi in sorted_nmp:
        p_disp = "—" if pi is None else str(pi)
        meths = ", ".join(sorted(by_nmp[(ni, mi, pi)]))
        cell_text.append([str(ni), str(mi), p_disp, _wrap_methods_cell(meths)])

    nrows = len(cell_text)
    fig_h = min(28, max(4.5, 0.52 * (nrows + 2) + 1.4))
    fig, ax = plt.subplots(figsize=(13.5, fig_h), facecolor="#FFFFFF")
    ax.axis("off")
    col_labels = ["N_local", "M_blocks", "p", "Methods"]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.09, 0.1, 0.06, 0.75],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 2.15)
    header_bg = "#E5E7EB"
    for j in range(len(col_labels)):
        c = tbl[(0, j)]
        c.set_facecolor(header_bg)
        c.set_text_props(fontweight="bold", color="#111827")
        c.set_edgecolor("#D1D5DB")
    for i in range(1, nrows + 1):
        stripe = "#FAFAFA" if i % 2 else "#FFFFFF"
        for j in range(4):
            c = tbl[(i, j)]
            c.set_facecolor(stripe)
            c.set_edgecolor("#E5E7EB")
            if j == 3:
                c.get_text().set_color("#1F2937")
    fig.suptitle(
        "Simulated configurations by problem shape (run_summaries.csv)",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        y=0.98,
        x=0.5,
        ha="center",
    )
    fig.text(
        0.5,
        0.02,
        "Unique (N_local, M_blocks, p, method); latest run_id kept when duplicates exist.",
        ha="center",
        fontsize=8,
        color="#6B7280",
    )
    plt.subplots_adjust(top=0.94, bottom=0.06, left=0.04, right=0.98)
    os.makedirs(os.path.dirname(out_public), exist_ok=True)
    os.makedirs(os.path.dirname(out_solutions), exist_ok=True)
    plt.savefig(out_public, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.savefig(out_solutions, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


def _metrics_bucket(r: dict) -> str:
    a = str(r.get("algorithm", "") or "").strip().lower()
    notes = str(r.get("notes", "") or "")
    if a == "classical":
        return "Classical (ILP)"
    if a == "qaoa":
        opt = str(r.get("optimizer", "") or "").strip().lower() or "?"
        if "execution_target=local" in notes:
            loc = "local"
        elif "execution_target=selene" in notes:
            loc = "Selene"
        else:
            loc = "other"
        return f"QAOA + {opt} ({loc})"
    if a == "dqi":
        return "DQI"
    return a or "unknown"


def _float_cell(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def generate_run_summary_metrics_table_png(
    csv_path: str = "SOLUTIONS/HEURISTICS/run_summaries.csv",
    out_public: str | None = None,
    out_solutions: str | None = None,
) -> None:
    """Aggregate wall-time and quality stats from run_summaries.csv (presentation table)."""
    out_public = out_public or os.path.join(
        "public", "plots", "heuristics", "run_summary_aggregate_metrics.png"
    )
    out_solutions = out_solutions or os.path.join(
        "SOLUTIONS", "HEURISTICS", "plots", "run_summary_aggregate_metrics.png"
    )
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    rt_by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        t = _float_cell(r.get("runtime_sec"))
        if t is not None:
            rt_by[_metrics_bucket(r)].append(t)

    wall_rows: list[list[str]] = []
    for k in sorted(rt_by.keys()):
        xs = rt_by[k]
        wall_rows.append(
            [
                k,
                str(len(xs)),
                f"{statistics.mean(xs):.3f}",
                f"{statistics.median(xs):.3f}",
                f"{min(xs):.3f}",
                f"{max(xs):.3f}",
            ]
        )

    loc_qaoa = [
        r
        for r in rows
        if str(r.get("algorithm", "") or "").strip() == "qaoa"
        and "execution_target=local" in str(r.get("notes", "") or "")
    ]
    approx: list[float] = []
    for r in loc_qaoa:
        bp = _float_cell(r.get("best_profit"))
        cp = _float_cell(r.get("classical_opt_profit"))
        if bp is not None and cp is not None and cp > 0:
            approx.append(bp / cp)
    feas: list[float] = []
    for r in loc_qaoa:
        ns = _float_cell(r.get("num_samples_total"))
        nf = _float_cell(r.get("num_samples_feasible"))
        if ns is not None and nf is not None and ns > 0:
            feas.append(nf / ns)

    cobyla_t = [
        _float_cell(r["runtime_sec"])
        for r in loc_qaoa
        if str(r.get("optimizer", "") or "").strip().lower() == "cobyla"
    ]
    cobyla_t = [x for x in cobyla_t if x is not None]
    spsa_t = [
        _float_cell(r["runtime_sec"])
        for r in loc_qaoa
        if str(r.get("optimizer", "") or "").strip().lower() == "spsa"
    ]
    spsa_t = [x for x in spsa_t if x is not None]

    cl_all = [
        _float_cell(r["runtime_sec"])
        for r in rows
        if str(r.get("algorithm", "") or "").strip() == "classical"
    ]
    cl_all = [x for x in cl_all if x is not None]

    cl_n10_m3 = [
        _float_cell(r["runtime_sec"])
        for r in rows
        if str(r.get("algorithm", "") or "").strip() == "classical"
        and str(r.get("N_local", "")).strip() in ("10", "10.0")
        and str(r.get("M_blocks", "")).strip() in ("3", "3.0")
    ]
    cl_n10_m3 = [x for x in cl_n10_m3 if x is not None]

    summary_rows: list[list[str]] = [
        ["Total rows in CSV", str(len(rows))],
        ["", ""],
        [
            "Local QAOA rows (any optimizer)",
            str(len(loc_qaoa)),
        ],
    ]
    if cobyla_t:
        summary_rows.append(
            [
                "Local QAOA + COBYLA: mean wall time (s)",
                f"{statistics.mean(cobyla_t):.2f} (n={len(cobyla_t)}, median {statistics.median(cobyla_t):.2f})",
            ]
        )
    if spsa_t:
        summary_rows.append(
            [
                "Local QAOA + SPSA: mean wall time (s)",
                f"{statistics.mean(spsa_t):.2f} (n={len(spsa_t)}, median {statistics.median(spsa_t):.2f})",
            ]
        )
    if approx:
        summary_rows.extend(
            [
                ["", ""],
                [
                    "Local QAOA: best_profit / classical_opt (mean, median)",
                    f"{statistics.mean(approx):.3f}, {statistics.median(approx):.3f} (n={len(approx)})",
                ],
                [
                    "Local QAOA: same ratio (min, max)",
                    f"{min(approx):.3f}, {max(approx):.3f}",
                ],
            ]
        )
    if feas:
        summary_rows.append(
            [
                "Local QAOA: feasibility rate mean (median)",
                f"{statistics.mean(feas):.3f} ({statistics.median(feas):.3f}), n={len(feas)}",
            ]
        )
    if cl_all:
        summary_rows.extend(
            [
                ["", ""],
                [
                    "Classical ILP: mean wall time (s)",
                    f"{statistics.mean(cl_all):.4f} (n={len(cl_all)}, median {statistics.median(cl_all):.4f})",
                ],
            ]
        )
    if len(cl_n10_m3) >= 2:
        summary_rows.append(
            [
                f"Classical repeat baseline n=10, m=3 ({len(cl_n10_m3)} runs)",
                f"mean {statistics.mean(cl_n10_m3)*1000:.1f} ms, stdev {statistics.pstdev(cl_n10_m3)*1000:.1f} ms",
            ]
        )

    fig = plt.figure(figsize=(13.5, max(7.5, 0.38 * (len(wall_rows) + len(summary_rows)) + 3.2)), facecolor="#FFFFFF")
    gs = fig.add_gridspec(2, 1, height_ratios=[len(wall_rows) + 1.2, len(summary_rows) + 1.0], hspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    ax1.axis("off")
    ax1.set_title(
        "Wall time by method bucket (runtime_sec)",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#111827",
        pad=8,
    )
    col_w = ["Method bucket", "N runs", "Mean (s)", "Median (s)", "Min (s)", "Max (s)"]
    t1 = ax1.table(
        cellText=wall_rows,
        colLabels=col_w,
        loc="upper center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.34, 0.09, 0.12, 0.12, 0.12, 0.12],
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(8)
    t1.scale(1.02, 1.95)
    for j in range(len(col_w)):
        c = t1[(0, j)]
        c.set_facecolor("#E5E7EB")
        c.set_text_props(fontweight="bold", color="#111827")
        c.set_edgecolor("#D1D5DB")
    for i in range(1, len(wall_rows) + 1):
        stripe = "#FAFAFA" if i % 2 else "#FFFFFF"
        for j in range(6):
            t1[(i, j)].set_facecolor(stripe)
            t1[(i, j)].set_edgecolor("#E5E7EB")

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title(
        "Summary metrics (local QAOA & classical)",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#111827",
        pad=8,
    )
    t2 = ax2.table(
        cellText=summary_rows,
        colLabels=["Metric", "Value"],
        loc="upper center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.42, 0.56],
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    t2.scale(1.02, 2.05)
    for j in range(2):
        c = t2[(0, j)]
        c.set_facecolor("#E5E7EB")
        c.set_text_props(fontweight="bold", color="#111827")
        c.set_edgecolor("#D1D5DB")
    for i in range(1, len(summary_rows) + 1):
        stripe = "#FAFAFA" if i % 2 else "#FFFFFF"
        for j in range(2):
            t2[(i, j)].set_facecolor(stripe)
            t2[(i, j)].set_edgecolor("#E5E7EB")

    fig.suptitle(
        "Aggregate benchmark metrics (run_summaries.csv)",
        fontsize=14,
        fontweight="bold",
        color="#111827",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "All CSV rows included; Selene / non-local QAOA appear in the wall-time table as separate buckets.",
        ha="center",
        fontsize=8,
        color="#6B7280",
    )
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.05, right=0.97)

    os.makedirs(os.path.dirname(out_public), exist_ok=True)
    os.makedirs(os.path.dirname(out_solutions), exist_ok=True)
    plt.savefig(out_public, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.savefig(out_solutions, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()


def generate_plots():
    df = pd.read_csv('SOLUTIONS/HEURISTICS/run_summaries.csv')
    
    # Filter out rows where n_total is missing
    df = df.dropna(subset=['n_total'])
    
    # Calculate Approximation Ratio
    df['approx_ratio'] = df['best_profit'] / df['classical_opt_profit']
    
    # Calculate Feasibility Rate
    df['feasibility_rate'] = df['num_samples_feasible'] / df['num_samples_total']
    
    # Create output directory
    os.makedirs('public/plots/heuristics', exist_ok=True)
    
    # 1. Approximation Ratio vs Problem Size
    plt.figure(figsize=(8, 5))
    
    # Plot Classical
    classical_df = df[df['algorithm'] == 'classical']
    if not classical_df.empty:
        sns.lineplot(data=classical_df, x='n_total', y='approx_ratio', marker='o', label='Classical', color=COLOR_CLASSICAL, errorbar=None)
    
    # Plot QAOA COBYLA
    cobyla_df = df[(df['algorithm'] == 'qaoa') & (df['optimizer'] == 'cobyla')]
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='approx_ratio', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
        
    # Plot QAOA SPSA
    spsa_df = df[(df['algorithm'] == 'qaoa') & (df['optimizer'] == 'spsa')]
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='approx_ratio', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Solution Quality: Approximation Ratio vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_vs_size.png', dpi=300)
    plt.close()
    
    # 2. Runtime vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not classical_df.empty:
        sns.lineplot(data=classical_df, x='n_total', y='runtime_sec', marker='o', label='Classical', color=COLOR_CLASSICAL, errorbar=None)
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='runtime_sec', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='runtime_sec', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Workflow Cost: Runtime vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_vs_size.png', dpi=300)
    plt.close()
    
    # 3. Feasibility Rate vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='feasibility_rate', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='feasibility_rate', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Constraint Handling: Feasibility Rate vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Feasibility Rate', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/feasibility_vs_size.png', dpi=300)
    plt.close()

    # 4. Circuit Resources vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='num_qubits', marker='s', label='Qubits (COBYLA)', color=COLOR_QAOA, linestyle='-', errorbar=None)
        sns.lineplot(data=cobyla_df, x='n_total', y='two_qubit_gate_count', marker='s', label='Two-Qubit Gates (COBYLA)', color=COLOR_QAOA, linestyle='--', errorbar=None)
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='num_qubits', marker='^', label='Qubits (SPSA)', color=COLOR_TRAVELERS_RED, linestyle='-', errorbar=None)
        sns.lineplot(data=spsa_df, x='n_total', y='two_qubit_gate_count', marker='^', label='Two-Qubit Gates (SPSA)', color=COLOR_TRAVELERS_RED, linestyle='--', errorbar=None)
        
    plt.title('Hardware Resources: Qubits & Gates vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Resource / Optimizer', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/resources_vs_size.png', dpi=300)
    plt.close()
    
    # 5. Approximation Ratio vs Circuit Depth (p)
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='p', y='approx_ratio', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='p', y='approx_ratio', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Depth Tradeoff: Approximation Ratio vs Circuit Depth (p)', fontsize=14, pad=15)
    plt.xlabel('Circuit Depth (p)', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.xticks([1, 2, 3])
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_vs_p.png', dpi=300)
    plt.close()
    
    # 6. Objective Evaluations vs Problem Size
    plt.figure(figsize=(8, 5))
    
    if not cobyla_df.empty:
        sns.lineplot(data=cobyla_df, x='n_total', y='num_objective_evals', marker='s', label='QAOA (COBYLA)', color=COLOR_QAOA, errorbar=('ci', 95))
    if not spsa_df.empty:
        sns.lineplot(data=spsa_df, x='n_total', y='num_objective_evals', marker='^', label='QAOA (SPSA)', color=COLOR_TRAVELERS_RED, errorbar=('ci', 95))
        
    plt.title('Optimization Effort: Objective Evaluations vs Problem Size', fontsize=14, pad=15)
    plt.xlabel('Problem Size (n_total)', fontsize=12)
    plt.ylabel('Objective Evaluations', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Algorithm / Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/evals_vs_size.png', dpi=300)
    plt.close()

    # 7. Runtime vs N_local for fixed M_blocks
    plt.figure(figsize=(8, 5))
    qaoa_df = df[df['algorithm'] == 'qaoa']
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='runtime_sec', hue='M_blocks', marker='o', palette='viridis', errorbar=None)
        
    plt.title('Runtime Scaling for Fixed Number of Packages (M)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Packages (M_blocks)')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_fixed_m.png', dpi=300)
    plt.close()
    
    # 8. Per-Block Qubits and Gates vs N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='num_qubits', marker='o', label='Qubits', color=COLOR_QAOA, errorbar=None)
        sns.lineplot(data=qaoa_df, x='N_local', y='two_qubit_gate_count', marker='s', label='Two-Qubit Gates', color=COLOR_TRAVELERS_RED, errorbar=None)
        
    plt.title('Per-Block Hardware Resources vs Block Size (N_local)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Resource')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/resources_vs_nlocal.png', dpi=300)
    plt.close()
    
    # 9. Runtime vs M_blocks for fixed N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='M_blocks', y='runtime_sec', hue='N_local', marker='o', palette='plasma', errorbar=None)
        
    plt.title('Runtime Scaling for Fixed Coverage per Package (N_local)', fontsize=14, pad=15)
    plt.xlabel('Packages (M_blocks)', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.legend(title='Coverage (N_local)')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_fixed_n.png', dpi=300)
    plt.close()

    # 9b. Runtime vs M_blocks — proportional shot budget (dashboard style, matches other HEURISTICS figures)
    plt.figure(figsize=(8, 5))
    local_mask = qaoa_df["notes"].fillna("").str.contains("execution_target=local")
    prop = qaoa_df[local_mask & (qaoa_df["optimizer"] == "cobyla")].copy()
    if not prop.empty:
        prop["M_int"] = prop["M_blocks"].astype(int)
        prop["shots_i"] = prop["num_samples_total"].astype(int)
        prop = prop[(prop["N_local"].astype(int) == 10) & (prop["p"].astype(int) == 2)]
        prop = prop[prop["shots_i"] == 3072 * prop["M_int"]]
        prop = prop.sort_values("run_id").groupby("M_int", as_index=False).last()
        prop = prop.sort_values("M_int")
    if not prop.empty and len(prop) >= 2:
        x = prop["M_int"].to_numpy(dtype=float)
        y = prop["runtime_sec"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x + intercept
        ss_res = float(np.sum((y - y_hat) ** 2))
        y_mean = float(np.mean(y))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        ax = plt.gca()
        sns.scatterplot(
            x=prop["M_int"],
            y=prop["runtime_sec"],
            s=110,
            marker="s",
            color=COLOR_QAOA,
            label="QAOA (COBYLA) measured",
            zorder=3,
            ax=ax,
        )
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            "--",
            color=COLOR_QAOA,
            linewidth=2,
            alpha=0.9,
            label="Linear fit",
        )
        ax.text(
            0.98,
            0.06,
            f"$R^2 = {r2:.4f}$\n≈ {slope:.1f} s / package\n$N_{{local}}=10$, $p=2$\nshots $= 3072 \\times M$",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#FFFFFF", edgecolor="#E5E7EB", alpha=0.96),
        )
        plt.title("Workflow Cost: Runtime vs Packages (Proportional Budget)", fontsize=14, pad=15)
        plt.xlabel("Packages (M_blocks)", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.legend(title="Series", loc="upper left")
    else:
        plt.title("Workflow Cost: Runtime vs Packages (Proportional Budget)", fontsize=14, pad=15)
        plt.xlabel("Packages (M_blocks)", fontsize=12)
        plt.ylabel("Runtime (seconds)", fontsize=12)
        plt.text(
            0.5,
            0.5,
            "No proportional-budget rows\n($N_{local}=10$, $p=2$, COBYLA, shots $=3072 \\times M_{blocks}$)",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=11,
            color="#6B7280",
        )
    plt.tight_layout()
    _pub_m = os.path.join("public", "plots", "heuristics", "runtime_vs_m_proportional_budget.png")
    _sol_m = os.path.join("SOLUTIONS", "HEURISTICS", "plots", "runtime_vs_m_blocks_n10_p2_cobyla_k3072.png")
    plt.savefig(_pub_m, dpi=300)
    plt.savefig(_sol_m, dpi=300)
    plt.close()

    # 10. 3D Runtime Landscape (Smooth Surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use SPSA as the representative for the smooth surface to avoid overlapping chaotic surfaces
    spsa_df = qaoa_df[qaoa_df['optimizer'] == 'spsa'].groupby(['N_local', 'M_blocks'])['runtime_sec'].mean().reset_index()
    
    if len(spsa_df) >= 3: # Need at least 3 points for a surface
        try:
            surf = ax.plot_trisurf(spsa_df['N_local'], spsa_df['M_blocks'], spsa_df['runtime_sec'], 
                                   cmap='plasma', edgecolor='none', alpha=0.9)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Runtime (s)')
        except Exception as e:
            print(f"Could not plot trisurf for runtime: {e}")
            ax.scatter(spsa_df['N_local'], spsa_df['M_blocks'], spsa_df['runtime_sec'], c=COLOR_TRAVELERS_RED, s=80)
                   
    ax.set_xlabel('Coverage per Package (N_local)', fontsize=10, labelpad=10)
    ax.set_ylabel('Packages (M_blocks)', fontsize=10, labelpad=10)
    ax.set_zlabel('Runtime (seconds)', fontsize=10, labelpad=10)
    ax.set_title('3D Runtime Landscape (QAOA SPSA)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/3d_runtime_surface.png', dpi=300)
    plt.close()
    
    # 11. 3D Approximation Ratio Landscape (Smooth Surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    spsa_approx_df = qaoa_df[qaoa_df['optimizer'] == 'spsa'].groupby(['N_local', 'M_blocks'])['approx_ratio'].mean().reset_index()
    
    if len(spsa_approx_df) >= 3:
        try:
            surf = ax.plot_trisurf(spsa_approx_df['N_local'], spsa_approx_df['M_blocks'], spsa_approx_df['approx_ratio'], 
                                   cmap='viridis', edgecolor='none', alpha=0.9)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Approximation Ratio')
        except Exception as e:
            print(f"Could not plot trisurf for approx ratio: {e}")
            ax.scatter(spsa_approx_df['N_local'], spsa_approx_df['M_blocks'], spsa_approx_df['approx_ratio'], c=COLOR_QAOA, s=80)
                   
    ax.set_xlabel('Coverage per Package (N_local)', fontsize=10, labelpad=10)
    ax.set_ylabel('Packages (M_blocks)', fontsize=10, labelpad=10)
    ax.set_zlabel('Approximation Ratio', fontsize=10, labelpad=10)
    ax.set_title('3D Solution Quality Landscape (QAOA SPSA)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/3d_approx_ratio_surface.png', dpi=300)
    plt.close()

    # 12. Profit Comparison (Scatter)
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.scatterplot(data=qaoa_df, x='classical_opt_profit', y='best_profit', hue='optimizer', style='optimizer', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, s=100, alpha=0.7)
        max_val = max(qaoa_df['classical_opt_profit'].max(), qaoa_df['best_profit'].max())
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Optimal (y=x)')
    plt.title('QAOA Best Profit vs Classical Optimal', fontsize=14, pad=15)
    plt.xlabel('Classical Optimal Profit', fontsize=12)
    plt.ylabel('QAOA Best Profit', fontsize=12)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/profit_scatter.png', dpi=300)
    plt.close()

    # 13. Approx Ratio Distribution by Optimizer
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.boxplot(data=qaoa_df, x='optimizer', y='approx_ratio', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'})
    plt.title('Approximation Ratio Distribution by Optimizer', fontsize=14, pad=15)
    plt.xlabel('Optimizer', fontsize=12)
    plt.ylabel('Approximation Ratio', fontsize=12)
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/approx_ratio_dist.png', dpi=300)
    plt.close()

    # 14. Runtime vs Objective Evals
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.scatterplot(data=qaoa_df, x='num_objective_evals', y='runtime_sec', hue='optimizer', style='optimizer', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, s=100, alpha=0.7)
    plt.title('Runtime vs Objective Evaluations', fontsize=14, pad=15)
    plt.xlabel('Objective Evaluations', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/runtime_vs_evals.png', dpi=300)
    plt.close()

    # 15. Feasibility vs Depth (p)
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='p', y='feasibility_rate', hue='optimizer', marker='o', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, errorbar=('ci', 95))
    plt.title('Feasibility Rate vs Circuit Depth (p)', fontsize=14, pad=15)
    plt.xlabel('Circuit Depth (p)', fontsize=12)
    plt.ylabel('Feasibility Rate', fontsize=12)
    plt.xticks([1, 2, 3])
    plt.ylim(0, 1.1)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/feasibility_vs_p.png', dpi=300)
    plt.close()

    # 16. Compiled Depth vs N_local
    plt.figure(figsize=(8, 5))
    if not qaoa_df.empty:
        sns.lineplot(data=qaoa_df, x='N_local', y='circuit_depth', hue='optimizer', marker='o', palette={'cobyla': COLOR_QAOA, 'spsa': COLOR_TRAVELERS_RED, 'random_batch': '#666666'}, errorbar=None)
    plt.title('Compiled Circuit Depth vs Block Size (N_local)', fontsize=14, pad=15)
    plt.xlabel('Coverage per Package (N_local)', fontsize=12)
    plt.ylabel('Compiled Circuit Depth', fontsize=12)
    plt.legend(title='Optimizer')
    plt.tight_layout()
    plt.savefig('public/plots/heuristics/compiled_depth_vs_nlocal.png', dpi=300)
    plt.close()

    # 17. Table: unique (N_local, M_blocks, p) with methods (presentation asset)
    generate_simulation_coverage_by_nmp_png()

    # 18. Table: aggregate wall time & summary metrics (presentation asset)
    generate_run_summary_metrics_table_png()

if __name__ == '__main__':
    generate_plots()
    print("Plots generated successfully.")
