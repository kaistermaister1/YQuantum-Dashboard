#!/usr/bin/env python3
"""Runtime vs problem size n_total = N_local × M_blocks with fixed QAOA budget.

Strict mode (default with ``--auto``): same ``num_samples_total``, ``p``, ``M_blocks``, λ,
and ``execution_target=local`` so scaling is not mixed with different shot budgets.

With ``--relax-shots``, every ``n`` that has classical + local COBYLA + local SPSA for the
chosen ``(m, p)`` is included; the latest row per series is used (shots and λ may differ).
Use ``--qaoa-lambda-norm`` (default ``mean`` when relaxed) to plot QAOA time divided by a
scalar derived from that row’s λ vector so penalty strength is less confounded; classical
stays in raw seconds on a second y-axis when λ-normalization is on.

Classical points use the same (N_local, M_blocks) as each QAOA point.

If you omit ``--m/--p/--shots/--lambda``, pass ``--auto`` to pick the (m, p, shots, λ)
with the largest set of n values where classical + local COBYLA + local SPSA all exist.

Requires: matplotlib, numpy.

Extrapolation (default on): QAOA series use a log–log (power-law) fit; classical uses a
linear fit in n_total. Dotted lines extend the fit; any “crossover” where extrapolated
QAOA wall time drops below classical is **not** a claim of real quantum advantage—only
the intersection of these simple models fit to the plotted points. Crossover markers are
omitted when QAOA is λ-normalized and classical uses a separate axis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_algorithm_comparison_table import lambdas_close, load_rows, parse_lambda_tuple

REPO_SOLUTIONS = Path(__file__).resolve().parent

CLASSICAL_GREEN = "#2D8C3C"
QAOA_COBYLA = "#0066CC"
QAOA_SPSA = "#00356B"
SHELL_BG = "#F3F4F6"


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


def is_local_qaoa(row: dict[str, str]) -> bool:
    if (row.get("algorithm") or "").strip() != "qaoa":
        return False
    opt = (row.get("optimizer") or "").strip().lower()
    if opt not in {"cobyla", "spsa"}:
        return False
    return "execution_target=local" in (row.get("notes") or "")


def lam_norm(raw: str) -> str:
    return (raw or "").strip()


def lambda_scale(row: dict[str, str], how: str) -> float:
    """Scalar from CSV λ column for normalizing QAOA runtime (penalty magnitude)."""
    t = parse_lambda_tuple(row.get("lambda") or "")
    if t is None or len(t) == 0:
        return 1.0
    v = np.abs(np.asarray(t, dtype=float))
    if how == "mean":
        s = float(np.mean(v))
    elif how == "sum":
        s = float(np.sum(v))
    elif how == "l2":
        s = float(np.linalg.norm(v))
    else:
        return 1.0
    return max(s, 1e-12)


def find_best_auto_relaxed(rows: list[dict[str, str]]) -> tuple[int, int, list[int]]:
    """Pick (m, p) with the most n values where classical + local COBYLA + local SPSA exist (shots/λ may vary)."""
    classical_ns = set()
    for r in rows:
        if (r.get("algorithm") or "").strip() != "classical":
            continue
        n, m = _i(r, "N_local"), _i(r, "M_blocks")
        if n is not None and m is not None:
            classical_ns.add((n, m))

    q = [r for r in rows if is_local_qaoa(r)]
    mp_set: set[tuple[int, int]] = set()
    for r in q:
        mm, pp = _i(r, "M_blocks"), _i(r, "p")
        if mm is not None and pp is not None:
            mp_set.add((mm, pp))

    best: tuple[int, int, int, int, list[int]] = (0, 0, 0, 0, [])
    for m, p in sorted(mp_set):
        ns = sorted(
            {
                _i(r, "N_local")
                for r in q
                if _i(r, "M_blocks") == m and _i(r, "p") == p and _i(r, "N_local") is not None
            }
        )
        ok: list[int] = []
        for n in ns:
            if (n, m) not in classical_ns:
                continue
            co = [
                r
                for r in q
                if _i(r, "N_local") == n
                and _i(r, "M_blocks") == m
                and _i(r, "p") == p
                and (r.get("optimizer") or "").strip().lower() == "cobyla"
            ]
            sp = [
                r
                for r in q
                if _i(r, "N_local") == n
                and _i(r, "M_blocks") == m
                and _i(r, "p") == p
                and (r.get("optimizer") or "").strip().lower() == "spsa"
            ]
            if co and sp:
                ok.append(int(n))
        if len(ok) < 2:
            continue
        span = max(ok) - min(ok)
        cand = (len(ok), m, p, span, ok)
        if cand > best:
            best = cand

    if best[0] < 2:
        raise SystemExit(
            "Could not find at least two n values with classical + local COBYLA + local SPSA "
            "for any (m, p). Add runs or use strict --auto without --relax-shots."
        )
    return best[1], best[2], best[4]


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (A, beta) with y ≈ A * x**beta (least squares in log space)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fit requires positive x and y")
    logx = np.log(x)
    logy = np.log(y)
    beta, logA = np.polyfit(logx, logy, 1)
    return float(np.exp(logA)), float(beta)


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (intercept, slope) for y ≈ slope * x + intercept (numpy.polyfit convention)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.polyfit(x, y, 1)
    slope, intercept = float(c[0]), float(c[1])
    return intercept, slope


def crossing_power_vs_constant(A: float, beta: float, c: float, x_lo: float, x_hi: float) -> float | None:
    """Find x in [x_lo, x_hi] where A*x**beta = c (positive QAOA time vs constant classical)."""

    def f(xv: float) -> float:
        return A * (xv**beta) - c

    xs = np.linspace(x_lo, x_hi, 512)
    fs = np.array([f(float(t)) for t in xs])
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            return float(xs[i])
        if fs[i] * fs[i + 1] < 0:
            lo, hi = float(xs[i]), float(xs[i + 1])
            flo, fhi = fs[i], fs[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if abs(fm) < 1e-9 * (abs(flo) + abs(fhi) + 1e-12):
                    return mid
                if flo * fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            return 0.5 * (lo + hi)
    return None


def crossing_power_vs_linear(
    A: float,
    beta: float,
    a: float,
    b: float,
    x_lo: float,
    x_hi: float,
    *,
    n_scan: int = 512,
) -> float | None:
    """Find x in [x_lo, x_hi] where A*x**beta = a + b*x (QAOA == classical), if bracket exists."""

    def f(xv: float) -> float:
        return A * (xv**beta) - (a + b * xv)

    xs = np.linspace(x_lo, x_hi, n_scan)
    fs = np.array([f(float(t)) for t in xs])
    for i in range(len(xs) - 1):
        if fs[i] == 0.0:
            return float(xs[i])
        if fs[i] * fs[i + 1] < 0:
            lo, hi = float(xs[i]), float(xs[i + 1])
            flo, fhi = fs[i], fs[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if abs(fm) < 1e-9 * (abs(flo) + abs(fhi) + 1e-12):
                    return mid
                if flo * fm <= 0:
                    hi, fhi = mid, fm
                else:
                    lo, flo = mid, fm
            return 0.5 * (lo + hi)
    return None


def find_best_auto(rows: list[dict[str, str]]) -> tuple[int, int, int, str, list[int]]:
    """Return (m, p, shots, lambda_raw, sorted_n_list) maximizing |n| with all three series."""
    classical_ns = set()
    for r in rows:
        if (r.get("algorithm") or "").strip() != "classical":
            continue
        n, m = _i(r, "N_local"), _i(r, "M_blocks")
        if n is not None and m is not None:
            classical_ns.add((n, m))

    q = [r for r in rows if is_local_qaoa(r)]
    # group (m,p,shots,lam) -> opt -> set of n
    from collections import defaultdict

    groups: dict[tuple[int, int, int, str], dict[str, set[int]]] = defaultdict(
        lambda: {"cobyla": set(), "spsa": set()}
    )
    for r in q:
        n, m, p = _i(r, "N_local"), _i(r, "M_blocks"), _i(r, "p")
        shots = _i(r, "num_samples_total")
        if n is None or m is None or p is None or shots is None or shots <= 0:
            continue
        opt = (r.get("optimizer") or "").strip().lower()
        lam = lam_norm(r.get("lambda") or "")
        groups[(m, p, shots, lam)][opt].add(n)

    best: tuple[int, int, int, int, list[int]] = (0, 0, 0, 0, [])
    best_key: tuple[int, int, int, str] | None = None
    for key, d in groups.items():
        m, p, shots, lam = key
        nc = d["cobyla"]
        ns = d["spsa"]
        common = sorted(nc & ns)
        ok = [n for n in common if (n, m) in classical_ns]
        if not ok:
            continue
        span = max(ok) - min(ok)
        # Prefer more n points; then larger M (richer packages); then larger p; then wider n span.
        cand = (len(ok), m, p, span, ok)
        if cand > best:
            best = cand
            best_key = key

    if not best_key or best[0] < 2:
        raise SystemExit(
            "Could not find at least two n values with classical + local COBYLA + local SPSA "
            "at the same (m, p, shots, λ). Pass explicit --m --p --shots --lambda, or add runs."
        )
    m, p, shots, lam = best_key
    return m, p, shots, lam, best[4]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summaries", type=Path, default=REPO_SOLUTIONS / "HEURISTICS" / "run_summaries.csv")
    ap.add_argument("--auto", action="store_true", help="Pick (m,p,shots,λ) with most matching n values.")
    ap.add_argument(
        "--relax-shots",
        action="store_true",
        help=(
            "Do not require the same num_samples_total (or the same λ) across n. "
            "Includes every n with classical + local COBYLA + SPSA for the chosen (m,p). "
            "Use with --qaoa-lambda-norm (default mean when relaxed) to reduce penalty-strength bias."
        ),
    )
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--p", type=int, default=None)
    ap.add_argument("--shots", type=int, default=None, help="Match CSV num_samples_total exactly (strict mode).")
    ap.add_argument("--lambda", dest="lambda_str", default=None, help="λ string as in CSV (list or scalar).")
    ap.add_argument(
        "--qaoa-lambda-norm",
        choices=["none", "mean", "sum", "l2"],
        default=None,
        help="Divide QAOA wall time by this λ statistic (default: mean if --relax-shots, else none). "
        "Classical stays in seconds on a second y-axis when this is not none.",
    )
    ap.add_argument("--linear-y", action="store_true")
    ap.add_argument(
        "--no-extrapolate",
        action="store_true",
        help="Do not draw dotted extrapolations or crossover markers.",
    )
    ap.add_argument(
        "--extrapolate-x-min",
        type=float,
        default=1.0,
        help="Extrapolation and crossover search start at this n_total (≥1).",
    )
    ap.add_argument(
        "--extrapolate-x-max-factor",
        type=float,
        default=12.0,
        help="Extrapolation grid ends at this * max n_total.",
    )
    ap.add_argument(
        "--classical-extrap",
        choices=["constant", "linear"],
        default="constant",
        help=(
            "How to extrapolate classical: constant median wall time (default, stable for noisy ILP times) "
            "or linear least-squares in n_total."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_SOLUTIONS / "HEURISTICS" / "plots")
    ap.add_argument("--output-name", default=None, help="PNG filename (default from m,p,shots).")
    args = ap.parse_args()

    rows = load_rows(args.summaries)

    if args.qaoa_lambda_norm is not None:
        qaoa_norm_how = args.qaoa_lambda_norm
    else:
        qaoa_norm_how = "mean" if args.relax_shots else "none"

    shots: int | None
    lam_raw: str
    lambda_target: tuple[float, ...] | None

    if args.auto:
        if args.relax_shots:
            m, p, n_list = find_best_auto_relaxed(rows)
            shots = None
            lam_raw = "(mixed λ; shots may differ per n)"
            lambda_target = None
        else:
            m, p, shots_i, lam_raw, n_list = find_best_auto(rows)
            shots = int(shots_i)
            lambda_target = parse_lambda_tuple(lam_raw) if lam_raw else None
    else:
        if args.m is None or args.p is None:
            raise SystemExit("Pass --m and --p, or use --auto.")
        m, p = args.m, args.p
        if args.relax_shots:
            shots = None
            lam_raw = "(mixed λ; shots may differ per n)"
            lambda_target = None
            n_list = []
            cand_n = sorted(
                {
                    _i(r, "N_local")
                    for r in rows
                    if _i(r, "N_local") is not None and _i(r, "M_blocks") == m
                }
            )
            for n in cand_n:
                if n is None:
                    continue
                cl = [
                    r
                    for r in rows
                    if (r.get("algorithm") or "").strip() == "classical"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                ]
                co = [
                    r
                    for r in rows
                    if is_local_qaoa(r)
                    and (r.get("optimizer") or "").strip().lower() == "cobyla"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                    and _i(r, "p") == p
                ]
                sp = [
                    r
                    for r in rows
                    if is_local_qaoa(r)
                    and (r.get("optimizer") or "").strip().lower() == "spsa"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                    and _i(r, "p") == p
                ]
                if cl and co and sp:
                    n_list.append(n)
            n_list.sort()
            if len(n_list) < 2:
                raise SystemExit(f"No n≥2 with classical + COBYLA + SPSA for m={m} p={p} (relaxed).")
        else:
            if args.shots is None or args.lambda_str is None:
                raise SystemExit("Strict mode needs --shots and --lambda, or use --relax-shots / --auto.")
            shots = int(args.shots)
            lam_raw = args.lambda_str.strip()
            lambda_target = parse_lambda_tuple(lam_raw)
            if lambda_target is None:
                raise SystemExit("Could not parse --lambda.")
            n_list = []
            cand_n = sorted(
                {
                    _i(r, "N_local")
                    for r in rows
                    if _i(r, "N_local") is not None and _i(r, "M_blocks") == m
                }
            )
            for n in cand_n:
                if n is None:
                    continue
                cl = [
                    r
                    for r in rows
                    if (r.get("algorithm") or "").strip() == "classical"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                ]
                co = [
                    r
                    for r in rows
                    if is_local_qaoa(r)
                    and (r.get("optimizer") or "").strip().lower() == "cobyla"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                    and _i(r, "p") == p
                    and _i(r, "num_samples_total") == shots
                ]
                sp = [
                    r
                    for r in rows
                    if is_local_qaoa(r)
                    and (r.get("optimizer") or "").strip().lower() == "spsa"
                    and _i(r, "N_local") == n
                    and _i(r, "M_blocks") == m
                    and _i(r, "p") == p
                    and _i(r, "num_samples_total") == shots
                ]
                co = [r for r in co if lambdas_close(parse_lambda_tuple(r.get("lambda") or "") or (), lambda_target)]
                sp = [r for r in sp if lambdas_close(parse_lambda_tuple(r.get("lambda") or "") or (), lambda_target)]
                if cl and co and sp:
                    n_list.append(n)
            n_list.sort()
            if len(n_list) < 2:
                raise SystemExit(
                    f"No n≥2 with classical + COBYLA + SPSA for m={m} p={p} shots={shots} and that λ. "
                    "Try --relax-shots."
                )

    # Resolve rows per n (latest run_id if duplicate — should not happen)
    def latest(rs: list[dict[str, str]]) -> dict[str, str]:
        return max(rs, key=lambda r: r.get("run_id", ""))

    n_totals: list[int] = []
    y_cl: list[float] = []
    y_co: list[float] = []
    y_sp: list[float] = []
    eval_co: list[int] = []
    eval_sp: list[int] = []
    shots_co: list[int] = []
    shots_sp: list[int] = []

    for n in n_list:
        classical = [r for r in rows if (r.get("algorithm") or "").strip() == "classical" and _i(r, "N_local") == n and _i(r, "M_blocks") == m]
        cobyla = [
            r
            for r in rows
            if is_local_qaoa(r)
            and (r.get("optimizer") or "").strip().lower() == "cobyla"
            and _i(r, "N_local") == n
            and _i(r, "M_blocks") == m
            and _i(r, "p") == p
            and (shots is None or _i(r, "num_samples_total") == shots)
        ]
        spsa = [
            r
            for r in rows
            if is_local_qaoa(r)
            and (r.get("optimizer") or "").strip().lower() == "spsa"
            and _i(r, "N_local") == n
            and _i(r, "M_blocks") == m
            and _i(r, "p") == p
            and (shots is None or _i(r, "num_samples_total") == shots)
        ]
        if lambda_target is not None:
            cobyla = [r for r in cobyla if lambdas_close(parse_lambda_tuple(r.get("lambda") or "") or (), lambda_target)]
            spsa = [r for r in spsa if lambdas_close(parse_lambda_tuple(r.get("lambda") or "") or (), lambda_target)]

        if not classical or not cobyla or not spsa:
            continue
        rc, rco, rsp = latest(classical), latest(cobyla), latest(spsa)
        rt_cl = _f(rc, "runtime_sec")
        rt_co = _f(rco, "runtime_sec")
        rt_sp = _f(rsp, "runtime_sec")
        if rt_cl is None or rt_co is None or rt_sp is None:
            continue
        n_totals.append(n * m)
        y_cl.append(rt_cl)
        sc_co = lambda_scale(rco, qaoa_norm_how) if qaoa_norm_how != "none" else 1.0
        sc_sp = lambda_scale(rsp, qaoa_norm_how) if qaoa_norm_how != "none" else 1.0
        y_co.append(rt_co / sc_co)
        y_sp.append(rt_sp / sc_sp)
        eval_co.append(_i(rco, "num_objective_evals") or 0)
        eval_sp.append(_i(rsp, "num_objective_evals") or 0)
        shots_co.append(_i(rco, "num_samples_total") or 0)
        shots_sp.append(_i(rsp, "num_samples_total") or 0)

    if len(n_totals) < 2:
        raise SystemExit("Fewer than two points after row selection.")

    x = np.array(n_totals, dtype=float)
    lam_src = lam_raw if args.auto or args.relax_shots else (args.lambda_str or "")
    lam_note = (lam_src[:77] + "…") if len(lam_src) > 80 else lam_src
    use_twin = qaoa_norm_how != "none"

    fig, ax_q = plt.subplots(figsize=(8.5, 5.4))
    fig.patch.set_facecolor("white")
    ax_q.set_facecolor(SHELL_BG)
    if use_twin:
        ax_c = ax_q.twinx()
        ax_c.spines["right"].set_edgecolor(CLASSICAL_GREEN)
        ax_c.tick_params(axis="y", labelcolor=CLASSICAL_GREEN)
        ax_c.grid(False)
    else:
        ax_c = ax_q

    y_cl_a = np.array(y_cl, dtype=float)
    y_co_a = np.array(y_co, dtype=float)
    y_sp_a = np.array(y_sp, dtype=float)

    ax_c.plot(x, y_cl_a, color=CLASSICAL_GREEN, marker="o", linewidth=2.0, markersize=7, label="Classical (ILP)")
    ax_q.plot(x, y_co_a, color=QAOA_COBYLA, marker="s", linewidth=2.0, markersize=7, label="QAOA COBYLA (local)")
    ax_q.plot(x, y_sp_a, color=QAOA_SPSA, marker="^", linewidth=2.0, markersize=7, label="QAOA SPSA (local)")

    cross_notes: list[str] = []
    x_ex_hi = float(x.max()) * float(args.extrapolate_x_max_factor)
    x_ex_lo = max(1.0, float(args.extrapolate_x_min))
    x_ex = np.linspace(x_ex_lo, x_ex_hi, 400)

    if not args.no_extrapolate:
        A_co, beta_co = fit_power_law(x, y_co_a)
        A_sp, beta_sp = fit_power_law(x, y_sp_a)

        if args.classical_extrap == "constant":
            a_cl = float(np.median(y_cl_a))
            b_cl = 0.0
            y_cl_e = np.full_like(x_ex, a_cl, dtype=float)
        else:
            icept, slope = fit_linear(x, y_cl_a)
            a_cl, b_cl = icept, slope
            y_cl_e = a_cl + b_cl * x_ex

        y_co_e = A_co * np.power(x_ex, beta_co)
        y_sp_e = A_sp * np.power(x_ex, beta_sp)
        if not args.linear_y:
            y_cl_e = np.maximum(y_cl_e, 1e-9)
            y_co_e = np.maximum(y_co_e, 1e-9)
            y_sp_e = np.maximum(y_sp_e, 1e-9)

        ax_c.plot(
            x_ex,
            y_cl_e,
            ":",
            color=CLASSICAL_GREEN,
            linewidth=1.6,
            alpha=0.9,
            label="Classical (median extrap.)" if args.classical_extrap == "constant" else "Classical (linear extrap.)",
        )
        ax_q.plot(
            x_ex,
            y_co_e,
            ":",
            color=QAOA_COBYLA,
            linewidth=1.6,
            alpha=0.9,
            label="COBYLA (extrap.)",
        )
        ax_q.plot(
            x_ex,
            y_sp_e,
            ":",
            color=QAOA_SPSA,
            linewidth=1.6,
            alpha=0.9,
            label="SPSA (extrap.)",
        )

        if use_twin:
            cross_notes.append(
                "Crossover markers skipped: QAOA y is λ-normalized; classical stays in raw seconds (dual y-axis)."
            )
        else:
            for idx, (name, Aq, beta, color) in enumerate(
                (
                    ("COBYLA", A_co, beta_co, QAOA_COBYLA),
                    ("SPSA", A_sp, beta_sp, QAOA_SPSA),
                )
            ):
                if args.classical_extrap == "constant":
                    xc = crossing_power_vs_constant(Aq, beta, a_cl, x_ex_lo, x_ex_hi)
                    yc = float(a_cl) if xc is not None else float("nan")
                else:
                    xc = crossing_power_vs_linear(Aq, beta, a_cl, b_cl, x_ex_lo, x_ex_hi)
                    yc = float(a_cl + b_cl * xc) if xc is not None else float("nan")
                if xc is not None and np.isfinite(xc) and x_ex_lo <= xc <= x_ex_hi:
                    ax_q.axvline(xc, color="#C41230", linestyle="--", linewidth=1.0, alpha=0.85)
                    ax_q.scatter([xc], [yc], color=color, s=36, zorder=5, edgecolors="#1F2937", linewidths=0.5)
                    ax_q.annotate(
                        f"{name}: extrap. QAOA = classical\nn_tot ≈ {xc:.2f}",
                        xy=(xc, yc),
                        xytext=(12, 18 - 28 * idx),
                        textcoords="offset points",
                        fontsize=7,
                        color="#374151",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
                    )
                    cross_notes.append(
                        f"{name}: extrapolated equal wall time at n_total ≈ {xc:.3f} (≈ {yc:.4g} s)"
                    )
                else:
                    cross_notes.append(
                        f"{name}: no crossover in [{x_ex_lo:.2g}, {x_ex_hi:.2g}] for power-law QAOA vs "
                        f"{'constant' if args.classical_extrap == 'constant' else 'linear'} classical"
                    )

    if not args.linear_y:
        ax_q.set_yscale("log")
        if use_twin:
            ax_c.set_yscale("log")

    log_note = "" if args.linear_y else ", log scale"
    ax_q.set_xlabel(r"Problem size $n_{\mathrm{total}} = N_{\mathrm{local}} \times M_{\mathrm{blocks}}$")
    if qaoa_norm_how == "none":
        ax_q.set_ylabel("Wall time (seconds)" + log_note)
    else:
        ax_q.set_ylabel(f"QAOA wall time ÷ λ ({qaoa_norm_how}) (s)" + log_note, color="#1F2937")
        ax_c.set_ylabel("Classical wall time (s)" + log_note, color=CLASSICAL_GREEN)

    if shots is None:
        fix_line = (
            f"Relaxed: M={m}, p={p}; latest run per series; shots COBYLA {shots_co} · SPSA {shots_sp} "
            "(may differ). Sub-second ILP times are noisy in the CSV."
        )
    else:
        fix_line = (
            f"Fixed: M={m}, p={p}, total shots={shots} (CSV num_samples_total), local QAOA, matching λ"
        )
    lam_line = f"λ-normalize QAOA: {qaoa_norm_how}" if qaoa_norm_how != "none" else ""
    subtitle = fix_line + "\n" + f"COBYLA evals: {eval_co} · SPSA evals: {eval_sp}"
    if lam_line:
        subtitle += "\n" + lam_line
    ax_q.set_title(r"Runtime vs $n_{\mathrm{total}}$ with matched QAOA budget" + "\n" + subtitle, fontsize=9)
    ax_q.grid(True, which="both", axis="y", color="#E5E7EB", linestyle="-", linewidth=0.7)
    ax_q.grid(True, which="major", axis="x", color="#E5E7EB", linestyle="-", linewidth=0.7)
    if use_twin:
        h1, l1 = ax_q.get_legend_handles_labels()
        h2, l2 = ax_c.get_legend_handles_labels()
        ax_q.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, framealpha=0.95, ncol=2)
    else:
        ax_q.legend(loc="upper left", fontsize=7, framealpha=0.95, ncol=2)
    ax_q.tick_params(axis="both", labelsize=9)
    ax_q.set_xticks(x)
    if not args.no_extrapolate:
        ax_q.set_xlim(left=max(0.5, x_ex_lo * 0.88), right=x_ex_hi * 1.02)

    fig.text(0.5, 0.02, f"λ note (truncated): {lam_note}", ha="center", fontsize=7, color="#374151")
    if not args.no_extrapolate:
        fig.text(
            0.5,
            0.055,
            "Dotted = power-law (QAOA) and "
            + ("flat median (classical)" if args.classical_extrap == "constant" else "linear (classical)")
            + "—illustrative only; not hardware quantum advantage.",
            ha="center",
            fontsize=6.5,
            color="#6B7280",
        )

    if args.output_name:
        out_name = args.output_name
    else:
        suf = ""
        if args.relax_shots:
            suf += "_relaxed"
        if qaoa_norm_how != "none":
            suf += f"_lamnorm_{qaoa_norm_how}"
        if shots is None:
            out_name = f"runtime_vs_n_total_m{m}_p{p}{suf}.png"
        else:
            out_name = f"runtime_vs_n_total_m{m}_p{p}_shots{shots}{suf}.png"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / out_name
    fig.subplots_adjust(bottom=0.18 if not args.no_extrapolate else 0.14)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out}")
    sh = shots if shots is not None else "relaxed"
    print(f"  m={m} p={p} shots={sh}  n_local list={n_list}  n_total={list(map(int, x))}  qaoa_λ_norm={qaoa_norm_how}")
    for line in cross_notes:
        print(f"  {line}")


if __name__ == "__main__":
    main()
