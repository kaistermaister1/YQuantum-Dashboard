#!/usr/bin/env python3
"""Generate a classic-style graph suite for hybrid DQI."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_benchmarks import benchmark_dqi_pipeline
from src.dqi_visualize import plot_convergence, plot_scaling_benchmark
from src.insurance_model import load_ltm_instance, subsample_problem
from src.qubo_block import build_qubo_block_for_package
from src.run_dqi import run_dqi_with_details


def _default_data_dir() -> Path:
    return ROOT.parent / "will" / "Travelers" / "docs" / "data" / "YQH26_data"


def _build_block(data_dir: Path, *, n_coverages: int, subsample_packages: int, package: int):
    problem = load_ltm_instance(data_dir)
    n_cov = min(int(n_coverages), int(problem.N))
    n_pkg = int(subsample_packages) if int(subsample_packages) > 0 else int(problem.M)
    n_pkg = min(n_pkg, int(problem.M))
    small = subsample_problem(problem, n_cov, n_pkg)
    if package < 0 or package >= small.M:
        raise ValueError(f"package {package} out of range for M={small.M}")
    block = build_qubo_block_for_package(small, package_index=int(package), penalty_weight=None)
    return small, block


def _run_hybrid(
    block,
    *,
    p: int,
    shots: int,
    seed: int,
    mixer: str = "h",
    statistic: str = "mean",
    gammas: list[float] | None = None,
    betas: list[float] | None = None,
):
    x, val, meta = run_dqi_with_details(
        block,
        p=int(p),
        shots=int(shots),
        seed=int(seed),
        mixer=str(mixer),
        statistic=str(statistic),
        execution="local",
        max_qubits=max(64, int(block.n_vars)),
        gammas=gammas,
        betas=betas,
    )
    return x, float(val), meta


def _convergence(
    block,
    *,
    p: int,
    shots: int,
    seed: int,
    n_eval: int,
) -> list[float]:
    hist: list[float] = []
    for i in range(int(n_eval)):
        _, val, _ = _run_hybrid(
            block,
            p=int(p),
            shots=int(shots),
            seed=int(seed) + i,
            mixer="h",
            statistic="mean",
        )
        hist.append(float(val))
    return hist


def _histogram_plot(block, meta, out_path: Path) -> None:
    counts = meta.run_result.stats_at_best.bitstring_counts
    energies: list[float] = []
    weights: list[float] = []
    for bitstring, c in counts.items():
        x = np.array([float(int(ch)) for ch in bitstring], dtype=float)
        energies.append(float(block.energy(x)))
        weights.append(float(c))
    e_arr = np.asarray(energies, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    w_arr = w_arr / np.sum(w_arr)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    bins = 24
    lo = float(np.min(e_arr))
    hi = float(np.max(e_arr))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.hist(e_arr, bins=np.linspace(lo - pad, hi + pad, bins + 1), weights=w_arr, color="#0066CC", edgecolor="white", linewidth=0.6, alpha=0.88)
    mean_e = float(np.sum(e_arr * w_arr))
    best_e = float(np.min(e_arr))
    ax.axvline(mean_e, color="#2D2926", linestyle="-", linewidth=2, label=f"sample mean = {mean_e:.4g}")
    ax.axvline(best_e, color="#2D8C3C", linestyle="--", linewidth=2, label=f"best in sample = {best_e:.4g}")
    ax.set_xlabel("QUBO energy  E(x) = x^T Q x + const")
    ax.set_ylabel("Fraction of shots")
    ax.set_title("Hybrid DQI energy histogram")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _landscape_plot(
    block,
    *,
    shots: int,
    seed: int,
    n_gamma: int,
    n_beta: int,
    out_png: Path,
    out_json: Path,
) -> None:
    gammas = np.linspace(0.0, np.pi, int(n_gamma), dtype=float)
    betas = np.linspace(0.0, np.pi, int(n_beta), dtype=float)
    Z = np.empty((len(betas), len(gammas)), dtype=float)
    rows: list[dict] = []
    k = 0
    total = len(gammas) * len(betas)
    for j, b in enumerate(betas):
        for i, g in enumerate(gammas):
            _, val, _ = _run_hybrid(
                block,
                p=1,
                shots=int(shots),
                seed=int(seed) + k,
                mixer="h",
                statistic="mean",
                gammas=[float(g)],
                betas=[float(b)],
            )
            Z[j, i] = float(val)
            rows.append({"gamma": float(g), "beta": float(b), "value": float(val)})
            k += 1
            if k % max(1, total // 10) == 0:
                print(f"  landscape {k}/{total} ...", flush=True)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[float(gammas[0]), float(gammas[-1]), float(betas[0]), float(betas[-1])],
        cmap="viridis",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.82)
    cb.set_label("Mean QUBO energy")
    ax.set_xlabel("gamma (rad)")
    ax.set_ylabel("beta (rad)")
    ax.set_title(f"Hybrid DQI p=1 landscape | {n_gamma}x{n_beta} grid")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "n_gamma": int(n_gamma),
                "n_beta": int(n_beta),
                "shots": int(shots),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@dataclass
class OptRow:
    method: str
    objective: float
    n_evaluations: int
    seconds: float
    p: int
    mixer: str


def _optimizer_benchmark(block, *, shots: int, seed: int, out_png: Path, out_csv: Path) -> None:
    search_space = [(1, "h"), (1, "rx"), (2, "h"), (2, "rx")]

    def eval_cfg(p: int, mixer: str, k: int) -> float:
        _, val, _ = _run_hybrid(block, p=p, shots=shots, seed=seed + k, mixer=mixer, statistic="mean")
        return float(val)

    rows: list[OptRow] = []

    t0 = time.perf_counter()
    best = None
    n_eval = 0
    for i, (p, mixer) in enumerate(search_space):
        v = eval_cfg(p, mixer, i)
        n_eval += 1
        cand = (v, p, mixer)
        if best is None or cand[0] < best[0]:
            best = cand
    rows.append(OptRow("grid", float(best[0]), n_eval, time.perf_counter() - t0, int(best[1]), str(best[2])))

    t1 = time.perf_counter()
    rng = np.random.default_rng(int(seed) + 999)
    best = None
    n_samples = 8
    for i in range(n_samples):
        p, mixer = search_space[int(rng.integers(0, len(search_space)))]
        v = eval_cfg(p, mixer, 100 + i)
        cand = (v, p, mixer)
        if best is None or cand[0] < best[0]:
            best = cand
    rows.append(OptRow("random", float(best[0]), n_samples, time.perf_counter() - t1, int(best[1]), str(best[2])))

    t2 = time.perf_counter()
    cur = (1, "h")
    evaluated: dict[tuple[int, str], float] = {}

    def get_val(cfg: tuple[int, str], idx: int) -> float:
        if cfg not in evaluated:
            evaluated[cfg] = eval_cfg(cfg[0], cfg[1], 200 + idx)
        return evaluated[cfg]

    improved = True
    idx = 0
    while improved:
        improved = False
        cur_val = get_val(cur, idx)
        idx += 1
        neighbors = [cfg for cfg in search_space if cfg != cur]
        best_nb = (cur_val, cur)
        for cfg in neighbors:
            v = get_val(cfg, idx)
            idx += 1
            if v < best_nb[0]:
                best_nb = (v, cfg)
        if best_nb[1] != cur:
            cur = best_nb[1]
            improved = True
    rows.append(OptRow("hillclimb", float(get_val(cur, idx)), len(evaluated), time.perf_counter() - t2, int(cur[0]), str(cur[1])))

    # plot
    names = [r.method for r in rows]
    colors = {"grid": "#0066CC", "random": "#5BA3D6", "hillclimb": "#2D8C3C"}
    bar_colors = [colors.get(n, "#455A64") for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.2), layout="constrained")

    axes[0, 0].bar(names, [r.objective for r in rows], color=bar_colors, edgecolor="white")
    axes[0, 0].set_ylabel("Best objective (lower is better)")
    axes[0, 0].set_title("Quality")
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    axes[0, 1].bar(names, [r.objective for r in rows], color=bar_colors, edgecolor="white")
    axes[0, 1].set_ylabel("Optimizer objective")
    axes[0, 1].set_title("What search minimized")
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    axes[1, 0].bar(names, [r.seconds for r in rows], color=bar_colors, edgecolor="white")
    axes[1, 0].set_ylabel("Wall time (s)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Runtime")
    axes[1, 0].grid(True, axis="y", alpha=0.25, which="both")

    axes[1, 1].bar(names, [r.n_evaluations for r in rows], color=bar_colors, edgecolor="white")
    axes[1, 1].set_ylabel("Hybrid DQI evaluations")
    axes[1, 1].set_title("Search cost")
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Hybrid DQI configuration benchmark", fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "objective", "n_evaluations", "seconds", "p", "mixer"])
        for r in rows:
            w.writerow([r.method, f"{r.objective:.17g}", r.n_evaluations, f"{r.seconds:.6f}", r.p, r.mixer])


def _scaling_rows(
    *,
    data_dir: Path,
    ns: list[int],
    package: int,
    subsample_packages: int,
    p: int,
    shots: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for n in ns:
        _, block = _build_block(data_dir, n_coverages=int(n), subsample_packages=int(subsample_packages), package=int(package))
        bench = benchmark_dqi_pipeline(
            block,
            p=int(p),
            shots=int(shots),
            dqi_seed=int(seed) + int(n),
            random_seed=int(seed) + 100 + int(n),
            brute_force_max_n=22,
            random_samples=int(shots),
            include_qaoa_baseline=False,
            include_local_search_baseline=True,
            mixer="h",
            statistic="mean",
            execution="local",
            max_qubits=max(64, int(block.n_vars)),
        )
        ref = float(bench["bruteforce"].best_value) if "bruteforce" in bench else None

        def pack(k: str) -> dict[str, float]:
            r = bench[k]
            extra = r.extra or {}
            cost = float(extra.get("n_energy_evaluations", extra.get("n_samples", 0)))
            return {"value": float(r.best_value), "time_s": float(r.runtime_sec), "cost": float(cost)}

        rows.append(
            {
                "n": int(n),
                "n_vars": int(block.n_vars),
                "reference_value": ref,
                "methods": {
                    "dqi": pack("dqi"),
                    "random": pack("random"),
                    "local_search": pack("local_search"),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_default_data_dir())
    ap.add_argument("--package", type=int, default=0)
    ap.add_argument("--subsample-packages", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shots", type=int, default=128)
    ap.add_argument("--convergence-evals", type=int, default=35)
    ap.add_argument("--convergence-n", type=int, default=10)
    ap.add_argument("--official-ns", type=int, nargs="*", default=[10], help="Rows for official-style comparison")
    ap.add_argument("--test-ns", type=int, nargs="*", default=[10, 20], help="Rows for test-style comparison")
    ap.add_argument("--landscape-n-gamma", type=int, default=4)
    ap.add_argument("--landscape-n-beta", type=int, default=4)
    ap.add_argument("--out-root", type=Path, default=ROOT / "artifacts")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    _, block10 = _build_block(
        data_dir,
        n_coverages=int(args.convergence_n),
        subsample_packages=int(args.subsample_packages),
        package=int(args.package),
    )

    out_root = args.out_root
    out_bench = out_root / "benchmark_official"
    out_hybrid = out_root / "benchmark_hybrid"
    out_root.mkdir(parents=True, exist_ok=True)
    out_bench.mkdir(parents=True, exist_ok=True)
    out_hybrid.mkdir(parents=True, exist_ok=True)

    # convergence counterpart
    conv_hist = _convergence(
        block10,
        p=2,
        shots=int(args.shots),
        seed=int(args.seed),
        n_eval=int(args.convergence_evals),
    )
    conv_path = out_root / "hybrid_dqi_convergence_10var.png"
    plot_convergence(conv_hist, out_path=conv_path, title="Hybrid DQI convergence (10 vars)")
    print("wrote:", conv_path)

    # histogram counterpart
    _, _, meta_hist = _run_hybrid(block10, p=2, shots=int(args.shots), seed=int(args.seed) + 500, mixer="h", statistic="mean")
    hist_path = out_hybrid / "hybrid_dqi_histogram_energy.png"
    _histogram_plot(block10, meta_hist, hist_path)
    print("wrote:", hist_path)

    # landscape counterpart
    land_png = out_hybrid / "hybrid_dqi_landscape.png"
    land_json = out_hybrid / "hybrid_dqi_landscape.json"
    _landscape_plot(
        block10,
        shots=max(64, int(args.shots) // 2),
        seed=int(args.seed) + 700,
        n_gamma=int(args.landscape_n_gamma),
        n_beta=int(args.landscape_n_beta),
        out_png=land_png,
        out_json=land_json,
    )
    print("wrote:", land_png)
    print("wrote:", land_json)

    # optimizer benchmark counterpart
    opt_png = out_hybrid / "hybrid_dqi_optimizer_benchmark.png"
    opt_csv = out_hybrid / "hybrid_dqi_optimizer_benchmark.csv"
    _optimizer_benchmark(block10, shots=int(args.shots), seed=int(args.seed) + 900, out_png=opt_png, out_csv=opt_csv)
    print("wrote:", opt_png)
    print("wrote:", opt_csv)

    # official/test comparison counterparts
    official_rows = _scaling_rows(
        data_dir=data_dir,
        ns=[int(n) for n in args.official_ns],
        package=int(args.package),
        subsample_packages=int(args.subsample_packages),
        p=2,
        shots=int(args.shots),
        seed=int(args.seed) + 1000,
    )
    official_plot = out_bench / "hybrid_dqi_official_sizes_comparison.png"
    plot_scaling_benchmark(
        official_rows,
        method_keys=("dqi", "random", "local_search"),
        out_path=official_plot,
        title_prefix="Hybrid official sizes benchmark",
    )
    print("wrote:", official_plot)

    test_rows = _scaling_rows(
        data_dir=data_dir,
        ns=[int(n) for n in args.test_ns],
        package=int(args.package),
        subsample_packages=int(args.subsample_packages),
        p=1,
        shots=int(args.shots),
        seed=int(args.seed) + 2000,
    )
    test_plot = out_bench / "hybrid_test_plot.png"
    plot_scaling_benchmark(
        test_rows,
        method_keys=("dqi", "random", "local_search"),
        out_path=test_plot,
        title_prefix="Hybrid DQI vs classical baselines",
    )
    print("wrote:", test_plot)

    rows_json = out_bench / "hybrid_replicated_plot_rows.json"
    rows_json.write_text(
        json.dumps(
            {
                "official_rows": official_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("wrote:", rows_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

