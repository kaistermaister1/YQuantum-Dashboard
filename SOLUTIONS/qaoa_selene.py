"""Selene-optimized Travelers package-local QAOA solver.

This runner reuses the QUBO construction from ``qaoa.py`` but changes the
execution strategy to better fit Nexus Selene:

* default to SPSA, which naturally batches ``theta+`` and ``theta-``
* submit those paired evaluations in a single Nexus execute job
* log in to Nexus once and reuse the same project reference
* optionally solve independent package-local blocks in parallel
"""

from __future__ import annotations

import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import qaoa as base_qaoa
    from qaoa_plots import save_training_loss_plot
except ModuleNotFoundError:
    import qaoa as base_qaoa
    from qaoa_plots import save_training_loss_plot


# Top-level choices
N_COVERAGES = base_qaoa.N_COVERAGES
M_PACKAGES = base_qaoa.M_PACKAGES
P_DEPTH = base_qaoa.P_DEPTH

# Selene-oriented QAOA settings
OPTIMIZER = "spsa"  # "spsa" is friendlier to remote execution than "cobyla"
SHOTS = base_qaoa.SHOTS
SEED = base_qaoa.SEED
MAX_QUBITS = base_qaoa.MAX_QUBITS
COBYLA_MAXITER = base_qaoa.COBYLA_MAXITER
SPSA_MAXITER = base_qaoa.SPSA_MAXITER
MAX_PARALLEL_PACKAGES = 1
NEXUS_PROJECT = "YQuantum QAOA"
NEXUS_VALID_CHECK = False
JOB_TIMEOUT_SECONDS = None

DATA_DIR = base_qaoa.DATA_DIR
PLOTS_DIR = base_qaoa.PLOTS_DIR


def default_loss_plot_path(n: int, m: int, p: int) -> Path:
    return PLOTS_DIR / f"qaoa_selene_training_loss_n{n}_m{m}_p{p}.png"


def _log(message: str) -> None:
    print(message, flush=True)


def _job_name(prefix: str, package_index: int | None = None) -> str:
    package_suffix = "" if package_index is None else f" package {package_index + 1}"
    return f"{prefix}{package_suffix} {time.time_ns()}"


@dataclass(frozen=True)
class SeleneSession:
    qnx: Any
    project: Any

    @classmethod
    def create(cls, project_name: str = NEXUS_PROJECT) -> "SeleneSession":
        try:
            import qnexus as qnx
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Selene execution requires the 'qnexus' package. Install it with 'pip install qnexus'."
            ) from exc

        qnx.login()
        project = qnx.projects.get_or_create(name=project_name)
        return cls(qnx=qnx, project=project)

    def make_config(self, n_qubits: int, seed: int):
        return self.qnx.models.SeleneConfig(
            n_qubits=int(n_qubits),
            simulator=self.qnx.models.StatevectorSimulator(seed=int(seed)),
            runtime=self.qnx.models.SimpleRuntime(seed=int(seed)),
            error_model=self.qnx.models.NoErrorModel(seed=int(seed)),
        )


def _stats_from_counts(block: base_qaoa.QuboBlock, counts: Counter[str]) -> base_qaoa.QaoaSampleStats:
    best_bitstring = min(
        counts,
        key=lambda bitstring: block.energy(np.array([int(char) for char in bitstring], dtype=float)),
    )
    best_energy = block.energy(np.array([int(char) for char in best_bitstring], dtype=float))
    return base_qaoa.QaoaSampleStats(dict(counts), best_bitstring, float(best_energy))


def _run_sources_on_selene(
    session: SeleneSession,
    block: base_qaoa.QuboBlock,
    sources: list[str],
    shots: int,
    seed: int,
    job_label: str,
) -> list[base_qaoa.QaoaSampleStats]:
    if block.n_vars > MAX_QUBITS:
        raise ValueError(f"Block uses {block.n_vars} qubits, which exceeds MAX_QUBITS={MAX_QUBITS}")

    hugr_refs = [
        session.qnx.hugr.upload(
            hugr_package=base_qaoa._compile_qaoa_hugr_from_source(source),
            name=_job_name(f"{job_label} HUGR", block.package_index),
            project=session.project,
        )
        for source in sources
    ]
    results = session.qnx.execute(
        programs=hugr_refs,
        n_shots=[int(shots)] * len(hugr_refs),
        n_qubits=[int(block.n_vars)] * len(hugr_refs),
        backend_config=session.make_config(block.n_vars, seed=seed),
        name=_job_name(job_label, block.package_index),
        project=session.project,
        valid_check=NEXUS_VALID_CHECK,
        timeout=JOB_TIMEOUT_SECONDS,
    )
    stats_list: list[base_qaoa.QaoaSampleStats] = []
    for result in results:
        counts = Counter(base_qaoa._shot_to_str(shot, block.n_vars) for shot in result.results)
        stats_list.append(_stats_from_counts(block, counts))
    return stats_list


def _run_single_source_on_selene(
    session: SeleneSession,
    block: base_qaoa.QuboBlock,
    source: str,
    shots: int,
    seed: int,
    job_label: str,
) -> base_qaoa.QaoaSampleStats:
    return _run_sources_on_selene(
        session=session,
        block=block,
        sources=[source],
        shots=shots,
        seed=seed,
        job_label=job_label,
    )[0]


def optimize_block_cobyla_selene(
    session: SeleneSession,
    block: base_qaoa.QuboBlock,
    p: int,
    shots: int,
    seed: int,
    evaluation_callback: base_qaoa.QaoaEvaluationCallback | None = None,
) -> base_qaoa.QaoaOptimizationResult:
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, math.pi, size=2 * p)
    best_theta = theta0.copy()
    best_stats: base_qaoa.QaoaSampleStats | None = None
    best_objective = float("inf")
    eval_index = 0
    trace = base_qaoa.QaoaOptimizationTrace()
    c_z, zz_terms = base_qaoa.qubo_to_ising_pauli_coefficients(block.Q)

    _log(
        f"[package {block.package_index + 1}] Starting Selene COBYLA on "
        f"{block.n_vars} qubits ({block.n_coverage} coverage + {block.n_slack} slack), "
        f"shots={shots}, maxiter={COBYLA_MAXITER}"
    )

    def objective(theta: np.ndarray) -> float:
        nonlocal best_theta, best_stats, best_objective, eval_index
        evaluation_number = eval_index + 1
        clipped = base_qaoa._clip_theta(theta)
        gammas, betas = base_qaoa._split_theta(clipped, p)
        source = base_qaoa._build_qaoa_source(c_z, zz_terms, gammas, betas)
        evaluation_start = time.perf_counter()
        stats = _run_single_source_on_selene(
            session=session,
            block=block,
            source=source,
            shots=shots,
            seed=seed + eval_index,
            job_label="Selene COBYLA",
        )
        value = base_qaoa._mean_sample_energy(block, stats)
        elapsed_seconds = time.perf_counter() - evaluation_start
        eval_index += 1
        note = ""
        improved = False
        if value < best_objective:
            best_objective = value
            best_theta = clipped.copy()
            best_stats = stats
            note = "new best"
            improved = True
        trace.record(value, best_stats=best_stats, improved=improved)
        base_qaoa._emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number,
            value,
            best_objective,
            stats,
            improved,
        )
        base_qaoa._report_evaluation(
            block,
            evaluation_number,
            value,
            best_objective,
            stats,
            elapsed_seconds,
            note=note,
        )
        return float(value)

    minimize(objective, theta0, method="COBYLA", options={"maxiter": int(COBYLA_MAXITER)})
    if best_stats is None:
        raise RuntimeError("Selene COBYLA did not produce any QAOA evaluations")
    _log(
        f"[package {block.package_index + 1}] Selene COBYLA finished after "
        f"{len(trace.objective_values)} evaluations with best_mean={best_objective:.6f}"
    )
    return base_qaoa.QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)


def optimize_block_spsa_selene(
    session: SeleneSession,
    block: base_qaoa.QuboBlock,
    p: int,
    shots: int,
    seed: int,
    evaluation_callback: base_qaoa.QaoaEvaluationCallback | None = None,
) -> base_qaoa.QaoaOptimizationResult:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, math.pi, size=2 * p)
    best_theta = theta.copy()
    best_stats: base_qaoa.QaoaSampleStats | None = None
    best_objective = float("inf")
    eval_index = 0
    trace = base_qaoa.QaoaOptimizationTrace()
    a = 0.15
    c = 0.12
    alpha = 0.602
    gamma_spsa = 0.101
    stability_A = 10.0
    c_z, zz_terms = base_qaoa.qubo_to_ising_pauli_coefficients(block.Q)

    _log(
        f"[package {block.package_index + 1}] Starting Selene SPSA on "
        f"{block.n_vars} qubits ({block.n_coverage} coverage + {block.n_slack} slack), "
        f"shots={shots}, maxiter={SPSA_MAXITER}"
    )

    for step in range(SPSA_MAXITER):
        ak = a / ((step + 1 + stability_A) ** alpha)
        ck = c / ((step + 1) ** gamma_spsa)
        delta = rng.choice([-1.0, 1.0], size=2 * p)

        theta_plus = base_qaoa._clip_theta(theta + ck * delta)
        theta_minus = base_qaoa._clip_theta(theta - ck * delta)
        source_plus = base_qaoa._build_qaoa_source(
            c_z,
            zz_terms,
            *base_qaoa._split_theta(theta_plus, p),
        )
        source_minus = base_qaoa._build_qaoa_source(
            c_z,
            zz_terms,
            *base_qaoa._split_theta(theta_minus, p),
        )
        evaluation_number = eval_index + 1
        evaluation_number_minus = eval_index + 2
        batch_start = time.perf_counter()
        stats_plus, stats_minus = _run_sources_on_selene(
            session=session,
            block=block,
            sources=[source_plus, source_minus],
            shots=shots,
            seed=seed + step,
            job_label=f"Selene SPSA step {step + 1}",
        )
        batch_elapsed = time.perf_counter() - batch_start
        value_plus = base_qaoa._mean_sample_energy(block, stats_plus)
        value_minus = base_qaoa._mean_sample_energy(block, stats_minus)
        eval_index += 2

        gradient = ((value_plus - value_minus) / (2.0 * ck)) * (1.0 / delta)
        theta = base_qaoa._clip_theta(theta - ak * gradient)

        improved_plus = False
        if value_plus < best_objective:
            best_objective = value_plus
            best_theta = theta_plus.copy()
            best_stats = stats_plus
            improved_plus = True
        best_plus = trace.record(value_plus, best_stats=best_stats, improved=improved_plus)
        base_qaoa._emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number,
            value_plus,
            best_plus,
            stats_plus,
            improved_plus,
        )
        base_qaoa._report_evaluation(
            block,
            evaluation_number,
            value_plus,
            best_plus,
            stats_plus,
            batch_elapsed,
            note=f"Selene SPSA step {step + 1} (+, batched){' new best' if improved_plus else ''}",
        )

        improved_minus = False
        if value_minus < best_objective:
            best_objective = value_minus
            best_theta = theta_minus.copy()
            best_stats = stats_minus
            improved_minus = True
        best_minus = trace.record(value_minus, best_stats=best_stats, improved=improved_minus)
        base_qaoa._emit_evaluation_callback(
            evaluation_callback,
            block,
            evaluation_number_minus,
            value_minus,
            best_minus,
            stats_minus,
            improved_minus,
        )
        base_qaoa._report_evaluation(
            block,
            evaluation_number_minus,
            value_minus,
            best_minus,
            stats_minus,
            batch_elapsed,
            note=f"Selene SPSA step {step + 1} (-, batched){' new best' if improved_minus else ''}",
        )

    if best_stats is None:
        raise RuntimeError("Selene SPSA did not produce any QAOA evaluations")
    _log(
        f"[package {block.package_index + 1}] Selene SPSA finished after "
        f"{len(trace.objective_values)} evaluations with best_mean={best_objective:.6f}"
    )
    return base_qaoa.QaoaOptimizationResult(theta=best_theta, stats=best_stats, trace=trace)


def optimize_block_selene(
    session: SeleneSession,
    block: base_qaoa.QuboBlock,
    p: int,
    shots: int,
    seed: int,
    evaluation_callback: base_qaoa.QaoaEvaluationCallback | None = None,
) -> base_qaoa.QaoaOptimizationResult:
    if OPTIMIZER == "cobyla":
        return optimize_block_cobyla_selene(
            session=session,
            block=block,
            p=p,
            shots=shots,
            seed=seed,
            evaluation_callback=evaluation_callback,
        )
    if OPTIMIZER == "spsa":
        return optimize_block_spsa_selene(
            session=session,
            block=block,
            p=p,
            shots=shots,
            seed=seed,
            evaluation_callback=evaluation_callback,
        )
    raise ValueError("OPTIMIZER must be 'cobyla' or 'spsa'")


def solve_qaoa_matrix(
    n: int = N_COVERAGES,
    m: int = M_PACKAGES,
    p: int = P_DEPTH,
    plot_output_path: str | Path | None = None,
) -> np.ndarray:
    workflow_start = time.perf_counter()
    resolved_plot_output_path = (
        Path(plot_output_path) if plot_output_path is not None else default_loss_plot_path(n=n, m=m, p=p)
    )
    session = SeleneSession.create(project_name=NEXUS_PROJECT)
    problem = base_qaoa.subsample_problem(base_qaoa.load_ltm_instance(DATA_DIR), n_coverages=n, m_packages=m)
    blocks = [base_qaoa.build_qubo_block_for_package(problem, package_index) for package_index in range(problem.M)]
    x_matrix = np.zeros((problem.N, problem.M), dtype=int)
    plot_series: list[tuple[str, list[float], list[float]]] = [("", [], []) for _ in range(problem.M)]

    _log(f"[workflow] Loading instance data from {DATA_DIR}")
    _log(
        f"[workflow] Solving {len(blocks)} package-local QUBO blocks on Selene "
        f"(N={problem.N} coverages, M={problem.M} packages, p={p}, optimizer={OPTIMIZER}, "
        f"parallel_packages={min(MAX_PARALLEL_PACKAGES, len(blocks))}, valid_check={NEXUS_VALID_CHECK})"
    )
    _log("[workflow] Selene SPSA batches theta+ and theta- into one Nexus job per optimization step.")

    def solve_single_block(package_index: int, block: base_qaoa.QuboBlock):
        block_start = time.perf_counter()
        block_summary = (
            f"[workflow] Package {package_index + 1}/{len(blocks)} block summary: "
            f"{block.n_vars} qubits = {block.n_coverage} coverage + {block.n_slack} slack"
        )
        if block.is_relaxed:
            block_summary += f" ({block.constraint_budget_summary()})"
        _log(block_summary)
        result = optimize_block_selene(
            session=session,
            block=block,
            p=p,
            shots=SHOTS,
            seed=SEED + package_index * 1000,
        )
        elapsed_seconds = time.perf_counter() - block_start
        return package_index, result, elapsed_seconds

    if len(blocks) == 1 or MAX_PARALLEL_PACKAGES <= 1:
        solved_blocks = [solve_single_block(package_index, block) for package_index, block in enumerate(blocks)]
    else:
        solved_blocks = []
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_PACKAGES, len(blocks))) as executor:
            futures = {
                executor.submit(solve_single_block, package_index, block): package_index
                for package_index, block in enumerate(blocks)
            }
            for future in as_completed(futures):
                solved_blocks.append(future.result())

    for package_index, result, elapsed_seconds in sorted(solved_blocks, key=lambda item: item[0]):
        coverage_bits = np.array([int(char) for char in result.stats.best_bitstring[:problem.N]], dtype=int)
        x_matrix[:, package_index] = coverage_bits
        plot_series[package_index] = (
            f"package {package_index + 1}",
            result.trace.objective_values,
            result.trace.best_objective_values,
        )
        _log(
            f"[workflow] Package {package_index + 1} complete in {elapsed_seconds:.2f}s with "
            f"best bitstring {result.stats.best_bitstring[:problem.N]} and sampled QUBO energy "
            f"{result.stats.best_qubo_energy:.6f}"
        )

    if plot_series:
        plot_path = save_training_loss_plot(plot_series, resolved_plot_output_path)
        _log(f"[workflow] Saved Selene QAOA training loss plot to {plot_path}")

    _log(f"[workflow] Finished all Selene package-local subproblems in {time.perf_counter() - workflow_start:.2f}s")
    return x_matrix


def main() -> np.ndarray:
    return solve_qaoa_matrix(n=N_COVERAGES, m=M_PACKAGES, p=P_DEPTH)


if __name__ == "__main__":
    M = main()
    print(M)
