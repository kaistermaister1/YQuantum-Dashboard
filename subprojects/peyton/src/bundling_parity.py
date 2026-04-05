"""Build parity-check matrices ``(B, v)`` for DQI from a :class:`BundlingProblem`.

The hackathon ILP (challenge brief / ``insurance_model.build_ilp``, Eqs. 27–31 analog) uses
mandatory families, optional families, package capacity, incompatibility, and dependencies.
The QUBO bridge in ``qubo_block.build_qubo_block_for_package`` introduces the **same**
slack variables and equalities for every constraint except mandatory families (which stay as
``(∑ x − 1)²`` on coverage bits only).

This module encodes **GF(2) linear equations** that match the **slack gadgets** used in that
QUBO (same variable order and slack placement). Those equations are **necessary conditions**
for feasibility in ℤ; they are **not** always sufficient (e.g. multi-bit capacity or
“exactly one of three” mandatory products cannot be captured by a single XOR). See Travelers
``04_dqi_pipeline.html`` on gadgets vs taking inequalities mod 2.

Loading data
------------
``ltm_coverages.csv`` (code-examples teaser) lists coverages only. The full **YQH26** instance
uses six ``instance_*.csv`` files under ``LTM/YQH26_data``. :func:`load_bundling_problem_for_challenge`
prefers that directory when it can be found next to your CSV path.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from insurance_model import BundlingProblem, InsuranceCoverage, load_ltm_instance
except ImportError:
    from src.insurance_model import BundlingProblem, InsuranceCoverage, load_ltm_instance

from src.dqi_core import parity_B_v_from_qubo
from src.qubo_block import build_qubo_block_for_package


@dataclass
class BundlingParityBuild:
    """Result of building ``B y = v (mod 2)`` for one package column."""

    B: np.ndarray
    v: np.ndarray
    n_coverage: int
    n_slack: int
    package_index: int
    variable_labels: list[str]
    """Index ``k`` is ``variable_labels[k]`` (coverage ``x_i`` or slack ``s_*``)."""

    mode: str
    notes: list[str] = field(default_factory=list)


def discover_yqh26_instance_dir(near_path: str | Path) -> Path | None:
    """If ``near_path`` is under a challenge tree, locate ``.../LTM/YQH26_data`` with ``instance_coverages.csv``."""
    p = Path(near_path).resolve()
    if p.is_file():
        p = p.parent
    candidates = [
        p.parent / "LTM" / "YQH26_data",
        p.parent.parent / "LTM" / "YQH26_data",
        p.parent.parent.parent / "LTM" / "YQH26_data",
        p / "YQH26_data",
        p / "LTM" / "YQH26_data",
    ]
    for c in candidates:
        if (c / "instance_coverages.csv").is_file():
            return c
    return None


def bundling_problem_from_ltm_coverages_csv(
    csv_path: str | Path,
    *,
    num_packages: int = 2,
    max_options_per_package: int = 7,
) -> BundlingProblem:
    """Load only ``ltm_coverages.csv``-style rows into a minimal :class:`BundlingProblem`.

    Expected columns: ``name``, ``family``, ``price``, ``take_rate``, ``margin_pct``, ``mandatory``
    (``mandatory`` may be ``True``/``False`` strings).

    **No** segment affinity, per-package discounts, dependencies, or incompatibility rules are
    loaded—use :func:`load_ltm_instance` on ``YQH26_data`` for the full YQH26 instance aligned
    with the challenge PDF.
    """
    path = Path(csv_path)
    coverages: list[InsuranceCoverage] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            fam = row["family"].strip()
            price = float(row["price"])
            take_rate = float(row["take_rate"])
            margin = float(row["margin_pct"])
            mand_raw = str(row.get("mandatory", "False")).strip().lower()
            mandatory = mand_raw in ("true", "1", "yes")
            coverages.append(
                InsuranceCoverage(
                    name=name,
                    family=fam,
                    price=price,
                    take_rate=take_rate,
                    contribution_margin_pct=margin,
                    is_mandatory_in_family=mandatory,
                )
            )
    return BundlingProblem(
        coverages=coverages,
        num_packages=int(num_packages),
        max_options_per_package=int(max_options_per_package),
        compatibility_rules=[],
        dependency_rules=[],
    )


def load_bundling_problem_for_challenge(
    data_path: str | Path,
    *,
    beta: float = 1.2,
) -> tuple[BundlingProblem, dict[str, Any]]:
    """Load a bundling problem suitable for parity / QUBO pipelines.

    * If ``data_path`` is a **directory** containing ``instance_coverages.csv``, calls
      :func:`load_ltm_instance` (full LTM / YQH26 folder).
    * If ``data_path`` is **``ltm_coverages.csv``** (or any CSV), tries
      :func:`discover_yqh26_instance_dir` first; on success loads ``YQH26_data`` via
      ``load_ltm_instance``. Otherwise falls back to :func:`bundling_problem_from_ltm_coverages_csv`.

    Returns ``(problem, meta)`` where ``meta["source"]`` describes which branch ran.
    """
    path = Path(data_path)
    if path.is_dir():
        prob = load_ltm_instance(path, beta=beta)
        return prob, {"source": "ltm_instance_dir", "path": str(path.resolve())}

    if not path.is_file():
        raise FileNotFoundError(path)

    yqh26 = discover_yqh26_instance_dir(path)
    if yqh26 is not None:
        prob = load_ltm_instance(yqh26, beta=beta)
        return prob, {
            "source": "yqh26_instance_dir",
            "path": str(yqh26.resolve()),
            "trigger_csv": str(path.resolve()),
        }

    prob = bundling_problem_from_ltm_coverages_csv(path)
    return prob, {
        "source": "ltm_coverages_csv_only",
        "path": str(path.resolve()),
        "warning": "No YQH26_data found nearby; dependencies, incompatibilities, and "
        "affinities are empty. Place CSV under challenge/code_examples/data and keep "
        "LTM/YQH26_data alongside, or pass the YQH26_data directory directly.",
    }


def _variable_labels(N: int, n_slack: int) -> list[str]:
    labels = [f"x_{i}" for i in range(N)]
    for s in range(n_slack):
        labels.append(f"s_{N + s}")
    return labels


def _plan_slack_counts(problem: BundlingProblem) -> tuple[int, int]:
    """Mirror slack **counts** in ``build_qubo_block_for_package`` (must stay in sync)."""
    N = problem.N
    K = problem.max_options_per_package
    slack_count = 0
    for _fam, indices in problem.optional_families.items():
        if len(indices) > 1:
            slack_count += 1
    cap_slacks = max(1, int(np.ceil(np.log2(K + 1))))
    slack_count += cap_slacks
    for rule in problem.compatibility_rules:
        if not rule.compatible:
            slack_count += 1
    slack_count += len(problem.dependency_rules)
    return slack_count, cap_slacks


def build_parity_B_v_gadgets(problem: BundlingProblem, package_index: int) -> BundlingParityBuild:
    """Construct ``B, v`` from the same slack gadgets as ``build_qubo_block_for_package``.

    Rows (all arithmetic mod 2):

    * **Mandatory family, |F| = 2:** ``x_a ⊕ x_b = 1`` (matches “exactly one” for two products).
    * **Mandatory family, |F| ≠ 2:** omitted (cannot be expressed as a single XOR on
      coverage bits alone); a note is appended.
    * **Optional family (|F| > 1):** ``(⊕_{i ∈ F} x_i) ⊕ s = 1`` with the same slack ``s`` as QUBO.
    * **Capacity:** ``(⊕_{i=0}^{N-1} x_i) ⊕ s_0 = K mod 2`` using only the **LSB** slack bit
      (``2^0``); necessary but not sufficient for ``∑ x + ∑ 2^b s_b = K``.
    * **Incompatibility:** ``x_i ⊕ x_j ⊕ s = 1``.
    * **Dependency:** ``x_j ⊕ x_i ⊕ s = 0`` for the gadget ``(x_j - x_i + s)²`` / ``x_j ≤ x_i``.

    Variable order: coverage indices ``0 .. N-1``, then slacks in the same order as
    ``build_qubo_block_for_package`` (optional-family slacks, then capacity slacks, then one
    slack per incompatibility, then one per dependency).
    """
    m = package_index
    if m < 0 or m >= problem.M:
        raise ValueError(f"package_index {m} out of range for M={problem.M}")

    N = problem.N
    K = int(problem.max_options_per_package)
    n_slack, cap_slacks = _plan_slack_counts(problem)
    n_total = N + n_slack
    labels = _variable_labels(N, n_slack)
    notes: list[str] = []

    rows: list[list[int]] = []
    v_list: list[int] = []
    seen_rows: set[tuple[int, ...]] = set()

    def add_row(cols: dict[int, int], rhs_mod2: int) -> None:
        r = [0] * n_total
        for idx, coeff in cols.items():
            if coeff % 2 != 0:
                r[idx] = 1
        key = tuple(r)
        if key in seen_rows:
            return
        seen_rows.add(key)
        rows.append(r)
        v_list.append(int(rhs_mod2) & 1)

    # --- Mandatory (no slack): only |F| == 2 is a single XOR for "exactly one"
    for fam_name, indices in problem.mandatory_families.items():
        if len(indices) == 2:
            i, j = int(indices[0]), int(indices[1])
            add_row({i: 1, j: 1}, 1)
        elif len(indices) > 2:
            notes.append(
                f"mandatory family {fam_name!r} has |F|={len(indices)}; "
                "no single XOR row (use QUBO penalty or a multi-ancilla encoding)."
            )

    slack_count = 0

    def next_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        idxs = list(range(start, start + n_bits))
        slack_count += n_bits
        return idxs

    # --- Optional families
    for fam_name, indices in problem.optional_families.items():
        if len(indices) <= 1:
            continue
        sidx = next_slack(1)[0]
        cols = {int(i): 1 for i in indices}
        cols[sidx] = 1
        add_row(cols, 1)

    # --- Capacity: sum x_i + sum 2^b s_b = K  =>  mod 2: xor x xor s_0 = K mod 2
    s_cap = next_slack(cap_slacks)
    cols_cap = {i: 1 for i in range(N)}
    cols_cap[s_cap[0]] = 1
    add_row(cols_cap, K & 1)

    # --- Incompatibility
    for rule in problem.compatibility_rules:
        if rule.compatible:
            continue
        i = problem.coverage_index(rule.coverage_i)
        j = problem.coverage_index(rule.coverage_j)
        si = next_slack(1)[0]
        add_row({i: 1, j: 1, si: 1}, 1)

    # --- Dependency x_j <= x_i
    for rule in problem.dependency_rules:
        i = problem.coverage_index(rule.requires)
        j = problem.coverage_index(rule.dependent)
        si = next_slack(1)[0]
        add_row({j: 1, i: 1, si: 1}, 0)

    assert slack_count == n_slack, "slack accounting mismatch (sync with qubo_block)"

    if not rows:
        B = np.zeros((0, n_total), dtype=np.int64)
        v = np.zeros(0, dtype=np.int64)
    else:
        B = np.asarray(rows, dtype=np.int64)
        v = np.asarray(v_list, dtype=np.int64)

    return BundlingParityBuild(
        B=B,
        v=v,
        n_coverage=N,
        n_slack=n_slack,
        package_index=m,
        variable_labels=labels,
        mode="gadgets",
        notes=notes,
    )


def build_parity_B_v_from_qubo_block(
    problem: BundlingProblem,
    package_index: int,
    *,
    tol: float = 1e-9,
) -> BundlingParityBuild:
    """Derive ``B, v`` from the **off-diagonal pattern** of the package QUBO (see ``parity_B_v_from_qubo``)."""
    block = build_qubo_block_for_package(problem, package_index)
    B, v = parity_B_v_from_qubo(block.Q, tol=tol)
    n_total = block.n_vars
    labels = _variable_labels(block.n_coverage, block.n_slack)
    return BundlingParityBuild(
        B=B,
        v=v,
        n_coverage=block.n_coverage,
        n_slack=block.n_slack,
        package_index=package_index,
        variable_labels=labels,
        mode="from_qubo",
        notes=[
            "Each row is x_a ⊕ x_b = 1 for a significant QUBO coupling; this is a heuristic "
            "and may not match integer ILP feasibility."
        ],
    )


def build_parity_B_v_for_package(
    problem: BundlingProblem,
    package_index: int,
    *,
    mode: Literal["gadgets", "from_qubo"] = "gadgets",
    qubo_tol: float = 1e-9,
) -> BundlingParityBuild:
    """Main entry: parity matrices for package ``package_index``."""
    if mode == "gadgets":
        return build_parity_B_v_gadgets(problem, package_index)
    if mode == "from_qubo":
        return build_parity_B_v_from_qubo_block(problem, package_index, tol=qubo_tol)
    raise ValueError(f"Unknown mode: {mode!r}")


def main() -> None:
    """Example: load from default Downloads path or ``YQH26_DATA`` env and print shapes."""
    import os
    import sys

    default_csv = Path(
        r"C:\Users\allie\Downloads\challenge\challenge\code_examples\data\ltm_coverages.csv"
    )
    data_arg = Path(os.environ.get("BUNDLING_DATA_PATH", default_csv))
    if len(sys.argv) > 1:
        data_arg = Path(sys.argv[1])

    problem, meta = load_bundling_problem_for_challenge(data_arg)
    print("load:", meta)
    m = 0
    g = build_parity_B_v_for_package(problem, m, mode="gadgets")
    print(f"package {m} gadgets: B shape {g.B.shape}, v shape {g.v.shape}")
    if g.notes:
        for line in g.notes:
            print(" note:", line)
    q = build_parity_B_v_for_package(problem, m, mode="from_qubo")
    print(f"package {m} from_qubo: B shape {q.B.shape}, v shape {q.v.shape}")


if __name__ == "__main__":
    main()
