"""Turn the insurance bundling ILP into a per-package QUBO (quadratic unconstrained binary optimization).

Why this exists
---------------
Quantum heuristics like QAOA usually need an **Ising / QUBO**: a single energy function
``E(x)`` over binary bits ``x``, to **minimize**. The hackathon ILP is instead
**maximize margin** subject to **linear** constraints (families, capacity, etc.).
This file builds the standard **penalty bridge**: move constraints into the objective as
squared terms scaled by a weight ``λ`` (``penalty_weight``), and **flip** the linear
profit to minimization. See ``Travelers/docs/02_ilp_to_qubo.html`` for the same story
with pictures.

What is a "block"?
------------------
The full problem has ``N`` coverages and ``M`` packages → ``N×M`` decision bits
``x[i,m]``. In the QUBO used here, **there is no coupling between different packages**:
each package column is its own subproblem. So we build **one** symmetric matrix
``Q`` per package ``m`` (a **block**). The full ``(NM)×(NM)`` picture would be
**block-diagonal** with ``M`` copies; this code never materializes that giant matrix,
it just returns each block. That matches the "exploit block structure" guidance in
the challenge materials.

What are the variables in one block?
------------------------------------
Indices ``0 .. N-1`` are the **coverage** bits for that package (include coverage ``i``
or not). Extra bits after that are **slack** variables: dummy binary variables used
only so an **inequality** can be rewritten as an **equality** and then squared.
Example: ``x_i + x_j ≤ 1`` becomes ``x_i + x_j + s = 1``; if both ``x_i`` and ``x_j``
are 1, no choice of ``s ∈ {0,1}`` satisfies the equality, so the squared penalty is
positive. Capacity ``∑ x_i ≤ K`` uses several slack bits so ``∑ x_i + (binary encoding
of slack) = K`` exactly.

What does ``Q`` mean?
---------------------
We store a **symmetric** ``Q`` and define (for binary ``x`` of length ``n``)::

    energy(x) = xᵀ Q x + constant_offset

The linear profit terms sit on the **diagonal** of ``Q`` as ``-c[i,m]`` (minimize
``-margin`` = maximize margin). Each squared penalty adds diagonal entries, pairwise
``(i,j)`` entries, and a piece ``λ·rhs²`` accumulated into ``constant_offset`` so the
numeric value of ``energy(x)`` matches the true ``-margin + λ·(violations)²`` expansion
without dropping constants that would shift the minimum.

Role of ``λ`` (penalty_weight)
------------------------------
If ``λ`` is too small, the solver can prefer **high margin** assignments that **break**
constraints because the profit outweighs the penalty. If ``λ`` is large enough, any
feasible ILP solution beats any infeasible one. The slides suggest starting around
``3 × max |c|``; the unit tests use a larger value on a tiny instance so brute-force
agrees with PuLP.

Main entry points
-----------------
``build_qubo_block_for_package(problem, m, ...)`` → one :class:`QuboBlock`.
``build_all_qubo_blocks(problem, ...)`` → ``M`` blocks, one per package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from insurance_model import BundlingProblem
except ImportError:
    from src.insurance_model import BundlingProblem


def _max_margin_coeff(problem: BundlingProblem, package_index: int) -> float:
    m = package_index
    vals = []
    for i, cov in enumerate(problem.coverages):
        delta_m = problem.get_discount(m)
        alpha_im = problem.get_affinity(i, m)
        beta = problem.price_sensitivity_beta
        vals.append(
            abs(
                cov.price
                * cov.contribution_margin_pct
                * (1 - delta_m)
                * cov.take_rate
                * alpha_im
                * (1 + beta * delta_m)
            )
        )
    return max(vals) if vals else 1.0


def default_penalty_weight(problem: BundlingProblem, package_index: int, factor: float = 3.0) -> float:
    """Kickoff-slide rule of thumb: start around ``factor * max |c_i|`` for that package."""
    return factor * _max_margin_coeff(problem, package_index)


def _add_squared_linear_penalty(
    Q: np.ndarray,
    coeffs: dict[int, float],
    rhs: float,
    lam: float,
) -> float:
    """Add ``lam * (sum_i coeffs[i]*x_i - rhs)**2`` to symmetric Q (minimize x.T @ Q @ x).

    Binary identity ``x_i^2 = x_i`` is used. Returns the **constant** term ``lam * rhs**2``
    so callers can accumulate a global offset (needed for correct energies / argmin).
    """
    idx = sorted(coeffs.keys())
    for i in idx:
        ai = coeffs[i]
        Q[i, i] += lam * (ai * ai - 2 * rhs * ai)
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            i, j = idx[ii], idx[jj]
            w = lam * coeffs[i] * coeffs[j]
            Q[i, j] += w
            Q[j, i] += w
    return lam * rhs * rhs


@dataclass
class QuboBlock:
    """One package-local QUBO block."""

    package_index: int
    """0-based package index this block belongs to."""

    Q: np.ndarray
    """Symmetric real matrix; energy ``x.T @ Q @ x`` for binary ``x`` (length ``n_vars``)."""

    n_coverage: int
    """``N`` (number of coverage decision bits in this block)."""

    n_slack: int
    """Number of slack binary variables appended after coverage indices."""

    coverage_offset: int
    """Always ``0``; slacks live at indices ``N .. N+n_slack-1``."""

    penalty_weight: float
    """``lambda`` used for all penalties in this block."""

    constant_offset: float = 0.0
    """Sum of ``lambda * rhs**2`` from squared penalties (see ``_add_squared_linear_penalty``)."""

    @property
    def n_vars(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        if x.shape[0] != self.n_vars:
            raise ValueError(f"x length {x.shape[0]} != n_vars {self.n_vars}")
        return float(x @ self.Q @ x) + self.constant_offset


def build_qubo_block_for_package(
    problem: BundlingProblem,
    package_index: int,
    penalty_weight: float | None = None,
) -> QuboBlock:
    """Construct the ``N + n_slack`` dimensional QUBO for one package.

    Penalties match the ILP in ``insurance_model.build_ilp`` for that column ``m``:

    * mandatory family: ``(sum_{i in F} x_i - 1)^2``
    * optional family (|F| > 1): ``(sum_{i in F} x_i + s - 1)^2``
    * capacity: ``(sum_i x_i + sum_b 2^b s_b - K)^2`` with ``ceil(log2(K+1))`` slacks
    * incompatibility: ``(x_i + x_j + s - 1)^2``
    * dependency: ``(x_j - x_i + s)^2``

    Objective (maximize margin) enters as **minimize** ``-sum_i c_{i,m} x_i`` on the diagonal
    via the same squared-linear helper (linear-only contribution).
    """
    m = package_index
    if m < 0 or m >= problem.M:
        raise ValueError(f"package_index {m} out of range for M={problem.M}")

    lam = (
        float(penalty_weight)
        if penalty_weight is not None
        else default_penalty_weight(problem, m)
    )

    N = problem.N
    K = problem.max_options_per_package

    slack_count = 0

    def alloc_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        idxs = list(range(start, start + n_bits))
        slack_count += n_bits
        return idxs

    # --- plan slacks (only counts dimensions; indices are fixed below) ---
    for _fam, indices in problem.optional_families.items():
        if len(indices) > 1:
            alloc_slack(1)

    cap_slacks = int(np.ceil(np.log2(K + 1)))
    if cap_slacks < 1:
        cap_slacks = 1
    alloc_slack(cap_slacks)

    for rule in problem.compatibility_rules:
        if not rule.compatible:
            alloc_slack(1)

    for _rule in problem.dependency_rules:
        alloc_slack(1)

    n_slack = slack_count
    n_total = N + n_slack
    Q = np.zeros((n_total, n_total), dtype=float)
    const_total = 0.0

    # --- objective: minimize -sum c_i x_i  =>  add (sum (-c_i) x_i - 0)^2 with lam_obj? ---
    # Use linear energy directly: Q[i,i] += -c_i for x.T Q x gives sum -c_i x_i^2 = sum -c_i x_i
    coverages = problem.coverages
    beta = problem.price_sensitivity_beta
    delta_m = problem.get_discount(m)
    for i in range(N):
        cov = coverages[i]
        alpha_im = problem.get_affinity(i, m)
        c_im = (
            cov.price
            * cov.contribution_margin_pct
            * (1 - delta_m)
            * cov.take_rate
            * alpha_im
            * (1 + beta * delta_m)
        )
        Q[i, i] -= c_im

    slack_count = 0

    def next_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        idxs = list(range(start, start + n_bits))
        slack_count += n_bits
        return idxs

    # Mandatory: (sum x_i - 1)^2
    for _fam, indices in problem.mandatory_families.items():
        coeffs = {i: 1.0 for i in indices}
        const_total += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    # Optional (|F|>1): (sum_{i in F} x_i + s - 1)^2
    for _fam, indices in problem.optional_families.items():
        if len(indices) <= 1:
            continue
        sidx = next_slack(1)[0]
        coeffs = {i: 1.0 for i in indices}
        coeffs[sidx] = 1.0
        const_total += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    # Capacity: sum_i x_i + sum_b 2^b s_b = K
    s_list = next_slack(cap_slacks)
    coeffs_cap: dict[int, float] = {i: 1.0 for i in range(N)}
    for b, si in enumerate(s_list):
        coeffs_cap[si] = float(2**b)
    const_total += _add_squared_linear_penalty(Q, coeffs_cap, float(K), lam)

    # Incompatibility
    for rule in problem.compatibility_rules:
        if rule.compatible:
            continue
        i = problem.coverage_index(rule.coverage_i)
        j = problem.coverage_index(rule.coverage_j)
        si = next_slack(1)[0]
        coeffs = {i: 1.0, j: 1.0, si: 1.0}
        const_total += _add_squared_linear_penalty(Q, coeffs, 1.0, lam)

    # Dependency x_j <= x_i  =>  (x_j - x_i + s)^2
    for rule in problem.dependency_rules:
        i = problem.coverage_index(rule.requires)
        j = problem.coverage_index(rule.dependent)
        si = next_slack(1)[0]
        coeffs = {j: 1.0, i: -1.0, si: 1.0}
        const_total += _add_squared_linear_penalty(Q, coeffs, 0.0, lam)

    assert slack_count == n_slack, "slack accounting mismatch"

    return QuboBlock(
        package_index=m,
        Q=Q,
        n_coverage=N,
        n_slack=n_slack,
        coverage_offset=0,
        penalty_weight=lam,
        constant_offset=const_total,
    )


def build_all_qubo_blocks(
    problem: BundlingProblem,
    penalty_weight: float | None = None,
) -> list[QuboBlock]:
    """One :class:`QuboBlock` per package ``m = 0 .. M-1``."""
    blocks: list[QuboBlock] = []
    for m in range(problem.M):
        lam_m = penalty_weight if penalty_weight is not None else default_penalty_weight(problem, m)
        blocks.append(build_qubo_block_for_package(problem, m, lam_m))
    return blocks
