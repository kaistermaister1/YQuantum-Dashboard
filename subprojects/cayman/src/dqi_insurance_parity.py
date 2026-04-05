"""GF(2) parity rows for insurance QUBO blocks (Travelers-style syndrome checks).

Each row is a XOR constraint ``(sum_{i: B[r,i]=1} x_i) mod 2 = rhs[r]`` on the **same**
variable ordering as :func:`qubo_block.build_qubo_block_for_package` (coverages then slacks).

These linear mod-2 relations are **necessary but not sufficient** for the full integer
slack equalities (e.g. mandatory ``sum x = 1`` with |F|>2, capacity with powers of two,
and dependency ``x_j - x_i + s = 0``). They are still useful as syndromes for DQI-style
post-selection and match the challenge narrative of XOR / parity checks.
"""

from __future__ import annotations

import numpy as np

try:
    from insurance_model import BundlingProblem
except ImportError:
    from src.insurance_model import BundlingProblem


def build_insurance_parity_B_rhs(problem: BundlingProblem, package_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``B`` (n_checks, n_vars) uint8 and ``rhs`` (n_checks,) uint8 in ``{0,1}``.

    Slack layout matches :func:`qubo_block.build_qubo_block_for_package`.
    """
    m = package_index
    if m < 0 or m >= problem.M:
        raise ValueError(f"package_index {m} out of range for M={problem.M}")

    N = problem.N
    K = problem.max_options_per_package

    slack_count = 0

    def alloc_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        idxs = list(range(start, start + n_bits))
        slack_count += n_bits
        return idxs

    cap_slacks = int(np.ceil(np.log2(K + 1)))
    if cap_slacks < 1:
        cap_slacks = 1

    for _fam, indices in problem.optional_families.items():
        if len(indices) > 1:
            alloc_slack(1)

    alloc_slack(cap_slacks)

    for rule in problem.compatibility_rules:
        if not rule.compatible:
            alloc_slack(1)

    for _rule in problem.dependency_rules:
        alloc_slack(1)

    n_vars = N + slack_count
    rows: list[list[int]] = []
    rhss: list[int] = []

    def add_row(cols: list[int], rhs: int) -> None:
        rows.append(cols)
        rhss.append(int(rhs) & 1)

    # Mandatory: weak XOR of all family bits = 1 (mod 2)
    for _fam, indices in problem.mandatory_families.items():
        add_row(list(indices), 1)

    slack_count = 0

    def next_slack(n_bits: int) -> list[int]:
        nonlocal slack_count
        start = N + slack_count
        idxs = list(range(start, start + n_bits))
        slack_count += n_bits
        return idxs

    # Optional (|F|>1): sum x + s = 1  ->  XOR = 1
    for _fam, indices in problem.optional_families.items():
        if len(indices) <= 1:
            continue
        sidx = next_slack(1)[0]
        add_row(list(indices) + [sidx], 1)

    # Capacity: integer sum x + sum 2^b s_b = K  ->  mod 2:  sum x + s_0 = K (mod 2)
    s_cap = next_slack(cap_slacks)
    cols_cap = list(range(N)) + [s_cap[0]]
    add_row(cols_cap, int(K) & 1)

    # Incompatibility: x_i + x_j + s = 1
    for rule in problem.compatibility_rules:
        if rule.compatible:
            continue
        i = problem.coverage_index(rule.coverage_i)
        j = problem.coverage_index(rule.coverage_j)
        si = next_slack(1)[0]
        add_row([i, j, si], 1)

    # Dependency x_j <= x_i  encoded as (x_j - x_i + s)^2 with (x_j + x_i + s) mod 2 = 0
    for rule in problem.dependency_rules:
        i = problem.coverage_index(rule.requires)
        j = problem.coverage_index(rule.dependent)
        si = next_slack(1)[0]
        add_row([i, j, si], 0)

    assert slack_count == n_vars - N

    B = np.zeros((len(rows), n_vars), dtype=np.uint8)
    for r, cols in enumerate(rows):
        for c in cols:
            B[r, c] ^= 1
    rhs = np.array(rhss, dtype=np.uint8)
    return B, rhs


def syndrome_ok(bitstring: str, B: np.ndarray, rhs: np.ndarray) -> bool:
    """True if all parity checks pass for problem bits (length ``B.shape[1]``)."""
    x = np.array([int(ch) for ch in bitstring], dtype=np.uint8)
    if x.shape[0] != B.shape[1]:
        return False
    syn = (B @ x) % 2
    return bool(np.all(syn == rhs))


def postselect_bitstring_counts(
    counts: dict[str, int],
    B: np.ndarray,
    rhs: np.ndarray,
    *,
    n_prob: int,
) -> tuple[dict[str, int], float]:
    """Keep shots whose first ``n_prob`` characters satisfy ``B @ x = rhs`` (mod 2).

    Returns ``(filtered_counts, keep_rate)`` where keys are **length-``n_prob``** bitstrings.
    """
    out: dict[str, int] = {}
    kept = 0
    total = 0
    for s, c in counts.items():
        total += int(c)
        prefix = s[:n_prob]
        if len(prefix) != n_prob:
            continue
        if syndrome_ok(prefix, B, rhs):
            out[prefix] = out.get(prefix, 0) + int(c)
            kept += int(c)
    rate = float(kept) / float(total) if total else 0.0
    return out, rate
