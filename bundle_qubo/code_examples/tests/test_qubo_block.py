"""Tests for src/qubo_block.py (run from ``code_examples`` directory).

Run:

    cd bundle_qubo/code_examples
    python -m unittest tests.test_qubo_block -v

Requires: numpy, pulp, and LTM CSVs (see ``_ltm_data_dir()``: sibling ``Travelers/`` clone or ``bundle_qubo/docs/data/``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Allow ``from src...`` when running unittest from code_examples/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.insurance_model import (  # noqa: E402
    BundlingProblem,
    load_ltm_instance,
    solve_ilp,
    subsample_problem,
)
from src.qubo_block import (  # noqa: E402
    QuboBlock,
    build_all_qubo_blocks,
    build_qubo_block_for_package,
    default_penalty_weight,
)


def _ltm_data_dir() -> Path:
    """``bundle_qubo/docs/...`` or sibling ``Travelers/docs/...``."""
    parent = _ROOT.parent
    candidates = [
        parent / "docs" / "data" / "YQH26_data",
        parent.parent / "Travelers" / "docs" / "data" / "YQH26_data",
    ]
    for p in candidates:
        if (p / "instance_coverages.csv").is_file():
            return p
    return candidates[1]


def _package_margin(problem: BundlingProblem, m: int, coverage_bits: np.ndarray) -> float:
    cov = np.asarray(coverage_bits, dtype=int).ravel()
    total = 0.0
    for i in range(problem.N):
        if not cov[i]:
            continue
        c = problem.coverages[i]
        delta_m = problem.get_discount(m)
        alpha_im = problem.get_affinity(i, m)
        beta = problem.price_sensitivity_beta
        total += (
            c.price
            * c.contribution_margin_pct
            * (1 - delta_m)
            * c.take_rate
            * alpha_im
            * (1 + beta * delta_m)
        )
    return float(total)


def _brute_force_argmin(block: QuboBlock) -> tuple[float, np.ndarray]:
    best_e = float("inf")
    best_x = None
    n = block.n_vars
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        e = block.energy(x)
        if e < best_e:
            best_e = e
            best_x = x.copy()
    assert best_x is not None
    return best_e, best_x


@unittest.skipUnless(_ltm_data_dir().is_dir(), f"Missing LTM data at {_ltm_data_dir()}")
class TestQuboBlockAgainstIlp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = subsample_problem(
            load_ltm_instance(_ltm_data_dir()),
            n_coverages=5,
            n_packages=2,
        )
        cls.ilp = solve_ilp(cls.problem)
        cls.lam = 5000.0

    def test_q_is_symmetric(self) -> None:
        for m in range(self.problem.M):
            b = build_qubo_block_for_package(self.problem, m, self.lam)
            np.testing.assert_allclose(b.Q, b.Q.T, rtol=0, atol=1e-9)

    def test_brute_force_matches_ilp_coverage(self) -> None:
        sol = self.ilp["solution_vector"]
        N = self.problem.N
        for m in range(self.problem.M):
            block = build_qubo_block_for_package(self.problem, m, self.lam)
            self.assertLessEqual(block.n_vars, 18, "brute-force test expects small block")
            e_min, x_min = _brute_force_argmin(block)
            cov_bf = x_min[:N].astype(int)
            cov_ilp = sol[m * N : (m + 1) * N].astype(int)
            np.testing.assert_array_equal(cov_bf, cov_ilp)
            margin_bf = _package_margin(self.problem, m, cov_bf)
            margin_ilp = _package_margin(self.problem, m, cov_ilp)
            self.assertAlmostEqual(margin_bf, margin_ilp, places=4)
            self.assertAlmostEqual(e_min, -margin_bf, places=3)

    def test_build_all_same_as_single(self) -> None:
        blocks = build_all_qubo_blocks(self.problem, self.lam)
        self.assertEqual(len(blocks), self.problem.M)
        for m, blk in enumerate(blocks):
            single = build_qubo_block_for_package(self.problem, m, self.lam)
            np.testing.assert_allclose(blk.Q, single.Q)
            self.assertAlmostEqual(blk.constant_offset, single.constant_offset)

    def test_default_lambda_positive(self) -> None:
        for m in range(self.problem.M):
            self.assertGreater(default_penalty_weight(self.problem, m), 0.0)


@unittest.skipUnless(_ltm_data_dir().is_dir(), f"Missing LTM data at {_ltm_data_dir()}")
class TestQuboBlockFullInstanceBuild(unittest.TestCase):
    """Smoke test: full LTM size builds without error (no brute force)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.problem = load_ltm_instance(_ltm_data_dir())

    def test_full_build(self) -> None:
        blocks = build_all_qubo_blocks(self.problem, penalty_weight=1e4)
        self.assertEqual(len(blocks), self.problem.M)
        for b in blocks:
            self.assertEqual(b.Q.shape[0], b.n_coverage + b.n_slack)
            np.testing.assert_allclose(b.Q, b.Q.T)


if __name__ == "__main__":
    unittest.main()
