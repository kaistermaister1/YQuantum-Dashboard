"""Tests for bundling parity matrix construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bundling_parity import build_parity_B_v_gadgets
from src.insurance_model import BundlingProblem, CompatibilityRule, InsuranceCoverage


class TestBundlingParityGadgets(unittest.TestCase):
    def test_mandatory_two_family_xor(self) -> None:
        covs = [
            InsuranceCoverage("a", "fam", 1.0, 0.5, 0.2, True),
            InsuranceCoverage("b", "fam", 1.0, 0.5, 0.2, True),
        ]
        p = BundlingProblem(
            coverages=covs,
            num_packages=1,
            max_options_per_package=1,
            compatibility_rules=[],
            dependency_rules=[],
        )
        out = build_parity_B_v_gadgets(p, 0)
        # mandatory XOR on (x0,x1) plus capacity parity row on (x0,x1,s0)
        self.assertEqual(out.n_slack, 1)
        self.assertEqual(out.B.shape, (2, 3))
        np.testing.assert_array_equal(out.B[0], [1, 1, 0])
        np.testing.assert_array_equal(out.B[1], [1, 1, 1])
        np.testing.assert_array_equal(out.v, [1, 1])

    def test_incompatibility_adds_slack_and_rows(self) -> None:
        covs = [
            InsuranceCoverage("a", "f1", 1.0, 0.5, 0.2, True),
            InsuranceCoverage("b", "f1", 1.0, 0.5, 0.2, True),
            InsuranceCoverage("c", "f2", 1.0, 0.5, 0.2, False),
        ]
        p = BundlingProblem(
            coverages=covs,
            num_packages=1,
            max_options_per_package=3,
            compatibility_rules=[CompatibilityRule("a", "c", False)],
            dependency_rules=[],
        )
        out = build_parity_B_v_gadgets(p, 0)
        n_tot = out.n_coverage + out.n_slack
        self.assertEqual(out.B.shape[1], n_tot)
        self.assertGreaterEqual(out.B.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
