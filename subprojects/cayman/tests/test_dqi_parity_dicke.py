"""Tests for GF(2) insurance parity and Dicke line generation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_dicke_prep import append_dicke_state_lines
from src.dqi_insurance_parity import build_insurance_parity_B_rhs, syndrome_ok
from src.dqi_core import _angle_literal
from src.insurance_model import BundlingProblem, InsuranceCoverage
from src.qubo_block import build_qubo_block_for_package


def _tiny_problem() -> BundlingProblem:
    return BundlingProblem(
        coverages=[
            InsuranceCoverage("a", "auto_base", 100.0, 0.5, is_mandatory_in_family=True),
            InsuranceCoverage("b", "home_base", 200.0, 0.5, is_mandatory_in_family=True),
        ],
        num_packages=1,
        max_options_per_package=3,
    )


class TestInsuranceParity(unittest.TestCase):
    def test_B_columns_match_qubo_block(self) -> None:
        prob = _tiny_problem()
        block = build_qubo_block_for_package(prob, 0, penalty_weight=10.0)
        B, rhs = build_insurance_parity_B_rhs(prob, 0)
        self.assertEqual(B.shape[1], block.n_vars)
        self.assertEqual(B.shape[0], len(rhs))

    def test_syndrome_ok_detects_xor(self) -> None:
        B = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        rhs = np.array([0, 1], dtype=np.uint8)
        self.assertTrue(syndrome_ok("001", B, rhs))
        self.assertTrue(syndrome_ok("110", B, rhs))
        self.assertFalse(syndrome_ok("010", B, rhs))


class TestDickePrep(unittest.TestCase):
    def test_dicke_emits_gates(self) -> None:
        lines: list[str] = []
        append_dicke_state_lines(lines, [0, 1, 2, 3], 2, _angle_literal)
        text = "\n".join(lines)
        self.assertIn("ry(q", text)
        self.assertIn("cx(q", text)
        self.assertIn("x(q", text)

    def test_dicke_angle_literal_roundtrip(self) -> None:
        th = 0.25 * math.pi
        lit = _angle_literal(th)
        back = float(lit) * 0.5 * math.pi
        self.assertAlmostEqual(back, th, places=12)
