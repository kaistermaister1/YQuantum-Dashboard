"""Smoke tests for the run_dqi API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_dqi import run_dqi, run_dqi_with_details

try:
    import guppylang  # noqa: F401
except ImportError:
    guppylang = None


@unittest.skipIf(guppylang is None, "guppylang not installed")
class TestRunDqiSmoke(unittest.TestCase):
    def test_run_dqi_returns_solution_and_value(self) -> None:
        Q = np.array([[1.1, -0.25], [-0.25, -0.4]], dtype=float)
        x, value = run_dqi(
            Q,
            p=1,
            optimizer="random",
            n_samples=8,
            shots=128,
            seed=3,
            rng_seed=4,
            max_qubits=8,
        )
        self.assertEqual(x.shape, (2,))
        self.assertTrue(np.all((x == 0.0) | (x == 1.0)))
        self.assertTrue(np.isfinite(value))

    def test_run_dqi_with_quboblock_metadata(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[0.3, 0.2], [0.2, -0.5]], dtype=float)
        block = QuboBlock(
            package_index=0,
            Q=q,
            n_coverage=2,
            n_slack=0,
            coverage_offset=0,
            penalty_weight=1.0,
            constant_offset=1.25,
        )
        x, value, meta = run_dqi_with_details(
            block,
            p=1,
            optimizer="random",
            n_samples=6,
            shots=96,
            seed=2,
            rng_seed=5,
            max_qubits=8,
        )
        self.assertEqual(x.shape, (2,))
        self.assertIsNotNone(meta.coverage_bits)
        self.assertEqual(len(meta.coverage_bits or ""), 2)
        self.assertAlmostEqual(meta.constant_offset, 1.25)
        self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
