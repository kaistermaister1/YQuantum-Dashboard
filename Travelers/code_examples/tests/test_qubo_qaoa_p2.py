"""Tests for ``qubo_qaoa_p1.run_qaoa_p2_on_block`` (Guppy p=2 smoke + bruteforce checks)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.qubo_qaoa_p1 import bruteforce_minimize_qubo, run_qaoa_p2_on_block

try:
    import guppylang  # noqa: F401
except ImportError:
    guppylang = None

try:
    import pulp  # noqa: F401
except ImportError:
    pulp = None


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestQaoaP2GuppySmoke(unittest.TestCase):
    def test_runs_two_qubit_block(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[0.3, 0.4], [0.4, -0.1]])
        block = QuboBlock(
            package_index=0,
            Q=q,
            n_coverage=2,
            n_slack=0,
            coverage_offset=0,
            penalty_weight=1.0,
            constant_offset=0.0,
        )
        stats = run_qaoa_p2_on_block(
            block,
            gamma1=0.25,
            beta1=0.45,
            gamma2=0.35,
            beta2=0.55,
            shots=128,
            seed=3,
            max_qubits=8,
        )
        self.assertEqual(stats.n_qubits, 2)
        self.assertEqual(sum(stats.bitstring_counts.values()), 128)
        self.assertEqual(len(stats.best_bitstring), 2)
        self.assertLess(stats.best_qubo_energy, float("inf"))

    def test_sample_energies_never_below_bruteforce_minimum(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[0.3, 0.4], [0.4, -0.1]])
        block = QuboBlock(
            package_index=0,
            Q=q,
            n_coverage=2,
            n_slack=0,
            coverage_offset=0,
            penalty_weight=1.0,
            constant_offset=2.5,
        )
        e_min, _ = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset)
        stats = run_qaoa_p2_on_block(
            block,
            gamma1=0.35,
            beta1=0.55,
            gamma2=0.2,
            beta2=0.4,
            shots=256,
            seed=7,
            max_qubits=8,
        )
        self.assertGreaterEqual(stats.best_qubo_energy, e_min - 1e-9)
        for s, _count in stats.bitstring_counts.items():
            x = np.array([float(int(c)) for c in s], dtype=float)
            self.assertGreaterEqual(block.energy(x), e_min - 1e-9)

    def test_angle_grid_finds_optimum_single_qubit(self) -> None:
        """Depth-2 QAOA on one Z term; coarse grid on (γ₁,β₁) with fixed second layer."""
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        e_min, _ = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset)
        self.assertAlmostEqual(e_min, 0.0)
        best_seen = float("inf")
        for gamma1 in np.linspace(0.25, 1.15, 3):
            for beta1 in np.linspace(0.25, 1.15, 3):
                stats = run_qaoa_p2_on_block(
                    block,
                    float(gamma1),
                    float(beta1),
                    gamma2=0.4,
                    beta2=0.5,
                    shots=300,
                    seed=0,
                    max_qubits=8,
                )
                best_seen = min(best_seen, stats.best_qubo_energy)
        self.assertLessEqual(abs(best_seen - e_min), 1e-9)


if __name__ == "__main__":
    unittest.main()
