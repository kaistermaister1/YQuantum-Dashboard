"""Tests for ``qubo_qaoa_p1`` (Ising map + Guppy p=1 smoke)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.qubo_qaoa_p1 import (
    bruteforce_minimize_qubo,
    ising_energy_for_bitstring,
    qubo_to_ising_pauli_coefficients,
    run_qaoa_p1_on_block,
    verify_ising_matches_qubo,
)

try:
    import guppylang  # noqa: F401
except ImportError:
    guppylang = None

try:
    import pulp  # noqa: F401
except ImportError:
    pulp = None


class TestIsingMap(unittest.TestCase):
    def test_random_matches(self) -> None:
        verify_ising_matches_qubo(np.array([[1.0, 0.5], [0.5, -2.0]]))
        rng = np.random.default_rng(42)
        for n in range(1, 6):
            a = rng.standard_normal((n, n))
            q = (a + a.T) * 0.5
            verify_ising_matches_qubo(q, rng=rng)

    def test_hand_two_qubit(self) -> None:
        q = np.array([[0.0, 1.0], [1.0, 0.0]])
        c_i, c_z, zz = qubo_to_ising_pauli_coefficients(q)
        self.assertAlmostEqual(ising_energy_for_bitstring(np.array([0.0, 0.0]), c_i, c_z, zz), 0.0)
        self.assertAlmostEqual(ising_energy_for_bitstring(np.array([1.0, 1.0]), c_i, c_z, zz), 2.0)

    def test_bruteforce_two_qubit_hand(self) -> None:
        q = np.array([[2.0, 1.0], [1.0, 2.0]])
        e_min, x = bruteforce_minimize_qubo(q)
        self.assertAlmostEqual(e_min, 0.0)
        np.testing.assert_array_equal(x, [0.0, 0.0])

    def test_bruteforce_includes_constant_offset(self) -> None:
        q = np.array([[1.0]])
        e0, _ = bruteforce_minimize_qubo(q, constant_offset=0.0)
        e1, _ = bruteforce_minimize_qubo(q, constant_offset=5.0)
        self.assertAlmostEqual(e1 - e0, 5.0)


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestQaoaP1GuppySmoke(unittest.TestCase):
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
        stats = run_qaoa_p1_on_block(block, gamma=0.35, beta=0.55, shots=128, seed=3, max_qubits=8)
        self.assertEqual(stats.n_qubits, 2)
        self.assertEqual(sum(stats.bitstring_counts.values()), 128)
        self.assertEqual(len(stats.best_bitstring), 2)
        self.assertLess(stats.best_qubo_energy, float("inf"))

    def test_sample_energies_never_below_bruteforce_minimum(self) -> None:
        """Every measured bitstring is evaluated with ``QuboBlock.energy``; none may beat the global QUBO min."""
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
        stats = run_qaoa_p1_on_block(block, gamma=0.35, beta=0.55, shots=256, seed=7, max_qubits=8)
        self.assertGreaterEqual(stats.best_qubo_energy, e_min - 1e-9)
        for s, _count in stats.bitstring_counts.items():
            x = np.array([float(int(c)) for c in s], dtype=float)
            self.assertGreaterEqual(block.energy(x), e_min - 1e-9)

    def test_angle_grid_finds_optimum_single_qubit(self) -> None:
        """For a 1-QUBO diagonal problem, a coarse (γ, β) grid should hit the ground state (stochastic but high prob)."""
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        e_min, _ = bruteforce_minimize_qubo(block.Q, constant_offset=block.constant_offset)
        self.assertAlmostEqual(e_min, 0.0)
        best_seen = float("inf")
        for gamma in np.linspace(0.25, 1.15, 3):
            for beta in np.linspace(0.25, 1.15, 3):
                stats = run_qaoa_p1_on_block(
                    block, float(gamma), float(beta), shots=300, seed=0, max_qubits=8
                )
                best_seen = min(best_seen, stats.best_qubo_energy)
        self.assertLessEqual(abs(best_seen - e_min), 1e-9)


if __name__ == "__main__":
    unittest.main()
