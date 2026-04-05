"""Tests for ``qubo_qaoa_optimize`` (mean energy helper + outer loops; Guppy optional)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.qubo_qaoa_p1 import QaoaP1SampleStats
from src.qubo_qaoa_optimize import (
    mean_sample_energy,
    repeat_optimize_qaoa_p1,
    sample_mean_energy_uncertainty,
    optimize_qaoa_p1_cobyla,
    optimize_qaoa_p1_grid,
    optimize_qaoa_p1_random,
    optimize_qaoa_p1_spsa,
    optimize_qaoa_p2_cobyla,
    optimize_qaoa_p2_grid,
    optimize_qaoa_p2_random,
    optimize_qaoa_p2_spsa,
)

try:
    import guppylang  # noqa: F401
except ImportError:
    guppylang = None

try:
    import pulp  # noqa: F401
except ImportError:
    pulp = None

try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None


class TestMeanSampleEnergy(unittest.TestCase):
    def test_weighted_mean_two_qubit(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[1.0, 0.0], [0.0, 2.0]])
        block = QuboBlock(0, q, 2, 0, 0, 1.0, 0.0)
        # E(00)=0, E(11)=1+2+0=3
        stats = QaoaP1SampleStats(
            n_qubits=2,
            shots=4,
            bitstring_counts={"00": 2, "11": 2},
            best_bitstring="00",
            best_qubo_energy=0.0,
            constant_offset=0.0,
        )
        self.assertAlmostEqual(mean_sample_energy(block, stats), 1.5)

    def test_sample_mean_se_two_level(self) -> None:
        from src.qubo_block import QuboBlock

        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        block = QuboBlock(0, q, 2, 0, 0, 1.0, 0.0)
        # E(00)=0, E(11)=2; 50/50 over 100 shots → mean 1, var = 1, std=1, se=0.1
        stats = QaoaP1SampleStats(
            n_qubits=2,
            shots=100,
            bitstring_counts={"00": 50, "11": 50},
            best_bitstring="00",
            best_qubo_energy=0.0,
            constant_offset=0.0,
        )
        u = sample_mean_energy_uncertainty(block, stats)
        self.assertAlmostEqual(u.mean, 1.0)
        self.assertAlmostEqual(u.std_iid, 1.0)
        self.assertAlmostEqual(u.se_mean, 0.1)
        self.assertEqual(u.n_shots, 100)


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP1Grid(unittest.TestCase):
    def test_small_grid_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p1_grid(
            block,
            n_gamma=2,
            n_beta=2,
            gamma_bounds=(0.2, 0.9),
            beta_bounds=(0.2, 0.9),
            shots=64,
            seed_offset=100,
            statistic="mean",
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 4)
        self.assertTrue(math.isfinite(res.objective_value))
        self.assertEqual(res.stats_at_best.n_qubits, 1)
        self.assertEqual(res.statistic, "mean")


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP1Random(unittest.TestCase):
    def test_random_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p1_random(
            block,
            n_samples=5,
            shots=48,
            rng_seed=42,
            seed_offset=200,
            statistic="best",
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 5)
        self.assertEqual(res.statistic, "best")


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP2Grid(unittest.TestCase):
    def test_tiny_grid_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p2_grid(
            block,
            n_gamma1=2,
            n_beta1=2,
            n_gamma2=2,
            n_beta2=2,
            shots=48,
            seed_offset=300,
            statistic="mean",
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 16)
        self.assertTrue(math.isfinite(res.objective_value))


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP2Random(unittest.TestCase):
    def test_random_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p2_random(
            block,
            n_samples=4,
            shots=40,
            rng_seed=7,
            seed_offset=400,
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 4)


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
@unittest.skipIf(scipy is None, "scipy not installed (needed for COBYLA)")
class TestOptimizeP1Cobyla(unittest.TestCase):
    def test_cobyla_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p1_cobyla(
            block,
            x0=(0.5, 0.5),
            gamma_bounds=(0.15, 1.2),
            beta_bounds=(0.15, 1.2),
            shots=48,
            seed_offset=500,
            maxiter=6,
            max_qubits=8,
        )
        self.assertGreater(res.n_evaluations, 0)
        self.assertTrue(math.isfinite(res.objective_value))


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP1Spsa(unittest.TestCase):
    def test_spsa_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p1_spsa(
            block,
            x0=(0.55, 0.55),
            maxiter=4,
            shots=40,
            rng_seed=99,
            seed_offset=600,
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 8)  # 2 evals per iteration
        self.assertTrue(math.isfinite(res.objective_value))


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
@unittest.skipIf(scipy is None, "scipy not installed (needed for COBYLA)")
class TestOptimizeP2Cobyla(unittest.TestCase):
    def test_cobyla_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p2_cobyla(
            block,
            x0=(0.4, 0.4, 0.4, 0.4),
            shots=40,
            seed_offset=700,
            maxiter=5,
            max_qubits=8,
        )
        self.assertGreater(res.n_evaluations, 0)
        self.assertTrue(math.isfinite(res.objective_value))


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestOptimizeP2Spsa(unittest.TestCase):
    def test_spsa_runs(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        res = optimize_qaoa_p2_spsa(
            block,
            maxiter=3,
            shots=36,
            rng_seed=3,
            seed_offset=800,
            max_qubits=8,
        )
        self.assertEqual(res.n_evaluations, 6)
        self.assertTrue(math.isfinite(res.objective_value))


@unittest.skipIf(guppylang is None, "guppylang not installed")
@unittest.skipIf(pulp is None, "pulp not installed (needed for QuboBlock)")
class TestRepeatOptimizeP1(unittest.TestCase):
    def test_two_grid_repeats_percentiles(self) -> None:
        from src.qubo_block import QuboBlock

        block = QuboBlock(0, np.array([[1.0]]), 1, 0, 0, 1.0, 0.0)
        summary = repeat_optimize_qaoa_p1(
            block,
            method="grid",
            n_repeats=2,
            base_seed_offset=9000,
            optimizer_kwargs={
                "n_gamma": 2,
                "n_beta": 2,
                "shots": 40,
                "max_qubits": 8,
                "statistic": "mean",
            },
        )
        self.assertEqual(len(summary.results), 2)
        self.assertEqual(summary.gamma.shape, (2,))
        pct = summary.angle_percentiles((50.0,))
        self.assertIn("p50", pct["gamma"])
        self.assertIn("p50", pct["beta"])


if __name__ == "__main__":
    unittest.main()
