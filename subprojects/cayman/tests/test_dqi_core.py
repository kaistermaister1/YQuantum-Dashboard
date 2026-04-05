"""Unit tests for DQI core math utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_core import (
    build_hamming_weight_penalty_qubo,
    ising_energy_for_bitstring,
    qubo_energy,
    qubo_to_ising,
)


class TestDqiCoreMath(unittest.TestCase):
    def test_qubo_to_ising_matches_random(self) -> None:
        rng = np.random.default_rng(42)
        for n in range(1, 7):
            a = rng.standard_normal((n, n))
            q = (a + a.T) * 0.5
            c_i, c_z, zz = qubo_to_ising(q)
            for _ in range(16):
                x = rng.integers(0, 2, size=n).astype(float)
                e_qubo = qubo_energy(x, q, constant_offset=0.0)
                e_ising = ising_energy_for_bitstring(x, c_i, c_z, zz)
                self.assertAlmostEqual(e_qubo, e_ising, places=8)

    def test_hamming_penalty_prefers_target_weight(self) -> None:
        n = 8
        k = 3
        Q = build_hamming_weight_penalty_qubo(n=n, target_weight=k, penalty=2.0)
        values = []
        for bits in range(1 << n):
            x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
            values.append((qubo_energy(x, Q), int(np.sum(x))))
        best_val = min(v for v, _ in values)
        best_weights = {w for v, w in values if abs(v - best_val) < 1e-9}
        self.assertEqual(best_weights, {k})


if __name__ == "__main__":
    unittest.main()
