"""Smoke tests for the run_dqi API."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqi_core import import_guppylang_with_workaround
from src.run_dqi import run_dqi, run_dqi_with_details

try:
    import_guppylang_with_workaround()
    _guppy_ok = True
except (ImportError, OSError, RuntimeError):
    _guppy_ok = False


def _skip_if_selene_env_missing(exc: BaseException) -> None:
    """Selene uses zig to compile; Windows needs MSVC + Windows SDK in PATH."""
    msg = str(exc)
    if "WindowsSdkNotFound" in msg or "zig command failed" in msg:
        raise unittest.SkipTest(
            "Selene emulator build skipped: install Visual Studio Build Tools with "
            "'Desktop development with C++' (includes Windows SDK), then retry."
        ) from exc


@unittest.skipIf(
    not _guppy_ok,
    "guppylang unavailable (not installed or wasmtime failed to load)",
)
class TestRunDqiSmoke(unittest.TestCase):
    def test_run_dqi_returns_solution_and_value(self) -> None:
        Q = np.array([[1.1, -0.25], [-0.25, -0.4]], dtype=float)
        try:
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
        except RuntimeError as e:
            _skip_if_selene_env_missing(e)
            raise
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
        try:
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
        except RuntimeError as e:
            _skip_if_selene_env_missing(e)
            raise
        self.assertEqual(x.shape, (2,))
        self.assertIsNotNone(meta.coverage_bits)
        self.assertEqual(len(meta.coverage_bits or ""), 2)
        self.assertAlmostEqual(meta.constant_offset, 1.25)
        self.assertTrue(np.isfinite(value))

    def test_run_dqi_non_variational_single_evaluation(self) -> None:
        Q = np.array([[1.1, -0.25], [-0.25, -0.4]], dtype=float)
        try:
            x, value, meta = run_dqi_with_details(
                Q,
                p=1,
                optimizer="cobyla",
                variational=False,
                fixed_gammas=[1.0],
                shots=64,
                seed=0,
                max_qubits=8,
                execution="local",
            )
        except RuntimeError as e:
            _skip_if_selene_env_missing(e)
            raise
        self.assertEqual(x.shape, (2,))
        self.assertEqual(meta.optimizer_result.n_evaluations, 1)
        self.assertEqual(len(meta.optimizer_result.history), 1)
        self.assertTrue(np.isfinite(value))

    def test_run_dqi_fixed_gammas_length_must_match_p(self) -> None:
        Q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        with self.assertRaises(ValueError):
            run_dqi(
                Q,
                p=2,
                optimizer="cobyla",
                variational=False,
                fixed_gammas=[1.0],
                shots=8,
                max_qubits=8,
                execution="local",
            )


if __name__ == "__main__":
    unittest.main()
