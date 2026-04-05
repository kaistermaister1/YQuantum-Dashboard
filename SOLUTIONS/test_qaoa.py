"""Hard-coded regression tests for SOLUTIONS.qaoa."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import SOLUTIONS.classical_baseline as classical_baseline
import SOLUTIONS.qaoa as qaoa
import SOLUTIONS.qaoa_plots as qaoa_plots


TOY_C_MATRIX = np.array(
    [
        [171.7867008, 152.889606144],
        [102.61060992, 133.42662144],
        [162.61518, 223.19908],
        [43.23062016, 46.45432512],
        [27.06941952, 24.21805056],
    ],
    dtype=float,
)

TOY_M_MATRIX = np.array(
    [
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
    ],
    dtype=int,
)

TOY_OBJECTIVE = 710.4906


def _toy_problem():
    full_problem = qaoa.load_ltm_instance(qaoa.DATA_DIR)
    return qaoa.subsample_problem(full_problem, n_coverages=5, m_packages=2)


def _bruteforce_block(block: qaoa.QuboBlock) -> tuple[float, np.ndarray]:
    best_energy = float("inf")
    best_x: np.ndarray | None = None
    for bits in itertools.product([0, 1], repeat=block.n_vars):
        x = np.array(bits, dtype=float)
        energy = block.energy(x)
        if energy < best_energy - 1e-9:
            best_energy = energy
            best_x = x.copy()
    if best_x is None:
        raise RuntimeError("Brute force failed to find a candidate")
    return best_energy, best_x


class TestToyBaseline(unittest.TestCase):
    def test_make_c_matrix_matches_hardcoded_toy_values(self) -> None:
        problem = _toy_problem()
        c_matrix = qaoa.make_c_matrix(problem)
        np.testing.assert_allclose(c_matrix, TOY_C_MATRIX, rtol=0, atol=1e-9)

    def test_qubo_block_zero_matches_expected_entries(self) -> None:
        problem = _toy_problem()
        block = qaoa.build_qubo_block_for_package(problem, 0)
        self.assertEqual(block.Q.shape, (8, 8))
        self.assertEqual(block.n_coverage, 5)
        self.assertEqual(block.n_slack, 3)
        self.assertAlmostEqual(block.constant_offset, 26283.3652224, places=6)
        self.assertAlmostEqual(block.Q[0, 0], -7386.8281344, places=6)
        self.assertAlmostEqual(block.Q[0, 1], 1030.7202048, places=6)
        self.assertAlmostEqual(block.Q[2, 3], 1030.7202048, places=6)
        self.assertAlmostEqual(block.Q[7, 7], -20614.404096, places=6)

    def test_toy_bruteforce_solution_matches_notebook_baseline(self) -> None:
        problem = _toy_problem()
        c_matrix = qaoa.make_c_matrix(problem)
        blocks = [qaoa.build_qubo_block_for_package(problem, m) for m in range(problem.M)]

        chosen_columns: list[np.ndarray] = []
        total_objective = 0.0
        for m, block in enumerate(blocks):
            _energy, x_best = _bruteforce_block(block)
            coverage_bits = x_best[: problem.N].astype(int)
            chosen_columns.append(coverage_bits)
            total_objective += float(c_matrix[:, m] @ coverage_bits)

        m_matrix = np.column_stack(chosen_columns)
        np.testing.assert_array_equal(m_matrix, TOY_M_MATRIX)
        self.assertAlmostEqual(total_objective, TOY_OBJECTIVE, places=4)

        chosen_names = [
            [problem.coverages[i].name for i in range(problem.N) if m_matrix[i, m] == 1]
            for m in range(problem.M)
        ]
        self.assertEqual(
            chosen_names,
            [
                ["auto_liability_basic", "homeowners"],
                ["auto_liability_basic", "homeowners"],
            ],
        )


class TestQaoaWiring(unittest.TestCase):
    def test_default_loss_plot_path_uses_plots_subfolder_and_dimensions(self) -> None:
        plot_path = qaoa.default_loss_plot_path(n=5, m=2, p=3)
        self.assertEqual(plot_path.parent.name, "plots")
        self.assertEqual(plot_path.name, "qaoa_training_loss_n5_m2_p3.png")

    def test_generated_source_uses_guppy_and_zz_phase(self) -> None:
        c_z = np.array([1.0, -0.5], dtype=float)
        zz_terms = [(0, 1, 0.25)]
        source = qaoa._build_qaoa_source(c_z, zz_terms, np.array([0.3]), np.array([0.4]))
        self.assertIn("from guppylang import guppy", source)
        self.assertIn("from guppylang.std.qsystem import rz, zz_phase", source)
        self.assertIn("zz_phase(q0, q1", source)

    def test_solve_qaoa_matrix_assembles_column_results(self) -> None:
        fake_stats = [
            qaoa.QaoaSampleStats({"10100000": 8}, "10100000", -1.0),
            qaoa.QaoaSampleStats({"10100000": 8}, "10100000", -1.0),
        ]

        def fake_optimize_block(block: qaoa.QuboBlock, p: int, shots: int, seed: int):
            trace = qaoa.QaoaOptimizationTrace(
                objective_values=[3.0, 2.0],
                best_objective_values=[3.0, 2.0],
            )
            return qaoa.QaoaOptimizationResult(
                theta=np.zeros(2 * p, dtype=float),
                stats=fake_stats[block.package_index],
                trace=trace,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "loss.png"
            with patch("SOLUTIONS.qaoa.optimize_block", side_effect=fake_optimize_block):
                m_matrix = qaoa.solve_qaoa_matrix(n=5, m=2, p=1, plot_output_path=plot_path)
            self.assertTrue(plot_path.exists())

        np.testing.assert_array_equal(m_matrix, TOY_M_MATRIX)

    def test_save_training_loss_plot_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "qaoa_loss.png"
            saved_path = qaoa_plots.save_training_loss_plot(
                [
                    ("package 1", [5.0, 3.0, 2.0], [5.0, 3.0, 2.0]),
                    ("package 2", [4.0, 4.5, 1.5], [4.0, 4.0, 1.5]),
                ],
                plot_path,
            )

            self.assertEqual(saved_path, plot_path)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)

    def test_save_training_loss_plot_uses_subplots_for_many_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "qaoa_loss_many.png"
            saved_path = qaoa_plots.save_training_loss_plot(
                [
                    ("package 1", [5.0, 3.0, 2.0], [5.0, 3.0, 2.0]),
                    ("package 2", [4.0, 4.5, 1.5], [4.0, 4.0, 1.5]),
                    ("package 3", [6.0, 3.5, 2.5], [6.0, 3.5, 2.5]),
                ],
                plot_path,
            )

            self.assertEqual(saved_path, plot_path)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 0)


class TestClassicalBaseline(unittest.TestCase):
    def test_solve_classical_baseline_matches_toy_notebook_result(self) -> None:
        result = classical_baseline.solve_classical_baseline(n=5, m=2)
        self.assertEqual(result["status"], "Optimal")
        self.assertAlmostEqual(result["profit"], TOY_OBJECTIVE, places=4)
        self.assertEqual(
            result["coverage_names"],
            [
                "auto_liability_basic",
                "auto_liability_enhanced",
                "homeowners",
                "condo_owners",
                "renters",
            ],
        )
        self.assertEqual(result["x_matrix"], TOY_M_MATRIX.tolist())
        self.assertEqual(result["solution_vector"], [1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(
            [package["coverages"] for package in result["packages"]],
            [
                ["auto_liability_basic", "homeowners"],
                ["auto_liability_basic", "homeowners"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
