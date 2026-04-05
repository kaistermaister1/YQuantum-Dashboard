"""Tests for the Kai QAOA grid history exporter."""

from __future__ import annotations

import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

import SOLUTIONS.qaoa as qaoa
import subprojects.kai.qaoa_grid_animation as qaoa_grid_animation


def _toy_problem():
    full_problem = qaoa.load_ltm_instance(qaoa.DATA_DIR)
    return qaoa.subsample_problem(full_problem, n_coverages=5, m_packages=2)


class TestQaoaGridAnimation(unittest.TestCase):
    def test_padded_lattice_helpers_match_inner_intersection_layout(self) -> None:
        self.assertEqual(qaoa_grid_animation.padded_lattice_shape(5, 2), (7, 4))
        self.assertEqual(qaoa_grid_animation.matrix_index_to_lattice_index(0, 0), (1, 1))
        self.assertEqual(qaoa_grid_animation.matrix_index_to_lattice_index(4, 1), (5, 2))

    def test_run_qaoa_with_matrix_history_collects_every_evaluated_matrix(self) -> None:
        fake_bitstrings = {
            0: ["10100000", "00100000"],
            1: ["00100000", "10100000"],
        }
        final_best_bitstrings = {
            0: "10100000",
            1: "10100000",
        }

        def fake_optimize_block(
            block: qaoa.QuboBlock,
            p: int,
            shots: int,
            seed: int,
            execution_target: str,
            evaluation_callback=None,
        ):
            trace = qaoa.QaoaOptimizationTrace()
            for eval_index, bitstring in enumerate(fake_bitstrings[block.package_index], start=1):
                stats = qaoa.QaoaSampleStats({bitstring: 8}, bitstring, float(-eval_index))
                if evaluation_callback is not None:
                    evaluation_callback(
                        block,
                        eval_index,
                        float(5 - eval_index),
                        float(5 - eval_index),
                        stats,
                        eval_index == 1,
                    )
                trace.record(float(5 - eval_index), best_stats=stats, improved=(eval_index == 1))
            return qaoa.QaoaOptimizationResult(
                theta=np.zeros(2 * p, dtype=float),
                stats=qaoa.QaoaSampleStats(
                    {final_best_bitstrings[block.package_index]: 8},
                    final_best_bitstrings[block.package_index],
                    -2.0,
                ),
                trace=trace,
            )

        with patch("subprojects.kai.qaoa_grid_animation.qaoa.optimize_block", side_effect=fake_optimize_block):
            history = qaoa_grid_animation.run_qaoa_with_matrix_history(n=5, m=2, p=1, execution_target="local")

        self.assertEqual(len(history.frames), 5)
        np.testing.assert_array_equal(
            history.final_matrix,
            np.array(
                [
                    [0, 1],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                ],
                dtype=int,
            ),
        )
        self.assertEqual(history.frames[0].iteration, 0)
        self.assertEqual(history.frames[1].iteration, 1)
        self.assertTrue(bool(history.frames[1].changed_mask[0, 0]))
        np.testing.assert_array_equal(
            history.frames[2].matrix,
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                    [0, 0],
                ],
                dtype=int,
            ),
        )
        np.testing.assert_array_equal(
            history.frames[3].matrix,
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                ],
                dtype=int,
            ),
        )
        self.assertTrue(bool(history.frames[4].changed_mask[0, 1]))

    def test_save_grid_history_text_writes_cpp_viewer_format(self) -> None:
        problem = _toy_problem()
        c_matrix = qaoa.make_c_matrix(problem)
        frames = [
            qaoa_grid_animation.MatrixFrame(
                iteration=0,
                package_index=None,
                matrix=np.zeros((problem.N, problem.M), dtype=int),
                changed_mask=np.zeros((problem.N, problem.M), dtype=bool),
                best_objective=None,
                best_qubo_energy=None,
                current_profit=0.0,
                improved=False,
                note="start",
            ),
            qaoa_grid_animation.MatrixFrame(
                iteration=1,
                package_index=0,
                matrix=np.array(
                    [
                        [1, 0],
                        [0, 0],
                        [1, 0],
                        [0, 0],
                        [0, 0],
                    ],
                    dtype=int,
                ),
                changed_mask=np.array(
                    [
                        [True, False],
                        [False, False],
                        [True, False],
                        [False, False],
                        [False, False],
                    ],
                    dtype=bool,
                ),
                best_objective=-1.0,
                best_qubo_energy=-2.0,
                current_profit=float(c_matrix[0, 0] + c_matrix[2, 0]),
                improved=True,
                note="package 1 eval 1",
            ),
        ]
        history = qaoa_grid_animation.QaoaGridHistory(
            problem=problem,
            c_matrix=c_matrix,
            frames=frames,
            final_matrix=frames[-1].matrix.copy(),
        )

        temp_root = Path(__file__).resolve().parent / ".tmp_test_artifacts"
        temp_root.mkdir(exist_ok=True)
        text_path = temp_root / f"qaoa_grid_history_{uuid.uuid4().hex}.txt"
        try:
            saved_path = qaoa_grid_animation.save_grid_history_text(history, text_path)
            self.assertEqual(saved_path, text_path)
            self.assertTrue(text_path.exists())

            text = text_path.read_text(encoding="utf-8")
            self.assertIn("grid_history_v1", text)
            self.assertIn("padding=1, lattice_rows=7, lattice_cols=4", text)
            self.assertIn("COEFFS", text)
            self.assertIn("FRAME 1 0 -1 -2", text)
            self.assertIn("NOTE\tpackage 1 eval 1", text)
            self.assertIn("CHANGE", text)
        finally:
            text_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
