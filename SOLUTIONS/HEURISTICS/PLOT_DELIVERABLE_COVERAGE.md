# Heuristics plots ↔ recorded metrics ↔ challenge deliverables

Columns in `run_summaries.csv`: `approximation ratio` (via `best_profit` / `classical_opt_profit`), `circuit_depth`, `two_qubit_gate_count`, `num_qubits`, `runtime_sec` + `num_samples_total` / `num_objective_evals`, feasibility (via `num_samples_feasible` / `num_samples_total`).

| Artifact | Primary metrics | Deliverable angle |
|----------|-----------------|-------------------|
| `plots/heuristics_summary*.png`, `plot_heuristics_summary.py` | profit vs classical, runtime, feasibility | Quality + cost + constraint handling (1–2) |
| `plots/algorithm_comparison_*.png`, `plot_algorithm_comparison_table.py` | side-by-side profit, runtime, shots | Method comparison (3) |
| `plots/profit_ratio_vs_n*.png`, `plot_profit_ratio_vs_n.py` | approximation ratio vs `N_local` | Quality scaling (1) |
| `plots/runtime_vs_n_total*.png`, `plot_runtime_vs_n_total.py` | runtime vs `n_total`, eval counts | Time-to-solution proxy vs size (2) |
| `plots/runtime_vs_n_classical_qaoa*.png`, `plot_runtime_vs_n.py` | classical vs QAOA runtime | Classical baseline (2–3) |
| **`plots/runtime_vs_m_blocks_n10_p2_cobyla_k3072.png`**, **`generate_plots.py` §9b** | `runtime_sec` vs `M_blocks` with `num_samples_total = k·M_blocks` (seaborn dashboard style) | **Linear scaling in packages `m`** when workload per block is held fixed (2) |
| **`plot_runtime_vs_m_blocks.py`** | same data selection; matplotlib styling for quick iteration | Same as above (2) |
| `public/plots/heuristics/*.png` (from `generate_plots.py`) | mixed aggregates on `n_total`, `p`, `M_blocks` | Dashboard overview (1–2) |
| `public/plots/heuristics/runtime_vs_m_proportional_budget.png` | written by **`generate_plots.py`** (same series as above) | QGars dashboard / slides |

**DQI:** table cells and plots show N/A until `algorithm=dqi` rows exist in `run_summaries.csv`.

**Regenerate proportional-`m` figure (dashboard look):** from repo root, `python generate_plots.py` (updates `public/plots/heuristics/runtime_vs_m_proportional_budget.png` and `SOLUTIONS/HEURISTICS/plots/runtime_vs_m_blocks_n10_p2_cobyla_k3072.png`). For the standalone matplotlib variant: `SOLUTIONS/plot_runtime_vs_m_blocks.py --auto`.
