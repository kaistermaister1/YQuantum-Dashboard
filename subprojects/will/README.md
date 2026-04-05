# Will — `subprojects/will/`

Two top-level trees, no duplicated QUBO/C++ trees:

| Path | Purpose |
|------|--------|
| **`Travelers/`** | Sponsor bundle: `docs/` (HTML + **YQH26_data** CSVs), `code_examples/` (classical notebooks + shared `src/`), `qubo_vis/` (C++ Raylib viewer + export script). |
| **`qaoa_python/`** | Will’s Python: Guppy/QAOA (`qubo_qaoa_*`, optimizers), tests, `requirements.txt` pattern same as `Travelers/code_examples/`. Notebook: `qaoa_python/notebooks/qaoa_guppy_template.ipynb`. |

Use **`Travelers/`** for data paths and classical baselines; use **`qaoa_python/`** for Selene/Guppy QAOA experiments and unit tests.
