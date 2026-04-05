# Will — `subprojects/will/`

| Path | Purpose |
|------|--------|
| **`Travelers/`** | Sponsor bundle: `docs/` (HTML + **YQH26_data** CSVs), `code_examples/` (classical notebooks + shared `src/`). |
| **`visualizations/qubo_vis/`** | C++ Raylib 3D QUBO-block viewer + `scripts/export_qubo_surface.py` (reads CSVs from `Travelers/docs/data/`, imports from `Travelers/code_examples/src`). |
| **`qaoa_python/`** | Guppy/QAOA (`qubo_qaoa`, optimizers), tests. Notebook: `qaoa_python/notebooks/qaoa_guppy_template.ipynb`. |

Use **`Travelers/`** for data and classical baselines; **`qaoa_python/`** for Selene/Guppy QAOA; **`visualizations/qubo_vis/`** for the mesh viewer and surface export.
