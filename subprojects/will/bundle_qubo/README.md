# Insurance bundling QUBO + visualizer (QGars / Travelers track)

This folder is the **canonical copy in the QGars repo** (`YQuantum-Dashboard`) so everyone can `git pull` and get the same code without a separate Travelers clone.

## Contents

- **`code_examples/`** — `qubo_block.py` (ILP → per-package QUBO), `insurance_model.py`, tests, `requirements.txt`.
- **`qubo_vis/`** — C++ Raylib 3D + matrix heatmap; export script writes `qubo_surface.txt` from LTM CSVs.

## Data

Point exports/tests at your LTM CSV directory (e.g. challenge `docs/data/YQH26_data/` if you also have the full materials repo). Paths in READMEs under `code_examples/` are written relative to that layout.

## QGars Dashboard

The Next.js app lives in **`QGarsDashboard/`** at the repo root. This Python/C++ tooling is separate; use it for QUBO generation, checks, and demo recordings.
