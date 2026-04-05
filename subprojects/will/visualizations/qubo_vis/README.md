# QUBO block visualizer (Raylib)

This is a small **C++** demo meant to **screen-record** for submissions: a **3D wireframe “landscape”** of **one package’s QUBO block** \(Q^{(m)}_{ij}\) (the \(m\)-th **column** of the bundling problem — **not** the full \((NM)\times(NM)\) matrix, which is **block-diagonal** across packages). Same spirit as [c-physics](https://github.com/kaistermaister1/c-physics) (Raylib, black background, line mesh, orbit camera).

- **Horizontal grid** \((i,j)\) = variable indices in the block (coverages + slacks).
- **Height** = entry \(Q_{ij}\), normalized by \(\max |Q|\) so the shape fits on screen.
- **Cyan vertical ticks** on the diagonal emphasize **linear** (\(Q_{ii}\)) vs **quadratic coupling** (off-diagonal); each pillar has a **screen label** (`cov` / `slk`) at its tip.
- **Right-hand legend**: what the height field means, what valleys/peaks mean, and a **table** mapping each index to coverage vs slack.
- Optional **gold markers**: binary assignment \(x\) (export with `x_present=1`) and reported **\(E(x)=x^\top Q x + \text{const}\)**.

## 1. Export real data from Python

From `Travelers/code_examples` (paths relative to `subprojects/will/`):

```bash
cd Travelers/code_examples
# Use the project venv if system Python lacks PuLP:
PY=./.venv/bin/python
PYTHONPATH=src $PY ../../visualizations/qubo_vis/scripts/export_qubo_surface.py \
  --data-dir ../docs/data/YQH26_data \
  --package 0 \
  --subsample-coverages 10 --subsample-packages 3 \
  -o ../../visualizations/qubo_vis/qubo_surface.txt
```

Omit subsampling for the full block (large \(n\) → dense mesh; better for screenshots of sparsity structure at moderate \(n\)).

## 2. Build the viewer

Install [Raylib](https://www.raylib.com/). On macOS:

```bash
brew install raylib
cd visualizations/qubo_vis
make
```

Linux: install `libraylib-dev` (or equivalent) so `pkg-config raylib` works, then `make`.

## 3. Run

```bash
./qubo_vis qubo_surface.txt
```

**Controls:** left-drag orbit, scroll zoom, **WASD** pan, **SPACE** recenter target, **R** toggle slow auto-spin (useful for video), **I** show/hide the right-hand info/legend panel (hidden by default), **Q** toggle **Print Q** (heatmap of the full matrix: blue = negative, red = positive, scaled by max\|Q\|; hover a cell for the exact value). While the mouse is over that panel, camera drag/zoom/pan is disabled so you can read the matrix.

## File format (`qubo_surface.txt`)

Text, `#` comments ignored:

1. Header: `n n_coverage n_slack package_index constant_offset`
2. `n` lines, each with `n` floats (rows of \(Q\))
3. `x_present` (`0` or `1`)
4. If `1`, one more line with `n` integers `0/1` (optional assignment)

The exporter writes `x_present` `0` by default; extend the script if you want a known feasible bitstring on the same indices as the QUBO block.

## Story mode (planned — narrative for a general audience)

Goal: **stepped scenes** with **Next / Back** (or chapter buttons) so a recording reads as a story: where the numbers come from, how they combine, and how that connects to **DQI** or **QAOA**. The current viewer is roughly the **late** chapter (one **QUBO block** and its matrix). We will add earlier and later chapters without throwing away this view.

**Intended sequence** (names can be tuned to match slides/code):

1. **Inputs / structure** — What is being optimized (bundles, coverages, packages); optional schematic (no heavy math).
2. **Build M** — However **M** is defined in your pipeline (e.g. margin-related or model matrix); animate or reveal **M** (heatmap / sparse pattern as appropriate).
3. **Build C** — Same for **C**; show it as a separate object before mixing.
4. **Combine M and C** — Visual merge (weighted sum, Hadamard, or whatever the model does) with **short on-screen caption** tied to `insurance_model` / docs.
5. **Weights** — Reveal objective weights (e.g. margins, discounts, affinities) and how they **scale** or **enter** the combined structure.
6. **Penalties** — Reveal constraint penalties (\(\lambda\), squared terms) as **additive** structure; connect to `qubo_block.py` / `02_ilp_to_qubo.html`.
7. **Full Q (conceptual)** — Assemble into the **symmetric Q** for minimization (before or without block truncation); optional **brief** full \((NM)\times(NM)\) **sparsity** or block-outline only so it stays readable.
8. **Block diagonal chop** — Animate **splitting into per-package blocks** (the hackathon “exploit block structure” message); align with **one block** = what this app already shows.
9. **Block field → Hamiltonian field** — **Visual transformation** from QUBO landscape to the **Ising / Pauli** picture (sign flips, \(s_i\in\{-1,1\}\), diagonal shifts) so the **same** problem looks like the **Hamiltonian** QAOA expects.
10. **QAOA / DQI** — Show **parameters**, **depth**, or **black-box** output: energy trace, best bitstring, or histogram; DQI path parallel if different UI.

**Implementation sketch (when we build it):**

- **Python** exports **one JSON (or folder) per scenario** with arrays per step (or deltas from previous step) so C++ can **interpolate** or **cross-fade** between scenes.
- **C++** holds a `StoryStep` index, **Prev/Next** hit-tests (Raylib `CheckCollisionPointRec` / `GuiButton` if we pull in **raygui**), and branches: `switch(step)` for which mesh / colors / captions to draw.
- **This repo’s** `qubo_block.py`, `insurance_model.py`, and `docs/02_ilp_to_qubo.html` are the **source of truth** for labels and formulas on screen.

We are **not** implementing all steps in one go; the list above is the shared checklist as we extend the viewer.
