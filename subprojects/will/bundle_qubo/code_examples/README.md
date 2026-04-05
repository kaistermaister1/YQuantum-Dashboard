# Code examples (classical + quantum pipeline)

## Classical optimum (PuLP)

```bash
cd code_examples
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
jupyter notebook notebooks/02_classical_baseline.ipynb
```

Run all cells. The notebook loads LTM data from **`../docs/data/YQH26_data/`** (path from `notebooks/`: `../../docs/data/YQH26_data`).

Generated artifacts are written under **`code_examples/data/`** (e.g. `classical_baseline.json`, `toy_landscape.png`).

## QUBO blocks (ILP → per-package Q)

`src/qubo_block.py` builds a **symmetric Q** for each package column so

`energy(x) = xᵀ Q x + constant`

matches the **penalty QUBO** in `docs/02_ilp_to_qubo.html` (mandatory / optional + slack / capacity + slack / incompatibility / dependency). Coverage bits are indices `0 … N-1`; slack bits follow. Use `build_qubo_block_for_package(problem, m)` or `build_all_qubo_blocks(problem)`.

**Test it** (from `code_examples/`, with `../docs/data/YQH26_data` present):

```bash
python -m unittest tests.test_qubo_block -v
```

## Quantum (Guppy + Selene)

The challenge targets **Quantinuum’s stack**: implement and simulate with **`guppylang`** (Guppy), using the **Selene** emulator (`guppylang.emulator`); use **Helios** when running on hardware. Install is included in `requirements.txt` (`pip install guppylang`). Official docs: [Guppy getting started](https://docs.quantinuum.com/guppy/getting_started.html).

**Division of labor:** classical ground truth (PuLP/CBC in the notebooks above) stays in ordinary Python. **QAOA / DQI circuits and execution** should be built and run through **Guppy**, not as a parallel Qiskit-only path, unless organizers say otherwise.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_generation.ipynb` | Export coverage CSV and per-size ILP `.npz` files |
| `02_classical_baseline.ipynb` | Solve all scaling instances with CBC; optional brute-force check for n=10 |
