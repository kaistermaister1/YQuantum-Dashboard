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

## QAOA p = 1 (Guppy + Selene)

`src/qubo_qaoa_p1.py` maps a :class:`QuboBlock`’s **Q** to a **commuting Z Hamiltonian** (via ``x_i → (I−Z_i)/2``), then runs **one** cost layer ``exp(−iγ H_C)`` and **X mixer** ``exp(−iβ ∑ X_i)`` on ``|+⟩^{⊗n}`` using **code-generated** Guppy (Selene emulator). **γ** and **β** are in **radians**.

```python
from pathlib import Path

from src.insurance_model import load_ltm_instance, subsample_problem
from src.qubo_block import build_qubo_block_for_package
from src.qubo_qaoa_p1 import run_qaoa_p1_on_block

# ``Travelers/`` (parent of ``code_examples/``) — adjust ``parents`` if ``__file__`` is deeper
travelers = Path(__file__).resolve().parent.parent
data = travelers / "docs" / "data" / "YQH26_data"
prob = subsample_problem(load_ltm_instance(data), n_coverages=5, n_packages=2)
block = build_qubo_block_for_package(prob, package_index=0, penalty_weight=5000.0)
stats = run_qaoa_p1_on_block(block, gamma=0.4, beta=0.6, shots=2000, seed=1)
print(stats.best_bitstring, stats.best_qubo_energy)
```

**Tests:** `python -m unittest tests.test_qubo_qaoa_p1 tests.test_qubo_qaoa_p2 -v` (needs `guppylang`; use `.venv` from `requirements.txt`).

**p = 2:** `run_qaoa_p2_on_block` in the same module (two cost + two mixer layers).

**Outer optimizer:** `src/qubo_qaoa_optimize.py` — grid / random search; **COBYLA** (SciPy `minimize`, derivative-free with bounds); **2SPSA** (two evals per step, good for noisy shot counts). p = 1: `optimize_qaoa_p1_*`; p = 2: `optimize_qaoa_p2_*`. Objective: **sample-mean** or **best-shot** energy per run (`statistic=`). For **error bars on the mean energy** at fixed angles, use `sample_mean_energy_uncertainty(block, stats)` (SE = std/√N from the histogram). For **angle / objective variability across stochastic reruns**, use `repeat_optimize_qaoa_p1` / `repeat_optimize_qaoa_p2` and `summary.angle_percentiles()`. Tests: `python -m unittest tests.test_qubo_qaoa_optimize -v`. Requires **SciPy** for COBYLA (`requirements.txt`).

## Quantum (Guppy + Selene)

The challenge targets **Quantinuum’s stack**: implement and simulate with **`guppylang`** (Guppy), using the **Selene** emulator (`guppylang.emulator`); use **Helios** when running on hardware. Install is included in `requirements.txt` (`pip install guppylang`). Official docs: [Guppy getting started](https://docs.quantinuum.com/guppy/getting_started.html).

**Division of labor:** classical ground truth (PuLP/CBC in the notebooks above) stays in ordinary Python. **QAOA / DQI circuits and execution** should be built and run through **Guppy**, not as a parallel Qiskit-only path, unless organizers say otherwise.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_generation.ipynb` | Export coverage CSV and per-size ILP `.npz` files |
| `02_classical_baseline.ipynb` | Solve all scaling instances with CBC; optional brute-force check for n=10 |
