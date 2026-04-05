# QAOA scale limits (this repo / current stack)

## Why large `--n` (many coverages per package) often “does not work”

Package-local QUBOs use **one binary per coverage** plus **slack qubits** for squared penalty constraints. Example from a failed run: **`n=20` coverages → 20 + 19 = 39 qubits** per block.

### 1. Nexus Selene (statevector)

- `qnexus` **execute** with Selene **statevector** rejects jobs above **26 qubits** per program (HTTP 400).
- Enforced in code via `SELENE_STATEVECTOR_MAX_QUBITS` in `SOLUTIONS/qaoa.py`; `heuristics.py --target selene` fails fast with a clear error if the block is larger.

### 2. Local Guppy / QuEST emulator (statevector)

- The local path uses a **full statevector** simulator. Memory scales as **\(2^{\text{qubits}}\)** (complex amplitudes).
- **39 qubits** implies on the order of **tebibytes** of RAM — not feasible on a laptop. A typical failure is QuEST reporting multi‑TiB allocation before the run aborts.

### What still works at large `n`

- **Classical ILP** (`heuristics.py --algorithm classical --n … --m …`) is fine for **n=20** and beyond; use it for baselines and approximation ratios.

### Challenge framing (sponsor materials)

- Full benchmark instances may use **N=20 coverages per package block** on **hardware**, where you execute a **physical circuit** — you are **not** holding a **\(2^{39}\)** vector on your laptop.
- **Exact classical simulation** of that same wide register is a different resource model than **device execution**.

### Practical commands

- **Selene / sim cap:** keep package-local qubits **≤ 26** (raise `n` only while `block.n_vars` stays under that; depends on slack count).
- **Local statevector:** stay in a **low‑20s qubit** regime on typical machines, or expect failure / swap.

## Compact QUBO mode (`--compact-qubo`)

The default bridge needs **39** qubits for **N=20**; compact mode needs **23** (fits Selene and local sim). **Full explanation** (math, constraint-by-constraint table, CLI): see **`COMPACT_QUBO.md`** in this folder.

**Caveat:** Different QUBO than the slack+equality teaching formulation; compare to classical and tag runs (`_cq1` in `run_id`, `notes` in summaries).
