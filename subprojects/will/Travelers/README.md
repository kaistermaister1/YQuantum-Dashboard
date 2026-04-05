# DQI vs QAOA for P&C Insurance Optimization

> **QGars copy:** This full challenge tree is versioned in the team repo **YQuantum-Dashboard** (so `git pull` gives everyone the same materials). Create **`code_examples/.venv`** locally: `cd code_examples && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

> **|Y>Quantum 2026 Hackathon Challenge**
> Sponsored by **Travelers Insurance + Quantinuum + LTM**

Comparing two quantum optimization algorithms -- Decoded Quantum Interferometry (DQI) and the Quantum Approximate Optimization Algorithm (QAOA) -- on a P&C insurance product bundling problem, targeting Quantinuum's trapped-ion H-series platform.

## The Problem

A P&C insurer sells individual coverage options (auto liability, collision, comprehensive, roadside, home dwelling, etc.). The goal: decide **which coverages to bundle into discount packages** to **maximize total contribution margin**.

This is a 0-1 Integer Linear Program (ILP) with constraints on coverage families, compatibility, dependencies, and package size. As the number of coverages and packages grows, the problem becomes intractable for classical solvers -- and a candidate for quantum speedup.

## Two Quantum Approaches

| | DQI | QAOA |
|---|---|---|
| **Approach** | Interference-based (no variational loop) | Variational (classical optimizer in the loop) |
| **Circuit** | Dicke state + syndrome encoding + BP1 decoder | Alternating cost/mixer layers |
| **Key mechanism** | Quantum Fourier transform on syndrome register | Parameter optimization to maximize expectation |
| **Constraint handling** | Parity check matrix B (max-XORSAT) | Penalty terms in cost Hamiltonian |
| **Hardware fit** | All-to-all connectivity avoids SWAP overhead in QFT | Native ZZPhase gate = one gate per cost interaction |

## Hackathon Challenge

Teams benchmark DQI vs QAOA on insurance bundling instances of increasing size:

See [challenge.docx](challenge.docx) for the full challenge specification.

**Classical ILP baseline (PuLP / CBC):** from `code_examples/`, run `pip install -r requirements.txt`, then open `code_examples/notebooks/02_classical_baseline.ipynb`. Instance CSVs are in `docs/data/YQH26_data/`.



## License

All solutions and source code are shared with challenge sponsors per |Y>Quantum 2026 rules.
