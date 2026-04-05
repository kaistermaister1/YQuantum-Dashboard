# Compact QUBO mode — how it works

This note explains **`--compact-qubo`** in `heuristics.py` and `build_qubo_block_for_package(..., compact_penalties=True)` in `SOLUTIONS/qaoa.py`.

## What problem it solves

Each **package-local** QUBO has:

- **`N`** bits for “include coverage *i* in this package or not.”
- Extra **slack** bits so inequality constraints can be written as equalities and squared in the standard penalty–QUBO bridge (see Travelers `02_ilp_to_qubo.html`).

For a **full LTM-style** subsample with **N = 20** coverages, the **default** bridge uses **16** slack bits for optional families, incompatibility, and dependency rules, plus **3** bits for the **capacity** \(\sum_i x_i \le K\) encoding — **19 slack + 20 logical = 39 qubits** per block.

That hits two walls in this repo:

1. **Nexus Selene statevector** allows at most **26** qubits per submitted program.
2. **Local** Guppy/QuEST **full statevector** memory grows as **\(2^{\text{qubits}}\)**; 39 qubits is not runnable on a laptop.

**Compact mode** keeps the **same economic objective** (negated margin on the diagonal) and the **same mandatory-family** encoding, but rewrites **optional / incompatibility / dependency** constraints using **only pairwise and quadratic terms on coverage bits** — **no slack** for those constraint classes. **Capacity** still uses the **existing binary slack** (for **K = 7**, **3** slack bits). So for **N = 20** you get **20 + 3 = 23** qubits, which fits Selene and is feasible for local simulation.

## Default vs compact (per block)

| Constraint class | Default (slack bridge) | Compact mode |
|------------------|-------------------------|--------------|
| **Mandatory family** (exactly one) | \((\sum_{i \in F} x_i - 1)^2\) — **no slack** | **Unchanged** |
| **Optional family** (at most one) | \(\big(\sum_{i \in F} x_i + s - 1\big)^2\) — **1 slack** | \(\lambda \sum_{i<j \in F} x_i x_j\) — **no slack** |
| **Capacity** \(\sum_i x_i \le K\) | \(\big(\sum_i x_i + \sum_b 2^b s_b - K\big)^2\) — **3 slacks** for K=7 | **Unchanged** |
| **Incompatible pair** | \((x_i + x_j + s - 1)^2\) — **1 slack** | \(\lambda\, x_i x_j\) — **no slack** |
| **Dependency** (dependent \(j\) needs prerequisite \(i\): \(x_j \le x_i\)) | \((x_j - x_i + s)^2\) style with slack | \(\lambda\,(x_j - x_i x_j)\) on the QUBO — **no slack** |

All penalties are scaled by the same **`λ`** per package as in the default builder (`default_penalty_weight` ≈ **3 × max |margin coefficient|** for that package).

## Energy model (reminder)

The block stores a symmetric **`Q`** and **`constant_offset`**. For binary **\(x \in \{0,1\}^n\)**,

\[
E(x) = x^\top Q x + \text{constant\_offset}.
\]

The QAOA pipeline **minimizes** \(E\) (profit enters as **negative** linear terms on the diagonal). Feasible ILP-style assignments should be **low** energy when **λ** is large enough; infeasible assignments get **penalty** energy from the constraint terms.

## Why optional “at most one” can use pairwise terms

For a family \(F\), add \(\lambda \sum_{i<j} x_i x_j\). If **two or more** coverages in \(F\) are 1, at least one pair contributes **λ** to the energy. If **zero or one** is 1, all pairwise products are 0. So the penalty is zero exactly for **at most one** selected (for that family), without introducing slack.

## Why dependency uses \(x_j - x_i x_j\)

Constraint **\(x_j \le x_i\)** (dependent **j** requires prerequisite **i**) is violated only when **\(x_j = 1\)** and **\(x_i = 0\)**. The term **\(\lambda (x_j - x_i x_j)\)** is **λ** in that case and **0** otherwise (check the four assignments). Expanded into **\(x^\top Q x\)** this adds a **linear** piece on **\(x_j\)** and an **off-diagonal** coupling between **\(i\)** and **\(j\)** (implemented in `_add_dependency_penalty_no_slack`).

## Why capacity still uses slack

A correct **cardinality cap** \(\sum_i x_i \le K\) without slack usually needs either **many** high-order terms or an **integer slack** encoding. The existing code uses a **small** binary expansion of slack (**\(\lceil \log_2(K+1) \rceil\)** bits). For **K = 7** that is **3** qubits — cheap compared to the **16** slacks compact mode removes.

## How to run

```bash
cd SOLUTIONS
PYTHONPATH=. ../.venv/bin/python heuristics.py \
  --algorithm qaoa --n 20 --m 3 --p 1 \
  --optimizer spsa --target selene --compact-qubo
```

- **`--target local`** uses the same compact **Q** with the local Guppy emulator.
- **`--compact-qubo`** is **off** by default; normal runs are unchanged.

## Wall time on Selene (SPSA vs COBYLA, tuning)

**SPSA** runs **two** objective evaluations per outer step (simultaneous perturbation **θ⁺** and **θ⁻**), then advances **θ**. With **`m`** package blocks, a full SPSA pass does about **`SPSA_MAXITER × 2 × m`** objective calls (each call is a circuit batch on Selene). **COBYLA** typically uses **one** evaluation per iteration and often stops before the iteration cap, so for similar caps it tends to use **fewer total jobs** than SPSA.

**What dominates:** Nexus wall clock is often **minutes per `execute` batch** (queue + simulator), not the number of shots. Defaults in `qaoa.py` (`SHOTS`, `SPSA_MAXITER`, `COBYLA_MAXITER`) are set for **shorter Selene runs**; for a higher-quality study, raise them or pass **`heuristics.py --shots N --spsa-maxiter N --cobyla-maxiter N`** without editing modules.

**If you need a very short smoke test** (order of tens of minutes, not hours): prefer **`--optimizer cobyla`**, reduce **`--m`** (e.g. **`--m 1`**), and/or pass aggressive **`--spsa-maxiter`** / **`--shots`**. A full **`n=20`, `m=3`, SPSA** job can still take **hours** when each Selene round-trip is slow; there is no substitute for fewer optimizer steps or fewer blocks when latency per job is large.

### Fast Selene mode (`--fast-selene`)

For **wall-clock** demos when each Nexus `execute` takes minutes, SPSA/COBYLA are the wrong tool: they issue **many** sequential jobs. **`--fast-selene`** instead:

1. Builds **several** QAOA programs (random angles, or a fixed **3×3** γ/β grid when **`--p 1`** and **`--fast-selene-strategy grid`**).
2. Submits them in **one** `execute()` per package (see `optimize_block_batched_theta_search_selene` in `qaoa_selene.py`).
3. Runs **multiple packages in parallel** (thread pool) so three blocks can overlap on the wire.

Default **`--fast-selene-trials 8`** with **`--shots`** defaulting to **48** when you omit **`--shots`**, targets **roughly one long round-trip per package** instead of dozens. This is **not** the same optimization quality as SPSA/COBYLA; use it for **smoke tests** and report **`random_batch`** / **`grid_batch`** in summaries.

For **small** **`N×M`** (below **`SELENE_FAST_SELENE_MIN_NM`** in `heuristics.py`, currently **41**), **`--fast-selene` is ignored** so Selene runs use normal **`--optimizer spsa`** or **`cobyla`** (override with **`--force-fast-selene`** if you really want the batched random path on a tiny instance).

Example (compact **N=20**, three packages, under ~10 minutes when parallel + batching behave well):

```bash
cd SOLUTIONS
MPLCONFIGDIR=/tmp/mpl-heur PYTHONUNBUFFERED=1 PYTHONPATH=. ../.venv/bin/python -u heuristics.py \
  --algorithm qaoa --n 20 --m 3 --p 1 --target selene --compact-qubo --fast-selene
```

Optional: **`--fast-selene-strategy grid`** (deterministic, 9 circuits per package, **`p=1` only**), or **`--fast-selene-trials 6`** / **`--shots 32`** to trim upload/sim work further.

**Artifacts:** `run_id` contains **`_cq1`**; `run_summaries.csv` **`notes`** mention compact mode.

## Relation to the teaching materials

The Travelers HTML / notebook story emphasizes **slack + equality** for inequalities because it is easy to explain and matches a common textbook bridge. **Compact mode** is an **alternative QUBO** that is smaller on the wire. For **large λ**, minimizers should still align with **feasible, high-margin** classical solutions, but **finite λ**, **sampling noise**, and **QAOA** non-convexity mean you should still **compare to PuLP** and report **approximation ratio** and **feasibility**.

## Code map

- **`SOLUTIONS/qaoa.py`**: `build_qubo_block_for_package(..., compact_penalties=...)`, helpers `_add_pairwise_at_most_one_penalty`, `_add_incompatible_pair_penalty`, `_add_dependency_penalty_no_slack`.
- **`SOLUTIONS/heuristics.py`**: `--compact-qubo` → `compact_qubo` → `build_qubo_block_for_package(..., compact_penalties=True)`.
- **Selene cap**: `SELENE_STATEVECTOR_MAX_QUBITS` in `qaoa.py` (still **26** for Nexus statevector).

See also **`QAOA_SCALE_LIMITS.md`** for simulator and API limits without the full penalty story.
