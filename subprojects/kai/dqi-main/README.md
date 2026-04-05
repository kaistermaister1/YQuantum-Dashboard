# Code for *“Towards Solving Industrial Integer Linear Programs with Decoded Quantum Interferometry”*

## Introduction

This repository accompanies the paper
> *“Towards Solving Industrial Integer Linear Programs with Decoded Quantum Interferometry”*, [arXiv:2509.08328](https://arxiv.org/abs/2509.08328).

It provides all the necessary code to:
1. Generate an industrial problem along with the corresponding mock data and formulate it as an Integer Linear Program (ILP).
2. Convert any ILP into a **max-XORSAT** instance.
3. Develop a detailed **quantum circuit implementation** of the Decoded Quantum Interferometry (DQI) algorithm using a **quantum version of belief propagation**, a heuristic algorithm for decoding LDPC codes.

---

## What this repository does

- Implements a **full quantum-circuit realization of DQI**, including a **quantum binary belief-propagation decoder**.
- Provides a **pipeline from industrial ILPs to max-XORSAT instances**, enabling DQI to tackle real-world optimization tasks.
- Benchmarks DQI performance against **classical Gurobi solvers** and **random sampling baselines**.
- Estimates **quantum resources** (qubits, gates, and circuit depth) to evaluate hardware feasibility and scaling behavior.

---

## How to cite

If you use this repository in your work, please cite:

**Francesc Sabater, Ouns El Harzli, Geert-Jan Besjes, Marvin Erdmann, Johannes Klepsch, Jonas Hiltrop, Jean-François Bobier, Yudong Cao, and Carlos A. Riofrío.**
*“Towards Solving Industrial Integer Linear Programs with Decoded Quantum Interferometry.”*
arXiv preprint **arXiv:2509.08328**, 2025.


## Create and activate a conda environment

To create the environment:

```bash
conda create -n dqi_env python=3.11
```
To activate the environment:

```bash
conda activate dqi_env
```
## Install dependencies

```bash
pip install -r requirements.txt
```

## GPU operation

If GPUs are available, update the jax library to run with the appropriate CUDA version of your system, for example

```bash
pip install --upgrade "jax[cuda12]"
```
Additionally, you should uncomment the lines:

```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```
in the files:

- experiment_empirical_vs_analytic.py
- experiment_performance_resources.py

## Running pipelines

To reproduce the numerical experiments of https://arxiv.org/pdf/2509.08328, run:
```bash
python -m pipelines.experiment_"name_of_experiment"
```

### `experiment_empirical_vs_analytic.py`
This script compares **empirical quantum simulation results** with **analytic predictions** for decoding performance on small random ILP-derived matrices.

- Generates random binary test cases for different sizes (e.g., 8×6, 6×4, 5×3).
- Computes expected constraint satisfaction using **belief propagation** (`expected_constrains_DQI`) and compares it to empirical averages from the **quantum DQI histogram simulation**.
- Stores results (empirical averages, errors, and expectations) in `.npz` files.
- Produces a **scatter plot with error bars** showing the agreement between analytic and empirical results, with a diagonal line indicating perfect match.

---

### `experiment_histogram_small_max_xorsat.py`
Analyzes **histograms of constraint satisfaction** for a small fixed binary matrix \( B \) with different decoding depths.

- Runs the **quantum DQI histogram simulation** for ℓ = 1, 2, 3 with 10⁴ measurement shots.
- Computes a baseline distribution from **random sampling**.
- Produces comparative histograms, normalized to probabilities.
- Highlights how the distribution of satisfied constraints evolves with decoding depth, showing differences between random baselines and quantum decoders.

---

### `experiment_performance_resources.py`
This is the **largest-scale experiment**, combining industrial ILP instances with benchmarking of performance and quantum resource costs.

- Transforms a JSON-formulated ILP into a binary matrix \( B \) using the max-XORSAT reduction.
- Generates multiple **downsized versions of \( B \)** with controlled marginals for benchmarking.
- Benchmarks the **Gurobi classical solver** on these instances, saving satisfaction ratios.
- Evaluates DQI decoding performance with two belief propagation variants (Gallager and LDPC), across ℓ = 1, 2, 3 and different iteration counts.
- Runs **random sampling benchmarks** for comparison.
- Performs **quantum resource estimation**, computing required qubits and gate counts for different depths and decoding levels.
- Outputs CSV and JSON summaries, and generates plots for **performance, resource usage, and gate composition**.

---

### `experiment_success_rates_2d_plots.py`
Benchmarks **decoder success rates** across varying problem sizes using random ILP instances.

- Generates random ILPs of increasing size, encodes them into binary matrices \( B \).
- For each decoder type (BP1, BP2, and GJ), runs success rate experiments over multiple iterations with 1000 random samples.
- Computes both **success rates** (percentage of satisfied constraints) and **errors**.
- Produces **2D heatmaps** (problem size × decoding depth ℓ) for success rates and error bars, saved as PNGs.
- Provides a **visual benchmarking landscape** showing how decoders scale with ILP size and decoding level.

# Advanced Usage

## Synthetic package data generation

First, use the below to generate the synthetic data. This only needs to be done once.

The process consists of two steps: first, build vehicles; second, measure take rates from these.

### Vehicle generation

Generate some vehicles:

```bash
python synthetic_data_generation/generate_vehicles.py -N 1000 --input_dir synthetic_data_generation/data --output_file synthetic_data_generation/data/test_vehicles.csv
```

Make sure that the number of cars is large enough, else take rates of pairs or even higher order tuples of options cannot be measured.
The script will randomly generate a take rate for each provided option, and then use that take rate to include (or not include) the option in a vehicle.
Conflicting options are never included.

This will also create test_vehicles_raw_take_rates.csv and test_vehicles_take_rates.csv. (Basically, it relies on the basename of your output filename.)

### Take rate data building

Next, generate some package take rates:

```bash
python synthetic_data_generation/build_package_take_rate_data.py --families_file synthetic_data_generation/data/families.csv  --options_file synthetic_data_generation/data/options.csv  --vehicles_file synthetic_data_generation/data/test_vehicles.csv  --template_file synthetic_data_generation/package_templates.yaml    --output_file pipelines/data/take_rates.parquet --output_format parquet
```

For more details on how to use the `package_templates.yaml` file to obtain different sizes of synthetic data, see the README in the `synthetic_data_generation` folder.

## Creating a ILP instance

To create a ILP instance from this synthetic data, run:

```bash
python pipelines/compute_milp_formulation.py
```

This will always return the same ILP instance if run on the same synthetic data. The instance will be stored in the file `pipeline/data/milp_formulation.json`. To generate a different instance, re-run the synthetic data generation with different parameters in `package_templates.yaml`.
