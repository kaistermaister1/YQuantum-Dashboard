"use client";

import Image from "next/image";
import { useEffect, useState } from "react";
import { PlotLightbox, type DashboardPlot, type DashboardPlotCategory } from "@/components/qaoa-plots-tab";

export const HEURISTICS_PLOT_PUBLIC_PATHS = {
  approx_ratio: "/plots/heuristics/approx_ratio_vs_size.png",
  runtime: "/plots/heuristics/runtime_vs_size.png",
  feasibility: "/plots/heuristics/feasibility_vs_size.png",
  resources: "/plots/heuristics/resources_vs_size.png",
  approx_p: "/plots/heuristics/approx_ratio_vs_p.png",
  evals: "/plots/heuristics/evals_vs_size.png",
  runtime_fixed_m: "/plots/heuristics/runtime_fixed_m.png",
  runtime_fixed_n: "/plots/heuristics/runtime_fixed_n.png",
  resources_nlocal: "/plots/heuristics/resources_vs_nlocal.png",
  runtime_3d: "/plots/heuristics/3d_runtime_surface.png",
  approx_ratio_3d: "/plots/heuristics/3d_approx_ratio_surface.png",
  profit_scatter: "/plots/heuristics/profit_scatter.png",
  approx_dist: "/plots/heuristics/approx_ratio_dist.png",
  runtime_evals: "/plots/heuristics/runtime_vs_evals.png",
  feasibility_p: "/plots/heuristics/feasibility_vs_p.png",
  compiled_depth: "/plots/heuristics/compiled_depth_vs_nlocal.png"
} as const;

export const HEURISTICS_PLOTS: DashboardPlot[] = [
  {
    id: "approx_ratio",
    category: "Solution Quality",
    title: "Solution Quality",
    subtitle: "Approximation Ratio vs Problem Size (n_total)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.approx_ratio,
    alt: "Line plot showing Approximation Ratio vs Problem Size.",
    summary: [
      "Approximation ratio degrades as problem size grows."
    ],
    description: [
      "Shows the ratio of the best profit found by QAOA to the classical optimal profit.",
      "Notice the degradation as the number of variables (n_total) increases from 10 to 50."
    ]
  },
  {
    id: "profit_scatter",
    category: "Solution Quality",
    title: "QAOA vs Classical Profit",
    subtitle: "Best Profit vs Classical Optimal Profit",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.profit_scatter,
    alt: "Scatter plot of QAOA profit vs Classical profit.",
    summary: [
      "Direct comparison of QAOA profits against the classical optimum."
    ],
    description: [
      "Plots the best profit found by QAOA against the true classical optimal profit for each run.",
      "Points falling on the dashed line represent perfect solutions. Points below the line show the optimality gap."
    ]
  },
  {
    id: "approx_dist",
    category: "Solution Quality",
    title: "Approximation Distribution",
    subtitle: "Approximation Ratio by Optimizer",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.approx_dist,
    alt: "Box plot of approximation ratios by optimizer.",
    summary: [
      "Distribution of solution quality across different optimizers."
    ],
    description: [
      "A box plot showing the spread of approximation ratios achieved by each classical optimizer.",
      "Highlights the variance and median performance of COBYLA vs SPSA across all runs."
    ]
  },
  {
    id: "approx_p",
    category: "Solution Quality",
    title: "Depth Tradeoff",
    subtitle: "Approximation Ratio vs Circuit Depth (p)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.approx_p,
    alt: "Line plot showing Approximation Ratio vs Circuit Depth (p).",
    summary: [
      "Increasing depth (p) generally improves solution quality."
    ],
    description: [
      "Explores the depth-quality tradeoff by plotting the approximation ratio against the circuit depth parameter (p).",
      "Higher p provides a richer parameter landscape, but makes classical optimization harder."
    ]
  },
  {
    id: "approx_ratio_3d",
    category: "Solution Quality",
    title: "3D Quality Landscape",
    subtitle: "Approximation Ratio vs Block Size and Packages",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.approx_ratio_3d,
    alt: "3D surface plot showing Approximation Ratio vs N_local and M_blocks.",
    summary: [
      "The 'landscape of difficulty' across both problem dimensions."
    ],
    description: [
      "Maps the approximation ratio across the full grid of problem sizes for the SPSA optimizer.",
      "Helps identify the 'sweet spot' where QAOA can still find high-quality solutions."
    ]
  },
  {
    id: "runtime",
    category: "Workflow Cost",
    title: "Workflow Cost",
    subtitle: "Runtime (seconds) vs Problem Size (n_total)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.runtime,
    alt: "Line plot showing Runtime vs Problem Size on a logarithmic scale.",
    summary: [
      "Runtime scales exponentially with problem size."
    ],
    description: [
      "Benchmarks the total workflow cost by plotting the runtime in seconds against the problem size.",
      "QAOA requires many circuit evaluations per optimizer step, leading to higher runtimes than the classical baseline."
    ]
  },
  {
    id: "runtime_fixed_m",
    category: "Workflow Cost",
    title: "Runtime Scaling (Fixed M)",
    subtitle: "Runtime vs Block Size (N_local) for Fixed Packages (M)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.runtime_fixed_m,
    alt: "Line plot showing Runtime vs N_local for different values of M_blocks.",
    summary: [
      "Increasing block size (N_local) drives exponential runtime growth."
    ],
    description: [
      "Isolates the effect of the block size (N_local) on the runtime for a fixed number of packages (M).",
      "The logarithmic scale reveals the exponential difficulty of optimizing larger sub-problems."
    ]
  },
  {
    id: "runtime_fixed_n",
    category: "Workflow Cost",
    title: "Runtime Scaling (Fixed N)",
    subtitle: "Runtime vs Packages (M_blocks) for Fixed Block Size (N_local)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.runtime_fixed_n,
    alt: "Line plot showing Runtime vs M_blocks for different values of N_local.",
    summary: [
      "Adding more packages scales runtime linearly."
    ],
    description: [
      "Isolates the effect of adding more packages (M_blocks) on the runtime for a fixed block size (N_local).",
      "Highlights the exact advantage of the block-diagonal structure: adding independent blocks scales linearly."
    ]
  },
  {
    id: "evals",
    category: "Workflow Cost",
    title: "Optimization Effort",
    subtitle: "Objective Evaluations vs Problem Size",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.evals,
    alt: "Line plot showing Objective Evaluations vs Problem Size.",
    summary: [
      "SPSA requires more evaluations than COBYLA."
    ],
    description: [
      "Tracks the number of circuit runs required by the classical optimizers as the problem size grows.",
      "SPSA requires more evaluations but is more robust to statistical noise than COBYLA."
    ]
  },
  {
    id: "runtime_evals",
    category: "Workflow Cost",
    title: "Runtime vs Evaluations",
    subtitle: "Runtime (seconds) vs Objective Evaluations",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.runtime_evals,
    alt: "Scatter plot of runtime vs objective evaluations.",
    summary: [
      "Runtime is heavily driven by the number of circuit evaluations."
    ],
    description: [
      "Plots the total wall-clock runtime against the number of objective evaluations.",
      "Confirms that the optimizer's evaluation count is the primary driver of QAOA's workflow cost."
    ]
  },
  {
    id: "runtime_3d",
    category: "Workflow Cost",
    title: "3D Runtime Landscape",
    subtitle: "Runtime vs Block Size (N_local) and Packages (M_blocks)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.runtime_3d,
    alt: "3D surface plot showing Runtime vs N_local and M_blocks.",
    summary: [
      "Runtime spikes along the N_local axis, not M_blocks."
    ],
    description: [
      "A comprehensive view of how wall-clock runtime scales across the entire grid of problem sizes.",
      "Notice how runtime shoots up dramatically as N_local increases, compared to the gradual incline along M_blocks."
    ]
  },
  {
    id: "resources",
    category: "Hardware Resources",
    title: "Total Hardware Resources",
    subtitle: "Qubits & Gates vs Problem Size",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.resources,
    alt: "Line plot showing Qubits and Two-Qubit Gates vs Problem Size.",
    summary: [
      "Two-qubit gate counts grow rapidly with problem size."
    ],
    description: [
      "Illustrates the hardware resources required by the QAOA circuits as the total problem size increases.",
      "Qubits scale linearly, while the two-qubit gate count (driven by ZZPhase penalty terms) grows more rapidly."
    ]
  },
  {
    id: "resources_nlocal",
    category: "Hardware Resources",
    title: "Per-Block Resources",
    subtitle: "Qubits & Gates vs Block Size (N_local)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.resources_nlocal,
    alt: "Line plot showing Qubits and Two-Qubit Gates vs N_local.",
    summary: [
      "Decomposing the problem drastically reduces per-block resources."
    ],
    description: [
      "Focuses on the hardware resources required for a single package-local sub-problem (N_local).",
      "Demonstrates how the block-diagonal structure makes it more feasible to run on near-term hardware."
    ]
  },
  {
    id: "compiled_depth",
    category: "Hardware Resources",
    title: "Compiled Circuit Depth",
    subtitle: "Compiled Depth vs Block Size (N_local)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.compiled_depth,
    alt: "Line plot showing Compiled Circuit Depth vs N_local.",
    summary: [
      "Compiled depth scales with block size."
    ],
    description: [
      "Shows how the compiled circuit depth grows as the number of coverages per package increases.",
      "Deeper circuits are more susceptible to hardware noise, impacting the final solution quality."
    ]
  },
  {
    id: "feasibility",
    category: "Constraint Handling",
    title: "Constraint Handling",
    subtitle: "Feasibility Rate vs Problem Size (n_total)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.feasibility,
    alt: "Line plot showing Feasibility Rate vs Problem Size.",
    summary: [
      "Feasibility drops sharply as problem size grows."
    ],
    description: [
      "Analyzes constraint handling by plotting the fraction of measured shots that decode to valid solutions.",
      "Highlights the difficulty of enforcing hard constraints via penalty terms at scale."
    ]
  },
  {
    id: "feasibility_p",
    category: "Constraint Handling",
    title: "Feasibility vs Depth",
    subtitle: "Feasibility Rate vs Circuit Depth (p)",
    src: HEURISTICS_PLOT_PUBLIC_PATHS.feasibility_p,
    alt: "Line plot showing Feasibility Rate vs Circuit Depth (p).",
    summary: [
      "Deeper circuits can improve the feasibility rate."
    ],
    description: [
      "Shows how increasing the circuit depth (p) affects the probability of sampling a valid solution.",
      "More layers allow the circuit to better learn the penalty structure and avoid invalid states."
    ]
  }
];

type HeuristicsPlotsTabProps = {
  onOpenPlot: (plot: DashboardPlot) => void;
};

export function HeuristicsPlotsTab({ onOpenPlot }: HeuristicsPlotsTabProps) {
  const categories: DashboardPlotCategory[] = [
    "Solution Quality",
    "Workflow Cost",
    "Hardware Resources",
    "Constraint Handling"
  ];

  return (
    <section className="plots-tab">
      <article className="plots-intro card-surface">
        <h2 className="plots-intro-title">Heuristics Scaling Analysis</h2>
        <p className="plots-intro-lead">
          These figures are generated from the data in <code className="plots-code">SOLUTIONS/HEURISTICS/run_summaries.csv</code>. 
          They benchmark solution quality, workflow cost, and constraint handling across different problem sizes.
        </p>
      </article>

      {categories.map((category) => {
        const categoryPlots = HEURISTICS_PLOTS.filter((plot) => plot.category === category);
        if (categoryPlots.length === 0) return null;

        return (
          <div key={category} className="plot-category">
            <h3 className="plot-category-title">{category}</h3>
            <div className="plots-grid">
              {categoryPlots.map((plot) => (
                <button
                  key={plot.id}
                  type="button"
                  className="plot-thumb"
                  aria-label={`Open full size: ${plot.title}`}
                  onClick={() => onOpenPlot(plot)}
                >
                  <span className="plot-thumb-frame">
                    <Image
                      src={plot.src}
                      alt=""
                      width={640}
                      height={360}
                      className="plot-thumb-image"
                      sizes="(max-width: 720px) 100vw, 320px"
                    />
                  </span>
                  <span className="plot-thumb-body">
                    <span className="plot-thumb-title">{plot.title}</span>
                    <span className="plot-thumb-subtitle">{plot.subtitle}</span>
                    {plot.summary.map((line, index) => (
                      <span key={index} className="plot-thumb-line">
                        {line}
                      </span>
                    ))}
                    <span className="plot-thumb-cta">View full size</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </section>
  );
}
