"use client";

import Image from "next/image";
import { useEffect } from "react";

/** Canonical PNGs live in `SOLUTIONS/PLOTS/`; Next serves them via symlinks under `public/plots/`. */
export const QAOA_PLOT_PUBLIC_PATHS = {
  benchmark:
    "/plots/qaoa_p1_benchmark_mean-sh256_pkg0_sub10x3_grid5x5_rand25_cobyla40_spsa25.png",
  histogram: "/plots/qaoa_p1_hist_gamma0.70_beta0.50_sh800.png"
} as const;

export type DashboardPlotId = "benchmark" | "histogram";

export type DashboardPlot = {
  id: DashboardPlotId;
  title: string;
  subtitle: string;
  src: string;
  alt: string;
  /** Short lines shown under the thumbnail */
  summary: string[];
  /** Full explanation in the lightbox */
  description: string[];
};

export const DASHBOARD_PLOTS: DashboardPlot[] = [
  {
    id: "benchmark",
    title: "Optimizer benchmark",
    subtitle:
      "p = 1 · mean objective · 256 shots/eval · pkg 0 · subsample 10×3 · grid 5×5 · rand 25 · COBYLA 40 · SPSA 25",
    src: QAOA_PLOT_PUBLIC_PATHS.benchmark,
    alt: "Bar charts comparing grid, random, COBYLA, and SPSA on best energy, objective, runtime, and evaluation count.",
    summary: [
      "Four bar panels: solution quality, objective, wall time, and Selene evaluation count.",
      "Compares angle-finding methods on the same QUBO block."
    ],
    description: [
      "This figure benchmarks classical outer-loop optimizers that search over QAOA angles (γ, β) for p = 1 on one insurance QUBO block. Each method runs through the same Guppy → Selene-sim stack on your machine; bars are not from Quantinuum hardware.",
      "Top left (best QUBO energy): how good the best observed bitstring was at the angles each method returned. Bars near the green dashed line (exact block minimum) are better; bars far above mean the angles rarely produced low-energy samples.",
      "Top right (optimizer objective): what the routine actually minimized—typically the sample mean energy when statistic=mean. It can stay high even if one lucky bitstring gave a decent “best,” which is why both panels matter.",
      "Bottom row: wall clock time and number of Selene evaluations. Interpret them together: a method with many evaluations and long runtime may win on quality only because it searched harder, not because the algorithm is inherently better.",
      "Single runs are noisy (shots, seeds). Use the CSV produced by the script alongside this PNG if you need exact numbers for a write-up."
    ]
  },
  {
    id: "histogram",
    title: "Energy histogram",
    subtitle: "γ = 0.70 rad · β = 0.50 rad · 800 shots · sample energy distribution",
    src: QAOA_PLOT_PUBLIC_PATHS.histogram,
    alt: "Histogram of QUBO energies from QAOA shots at fixed angles.",
    summary: [
      "Distribution of energies from repeated shots at one (γ, β) point.",
      "Shows how often the circuit lands near optimal vs high energy."
    ],
    description: [
      "This histogram summarizes many measurement shots at a single choice of QAOA angles. Each bar groups bitstrings by their QUBO energy under the same block used elsewhere in the pipeline.",
      "Use it to read typical behavior at those angles: a peak far from the brute-force minimum means most samples are high-energy even if an occasional shot hits a good string. Reference lines for mean energy, best observed energy, and exact minimum (when available) anchor the story.",
      "Regenerate with plot_qaoa_results.py histogram when you change γ, β, shots, or the underlying block so the figure matches your narrative."
    ]
  }
];

type QaoaPlotsTabProps = {
  onOpenPlot: (plot: DashboardPlot) => void;
};

export function QaoaPlotsTab({ onOpenPlot }: QaoaPlotsTabProps) {
  return (
    <section className="plots-tab">
      <article className="plots-intro card-surface">
        <h2 className="plots-intro-title">QAOA figures</h2>
        <p className="plots-intro-lead">
          Source images are stored under{" "}
          <code className="plots-code">SOLUTIONS/PLOTS/</code> (parameterized filenames); the app loads them through{" "}
          <code className="plots-code">public/plots</code> symlinks. Figures come from{" "}
          <code className="plots-code">subprojects/will/qaoa_python</code> (local Selene-sim). Open a tile for the full
          image and how to read it.
        </p>
      </article>

      <div className="plots-grid">
        {DASHBOARD_PLOTS.map((plot) => (
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
    </section>
  );
}

type PlotLightboxProps = {
  plot: DashboardPlot | null;
  onClose: () => void;
};

export function PlotLightbox({ plot, onClose }: PlotLightboxProps) {
  useEffect(() => {
    if (!plot) {
      return;
    }

    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [plot, onClose]);

  if (!plot) {
    return null;
  }

  return (
    <div className="overlay overlay--plot" role="presentation" onClick={onClose}>
      <div
        className="modal modal--plot"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`plot-lightbox-title-${plot.id}`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="plot-lightbox-head">
          <div>
            <h2 id={`plot-lightbox-title-${plot.id}`} className="plot-lightbox-title">
              {plot.title}
            </h2>
            <p className="plot-lightbox-subtitle">{plot.subtitle}</p>
          </div>
          <button type="button" className="button button-soft plot-lightbox-close" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="plot-lightbox-image-wrap">
          <Image
            src={plot.src}
            alt={plot.alt}
            width={1200}
            height={900}
            className="plot-lightbox-image"
            sizes="(max-width: 960px) 100vw, 900px"
            priority
          />
        </div>

        <div className="plot-lightbox-copy">
          <h3 className="plot-lightbox-section-label">What this shows</h3>
          {plot.description.map((paragraph, index) => (
            <p key={index} className="plot-lightbox-paragraph">
              {paragraph}
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}
