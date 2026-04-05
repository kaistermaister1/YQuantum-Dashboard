"use client";

import { useMemo, useState } from "react";
import type { Yqh26AffinityRow, Yqh26Coverage, Yqh26DashboardData, Yqh26Package } from "@/lib/yqh26-data";

type Yqh26DataTabProps = {
  dataset: Yqh26DashboardData;
};

const PRICE_SENSITIVITY_BETA = 1.2;

function titleize(value: string) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(0)}%`;
}

function formatDecimal(value: number) {
  return value.toFixed(2);
}

function summarizeCims(values: number[]) {
  if (values.length === 0) {
    return "c avg 0.00";
  }

  const total = values.reduce((sum, value) => sum + value, 0);
  const average = total / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);

  if (values.length === 1) {
    return `c ${formatDecimal(values[0])}`;
  }

  return `c avg ${formatDecimal(average)} | ${formatDecimal(min)}-${formatDecimal(max)}`;
}

function computeCim(coverage: Yqh26Coverage, affinity: number, discount: number) {
  return (
    coverage.priceUsd *
    coverage.marginPct *
    (1 - discount) *
    coverage.takeRate *
    affinity *
    (1 + PRICE_SENSITIVITY_BETA * discount)
  );
}

function getAffinityValue(row: Yqh26AffinityRow | undefined, packageName: string) {
  return row?.values.find((value) => value.packageName === packageName)?.value ?? 0;
}

function buildPackageTooltip(pkg: Yqh26Package, dataset: Yqh26DashboardData, affinityMap: Map<string, Yqh26AffinityRow>) {
  const cimValues = dataset.coverages.map((coverage) =>
    computeCim(coverage, getAffinityValue(affinityMap.get(coverage.name), pkg.name), pkg.discount)
  );

  return [`${pkg.name}`, `delta ${formatDecimal(pkg.discount)} | ${formatPercent(pkg.discount)} off`, summarizeCims(cimValues)].join(
    "\n"
  );
}

export function Yqh26DataTab({ dataset }: Yqh26DataTabProps) {
  const [bundleCount, setBundleCount] = useState(dataset.packages.length);

  const affinityMap = useMemo(
    () => new Map(dataset.affinityMatrix.map((row) => [row.coverageName, row])),
    [dataset.affinityMatrix]
  );

  const sortedFamilies = useMemo(
    () =>
      [...dataset.familySummaries].sort((left, right) => {
        if (left.type !== right.type) {
          return left.type === "Mandatory" ? -1 : 1;
        }

        return left.family.localeCompare(right.family);
      }),
    [dataset.familySummaries]
  );

  const displayedPackages = dataset.packages.slice(0, bundleCount);

  return (
    <section className="dataset-tab dataset-tab-compact">
      <section className="dataset-summary-grid dataset-summary-grid-tight">
        <article className="quick-list-card family-table-card card-surface">
          <h3>Coverage Names by Family</h3>
          <div className="table-shell concise-table-shell">
            <table className="data-table concise-family-table">
              <thead>
                <tr>
                  <th>Family</th>
                  <th>Type</th>
                  <th>Coverage Count</th>
                  <th>Coverage Names</th>
                </tr>
              </thead>
              <tbody>
                {sortedFamilies.map((family) => {
                  const familyCoverages = dataset.coverages.filter((coverage) => coverage.family === family.family);

                  const familyTooltip = [
                    `${titleize(family.family)}`,
                    `${family.type.toLowerCase()} | ${family.count} coverages`,
                    summarizeCims(
                      familyCoverages.flatMap((coverage) =>
                        displayedPackages.map((pkg) =>
                          computeCim(coverage, getAffinityValue(affinityMap.get(coverage.name), pkg.name), pkg.discount)
                        )
                      )
                    )
                  ].join("\n");

                  return (
                    <tr key={family.family}>
                      <td>
                        <div className="family-name-stack" title={familyTooltip}>
                          <strong>{titleize(family.family)}</strong>
                        </div>
                      </td>
                      <td>
                        <span className={`pill ${family.type === "Mandatory" ? "pill-mandatory" : "pill-optional"}`}>
                          {family.type}
                        </span>
                      </td>
                      <td>{family.count}</td>
                      <td className="coverage-name-cell">
                        <div className="name-pill-grid">
                          {family.coverageNames.map((coverageName) => {
                            const coverage = dataset.coverages.find((item) => item.name === coverageName);
                            const tooltip = coverage
                              ? [
                                  `${titleize(coverage.name)}`,
                                  `$${coverage.priceUsd.toFixed(0)} | ${formatPercent(coverage.marginPct)} margin | ${formatPercent(coverage.takeRate)} take`,
                                  summarizeCims(
                                    displayedPackages.map((pkg) =>
                                      computeCim(
                                        coverage,
                                        getAffinityValue(affinityMap.get(coverage.name), pkg.name),
                                        pkg.discount
                                      )
                                    )
                                  )
                                ].join("\n")
                              : titleize(coverageName);

                            return (
                              <span
                                key={coverageName}
                                className="name-pill name-pill-subtle hover-chip"
                                data-tooltip={tooltip}
                              >
                                {titleize(coverageName)}
                              </span>
                            );
                          })}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </article>

        <article className="quick-list-card quick-list-card-sticky card-surface">
          <div className="bundles-head">
            <h3>Package Names</h3>
            <div className="bundle-toggle">
              <label htmlFor="bundle-count">m = {bundleCount}</label>
              <input
                id="bundle-count"
                type="range"
                min={1}
                max={dataset.packages.length}
                value={bundleCount}
                onChange={(event) => setBundleCount(Number(event.target.value))}
              />
            </div>
          </div>

          <div className="name-pill-grid" aria-label="Displayed YQH26 package names">
            {displayedPackages.map((pkg) => {
              const tooltip = buildPackageTooltip(pkg, dataset, affinityMap);

              return (
                <span key={pkg.name} className="name-pill hover-chip" data-tooltip={tooltip}>
                  {pkg.name}
                </span>
              );
            })}
          </div>
        </article>
      </section>
    </section>
  );
}
