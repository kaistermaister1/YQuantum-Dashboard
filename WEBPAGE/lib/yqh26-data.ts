import { access, readFile } from "node:fs/promises";
import path from "node:path";

type CsvRow = Record<string, string>;

export type Yqh26Coverage = {
  coverageIndex: number;
  name: string;
  family: string;
  mandatory: boolean;
  priceUsd: number;
  takeRate: number;
  marginPct: number;
};

export type Yqh26Package = {
  packageIndex: number;
  name: string;
  discount: number;
  maxOptions: number;
};

export type Yqh26Dependency = {
  requiredCoverageIndex: number;
  requiredCoverageName: string;
  dependentCoverageIndex: number;
  dependentCoverageName: string;
};

export type Yqh26IncompatiblePair = {
  coverageAIndex: number;
  coverageAName: string;
  coverageBIndex: number;
  coverageBName: string;
};

export type Yqh26Parameters = {
  N: number;
  M: number;
  nVars: number;
  maxOptionsPerPackageK: number;
  nMandatoryFamilies: number;
  nOptionalFamilies: number;
  nIncompatiblePairs: number;
  nDependencyRules: number;
};

export type Yqh26FamilySummary = {
  family: string;
  type: "Mandatory" | "Optional";
  count: number;
  coverageNames: string[];
  minPrice: number;
  maxPrice: number;
  avgPrice: number;
  avgTakeRate: number;
  avgMarginPct: number;
  avgAffinity: number;
};

export type Yqh26PackageSummary = {
  name: string;
  discount: number;
  maxOptions: number;
  avgAffinity: number;
  topFamily: string;
  topFamilyScore: number;
  topCoverage: string;
  topCoverageScore: number;
  highAffinityCount: number;
};

export type Yqh26AffinityRow = {
  coverageName: string;
  family: string;
  values: Array<{
    packageName: string;
    value: number;
  }>;
};

export type Yqh26DashboardData = {
  overview: {
    totalCoverages: number;
    totalFamilies: number;
    totalPackages: number;
    totalDependencies: number;
    totalIncompatibilities: number;
    totalBasePrice: number;
    averagePrice: number;
    averageTakeRate: number;
    averageMarginPct: number;
    N: number;
    M: number;
    n: number;
    K: number;
  };
  coverages: Yqh26Coverage[];
  packages: Yqh26Package[];
  dependencies: Yqh26Dependency[];
  incompatibilities: Yqh26IncompatiblePair[];
  parameters: Yqh26Parameters;
  familySummaries: Yqh26FamilySummary[];
  packageSummaries: Yqh26PackageSummary[];
  affinityMatrix: Yqh26AffinityRow[];
};

function parseCsv(text: string): CsvRow[] {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headers = lines[0].split(",").map((value) => value.trim());

  return lines.slice(1).map((line) => {
    const values = line.split(",").map((value) => value.trim());

    return headers.reduce<CsvRow>((row, header, index) => {
      row[header] = values[index] ?? "";
      return row;
    }, {});
  });
}

function toNumber(value: string) {
  return Number.parseFloat(value);
}

function toInteger(value: string) {
  return Number.parseInt(value, 10);
}

function toBoolean(value: string) {
  return value.toLowerCase() === "true";
}

function mean(values: number[]) {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function round(value: number, digits = 3) {
  return Number(value.toFixed(digits));
}

const dataDirectory = path.join(
  process.cwd(),
  "..",
  "subprojects",
  "will",
  "Travelers",
  "docs",
  "data",
  "YQH26_data"
);

async function findDataDirectory() {
  await access(path.join(dataDirectory, "instance_coverages.csv"));
  return dataDirectory;
}

async function readDataFile(fileName: string) {
  const dataDirectory = await findDataDirectory();
  return readFile(path.join(dataDirectory, fileName), "utf8");
}

export async function loadYqh26DashboardData(): Promise<Yqh26DashboardData> {
  const [
    coveragesCsv,
    packagesCsv,
    dependenciesCsv,
    incompatiblePairsCsv,
    parametersCsv,
    segmentAffinityCsv
  ] = await Promise.all([
    readDataFile("instance_coverages.csv"),
    readDataFile("instance_packages.csv"),
    readDataFile("instance_dependencies.csv"),
    readDataFile("instance_incompatible_pairs.csv"),
    readDataFile("instance_parameters.csv"),
    readDataFile("instance_segment_affinity.csv")
  ]);

  const coverages = parseCsv(coveragesCsv).map<Yqh26Coverage>((row) => ({
    coverageIndex: toInteger(row.coverage_index),
    name: row.name,
    family: row.family,
    mandatory: toBoolean(row.mandatory),
    priceUsd: toNumber(row.price_usd),
    takeRate: toNumber(row.take_rate),
    marginPct: toNumber(row.margin_pct)
  }));

  const packages = parseCsv(packagesCsv).map<Yqh26Package>((row) => ({
    packageIndex: toInteger(row.package_index),
    name: row.name,
    discount: toNumber(row.discount),
    maxOptions: toInteger(row.max_options)
  }));

  const dependencies = parseCsv(dependenciesCsv).map<Yqh26Dependency>((row) => ({
    requiredCoverageIndex: toInteger(row.required_coverage_index),
    requiredCoverageName: row.required_coverage_name,
    dependentCoverageIndex: toInteger(row.dependent_coverage_index),
    dependentCoverageName: row.dependent_coverage_name
  }));

  const incompatibilities = parseCsv(incompatiblePairsCsv).map<Yqh26IncompatiblePair>((row) => ({
    coverageAIndex: toInteger(row.coverage_a_index),
    coverageAName: row.coverage_a_name,
    coverageBIndex: toInteger(row.coverage_b_index),
    coverageBName: row.coverage_b_name
  }));

  const parametersRow = parseCsv(parametersCsv)[0];
  const parameters: Yqh26Parameters = {
    N: toInteger(parametersRow.N),
    M: toInteger(parametersRow.M),
    nVars: toInteger(parametersRow.n_vars),
    maxOptionsPerPackageK: toInteger(parametersRow.max_options_per_package_K),
    nMandatoryFamilies: toInteger(parametersRow.n_mandatory_families),
    nOptionalFamilies: toInteger(parametersRow.n_optional_families),
    nIncompatiblePairs: toInteger(parametersRow.n_incompatible_pairs),
    nDependencyRules: toInteger(parametersRow.n_dependency_rules)
  };

  const affinityRows = parseCsv(segmentAffinityCsv);
  const affinityLookup = new Map<string, Record<string, number>>();

  affinityRows.forEach((row) => {
    const scores = packages.reduce<Record<string, number>>((accumulator, pkg) => {
      accumulator[pkg.name] = toNumber(row[pkg.name]);
      return accumulator;
    }, {});

    affinityLookup.set(row.coverage, scores);
  });

  const familyGroups = new Map<string, Yqh26Coverage[]>();
  coverages.forEach((coverage) => {
    const group = familyGroups.get(coverage.family) ?? [];
    group.push(coverage);
    familyGroups.set(coverage.family, group);
  });

  const familySummaries = [...familyGroups.entries()]
    .map<Yqh26FamilySummary>(([family, members]) => {
      const prices = members.map((coverage) => coverage.priceUsd);
      const takeRates = members.map((coverage) => coverage.takeRate);
      const margins = members.map((coverage) => coverage.marginPct);
      const affinities = members.flatMap((coverage) => {
        const scores = affinityLookup.get(coverage.name) ?? {};
        return packages.map((pkg) => scores[pkg.name] ?? 0);
      });

      return {
        family,
        type: members.some((coverage) => coverage.mandatory) ? "Mandatory" : "Optional",
        count: members.length,
        coverageNames: members.map((coverage) => coverage.name),
        minPrice: Math.min(...prices),
        maxPrice: Math.max(...prices),
        avgPrice: round(mean(prices), 2),
        avgTakeRate: round(mean(takeRates), 3),
        avgMarginPct: round(mean(margins), 3),
        avgAffinity: round(mean(affinities), 3)
      };
    })
    .sort((left, right) => left.family.localeCompare(right.family));

  const packageSummaries = packages.map<Yqh26PackageSummary>((pkg) => {
    const familyScores = [...familyGroups.entries()].map(([family, members]) => ({
      family,
      score: mean(
        members.map((coverage) => {
          const scores = affinityLookup.get(coverage.name) ?? {};
          return scores[pkg.name] ?? 0;
        })
      )
    }));

    const topFamily = familyScores.reduce((best, current) => (current.score > best.score ? current : best), familyScores[0]);

    const topCoverage = coverages.reduce(
      (best, coverage) => {
        const score = affinityLookup.get(coverage.name)?.[pkg.name] ?? 0;
        return score > best.score ? { name: coverage.name, score } : best;
      },
      { name: coverages[0]?.name ?? "", score: 0 }
    );

    const allScores = coverages.map((coverage) => affinityLookup.get(coverage.name)?.[pkg.name] ?? 0);

    return {
      name: pkg.name,
      discount: pkg.discount,
      maxOptions: pkg.maxOptions,
      avgAffinity: round(mean(allScores), 3),
      topFamily: topFamily.family,
      topFamilyScore: round(topFamily.score, 3),
      topCoverage: topCoverage.name,
      topCoverageScore: round(topCoverage.score, 3),
      highAffinityCount: allScores.filter((score) => score >= 1.1).length
    };
  });

  const affinityMatrix = coverages.map<Yqh26AffinityRow>((coverage) => ({
    coverageName: coverage.name,
    family: coverage.family,
    values: packages.map((pkg) => ({
      packageName: pkg.name,
      value: affinityLookup.get(coverage.name)?.[pkg.name] ?? 0
    }))
  }));

  const prices = coverages.map((coverage) => coverage.priceUsd);
  const takeRates = coverages.map((coverage) => coverage.takeRate);
  const margins = coverages.map((coverage) => coverage.marginPct);

  return {
    overview: {
      totalCoverages: coverages.length,
      totalFamilies: familyGroups.size,
      totalPackages: packages.length,
      totalDependencies: dependencies.length,
      totalIncompatibilities: incompatibilities.length,
      totalBasePrice: round(prices.reduce((sum, value) => sum + value, 0), 2),
      averagePrice: round(mean(prices), 2),
      averageTakeRate: round(mean(takeRates), 3),
      averageMarginPct: round(mean(margins), 3),
      N: parameters.N,
      M: parameters.M,
      n: parameters.nVars,
      K: parameters.maxOptionsPerPackageK
    },
    coverages,
    packages,
    dependencies,
    incompatibilities,
    parameters,
    familySummaries,
    packageSummaries,
    affinityMatrix
  };
}
