import path from "node:path";

import type { NextConfig } from "next";

const repoRoot = path.resolve(process.cwd(), "..");

const nextConfig: NextConfig = {
  typedRoutes: true,
  outputFileTracingRoot: repoRoot,
  turbopack: {
    root: repoRoot
  }
};

export default nextConfig;
