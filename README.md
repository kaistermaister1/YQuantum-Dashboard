# YQuantum Dashboard

This repository is organized for a root-level Vercel deployment.

- Dashboard app: root-level Next.js project in `app/`, `components/`, `lib/`, and `public/`
- Subprojects: `subprojects/` contains member-owned work
- Docs: `docs/` contains the design brief and dashboard notes

## Subprojects

- `subprojects/will/Travelers/` is the primary challenge-materials, data, classical `code_examples`, and `qubo_vis` workspace
- `subprojects/will/qaoa_python/` is Will’s Python package (QAOA on `QuboBlock`, optimizers, tests) plus the Guppy notebook under `qaoa_python/notebooks/`
- `subprojects/kai/` holds Kai's files
- `subprojects/cayman/` holds Cayman's files
- `subprojects/peyton/` holds Peyton's files

## Local Development

```bash
npm install
npm run dev
```

The dashboard reads YQH26 data from `subprojects/will/Travelers/docs/data/YQH26_data`.
