# YQuantum Dashboard

This repository is organized for a root-level Vercel deployment.

- Dashboard app: root-level Next.js project in `app/`, `components/`, `lib/`, and `public/`
- Subprojects: `subprojects/` contains member-owned work
- Docs: `docs/` contains the design brief and dashboard notes

## Subprojects

- `subprojects/will/Travelers/` is the primary challenge-materials and data workspace
- `subprojects/will/bundle_qubo/` is the smaller QUBO-focused companion workspace
- `subprojects/kai/` holds Kai's files
- `subprojects/cayman/` holds Cayman's files
- `subprojects/peyton/` holds Peyton's files

## Local Development

```bash
npm install
npm run dev
```

The dashboard reads YQH26 data from `subprojects/will/Travelers/docs/data/YQH26_data`.
