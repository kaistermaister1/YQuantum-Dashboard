# YQuantum Board

Simple five-deck kanban board built for `Team`, `Kai`, `Cayman`, `Peyton`, and `Will`.

## What it does

- Shows all five decks side by side
- Lets anyone add, edit, delete, and move cards
- Keeps data out of the codebase when `DATABASE_URL` is configured
- Falls back to browser `localStorage` when a database is not configured yet

## Local Development

```bash
npm install
npm run dev
```

The app now runs from the repository root and reads the YQH26 dataset from `subprojects/will/Travelers/docs/data/YQH26_data`.

## Deploying on Vercel

1. Import the repository in Vercel with the project root set to `.`
2. Add `DATABASE_URL` if you want shared board storage
3. Deploy

Once `DATABASE_URL` is present, cards are stored in Postgres instead of the browser, so shipping new code will not reset the board.
