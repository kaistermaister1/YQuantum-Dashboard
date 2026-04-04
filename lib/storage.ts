import { neon } from "@neondatabase/serverless";
import { BoardCard, DeckKey } from "@/lib/board";

declare global {
  var __yquantumBoardSchemaReady: boolean | undefined;
}

type BoardRow = {
  id: string;
  deck: DeckKey;
  title: string;
  notes: string;
  position: number;
  createdAt: string;
  updatedAt: string;
};

function getSql() {
  const databaseUrl = process.env.DATABASE_URL;

  if (!databaseUrl) {
    throw new Error("DATABASE_URL is not configured.");
  }

  return neon(databaseUrl);
}

async function ensureSchema() {
  if (globalThis.__yquantumBoardSchemaReady) {
    return;
  }

  const sql = getSql();

  await sql`
    CREATE TABLE IF NOT EXISTS cards (
      id TEXT PRIMARY KEY,
      deck TEXT NOT NULL,
      title TEXT NOT NULL,
      notes TEXT NOT NULL DEFAULT '',
      position DOUBLE PRECISION NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `;

  await sql`
    CREATE INDEX IF NOT EXISTS cards_deck_position_idx
    ON cards (deck, position)
  `;

  globalThis.__yquantumBoardSchemaReady = true;
}

function mapRows(rows: BoardRow[]): BoardCard[] {
  return rows.map((row) => ({
    id: row.id,
    deck: row.deck,
    title: row.title,
    notes: row.notes,
    position: row.position,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt
  }));
}

export function hasDatabase() {
  return Boolean(process.env.DATABASE_URL);
}

export async function listCards() {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`
    SELECT
      id,
      deck,
      title,
      notes,
      position,
      created_at AS "createdAt",
      updated_at AS "updatedAt"
    FROM cards
    ORDER BY deck ASC, position ASC, updated_at ASC
  `) as BoardRow[];

  return mapRows(rows);
}

export async function createCard(input: {
  id: string;
  deck: DeckKey;
  title: string;
  notes: string;
  position: number;
}) {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`
    INSERT INTO cards (id, deck, title, notes, position)
    VALUES (${input.id}, ${input.deck}, ${input.title}, ${input.notes}, ${input.position})
    RETURNING
      id,
      deck,
      title,
      notes,
      position,
      created_at AS "createdAt",
      updated_at AS "updatedAt"
  `) as BoardRow[];

  return mapRows(rows)[0];
}

export async function updateCard(id: string, updates: Partial<Pick<BoardCard, "deck" | "title" | "notes" | "position">>) {
  await ensureSchema();
  const sql = getSql();
  const rows = (await sql`
    UPDATE cards
    SET
      deck = COALESCE(${updates.deck ?? null}, deck),
      title = COALESCE(${updates.title ?? null}, title),
      notes = COALESCE(${updates.notes ?? null}, notes),
      position = COALESCE(${updates.position ?? null}, position),
      updated_at = NOW()
    WHERE id = ${id}
    RETURNING
      id,
      deck,
      title,
      notes,
      position,
      created_at AS "createdAt",
      updated_at AS "updatedAt"
  `) as BoardRow[];

  return mapRows(rows)[0] ?? null;
}

export async function deleteCard(id: string) {
  await ensureSchema();
  const sql = getSql();
  await sql`DELETE FROM cards WHERE id = ${id}`;
}
