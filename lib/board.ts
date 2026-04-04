export const DECKS = [
  { key: "team", label: "Team", subtitle: "Shared priorities and loose ends" },
  { key: "kai", label: "Kai", subtitle: "Kai's active cards" },
  { key: "cayman", label: "Cayman", subtitle: "Cayman's active cards" },
  { key: "peyton", label: "Peyton", subtitle: "Peyton's active cards" },
  { key: "will", label: "Will", subtitle: "Will's active cards" }
] as const;

export type DeckKey = (typeof DECKS)[number]["key"];

export type BoardCard = {
  id: string;
  deck: DeckKey;
  title: string;
  notes: string;
  position: number;
  createdAt: string;
  updatedAt: string;
};

export const DECK_KEY_SET = new Set<DeckKey>(DECKS.map((deck) => deck.key));
export const LOCAL_STORAGE_KEY = "yquantum-board:v1";

export function isDeckKey(value: string): value is DeckKey {
  return DECK_KEY_SET.has(value as DeckKey);
}

export function sortCards(cards: BoardCard[]) {
  return [...cards].sort((left, right) => {
    if (left.deck !== right.deck) {
      return DECKS.findIndex((deck) => deck.key === left.deck) - DECKS.findIndex((deck) => deck.key === right.deck);
    }

    if (left.position !== right.position) {
      return left.position - right.position;
    }

    return left.updatedAt.localeCompare(right.updatedAt);
  });
}

export function cardsForDeck(cards: BoardCard[], deck: DeckKey) {
  return sortCards(cards).filter((card) => card.deck === deck);
}

export function computeInsertedPosition(cards: BoardCard[], insertIndex: number) {
  const previous = cards[insertIndex - 1];
  const next = cards[insertIndex];

  if (!previous && !next) {
    return 1024;
  }

  if (!previous && next) {
    return next.position - 1024;
  }

  if (previous && !next) {
    return previous.position + 1024;
  }

  return (previous.position + next.position) / 2;
}

export function createLocalCard(input: {
  id?: string;
  deck: DeckKey;
  title: string;
  notes: string;
  position: number;
}): BoardCard {
  const timestamp = new Date().toISOString();

  return {
    id: input.id ?? crypto.randomUUID(),
    deck: input.deck,
    title: input.title.trim(),
    notes: input.notes.trim(),
    position: input.position,
    createdAt: timestamp,
    updatedAt: timestamp
  };
}
