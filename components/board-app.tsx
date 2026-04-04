"use client";

import { FormEvent, useEffect, useMemo, useState, useTransition } from "react";
import {
  BoardCard,
  DeckKey,
  LOCAL_STORAGE_KEY,
  cardsForDeck,
  computeInsertedPosition,
  createLocalCard,
  DECKS,
  isDeckKey,
  sortCards
} from "@/lib/board";

type StorageMode = "local" | "remote";

type BoardAppProps = {
  storageMode: StorageMode;
};

type DraftState = Record<DeckKey, { title: string; notes: string }>;

type EditingState = {
  id: string;
  title: string;
  notes: string;
  deck: DeckKey;
};

const EMPTY_DRAFTS = DECKS.reduce(
  (accumulator, deck) => {
    accumulator[deck.key] = { title: "", notes: "" };
    return accumulator;
  },
  {} as DraftState
);

function loadLocalCards() {
  if (typeof window === "undefined") {
    return [];
  }

  const raw = window.localStorage.getItem(LOCAL_STORAGE_KEY);
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as BoardCard[];
    return sortCards(parsed.filter((card) => isDeckKey(card.deck)));
  } catch {
    return [];
  }
}

function saveLocalCards(cards: BoardCard[]) {
  window.localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(sortCards(cards)));
}

export function BoardApp({ storageMode }: BoardAppProps) {
  const [cards, setCards] = useState<BoardCard[]>([]);
  const [drafts, setDrafts] = useState<DraftState>(EMPTY_DRAFTS);
  const [editing, setEditing] = useState<EditingState | null>(null);
  const [dragState, setDragState] = useState<{ cardId: string; deck: DeckKey } | null>(null);
  const [activeDrop, setActiveDrop] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("Ready to move cards.");
  const [isSaving, startSavingTransition] = useTransition();

  const lastSyncedText = useMemo(() => {
    return storageMode === "remote" ? "Live sync every 3 seconds" : "Stored in this browser only";
  }, [storageMode]);

  async function syncRemoteBoard() {
    const response = await fetch("/api/board", { cache: "no-store" });

    if (!response.ok) {
      throw new Error("Unable to refresh the board right now.");
    }

    const payload = (await response.json()) as { cards: BoardCard[] };
    setCards(sortCards(payload.cards));
  }

  useEffect(() => {
    if (storageMode === "local") {
      const localCards = loadLocalCards();
      setCards(localCards);
      setStatusMessage("Local mode is on. Add a database when you're ready to share across laptops.");

      const handleStorage = (event: StorageEvent) => {
        if (event.key === LOCAL_STORAGE_KEY) {
          setCards(loadLocalCards());
        }
      };

      window.addEventListener("storage", handleStorage);
      return () => window.removeEventListener("storage", handleStorage);
    }

    let cancelled = false;

    const load = async () => {
      try {
        const response = await fetch("/api/board", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("Unable to reach the shared board.");
        }

        const payload = (await response.json()) as { cards: BoardCard[] };
        if (!cancelled) {
          setCards(sortCards(payload.cards));
          setStatusMessage("Shared board is live. Changes from the team will keep rolling in.");
        }
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(error instanceof Error ? error.message : "Board sync failed.");
        }
      }
    };

    void load();
    const intervalId = window.setInterval(() => {
      void load();
    }, 3000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [storageMode]);

  const deckMap = useMemo(() => {
    return DECKS.reduce(
      (accumulator, deck) => {
        accumulator[deck.key] = cardsForDeck(cards, deck.key);
        return accumulator;
      },
      {} as Record<DeckKey, BoardCard[]>
    );
  }, [cards]);

  function updateLocalCards(nextCards: BoardCard[]) {
    const sorted = sortCards(nextCards);
    setCards(sorted);
    saveLocalCards(sorted);
  }

  async function runRemoteMutation<T>(action: () => Promise<T>, successMessage: string) {
    startSavingTransition(() => {
      setStatusMessage("Saving changes...");
    });

    try {
      await action();
      await syncRemoteBoard();
      setStatusMessage(successMessage);
    } catch (error) {
      await syncRemoteBoard().catch(() => undefined);
      setStatusMessage(error instanceof Error ? error.message : "Something went wrong.");
    }
  }

  function setDraft(deck: DeckKey, field: "title" | "notes", value: string) {
    setDrafts((current) => ({
      ...current,
      [deck]: {
        ...current[deck],
        [field]: value
      }
    }));
  }

  async function handleCreate(deck: DeckKey, event: FormEvent) {
    event.preventDefault();
    const draft = drafts[deck];
    const title = draft.title.trim();
    const notes = draft.notes.trim();

    if (!title) {
      setStatusMessage("A card needs a title before it can land on the board.");
      return;
    }

    const newCard = createLocalCard({
      deck,
      title,
      notes,
      position: computeInsertedPosition(deckMap[deck], deckMap[deck].length)
    });

    setDrafts((current) => ({
      ...current,
      [deck]: { title: "", notes: "" }
    }));

    if (storageMode === "local") {
      updateLocalCards([...cards, newCard]);
      setStatusMessage(`Added a card to ${DECKS.find((entry) => entry.key === deck)?.label}.`);
      return;
    }

    setCards((current) => sortCards([...current, newCard]));

    await runRemoteMutation(async () => {
      const response = await fetch("/api/cards", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: newCard.id,
          deck,
          title,
          notes,
          position: newCard.position
        })
      });

      if (!response.ok) {
        throw new Error("Could not create that card.");
      }
    }, `Added a card to ${DECKS.find((entry) => entry.key === deck)?.label}.`);
  }

  async function saveCard(cardId: string, updates: Partial<Pick<BoardCard, "title" | "notes" | "deck" | "position">>, successMessage: string) {
    if (storageMode === "local") {
      updateLocalCards(
        cards.map((card) =>
          card.id === cardId
            ? {
                ...card,
                ...updates,
                updatedAt: new Date().toISOString()
              }
            : card
        )
      );
      setStatusMessage(successMessage);
      return;
    }

    setCards((current) =>
      sortCards(
        current.map((card) =>
          card.id === cardId
            ? {
                ...card,
                ...updates,
                updatedAt: new Date().toISOString()
              }
            : card
        )
      )
    );

    await runRemoteMutation(async () => {
      const response = await fetch(`/api/cards/${cardId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(updates)
      });

      if (!response.ok) {
        throw new Error("Could not save that card.");
      }
    }, successMessage);
  }

  async function handleDelete(cardId: string) {
    if (storageMode === "local") {
      updateLocalCards(cards.filter((card) => card.id !== cardId));
      setEditing(null);
      setStatusMessage("Card removed.");
      return;
    }

    setCards((current) => current.filter((card) => card.id !== cardId));
    setEditing(null);

    await runRemoteMutation(async () => {
      const response = await fetch(`/api/cards/${cardId}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        throw new Error("Could not delete that card.");
      }
    }, "Card removed.");
  }

  async function moveCard(cardId: string, targetDeck: DeckKey, insertIndex: number) {
    const movingCard = cards.find((card) => card.id === cardId);
    if (!movingCard) {
      return;
    }

    const destinationCards = cardsForDeck(
      cards.filter((card) => card.id !== cardId || card.deck !== targetDeck),
      targetDeck
    );
    const nextPosition = computeInsertedPosition(destinationCards, insertIndex);

    await saveCard(
      cardId,
      {
        deck: targetDeck,
        position: nextPosition
      },
      `Moved card to ${DECKS.find((deck) => deck.key === targetDeck)?.label}.`
    );
  }

  return (
    <main className="page-shell">
      <div className="page-frame">
        <section className="hero">
          <p className="eyebrow">YQuantum shared workflow</p>
          <h1>Five decks, one board, and zero deploy-related data wipes.</h1>
          <p className="hero-copy">
            This board is set up for `Team`, `Kai`, `Cayman`, `Peyton`, and `Will`. In shared mode it keeps card
            data outside the codebase, so updating the site won&apos;t erase the actual work.
          </p>
          <div className="hero-meta">
            <span className="pill pill-strong">{storageMode === "remote" ? "Shared database mode" : "Local browser mode"}</span>
            <span className="pill">{lastSyncedText}</span>
            <span className="pill">{isSaving ? "Saving..." : "Ready"}</span>
          </div>
        </section>

        <section className="boards-wrap">
          <div className="boards-grid">
            {DECKS.map((deck) => {
              const deckCards = deckMap[deck.key];
              return (
                <section className="deck" data-color={deck.key} key={deck.key}>
                  <header className="deck-head">
                    <div>
                      <h2 className="deck-title">{deck.label}</h2>
                      <p className="deck-subtitle">{deck.subtitle}</p>
                    </div>
                    <div className="count-badge">{deckCards.length}</div>
                  </header>

                  <form className="composer" onSubmit={(event) => void handleCreate(deck.key, event)}>
                    <input
                      className="input"
                      value={drafts[deck.key].title}
                      onChange={(event) => setDraft(deck.key, "title", event.target.value)}
                      placeholder={`Add a card to ${deck.label}`}
                      maxLength={120}
                    />
                    <textarea
                      className="textarea"
                      value={drafts[deck.key].notes}
                      onChange={(event) => setDraft(deck.key, "notes", event.target.value)}
                      placeholder="Notes, links, context, next step..."
                    />
                    <div className="composer-row">
                      <button className="button button-primary" type="submit">
                        Add card
                      </button>
                    </div>
                  </form>

                  <div className="cards">
                    <div
                      className="drop-zone"
                      data-active={activeDrop === `${deck.key}:0`}
                      onDragOver={(event) => {
                        event.preventDefault();
                        setActiveDrop(`${deck.key}:0`);
                      }}
                      onDragLeave={() => setActiveDrop((current) => (current === `${deck.key}:0` ? null : current))}
                      onDrop={(event) => {
                        event.preventDefault();
                        if (dragState) {
                          void moveCard(dragState.cardId, deck.key, 0);
                        }
                        setDragState(null);
                        setActiveDrop(null);
                      }}
                    />

                    {deckCards.map((card, index) => (
                      <div key={card.id}>
                        <article
                          className="card"
                          data-dragging={dragState?.cardId === card.id}
                          draggable
                          onDragStart={() => {
                            setDragState({ cardId: card.id, deck: card.deck });
                          }}
                          onDragEnd={() => {
                            setDragState(null);
                            setActiveDrop(null);
                          }}
                        >
                          <h3 className="card-title">{card.title}</h3>
                          {card.notes ? <p className="card-notes">{card.notes}</p> : null}
                          <div className="card-row">
                            <button
                              className="mini-button"
                              type="button"
                              onClick={() =>
                                setEditing({
                                  id: card.id,
                                  title: card.title,
                                  notes: card.notes,
                                  deck: card.deck
                                })
                              }
                            >
                              Edit
                            </button>
                            <button
                              className="mini-button"
                              type="button"
                              onClick={() => {
                                const currentIndex = deckCards.findIndex((entry) => entry.id === card.id);
                                if (currentIndex > 0) {
                                  void moveCard(card.id, deck.key, currentIndex - 1);
                                }
                              }}
                            >
                              Up
                            </button>
                            <button
                              className="mini-button"
                              type="button"
                              onClick={() => {
                                const currentIndex = deckCards.findIndex((entry) => entry.id === card.id);
                                if (currentIndex < deckCards.length - 1) {
                                  void moveCard(card.id, deck.key, currentIndex + 2);
                                }
                              }}
                            >
                              Down
                            </button>
                          </div>
                        </article>

                        <div
                          className="drop-zone"
                          data-active={activeDrop === `${deck.key}:${index + 1}`}
                          onDragOver={(event) => {
                            event.preventDefault();
                            setActiveDrop(`${deck.key}:${index + 1}`);
                          }}
                          onDragLeave={() =>
                            setActiveDrop((current) => (current === `${deck.key}:${index + 1}` ? null : current))
                          }
                          onDrop={(event) => {
                            event.preventDefault();
                            if (dragState) {
                              void moveCard(dragState.cardId, deck.key, index + 1);
                            }
                            setDragState(null);
                            setActiveDrop(null);
                          }}
                        />
                      </div>
                    ))}

                    {deckCards.length === 0 ? (
                      <div className="deck-empty">
                        Drag cards here or add the first one above.
                        <br />
                        The deck will stay in place even when the board is empty.
                      </div>
                    ) : null}
                  </div>
                </section>
              );
            })}
          </div>
        </section>

        <section className="status">
          <p className="status-note">
            <strong>Status:</strong> {statusMessage}
            {storageMode === "local" ? " Add a database URL when you're ready for shared edits across laptops." : null}
          </p>
        </section>
      </div>

      {editing ? (
        <div className="overlay" role="presentation" onClick={() => setEditing(null)}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <h2>Edit card</h2>
            <div className="modal-grid">
              <input
                className="input"
                value={editing.title}
                onChange={(event) =>
                  setEditing((current) => (current ? { ...current, title: event.target.value } : current))
                }
                placeholder="Card title"
                maxLength={120}
              />
              <textarea
                className="textarea"
                value={editing.notes}
                onChange={(event) =>
                  setEditing((current) => (current ? { ...current, notes: event.target.value } : current))
                }
                placeholder="Notes"
              />
              <select
                className="select"
                value={editing.deck}
                onChange={(event) => {
                  const nextDeck = event.target.value;
                  if (isDeckKey(nextDeck)) {
                    setEditing((current) => (current ? { ...current, deck: nextDeck } : current));
                  }
                }}
              >
                {DECKS.map((deck) => (
                  <option key={deck.key} value={deck.key}>
                    {deck.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="modal-actions">
              <button className="button button-alert" type="button" onClick={() => void handleDelete(editing.id)}>
                Delete card
              </button>
              <div className="modal-actions-right">
                <button className="button button-soft" type="button" onClick={() => setEditing(null)}>
                  Close
                </button>
                <button
                  className="button button-primary"
                  type="button"
                  onClick={async () => {
                    const nextDeckCards = cardsForDeck(
                      cards.filter((card) => card.id !== editing.id || card.deck !== editing.deck),
                      editing.deck
                    );

                    await saveCard(
                      editing.id,
                      {
                        title: editing.title.trim() || "Untitled card",
                        notes: editing.notes.trim(),
                        deck: editing.deck,
                        position: computeInsertedPosition(nextDeckCards, nextDeckCards.length)
                      },
                      "Card updated."
                    );
                    setEditing(null);
                  }}
                >
                  Save changes
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
