"use client";

import { useEffect, useMemo, useState } from "react";
import { CollaborationStrip } from "@/components/collaboration-strip";
import { PlotLightbox, type DashboardPlot } from "@/components/qaoa-plots-tab";
import { Yqh26DataTab } from "@/components/yqh26-data-tab";
import { HeuristicsPlotsTab } from "@/components/heuristics-plots-tab";
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
import type { Yqh26DashboardData } from "@/lib/yqh26-data";

type StorageMode = "local" | "remote";
type ViewKey = "home" | "dataset" | "heuristics";

type BoardAppProps = {
  storageMode: StorageMode;
  dataset: Yqh26DashboardData;
};

const DECK_SUBTEXT: Record<DeckKey, string> = {
  team: "Shared Board",
  kai: "Researcher Est 2012",
  cayman: "Asian Workhorse",
  peyton: "Master Qiskit",
  will: "Abstract Thinker"
};

function DeckGlyph({ deck }: { deck: DeckKey }) {
  switch (deck) {
    case "team":
      return (
        <svg aria-hidden="true" className="deck-glyph" viewBox="0 0 24 24">
          <rect x="4" y="5" width="6" height="6" rx="1.5" />
          <rect x="14" y="5" width="6" height="6" rx="1.5" />
          <rect x="4" y="13" width="6" height="6" rx="1.5" />
          <rect x="14" y="13" width="6" height="6" rx="1.5" />
        </svg>
      );
    case "kai":
      return (
        <svg aria-hidden="true" className="deck-glyph" viewBox="0 0 24 24">
          <circle cx="10.5" cy="10.5" r="5.5" />
          <path d="M15 15l4 4" />
        </svg>
      );
    case "cayman":
      return (
        <svg aria-hidden="true" className="deck-glyph" viewBox="0 0 24 24">
          <path d="M13 3L6 13h5l-1 8 8-11h-5l0-7z" />
        </svg>
      );
    case "peyton":
      return (
        <svg aria-hidden="true" className="deck-glyph" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="2.1" />
          <ellipse cx="12" cy="12" rx="8" ry="3.5" />
          <ellipse cx="12" cy="12" rx="8" ry="3.5" transform="rotate(60 12 12)" />
          <ellipse cx="12" cy="12" rx="8" ry="3.5" transform="rotate(120 12 12)" />
        </svg>
      );
    case "will":
      return (
        <svg aria-hidden="true" className="deck-glyph" viewBox="0 0 24 24">
          <path d="M12 4l7 4v8l-7 4-7-4V8l7-4z" />
          <path d="M5 8l7 4 7-4" />
          <path d="M12 12v8" />
        </svg>
      );
  }
}

type CardModalState =
  | {
      mode: "create";
      deck: DeckKey;
      title: string;
    }
  | {
      mode: "edit";
      deck: DeckKey;
      id: string;
      title: string;
    };

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

export function BoardApp({ storageMode, dataset }: BoardAppProps) {
  const [cards, setCards] = useState<BoardCard[]>([]);
  const [cardModal, setCardModal] = useState<CardModalState | null>(null);
  const [dragState, setDragState] = useState<{ cardId: string; deck: DeckKey } | null>(null);
  const [activeDrop, setActiveDrop] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewKey>("home");
  const [activePlot, setActivePlot] = useState<DashboardPlot | null>(null);

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
      setCards(loadLocalCards());

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
        }
      } catch {
        if (!cancelled) {
          setCards([]);
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

  async function runRemoteMutation(action: () => Promise<void>) {
    try {
      await action();
      await syncRemoteBoard();
    } catch {
      await syncRemoteBoard().catch(() => undefined);
    }
  }

  async function createCardInDeck(deck: DeckKey, title: string) {
    const nextTitle = title.trim();

    if (!nextTitle) {
      return;
    }

    const newCard = createLocalCard({
      deck,
      title: nextTitle,
      notes: "",
      position: computeInsertedPosition(deckMap[deck], deckMap[deck].length)
    });

    if (storageMode === "local") {
      updateLocalCards([...cards, newCard]);
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
          title: nextTitle,
          notes: "",
          position: newCard.position
        })
      });

      if (!response.ok) {
        throw new Error("Could not create that card.");
      }
    });
  }

  async function saveCard(cardId: string, updates: Partial<Pick<BoardCard, "title" | "deck" | "position">>) {
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
    });
  }

  async function deleteCardById(cardId: string) {
    if (storageMode === "local") {
      updateLocalCards(cards.filter((card) => card.id !== cardId));
      return;
    }

    setCards((current) => current.filter((card) => card.id !== cardId));

    await runRemoteMutation(async () => {
      const response = await fetch(`/api/cards/${cardId}`, {
        method: "DELETE"
      });

      if (!response.ok) {
        throw new Error("Could not delete that card.");
      }
    });
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

    await saveCard(cardId, {
      deck: targetDeck,
      position: nextPosition
    });
  }

  const headerCopy =
    activeView === "home"
      ? {
          kicker: "Q-gars Workspace",
          quote:
            "Track team work across the board, then switch tabs to inspect the full YQH26 insurance bundling instance or QAOA result figures."
        }
      : activeView === "dataset"
        ? {
            kicker: "Travelers Challenge Summary",
            quote: "Review the Travelers Challenge dataset summary, package structure, and family coverage details."
          }
        : activeView === "heuristics"
          ? {
              kicker: "Heuristics Scaling Analysis",
              quote: "Analyze solution quality, workflow cost, and constraint handling across problem sizes."
            }
          : {
              kicker: "",
              quote: ""
            };

  return (
    <main className="page-shell">
      <div className="page-frame">
        <header className="page-header">
          <div className="page-header-layout">
            <div className="page-header-copy">
              <p className="page-kicker">{headerCopy.kicker}</p>
              <h1>QGars x Travelers 2026 - quantum advantage in optimization</h1>
              <p className="page-quote">{headerCopy.quote}</p>
            </div>

            <div className="page-header-brand">
              <span className="page-header-brand-label">In collaboration with</span>
              <CollaborationStrip />
            </div>
          </div>

          <nav className="top-tabs" aria-label="Workspace sections">
            <button
              type="button"
              className={`top-tab ${activeView === "home" ? "top-tab-active" : ""}`}
              aria-current={activeView === "home" ? "page" : undefined}
              onClick={() => setActiveView("home")}
            >
              Home
            </button>
            <button
              type="button"
              className={`top-tab ${activeView === "dataset" ? "top-tab-active" : ""}`}
              aria-current={activeView === "dataset" ? "page" : undefined}
              onClick={() => setActiveView("dataset")}
            >
              YQH26 Data
            </button>
            <button
              type="button"
              className={`top-tab ${activeView === "heuristics" ? "top-tab-active" : ""}`}
              aria-current={activeView === "heuristics" ? "page" : undefined}
              onClick={() => setActiveView("heuristics")}
            >
              Heuristics Plots
            </button>
          </nav>
        </header>

        {activeView === "home" ? (
          <section className="boards-wrap">
            <div className="boards-grid">
              {DECKS.map((deck) => {
                const deckCards = deckMap[deck.key];

                return (
                  <section className="deck" data-color={deck.key} key={deck.key}>
                    <header className="deck-head">
                      <div>
                        <h2 className="deck-title">{deck.label}</h2>
                        <p className="deck-subtitle">
                          <DeckGlyph deck={deck.key} />
                          <span>{DECK_SUBTEXT[deck.key]}</span>
                        </p>
                      </div>
                      <div className="count-badge">{deckCards.length}</div>
                    </header>

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
                            onClick={() =>
                              setCardModal({
                                mode: "edit",
                                deck: card.deck,
                                id: card.id,
                                title: card.title
                              })
                            }
                            onDragStart={() => {
                              setDragState({ cardId: card.id, deck: card.deck });
                            }}
                            onDragEnd={() => {
                              setDragState(null);
                              setActiveDrop(null);
                            }}
                          >
                            <h3 className="card-title">{card.title}</h3>
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
                    </div>

                    <button
                      className="add-button"
                      type="button"
                      onClick={() =>
                        setCardModal({
                          mode: "create",
                          deck: deck.key,
                          title: ""
                        })
                      }
                    >
                      <span className="add-button-icon" aria-hidden="true">
                        +
                      </span>
                      <span>Add card</span>
                    </button>
                  </section>
                );
              })}
            </div>
          </section>
        ) : activeView === "dataset" ? (
          <div className="dataset-wrap">
            <Yqh26DataTab dataset={dataset} />
          </div>
        ) : activeView === "heuristics" ? (
          <div className="dataset-wrap plots-wrap">
            <HeuristicsPlotsTab onOpenPlot={setActivePlot} />
          </div>
        ) : null}
      </div>

      <PlotLightbox plot={activePlot} onClose={() => setActivePlot(null)} />

      {activeView === "home" && cardModal ? (
        <div className="overlay" role="presentation" onClick={() => setCardModal(null)}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
            <h2>{cardModal.mode === "create" ? "Add card" : "Edit card"}</h2>
            <input
              autoFocus
              className="input"
              value={cardModal.title}
              onChange={(event) =>
                setCardModal((current) => (current ? { ...current, title: event.target.value } : current))
              }
              placeholder="Task title"
              maxLength={120}
              onKeyDown={(event) => {
                if (event.key !== "Enter" || !cardModal.title.trim()) {
                  return;
                }

                event.preventDefault();

                if (cardModal.mode === "create") {
                  void createCardInDeck(cardModal.deck, cardModal.title);
                } else {
                  void saveCard(cardModal.id, { title: cardModal.title.trim() });
                }

                setCardModal(null);
              }}
            />

            <div className="modal-actions">
              {cardModal.mode === "edit" ? (
                <button
                  className="button button-alert"
                  type="button"
                  onClick={async () => {
                    await deleteCardById(cardModal.id);
                    setCardModal(null);
                  }}
                >
                  Delete
                </button>
              ) : (
                <span />
              )}

              <div className="modal-actions-right">
                <button className="button button-soft" type="button" onClick={() => setCardModal(null)}>
                  Cancel
                </button>
                <button
                  className="button button-primary"
                  type="button"
                  onClick={async () => {
                    if (!cardModal.title.trim()) {
                      return;
                    }

                    if (cardModal.mode === "create") {
                      await createCardInDeck(cardModal.deck, cardModal.title);
                    } else {
                      await saveCard(cardModal.id, { title: cardModal.title.trim() });
                    }

                    setCardModal(null);
                  }}
                >
                  {cardModal.mode === "create" ? "Add" : "Save"}
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
