"use client";

import { useEffect, useMemo, useState } from "react";
import { CollaborationStrip } from "@/components/collaboration-strip";
import { PlotLightbox, type DashboardPlot } from "@/components/qaoa-plots-tab";
import { Yqh26DataTab } from "@/components/yqh26-data-tab";
import { HeuristicsPlotsTab } from "@/components/heuristics-plots-tab";
import { QuboVisLightbox } from "@/components/qubo-vis-tab";
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
  const [showQuboVis, setShowQuboVis] = useState(false);

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
          quote: "Quantum advantage in optimization"
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
              <h1>QGars x Travelers 2026</h1>
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
            <button
              type="button"
              className="top-tab"
              onClick={() => setShowQuboVis(true)}
            >
              QUBO Visualizer
            </button>
          </nav>
        </header>

        {activeView === "home" ? (
          <section className="home-overview card-surface" style={{ padding: '3rem', margin: '2rem auto', lineHeight: '1.7', fontSize: '1.1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
              <div>
                <h2 style={{ fontSize: '2rem', marginBottom: '0.5rem', color: 'var(--yale-blue)' }}>BYU QGars - YQuantum 2026</h2>
                <p style={{ fontSize: '1.2rem', color: 'var(--text-muted)', margin: 0 }}><strong>Travelers &times; Quantinuum &times; LTM</strong></p>
              </div>
              <a href="/YQUANTUM_2026.pdf" target="_blank" rel="noopener noreferrer" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none', color: 'var(--travelers-red)', fontWeight: 600, fontSize: '1.1rem', transition: 'opacity 0.2s' }} onMouseOver={(e) => e.currentTarget.style.opacity = '0.8'} onMouseOut={(e) => e.currentTarget.style.opacity = '1'}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                Read Full Report
              </a>
            </div>

            <p>We had an amazing time learning to apply two quantum algorithms to real insurance problems. This is our collective gained knowledge over the last 24 hours.</p>

            <h3 style={{ fontSize: '1.5rem', marginTop: '2.5rem', marginBottom: '1rem', color: 'var(--travelers-red)' }}>Integer Linear Programming in Insurance</h3>
            
            <p><strong>Marbles.</strong> Imagine many marbles of different colors. You are arranging them into gift bags; each recipient needs a blue or red marble, and optionally some other colors as well.</p>
            
            <p>This analogy helped us understand, to some degree, the confusing world of insurance. From the company's perspective, they have several coverages (marbles) of several families (colors) to distribute into discounted packages. Abiding by some rules of mixing coverages into bundles, the company aims to maximize profit by choosing which packages to offer to their audience. This can be formulated mathematically as follows.</p>
            
            <p>Let <em>n</em> and <em>m</em> be indices labeling coverages and packages. Let <em>M<sub>0</sub></em> be an <em>n &times; m</em> matrix, where <em>M<sub>ij</sub> &isin; {"{0,1}"}</em> indicates whether coverage <em>i</em> is in bundle <em>j</em>. The goal is to find an <em>M<sub>0</sub></em> that maximizes profit subject to some restraints; that is, minimize</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              -&sum;<sub>i,j</sub> (M &times; C)<sub>ij</sub>
            </div>
            
            <p>subject to certain conditions on <em>M<sub>ij</sub></em>, where <em>C<sub>ij</sub></em> encodes the average revenue from the <em>i</em>th coverage being in the <em>j</em>th bundle. This is a textbook ILP, or integer linear programming method. This turns out to be NP-hard, as the coefficients need to be 0 or 1. The proceeding quantum algorithms aim to solve this combinatorial problem with the high dimensionality of quantum computing.</p>

            <h3 style={{ fontSize: '1.5rem', marginTop: '2.5rem', marginBottom: '1rem', color: 'var(--travelers-red)' }}>Quantum Approximate Optimization Algorithm</h3>
            
            <p>QAOA is a sneaky beast. We spent the first 9 hours understanding how the algorithm worked. The idea is simple enough: construct a matrix whose eigenvalues are the objective values of the feasible set and eigenvectors corresponding to the lowest one. We begin by encoding the ILP constraints into the objective function by introducing a large scalar penalty <em>&lambda;</em>. The domain of the objective for the quantum computer is all points in</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              {"{0,1}"}<sup>&otimes; n</sup>,
            </div>
            
            <p>many of which are not feasible. By using nonnegative slack variables, we can ensure that the objective function does not converge to a false minimum by adding a punishment proportional to <em>&lambda;</em> for such points.</p>
            
            <p>Mathematically,</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              f(y) = -&sum; c<sub>i</sub> M<sub>i</sub> + &lambda; F(x),
            </div>
            
            <p>where <em>F(x)=0</em> represents the feasible set. Expanding,</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              f(y) = &sum;<sub>i</sub> Q<sub>ii</sub> y<sub>i</sub> + &sum;<sub>i&lt;j</sub> Q<sub>ij</sub> y<sub>i</sub> y<sub>j</sub>
            </div>
            
            <p>up to a constant. This form is useful after making the substitution</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              y<sub>i</sub> = (I - Z<sub>i</sub>) / 2.
            </div>
            
            <p>Associating <em>y<sub>i</sub> y<sub>j</sub></em> with <em>ZZ</em> and <em>y<sub>i</sub></em> with <em>Z</em> gates, the coefficients <em>Q</em> of this objective function are used to construct precisely the matrix whose eigenvalues are the objective values of the feasible points. This matrix is called a cost Hamiltonian <em>H<sub>c</sub></em>, though seemingly, the term's only relation to energy is the objective to minimize.</p>
            
            <p>To implement <em>H<sub>c</sub></em> on a circuit, we use the unitary matrix</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              U(&gamma;) = e<sup>i&gamma; H<sub>c</sub></sup>,
            </div>
            
            <p>as <em>H<sub>c</sub></em> is not necessarily Hermitian. After applying on an equal superposition of states, the parameter <em>&gamma;</em> determines the magnitude of the phase picked up by the basis eigenstates; indeed,</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              U(&gamma;)|E&rang; = e<sup>i&gamma; E</sup>|E&rang;.
            </div>
            
            <p>Finally, this phase is converted into a magnitude after applying</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              U(&beta;) = e<sup>i&beta; &sum;<sub>i</sub> X<sub>i</sub></sup>.
            </div>
            
            <p>To see this concretely, consider</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              1/&radic;2 (|0&rang; + |1&rang;),
            </div>
            
            <p>where <em>|0&rang;</em> and <em>|1&rang;</em> have corresponding eigenvalues, or equivalently, objective function values. Applying <em>U(&gamma;)</em> gives a relative phase:</p>
            
            <div style={{ textAlign: 'center', margin: '1.5rem 0', fontSize: '1.3em', fontFamily: 'serif' }}>
              1/&radic;2 (e<sup>i&gamma; E<sub>0</sub></sup>|0&rang; + e<sup>i&gamma; E<sub>1</sub></sup>|1&rang;).
            </div>
            
            <p>This relative phase is what allows the amplitudes to change after applying an <em>X</em>-based mixing gate. A weakness in QAOA is periodicity; <em>E<sub>0</sub></em> and <em>E' = E<sub>0</sub> + 2&pi;</em> apply the same phase though their objective values differ. This can be addressed by applying <em>U(&gamma;)</em> and <em>U(&beta;)</em> successively.</p>

            <h3 style={{ fontSize: '1.5rem', marginTop: '2.5rem', marginBottom: '1rem', color: 'var(--travelers-red)' }}>DQI</h3>
            
            <p>Decoded quantum interferometry is a more recent method to minimize an objective function. Constraints are encoded into a collection of XOR operations and implemented through phase and CNOT gates. The most difficult part is finding the matrix <em>B</em>, which contains information on the XOR-translated constraints. Our DQI implementation worked, but took too long for our laptops (or even Selene) to run in a reasonable amount of time. Instead, we implemented a DQI-inspired QAOA approach, taking ideas from parity and syndrome qubits.</p>
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
      <QuboVisLightbox isOpen={showQuboVis} onClose={() => setShowQuboVis(false)} />
    </main>
  );
}
