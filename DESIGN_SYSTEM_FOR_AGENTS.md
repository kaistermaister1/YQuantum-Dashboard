# |Y>Quantum 2026 — design system brief (for AI agents & humans)

Use this file so websites, dashboards, notebooks, and internal tools **visually align** with the challenge materials, especially **`YQH26.pdf`** (Yale Quantum Hackathon 2026 / P&C bundling problem statement).

**In this repo (YQuantum-Dashboard):** This file is at the project root. Paths like `Travelers/YQH26.pdf` and `Travelers/docs/*.html` point to the **official challenge materials** repository. Use a local clone of that repo for side-by-side visual reference; **tokens, layout rules, and the paste-ready prompt below are self-contained** and do not require those files to be present here.

## Canonical references (implementations to mimic)

1. **Primary:** Open `Travelers/YQH26.pdf` and match **density, hierarchy, and seriousness** of a technical internal report: clear section numbering, readable tables, restrained decoration, strong typographic hierarchy, minimal “startup” styling.
2. **Token + component reference (already in-repo):** The static pages under `Travelers/docs/` encode approved colors and patterns. Prefer copying their `:root` variables and layout idioms rather than inventing new palettes.
   - **Document / tool UI (light):** `Travelers/docs/01_insurance_bundling.html` — header bar, grid layout, cards, family badges.
   - **Dashboard / charts:** `Travelers/docs/06_scaling.html` — section labels, assumption boxes, chart legends.
   - **Full-screen dark deck:** `Travelers/docs/index.html` / `00_kickoff_slides.html` — only when the product is explicitly presentation-style.

**Rule of thumb:** If it is a **dashboard, app, or report viewer**, use the **light** pattern (gray background, charcoal text, red/Yale/blue accents). Reserve the **charcoal full-bleed slide** look for **presentations**.

---

## Design tokens (copy into CSS or design tools)

These values are consolidated from `Travelers/docs/*.html`. Treat them as **defaults** unless sponsor materials specify otherwise.

| Token | Hex | Role |
|--------|-----|------|
| `--accent-red` | `#E31837` | Primary brand accent (Travelers red): CTAs, key metrics, section labels, borders |
| `--accent-dark` | `#B8132D` | Hover / pressed states for red controls |
| `--accent-red-light` | `#f4a0ad` | Rare: soft highlights only |
| `--accent-red-bg` | `#fdf0f2` | Subtle callout / assumption panel background |
| `--platform-blue` | `#0066CC` | **QAOA** / “platform” association in diagrams and legends |
| `--yale-blue` | `#00356B` | **DQI** / Yale association; deep blue in gradients and text |
| `--charcoal` | `#2D2926` | Primary body text and dark header gradients |
| `--light-gray` | `#F5F5F5` | Page / app background |
| `--white` | `#FFFFFF` | Cards, panels, table backgrounds |
| `--green` / `--classical-green` | `#2D8C3C` | Classical / baseline / “success” contrast in charts (when needed) |
| `--border` | `#dddddd` | Card and table borders |
| Shadow (soft) | `rgba(0,0,0,0.08)` | Card elevation |

**Optional categorical colors** (coverage families, multi-series charts only — from `01_insurance_bundling.html`): use the defined optional-family palette there; do **not** introduce rainbow gradients unrelated to data categories.

### CSS `:root` starter (drop-in)

```css
:root {
  --accent-red: #E31837;
  --accent-dark: #B8132D;
  --accent-red-light: #f4a0ad;
  --accent-red-bg: #fdf0f2;
  --platform-blue: #0066CC;
  --yale-blue: #00356B;
  --charcoal: #2D2926;
  --light-gray: #F5F5F5;
  --white: #FFFFFF;
  --green: #2D8C3C;
  --border: #dddddd;
  --shadow-soft: rgba(0, 0, 0, 0.08);
}
```

---

## Typography

- **Font stack:** `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif` (matches docs).
- **Headings:** Tight letter-spacing on large titles (`letter-spacing: -0.5px` for main H1). Weights **600–800** for hierarchy; avoid ultra-light weights.
- **Section labels:** Uppercase, **11–14px**, bold, **wide letter-spacing** (~1–1.5px), color **`--accent-red`** or charcoal depending on background.
- **Body:** **14–16px** for UI; line-height ~1.5. Prefer **left-aligned** blocks for long technical copy (like the PDF), not centered walls of text.

---

## Layout & surfaces

### App / dashboard shell (light)

- **Page background:** `--light-gray`; optional subtle grid (see `01_insurance_bundling.html` background-image pattern) — very low contrast only.
- **Top header:** Horizontal gradient `charcoal → yale-blue` (135deg), white text, **4px bottom accent**: gradient strip `red → platform-blue → yale-blue` *or* solid red bar (both appear in-repo — pick one per product and stay consistent).
- **Content:** Max-width ~`1920px` centered; generous horizontal padding (`20–40px`).
- **Cards:** White background, `1px solid var(--border)`, `border-radius: 6–8px`, left accent border optional for emphasis; hover = slight shadow + small translate (see coverage cards).

### Tables

- Clear header row: dark or red-accent header text; zebra or row hover optional; **borders** visible (report-like), not borderless “spreadsheet glam.”
- Align numbers right; align text left; keep **compact** row height for data-heavy views (PDF-like).

### Charts

- Legend colors: **QAOA = platform blue**, **DQI = Yale blue**, **classical = green** when comparing approaches (see `06_scaling.html`).
- Prefer **restrained** color count; background remains light gray or white.

---

## Content & naming

- **Hackathon name:** `|Y>Quantum 2026` (ket notation) where appropriate.
- **Challenge title:** “P&C Insurance Product Bundling Optimization” (or shortened consistently in UI).
- **Sponsors:** Travelers, Quantinuum, LTM — use sponsor-provided logos when available; do not redraw trademarks.

---

## Do / don’t

**Do**

- Reuse the token table above for all new UI.
- Match the **professional report** tone of `YQH26.pdf`: structure, tables, and clarity over decoration.
- Keep **algorithm colors consistent** across charts, badges, and docs.

**Don’t**

- Introduce unrelated brand palettes (neon cyber, purple gradients, glassmorphism-heavy marketing sites).
- Use red for large background fills except small banners, labels, or primary buttons.
- Center long problem descriptions; the PDF is document-style.

---

## Paste-ready system prompt (for Cursor / other agents)

Copy everything inside the block below into a project rule, agent instruction, or chat system context when starting UI work:

```
You are implementing UI for the |Y>Quantum 2026 hackathon (Travelers + Quantinuum + LTM).
Visual target: match the technical internal-report aesthetic of Travelers/YQH26.pdf — clear hierarchy, compact tables, restrained color, no flashy marketing styling.

Mandatory design tokens (use as CSS variables or exact hex):
- accent red #E31837, accent dark #B8132D, platform blue #0066CC (QAOA), Yale blue #00356B (DQI), charcoal #2D2926, light gray #F5F5F5, white #FFFFFF, classical green #2D8C3C, border #dddddd.

Layout defaults for apps/dashboards (not slide decks):
- Background: light gray; text: charcoal.
- Top header: gradient charcoal to Yale blue, white type, thin accent strip using red/blue/Yale gradient OR solid red bottom border (pick one and keep consistent).
- Cards: white, 1px #ddd border, 6–8px radius, subtle shadow on hover.
- Typography: system UI stack; section labels uppercase with letter-spacing, often in accent red.
- Charts: QAOA=blue #0066CC, DQI=Yale #00356B, classical=green #2D8C3C when comparing methods.

Before finishing, mentally diff your result against Travelers/docs/01_insurance_bundling.html and 06_scaling.html — same palette and seriousness. Do not invent a new theme.
```

---

## Maintenance

When sponsor PDFs or slides update, **update this file** and, in the challenge-materials repo, the `:root` blocks in `Travelers/docs/*.html`, so agents and humans stay aligned.
