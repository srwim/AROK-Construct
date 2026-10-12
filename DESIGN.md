# AROK Construct: design system

Applies to the HTML timeline (`construct view`) and anything else Construct
renders. Grounded in Robert's Design Library; reference ids are listed at the
end.

## Scene

An MSP engineer at a desk mid-afternoon, the timeline open beside their
editor on a 24-inch monitor, reading two branches of a run side by side to
find where an agent went wrong; later a compliance reviewer opens the same
file on a laptop, or prints it.

Daylight, long reading, paper as a likely destination: **light, warm sand is
the default**; dark (charcoal, cream ink) follows the system preference.

## Concept: contour

Like a topographic plate: dense, quiet lines for everything that matched,
and one burnt-orange trace marking where the runs part ways. Identical steps
merge into a single muted row; diverged steps split into columns with the
evidence of what changed directly beneath.

## Color (OKLCH, restrained)

Neutrals are tinted: ink toward forest (hue 150), grounds toward sand (hue 85).
One accent, burnt orange, used only for divergence and errors (under 10% of
any screen). Diff colors are functional, not decorative.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--ground` | `oklch(0.955 0.012 85)` | `oklch(0.205 0.01 150)` | page |
| `--raised` | `oklch(0.978 0.008 85)` | `oklch(0.24 0.012 150)` | expanded evidence |
| `--ink` | `oklch(0.25 0.02 150)` | `oklch(0.92 0.018 85)` | text |
| `--ink-2` | `oklch(0.47 0.015 150)` | `oklch(0.74 0.014 85)` | secondary text |
| `--ink-3` | `oklch(0.53 0.012 150)` | `oklch(0.62 0.012 85)` | captions, identical rows (4.5:1 on ground) |
| `--rule` | `oklch(0.86 0.014 85)` | `oklch(0.33 0.012 150)` | hairlines |
| `--accent` | `oklch(0.63 0.17 45)` | `oklch(0.72 0.15 52)` | divergence marks, spine |
| `--accent-ink` | `oklch(0.5 0.15 42)` | `oklch(0.8 0.12 55)` | accent text (AA on ground) |
| `--del` / `--del-bg` | `oklch(0.45 0.13 30)` / `oklch(0.91 0.045 30)` | `oklch(0.86 0.07 30)` / `oklch(0.33 0.06 30)` | removed |
| `--ins` / `--ins-bg` | `oklch(0.38 0.08 150)` / `oklch(0.91 0.045 150)` | `oklch(0.88 0.07 150)` / `oklch(0.32 0.05 150)` | added |

Diff color is never the only signal: removed text is struck through, and
diff lines keep their `-` / `+` prefixes.

## Type

System stacks only: the timeline must render offline and make no requests.

- **Interface and body:** `system-ui` (Segoe UI Variable on Windows, SF on
  macOS). 14px base, 1.5 line height, tabular numerals for step numbers.
- **Title:** the run names, in the system serif stack (`"Iowan Old Style",
  "Palatino Linotype", Palatino, Georgia, serif`), 28px, regular weight. The
  only serif on the page: editorial serif over light grotesk, kept to one line.
- **Technical:** `ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas,
  monospace` at 12px for hashes, JSON, and diffs. Never for prose.

Scale steps: 12 / 14 / 16 / 28 (no flat steps). Prose capped at 72ch.

## Layout

- A grid: a narrow step gutter, then one column per run. Max width 1200px,
  16px side gutter on phones.
- The gutter carries a 1px vertical spine (the "contour"): `--rule` through
  identical steps, `--accent` from the first divergence on. A small filled
  dot marks each diverged step. No colored side stripes on rows.
- Identical steps (same request and response in every run) render as one
  muted row spanning all run columns.
- Diverged steps render one cell per run, then a full-width "what changed"
  block: request diff, response word diff, tool-call diff (or effect args and
  result). After the first divergence, request diffs repeat the inherited
  cause, so they are folded behind a disclosure; responses stay open.
- Rows are separated by hairlines, not cards. Evidence opens inline with
  `<details>`; no modals.
- Under 720px, run cells stack within a step, each labeled with its run name.

## Components

- **Header:** run names as the title; one sentence stating the outcome ("Identical
  through step 0; diverges at step 1."), with a link to the first divergence.
  Beneath, one line of facts per run: mode, parent, model substitution, when.
- **Step cell:** kind and target ("LLM call /v1/chat/completions", "Effect
  search"), source ("live" or "replayed"), model when present, HTTP status
  when not 200 (in `--accent-ink`), a two-line response summary, and an
  "Evidence" disclosure with the exact request and response.
- **Tool calls in summaries:** `search(query: "volcanoes", limit: 3)`, not raw JSON.
- **Footer:** database path, Construct version, generation time. Provenance.

## States

- **One run:** no merging, no change blocks; header reads "N steps: L live, R replayed."
- **Empty run:** "This run recorded no steps."
- **Uneven lengths:** the missing cell reads "No step N in this run."
- **Three or more runs:** divergence is marked; change blocks appear only for two.
- **Non-JSON or error responses:** shown raw, status in accent ink.
- **Print:** all evidence expands, backgrounds drop, the accent spine stays.

## Bans (from impeccable, enforced here)

No side-stripe borders, gradient text, glassmorphism, hero metrics, identical
card grids, or modals. No em dashes in copy. No external fonts, scripts, or images.

## References

From the Design Library (`srwim/aperture-library`, `design/refs/<id>.jpg`):

- `66fe402684` Neue Montreal / RRA: warm grey-sand page on charcoal, strict grid, one vermilion accent.
- `4daa29cdfc` Contour Ridge, Orange Sun: dense muted lines, one orange trace marking the path.
- `6856321c36` Red Disc Over Cold Sea: a single hairline vertical rule organizing the frame.
- `f5ef479fa7` Nolan Ghost Print: mono only for tiny technical captions on a warm ground.
- `924e5864ec` 03 Functional Typography: quiet spacing, hierarchy by scale and weight.
