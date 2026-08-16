# CLAUDE.md — Global

Personal guidance applied to every Claude Code session across all projects.
Project-level `CLAUDE.md` files layer on top of this and may add specifics.

## Generated-document conventions

Any documentation, notes, plans, research artifacts, or runnable exploration
scripts that Claude produces follow a single naming scheme so output stays
consistent across projects and sessions.

### Folders

- One top-level folder per artifact *category*, named `Claude-<category>`.
- `<category>` is a short lowercase noun describing the kind of artifact.
  Common categories:
  - `Claude-docs/`           — durable codebase reference documentation
  - `Claude-research/`       — research proposals, brainstorms, literature surveys
  - `Claude-plans/`          — short-form implementation plans (single-doc)
  - `Claude-implementation/` — multi-doc implementation plans bundled per session (see *Dated session subfolders* below)
  - `Claude-reviews/`        — code reviews / audits
  - `Claude-unit-scripts/`   — runnable Jupyter notebooks that explore the codebase in isolation
- Add new categories the same way as needs arise.

### Files

- Name every file `NN-topic.ext`:
  - `NN` — two-digit zero-padded sequence number giving reading order.
  - `topic` — lowercase, words joined by hyphens (kebab-case).
  - `ext` —
    - `.md` for documents.
    - `.ipynb` for exploration artifacts in `Claude-unit-scripts/`
      (Jupyter notebooks let the user inspect variables and re-run cells in
      place — **prefer notebooks over plain `.py` scripts** here, even when
      the file is fully linear).
    - The artifact's own extension (`.py`, `.sh`, etc.) for any other
      runnable artifact.
- `00-index.md` is reserved as each folder's table of contents.
- Hyphens only — no underscores, no capitals, no spaces anywhere in a
  folder or file name.
- For journal-style folders (dated logs rather than an ordered set), use
  `YYYY-MM-DD-topic.md` instead of the `NN-` prefix. Pick one mode per
  folder; never mix the two.

### Iteration snapshots vs. living docs

Within a category folder, files come in two kinds:

- **Iteration snapshots** — numbered `NN-topic.md`. Each captures a discrete
  step in the work (a brainstorm, a design decision, a result writeup) and is
  treated as immutable once written; new thinking goes into a new numbered file.
- **Living docs** — prefixed `XX-` instead of a sequence number (e.g.
  `XX-research-summary.md`, `XX-status.md`). Updated continuously to reflect
  current state — running summaries, status boards, in-progress notes. They
  sort to the end alphabetically so the numbered iteration history stays at
  the top of `ls`.

Each `XX-` doc should end with an `## Update log` section where every
material change is recorded as a one-liner: `YYYY-MM-DD — <change>`.

### Dated session subfolders

For category folders where each *session* produces a bundle of related docs
that belong together (e.g. an implementation plan spanning overview, loss
formulation, file changes, and milestones), group the session inside a
timestamped subfolder rather than flattening everything into the category
root:

    Claude-<category>/
      YYYY-MM-DD-HHMM/
        00-index.md
        01-<topic>.md
        02-<topic>.md
        XX-<topic>.md          # living doc for this session, if any

Why subfolders rather than flat numbering at the category root:

- Bundles stay atomic — a new session never has to renumber an old one.
- The folder name is itself the timestamp, so chronological order is obvious.
- Revising the same task later (e.g. a second-pass plan) lives in a new
  subfolder, leaving the original snapshot intact.

Inside the subfolder use the standard `NN-topic.md` / `XX-topic.md` rules.
Always include `HHMM` in the folder name (not just `YYYY-MM-DD`) so multiple
sessions on the same day disambiguate without judgment calls.

Skip dated subfolders when a category only ever produces a single ordered
stream of files (research arcs, doc sets) — flat numbering at the category
root is simpler there. The natural fit for this pattern is
`Claude-implementation/` (one subfolder per implementation plan) and
`Claude-reviews/` (one subfolder per review), but it applies to any category
whose unit of work is a session rather than a single doc.

### Math notation

Write math in LaTeX delimiters — never in backticks or unlabeled code fences —
so renderers (GitHub, VS Code preview, Obsidian, Add-Ons) display it as proper
math instead of monospace text.

- Inline math: `$...$` — e.g. `$K = 4$`, `$\alpha, \beta$`, `$o_T$`,
  `$\mathcal{L}_{\text{thought}}$`.
- Display equations: `$$...$$` on their own line(s).
- Inside a bulleted/numbered list, leave a blank line before a `$$...$$`
  block so the renderer treats it as display math rather than inlining it
  into the bullet.
- Keep actual code (Python, shell, pseudocode) in fenced blocks with a
  language tag. Math expressions inside backticks render as code text and
  Greek letters, sub/superscripts, and operators are lost.

### Notebook structure (for `.ipynb` artifacts)

Each generated notebook should have a consistent cell layout so the user
can step through it easily:

1. **First cell — markdown.** The notebook's docstring: title, one-line
   purpose, what it shows, any external dependencies (servers, API keys, GPU).
2. **Second cell — code.** All imports and setup needed for the rest of the
   notebook.
3. **Per section — alternating markdown + code cells:**
   - One markdown cell with a `## Section N: title` header.
   - One code cell containing only that section's body.

Keep each section's code cell focused on a single concept so the user can
run, edit, and re-run any one cell without rerunning the whole notebook.

### Example

    Claude-research/
      00-index.md
      01-literature-survey.md
      02-action-distillation-objective.md
      03-sod-comparison.md
      XX-research-summary.md

    Claude-implementation/
      2026-05-19-1758/
        00-index.md
        01-experiment-overview.md
        02-loss-formulation.md
        03-implementation-changes.md
        04-training-and-evaluation.md
        05-milestones-and-risks.md
      2026-06-02-0930/
        00-index.md
        01-...

    Claude-unit-scripts/
      00-index.md
      01-explore-tool-class.ipynb
      02-explore-agent-memory.ipynb
