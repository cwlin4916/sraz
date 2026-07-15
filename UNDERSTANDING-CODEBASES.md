# Understanding a Codebase — Principles

*General method for comprehending an unfamiliar codebase, top-down. These
principles are independent of any particular project; a worked illustration
drawn from this repo is kept in an appendix at the end.*

## Principle 0 — top-down before bottom-up

Understand the **components and how they fit together** before diving into any
one component's implementation. Build the mental map first, so that every
implementation detail you read later has a place to hang. Reading internals
before you have the map is how you spend an hour on a function before realizing
it wasn't the one that mattered.

A corollary for *reading order*: start with the shortest "what is this and why"
overview, then the component map / architecture doc. That is usually enough to
understand a system in general. Go deeper into a specific component only when a
task gives you a concrete reason to.

## The core lens: data flow — and its limits

"What are the components, what is each one's input/output, and how does data
flow through?" is **the single highest-leverage lens**, because it forces you to
find the **seams** — the interfaces between components. Once you know a
component's I/O, you can reason about the whole system *without reading that
component's internals*. That is exactly the abstraction that lets you
legitimately defer implementation details.

Well-designed systems make this explicit: the interface/abstract-base-class
boundaries usually *are* the I/O contracts, and the rest of the system talks to
components only through them.

But data flow alone is incomplete. Treat it as **one of four complementary
views**.

## The four views

| View | Question it answers |
| --- | --- |
| **Data flow** | *What* moves between components |
| **Control flow** | *Who drives whom, in what order, how many times* |
| **State & ownership** | *What is mutated in place, who owns it, how long it lives* |
| **Contracts / invariants** | *What must be true at a boundary, beyond the types* |

The unifying idea: lead with data flow, but remember that **two of the other
three views do not live on the boundaries where data-flow thinking looks** —
control flow is system-wide orchestration, and state/ownership is often about
*internal* mutation. Bugs cluster in all three of the non-data views.

## Two refinements to the method

### Refinement 1 — trace one concrete value (a *method* point)

Do **not** merely catalog the data structures at each interface — that is still
a *static* inventory. Instead, **pick one concrete value and follow that single
instance through the entire path.**

- Static inventory: "X is an array of shape N that goes from component A to B."
- Dynamic trace: "*this* input becomes *these* intermediate values, which are
  combined into *this* output, which feeds *that* next stage."

The trace is what turns a list of classes into *roles in a story*. It is the
part most people skip, and it is where understanding solidifies.

### Refinement 2 — cover the non-data views (a *coverage* point)

Watching the boundaries captures only the **contracts** view (one of four). The
two you are most likely to miss entirely by thinking in data flow:

- **Control flow** — not a boundary detail at all. Timing, ordering, and the
  driving loop. *Data flow shows the pipe; control flow shows the pump.*
- **State & ownership** — often internal, not at a boundary. What is mutated,
  who owns it, its lifetime.

The boundary **invariants** are the third piece — real, but the smallest of the
three.

## Compact restatement

| | Not this | But this |
| --- | --- | --- |
| **Refinement 1** | Catalog the data structures at each interface (static) | Trace one concrete value through the whole flow (dynamic) |
| **Refinement 2** | Just watch the boundaries | Add two non-data views — control flow (timing/ordering) and state/ownership (mutation/lifetime); boundary invariants are the third |

## A practical checklist

1. **Static structure first** — what are the components, and what *can* call
   what (the module/dependency graph).
2. **One dynamic trace** — pick the main operation and follow a concrete value
   through it end to end.
3. **Annotate the seams with contracts** — for each interface, record the
   *invariant*, not just the type.
4. **List cross-cutting concerns separately** — determinism/seeding,
   concurrency, caching, error handling. These belong to no single component.

Two vocabulary distinctions worth keeping straight:

- *Static* structure (what **may** call what) vs. *dynamic* flow (what
  **actually** runs for a given operation) — different questions, both needed.
- *Data* flow (what moves) vs. *control* flow (what drives it) — complementary
  halves of the same picture.

---

## Appendix — worked illustration (this repo, `sraz`)

The same principles applied concretely to this codebase, an AlphaZero engine +
symbolic-regression game. See `Claude-docs/` for the full component reference.

**Reading order here:** `Claude-docs/01-overview.md` (what it is, engine-vs-
instance split) → `Claude-docs/02-architecture.md` (component map + connection
flow). That pair suffices for a general understanding; `03`–`05` are deeper
dives to defer.

**The one dynamic trace worth following** (Refinement 1): a single self-play
game emits `(obs, move_probs, discounted_return)` tuples → they land in the
Trainer's replay window → get flattened → become the batch that `net.train`
fits. Within one move: `obs` → `net.predict` → prior + value → MCTS refines it
into `move_probs`.

**The four views, instantiated:**

| View | Example in `sraz` |
| --- | --- |
| Data flow | The experience tuples flowing from self-play into the replay window |
| Control flow | N games per iteration; a fresh MCTS per move; a 10-iteration sliding replay window |
| State & ownership | MCTS mutates the game in place and rewinds via `stash_state`/`unstash_state` — nothing crosses an interface, yet it is central |
| Contracts / invariants | "the policy must be masked and renormalized"; "stash must round-trip every field" |

Notice that the state/ownership example (in-place mutation + rewind) is entirely
internal to MCTS — a pure data-flow reading would miss it, which is the whole
point of Refinement 2.
