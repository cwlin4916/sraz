# The informativeness × deception framework

*Snapshot, 2026-07-15. The core conceptual tool for this research; refines the
"informative intermediate states" idea from [01](01-brainstorm-directions.md).*

## Why this doc

AlphaZero is MCTS guided by a policy/value network. To study how the *search*
and the *net* interact, we use symbolic regression as a controllable testbed and
ask one question about a task: **how does the value signal behave as an
expression is built piece by piece?** That question splits into two *independent*
axes. Where a task sits on these axes predicts which part of the MCTS↔NN loop is
stressed — and whether AlphaZero should work at all.

**Vocabulary.** A *state* $s$ is a partial derivation (e.g. `+ S S` — committed
to a sum, blanks unfilled). From $s$, many completions are possible; each
finished expression earns a reward (clipped $R^2$). The "value" of $s$ is,
loosely, the score reachable from it. Both axes are about how that value is (or
isn't) revealed as you build.

## Axis 1 — Informativeness (can a half-built expression's promise be read?)

**Definition.** If you finish $s$ with a *weak* policy (random, or early
self-play), does the resulting score correlate with how good $s$ could be if
finished *well*?

- **Informative:** good skeletons score okay-ish even when finished sloppily,
  and bad ones score badly regardless. Partial states are rankable *without*
  already knowing the perfect completion.
- **Uninformative:** $s$ is worthless unless completed exactly right, which weak
  play almost never does — so every partial state looks equally bad, and the
  promise stays hidden until the last piece clicks.

**SR examples.**
- *Uninformative:* `* C3 sin * C4 x` (a lone sine) must nail the frequency to fit
  at all; finished carelessly it scores $-1$. So "I started a sine term" looks
  like garbage even though the right completion is part of the $0.9873$ frontier.
- *Informative:* a sum-of-simple-terms target where a partial expression already
  containing the dominant term (e.g. $C_2 x^2$) still scores decently under most
  completions — quality leaks through.

**Why it matters for AlphaZero.** The value net learns $V(s)$ from self-play
returns, and early self-play is nearly random. Informative → a clean value
signal from a cold start → the net can *guide* the search. Uninformative → the
net's early targets are noise → it never learns to rank skeletons → MCTS flies
blind.

**Metric (policy-free).** For many states, average the return over $K$ random
completions and correlate it with the best-achievable return. High correlation =
informative. This is a property of the *task*, measured with no policy.

## Axis 2 — Non-deceptiveness (does the local gradient point toward the optimum?)

**Definition.** As you add the *right* pieces toward the best expression, does
the observable score go **up at every step**, or must you accept something
*worse* (or forgo a tempting easy win) before the payoff arrives?

- **Non-deceptive:** every correct step improves the score; greedy hill-climbing
  basically works; no traps.
- **Deceptive:** an easy-to-reach state scores *well* but is a dead end, and the
  path to the true optimum runs through *worse*-scoring states; the local
  gradient points *away* from the global best.

**SR examples.**
- *Deceptive (the current game):* `C1 x` is a **one-move** terminal at $0.8729$ —
  a cheap, tempting reward. Beating it requires `+ S S` (compose), committing to
  build two sub-expressions whose half-built forms score worse or fail. The
  immediate reward lures the policy into terminating early, away from the
  deferred, larger payoff.
- *Non-deceptive:* additive target $C_0 + C_1 x + C_2 x^2$ — $C_1 x$ ($0.87$) →
  $+\,C_2 x^2$ ($0.93$) → $+\,C_0$ ($0.97$); every step up, no temptation to
  stop early.

**Why it matters for AlphaZero.** Even if states are *eventually* informative, a
trap captures the **policy**. Once self-play concentrates on the easy win it
stops visiting the good branch, so the value net never gets the data that would
prove the branch's worth — the trap seals itself. Deception is an **exploration**
problem, distinct from informativeness (a **value-signal** problem).

**Metric.** Is there a short-to-reach terminal whose score exceeds the value of
the on-path partial states? Equivalently: does the greedy-immediate action at
each state head *toward* or *away* from the global optimum?

## The 2×2 — the experimental design

The axes are independent, so a task lands in one of four cells, each stressing a
different part of the loop:

| | Non-deceptive (no trap) | Deceptive (trap) |
| --- | --- | --- |
| **Informative** (value readable) | *Easiest.* Value net learns fast, policy climbs monotonically → AZ should shine even at 25 sims. | Value net *can* learn, but only after exploration escapes the trap → the regime where **exploration knobs** (Dirichlet, temperature, `max`-backup, more sims) matter most. |
| **Uninformative** (value hidden) | No trap, but the net can't help → rely on **raw search**; the NN adds little. | *Worst case — the current game.* Value hidden **and** a trap → AZ fails (observed). |

**Mapping to the two networks (why this isolates the interaction):**
- Axis 1 (informativeness) ↔ whether the **value network** can do its job.
- Axis 2 (deception) ↔ whether the **policy net + search exploration** can avoid
  the trap and cover the good branch.

Moving an SR instance around this 2×2 independently stresses the value net and
the policy/search — which is exactly how we tease apart their interaction.

## Caveat: for AlphaZero, informativeness is partly policy-dependent

AZ's *effective* informativeness is a property of the self-play **distribution**,
not the task alone — a branch can start uninformative because it is undersampled,
then become informative once explored. So it is not a pure task knob; that
coupling is itself part of what we study. The rollout-correlation metric is
valuable precisely because it is policy-free: a static, task-side ground truth to
compare the trained value net against.

## Immediate implication

The current game sits in the **worst corner** (uninformative + deceptive). The
cleanest first experiment: construct a **best-corner** instance (the additive
target), confirm AZ climbs the ladder even at a low simulation budget, then walk
it toward the hard corners **one axis at a time** to see which mechanism breaks
first.
