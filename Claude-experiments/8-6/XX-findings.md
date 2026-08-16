# c_exploration sweep — findings (sine, problem_seed 42)

Living doc for the 2026-08-06 experiment. Artifacts in this folder:

| file | what |
|---|---|
| `c-sweep_sine_seed42.{json,png}` | sweep at the shipped `rollout_budget=500` |
| `c-sweep_sine_seed42_rb5000.{json,png}` | same sweep, `rollout_budget=5000` |
| `c-sweep.log`, `c-sweep-rb5000.log` | full console output |

Script: `Claude-scripts/06-c-exploration-sweep.py`.
Grid: $c \in \{0.25, 0.5, 1, 2, 3, 5, 8, 15\}$ × sims $\in \{25, 100, 400\}$,
24 paired episodes per cell (identical episode seeds across cells, so
differences are attributable to $c$).

Reference rungs, exact from the enumeration in `Claude-scripts/01`/`02`:

$$R^2_{\text{linear}} = 0.8729 \;(\texttt{C1*x}) \quad<\quad R^2_{\text{affine}} = 0.9575 \;(\texttt{C0+C1*x}) \quad<\quad V^* = 0.9966$$

## Headline

**The sine is never found. Not in one cell, not in one episode, at any $c$.**
0/24 cells reached a `sin` term in either sweep. The best rung reached anywhere
in this sweep is affine, $R^2 = 0.9575$ — still $0.039$ short of $V^*$.

**Lower $c$ produced better equations than the default.** That is backwards from
the usual reading of $c$, and the "catch" section below shows it is an artifact
rather than a real effect.

| $c$ | converged equation (unstarved budget) |
|---|---|
| 0.25 | affine, $0.9575$ — **24/24 episodes at every sims budget** |
| 0.5 | mixed: mostly linear, 2–4/24 affine |
| 1 – 8 | linear, $0.8729$, 24/24 |
| 15 | affine at 25 sims; linear at 100 and 400 |

## Why: measured root visit distributions (sims = 100, `rollout_budget=5000`)

```
c=0.25   N= 99  + S S          <- 99/100 visits into ONE branch
         N=  1  C0
         N=  0  * C1 x         <- the linear terminal is NEVER visited

c=1      N= 48  * C1 x         <- discovers it, Qtilde=1.0, locks on
         N= 30  * C2 * x x
         N= 12  + S S

c=15     N= 18  * C1 x         <- near-uniform: 18,18,16,13,13,12,10
         N= 18  * C2 * x x        breadth, but nothing mined deeply
         N= 16  + S S
```

Three regimes:

- **Small $c$** — selection is effectively greedy on $\tilde{Q}$. Simulation 1
  has every UCB numerically equal, so `argmax` breaks the tie to the **lowest
  action index**, which is `+ S S`. Its first rollout returns $\approx 0.70$,
  $\tilde{Q}$ pins to $1.0$, and the bonus is far too small to ever pull away.
  The search commits 99% of its budget to that one branch and mines it deeply,
  finding `C0+C0+...+C1*x`. It never even *sees* `* C1 x`.
- **$c \approx 1$** — enough breadth to discover `* C1 x`, a *terminal* whose
  exact $R^2 = 0.873$ never decays. $\tilde{Q} = 1.0$ forever, so it wins.
- **Large $c$** — visits spread almost uniformly. Genuine breadth, but no branch
  gets the depth needed to reach a good completion, and `* C1 x` still takes the
  visit-count argmax by a hair.

## The catch: the small-$c$ win is a tie-break artifact

`+ S S` is action **index 0**. The $c = 0.25$ result depends entirely on the
first-simulation tie going to it — every episode starts identically because the
tie-break is deterministic, which is exactly why the result is a suspiciously
clean 24/24.

**If the grammar listed `* C1 x` first, small $c$ would lock onto the linear
answer and be strictly worse.** This is luck about production ordering, not a
property of low exploration. Do not report "lower $c$ helps" without this
caveat. A proper test would randomise the production order (or the tie-break)
and re-run — see "Open" below.

## Rollout budget matters, and the shipped default is starved

Raising `rollout_budget` 500 → 5000 lifted cells beating linear from **3/24 to
8/24**, and made $c=0.25$ robustly affine at *all* sims budgets instead of only
at 25.

The shipped default of 500 steps is shared across every simulation of one move
and is exhausted after ~6 leaf evaluations (each 20-rollout evaluation costs
80–140 steps). Measured on a 40-simulation search: **5 leaf evaluations got a
real rollout mean, 5 fell back to the net's constant $0.0$**, and the rest hit
terminals. Consequence: raising `n_simulations` buys tree depth but **zero
additional rollout information**, so the earlier "more search is worse" result
should be stated as *more search over a fixed, very small information budget is
worse*.

## Animations of the two extremes

`mcts-ucb_sine_seed42_c0p25_rb5000.gif` and `..._c15_rb5000.gif` (40 frames,
1.5 fps, from `Claude-scripts/05-mcts-ucb-animation.py --c-exploration ...`).
Each frame decomposes every root action's selection score into
$\tilde{Q}(a)$ + $u(a)$ as of *before* that simulation, alongside the depth-2
tree and a UCB-vs-simulation time series.

| | $c = 0.25$ | $c = 15$ |
|---|---|---|
| `* C1 x` first selected at | **never** (40 sims) | simulation **3** |
| greedy first move | `+ S S` | `* C1 x` |
| $q_{\min}, q_{\max}$ | $0.000, 0.9575$ | $-1.000, 0.9227$ |
| $\tilde{Q}$ of a zero-$Q$ action | $0.000$ (floor) | $+0.520$ (mid-pack) |
| $u$ magnitude | $\approx 0.23$ | $\approx 2.3 - 2.7$ |

At $c = 0.25$ the exploration term never exceeds $0.23$ while `+ S S` sits at
$\tilde{Q} = 1.0$, so nothing can dislodge it; $q_{\max} = 0.9575$ shows the
search did reach the affine optimum *inside* that one branch.

At $c = 15$ the bonus ($\approx 2.5$) is **two to three times the entire width of
the exploit range** ($\tilde{Q} \in [0, 1]$), so selection is effectively
round-robin and $\tilde{Q}$ is decorative.

### The important one: breadth is not the bottleneck

At $c = 15$ the search **does** try the sine production — `* C3 sin * C4 x`
reaches $N = 4$ — and its backed-up value comes home at exactly $Q = -1.000$,
the clip floor. So high $c$ finds the branch and then correctly rejects it *by
its own lights*, because random completions underneath it are terrible.

That is direct evidence that the failure is the **leaf evaluator**, not
insufficient exploration. No setting of $c$ can fix a value estimate that reports
$-1$ for the branch containing $V^* = 0.997$.

## Open

- Randomise production order / tie-break to confirm the small-$c$ result is an
  artifact.
- $c$ is inert at 100+ sims in the starved sweep — worth confirming that is
  entirely budget starvation.
- Untested knobs that target this failure directly: `backup_rule="max"`/`"topk"`
  (stops the $+\,S\,S$ decay from $0.70 \to 0.25$) and `rollout_mode="max"`
  (optimistic leaf estimate). Both already implemented.
- Nothing here tests the learned net; $V^*$ may only be reachable with a value
  head that generalises across states.

## Update log

- 2026-08-06 — created; both sweeps run, root visit distributions measured,
  tie-break artifact identified.
- 2026-08-06 — added $c = 0.25$ / $c = 15$ UCB animations; found that $c = 15$
  visits the sine branch and gets $Q = -1.0$ back, pinning the failure on the
  leaf evaluator rather than on exploration.
