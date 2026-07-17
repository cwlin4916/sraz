# Deep-research prompt — next minimal working plan for AlphaZero-on-SR

*Paste everything below into a deep-research LLM. It is self-contained; you cannot see my files, so §A and §B carry all context.*

---

## §A — My current research context

- **Active project — search vs. learning in single-player synthesis (AlphaZero_PP → `sraz`).** Single-player AlphaZero (MCTS + policy/value net) for combinatorial/mathematical construction. The current instance in focus, `sraz`, is symbolic regression posed as a grammar-derivation game (§B). **Open question I care about most: does the neural net actually contribute, or is MCTS noise doing the solving — and where does the AlphaZero recipe structurally break?**
## §B — References

**B1 — The SR grammar game and its first training run.** Symbolic regression as a single-player MDP. State: a 15-slot token buffer holding a context-free sentential form; action `a=(i,j)` rewrites nonterminal at slot `i` with production `p_j` (105 flat actions). Legality mask (verbatim):

$$\boxed{\,m(s)_{ij} = 1 \iff s_i \text{ is a nonterminal} \;\wedge\; \mathrm{lhs}(p_j) = s_i \;\wedge\; \ell(s) + \lvert\mathrm{rhs}(p_j)\rvert - 1 \le 14\,}$$

The seven productions (the only nonterminal is `S`):

| $j$ | production $p_j$ | $\lvert\mathrm{rhs}\rvert$ |
|---|---|---|
| 0 | `S -> '+' S S` | 3 |
| 1 | `S -> 'C0'` | 1 |
| 2 | `S -> '*' 'C1' 'x'` | 3 |
| 3 | `S -> '*' 'C2' '*' 'x' 'x'` | 5 |
| 4 | `S -> '*' S S` | 3 |
| 5 | `S -> '/' S S` | 3 |
| 6 | `S -> '*' 'C3' 'sin' '*' 'C4' 'x'` | 6 |

The game chooses only *structure*; constants $C_k$ stay symbolic and are fitted (lmfit least squares, all coords init $2.5$) at termination. Reward (verbatim):

$$\boxed{\,\rho(\pi) = \mathrm{clip}\!\big(R^2,\, -1,\, 1\big), \qquad R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}}\,}$$

Hidden target $y(x) = 4\sin(4x) + C_0^* + C_1^* x + C_2^* x^2$, $C^*\sim U[1,4]^3$ fixed per instance, sampled at 41 points on $[1,3]$ (deterministic by default; the reward fails soft to $-1$ and is cached per sentence). The 14-token cap makes the 18-token generating form **inexpressible**; the measured score ladder (seed 42):

| structure | tokens | $\rho$ |
|---|---|---|
| $C_0$ | 1 | $0.0000$ |
| $C_3\sin(C_4x)$ | 6 | $-1.0000$ (fit fails) |
| $C_2x^2$ | 5 | $0.8409$ |
| $C_1x$ | 3 | $0.8729$ |
| $C_1x + C_2x^2$ | 7 | $0.9316$ |
| $C_0 + C_1x + C_2x^2$ | 9 | $0.9731$ |
| $C_3\sin(C_4x) + C_0 + C_2x^2$ (frontier) | 14 | $0.9873$ |
| generating form | 18 | $0.9996$ — unreachable |

**First run (default: 10 iters × 20 self-play games × 25 MCTS sims, MLP 2×128):** converges in iteration 1 onto the single-production sentence $C_1x$ ($\rho=0.8729$) and never escapes, though the expressible ceiling is $0.9873$. A textbook premature-convergence collapse.

**B2 — A verified literature map (≈79 records) placing the repo's five design commitments.** (1) *MCTS/AlphaZero for SR*: nearest relatives — Symbolic Physics Learner (grammar-as-MDP MCTS, network-free), SR-GPT (an explicit AlphaZero learn-and-search loop, but over vocabulary tokens), TPSR (MCTS as decode-time planner over a pretrained transformer), RSRM, and Huang et al. 2025 (argues SR wants an *extreme-bandit* backup, not mean-UCB, since the goal is the single best expression). (2) *Deep symbolic regression*: Petersen et al.'s DSR — RNN emits tokens, inner BFGS fits constants, a *risk-seeking* policy gradient optimizes squashed NRMSE with no tree search; Landajuela et al. name "early commitment" (premature entropy collapse) and "initialization bias to short sequences" and fix them with hierarchical entropy + a soft length prior; GFN-SR samples trees ∝ reward. (3) *Grammar-constrained search*: Lagramge, ProGED, grammatical evolution, Grammar-VAE, Yin & Neubig — all ancestors of the legality mask. (4) *Single-player AlphaZero*: AlphaTensor, AlphaDev, Ranked Reward (binarize score against a moving percentile), Gumbel AlphaZero (visit-count targets can fail to improve at low sim counts), DeepCubeA, FunSearch. (5) *Inner fit oracle*: LM constant-fitting (Kommenda), the "Good Structure, Bad Score" bottleneck. Also §6 exhaustive grammar enumeration + MDL (the capped space is finite and auditable), and §7 a fix menu for the collapse grouped by engine component: search-side (Gumbel, AmEx-MCTS, extreme-bandit), reward-side (Ranked Reward, count bonuses, novelty), target-side (regularized-policy $\bar\pi$ targets, greedy value targets, priority-queue training), population-side (Go-Explore, GFlowNets, island diversity). Positioning claim: the *combination* — grammar-production actions with agent-chosen rewrite position, from-scratch single-player AlphaZero, deterministic cached fit-oracle reward — is unoccupied, and the instance is auditable in a way benchmark-scale relatives are not.

**B3 — Rollout-signal diagnostic (three toy games).** Measures how well average random-rollout value from a partial state correlates with the achievable optimum:

| Game | Rollout correlation | AlphaZero outcome |
|---|---|---|
| Path building (Go-like) | **0.89** | works |
| Combination lock | 0.62 | struggles |
| Symbolic regression | **0.51** | fails |
| Go / Chess (literature) | ~0.95 / ~0.90 | superhuman |

On the *optimal path*, the true-optimal state's rank by average-completion score degrades with depth: for SR it is 1st at depth 1 but ranks 8th/32 by depth 3 and 25th/256 by depth 6 (while remaining 1st by achievable optimum). The MCTS value bootstrap therefore has almost no signal to learn from — good partial states look mediocre on average.

**B4 — Structural diagnosis: compositional credit assignment (why AlphaDev works, SR does not).** MCTS+value-net needs two properties. **A — evaluable partial states.** **B — compositional structure:** B1 actions have local scope, B2 sub-goals are independently attackable, B3 progress accumulates (a good step is not later undone). AlphaDev (assembly-program search) satisfies both — one instruction writes one memory cell (B1), the target decomposes over N test-cases × array positions (B2), correct placements are monotone (B3) — and its dense per-step reward is a *consequence* of B, not a free knob. SR fails: grammar-growth violates A (a partial tree `((?+?)+(?*?))` has no output) and B; GP-mutation restores A (complete expressions) but still violates B (a subtree rewrite is non-local; a single function evaluated at N points has no independent sub-goal partition; mutations overwrite each other). Per-data-point residual is a decomposable *signal* but the *actions* don't decompose over it. Two coherent exits: **(a)** change the representation to *additive fragment composition* $f=\sum_i g_i(x)$ so A+B hold and MCTS works — but this distorts compact, deeply-nested physics targets toward basis expansions; **(b)** change the algorithm to *DSR-style autoregressive policy gradient* over the hierarchical grammar, which needs only reward-at-completion and preserves compact nested structure. For the project's goal of interpretable equations, (b) is argued the more natural match.

## §C — Your task

Study §A and §B. **Begin each part with clear questions and goal.** The task is explicit and three-fold:

**i) What a systematic study would look like.** Lay out how a rigorous, systematic investigation of *where and why the AlphaZero recipe breaks on symbolic regression* would be structured — the axes to vary, the controls and ablations, the ground-truth baselines (the capped grammar is finite and exhaustively enumerable), and what would count as evidence that the neural net — rather than MCTS noise — is doing the work.

**ii) Orient the literature.** Place this problem in the SR and RL literature of §B2: who has hit the same premature-convergence / weak-rollout-signal wall, what they concluded, which methods sidestep it (extreme-bandit backup, Ranked Reward, DSR-style policy gradient, GFlowNets), and where this repo's exact setup — grammar-production actions with agent-chosen rewrite position, from-scratch single-player AlphaZero, deterministic cached fit-oracle reward — sits relative to them.

**iii) One explicit, minimal next-to-do plan — with justification.** This is the deliverable I most want. Give me a *single*, minimal, runnable next step — the smallest concrete experiment I should do next — and justify it: why it is the right next move, what result would confirm or overturn the current diagnosis, and what each outcome would tell me to do. Keep it minimal — one plan, not a menu of options to survey. The two diagnostics below are *context* the plan should be grounded in, not things to expand:
- **Peter1 / §B3 (context)** — the rollout-correlation (0.51) and rank-of-optimal-state-vs-depth signal-loss measurements, currently shown only on toy games, not the actual `sraz` grammar.
- **Peter2 / §B4 (context)** — the A+B compositional-credit-assignment framework and its two exits: additive-fragment composition that satisfies A+B, vs. DSR-style policy gradient that drops it.

In working through i) and ii) — the groundwork the plan in iii) rests on — weigh these axes I care about:

- **SR literature — the function menu.** What primitive operator/function sets do SR systems actually search over (e.g. protected division, `sin/exp/log`, unary vs binary libraries, dimensional constraints), and how does that choice shape which targets are reachable vs. which collapse? Help me understand the SR literature's conventions here.
- **RL literature — what MCTS buys.** What is the measured contribution of tree search *over no-search* (policy-gradient/autoregressive) in these settings, and under what signal conditions does search stop helping? Connect to §B3/§B4.
- **RL literature — the neural net's role.** Policy vs. value head — which one carries the load, and what evidence separates "the net learned something" from "MCTS noise found it"?
- **Diagnostics / key plots.** What are the canonical plots and ablations this literature uses to *demonstrate* where a method breaks (rollout-correlation curves, rank-vs-depth, reward-vs-simulations, exploration/entropy traces, exhaustive-sweep ground-truth ranking)?
- **Theory.** What does theory say about the collapse — visit-count targets as degraded regularized-policy-optimization at low sim counts, extreme- vs mean-bandit objectives, GFlowNet reward-proportional sampling — and what minimal experiment would discriminate between these diagnoses?
