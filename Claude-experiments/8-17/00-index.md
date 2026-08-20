# 8-17 — The target family under `ADDITIVE_GRAMMAR`: linear members

Experiments on the controlled family of `writeup-milton/writeup.pdf`, starting
with the four linear targets. The question the folder works toward is the
writeup's own: **does pure MCTS succeed on this family, and where?** This first
pass answers the prerequisite question instead — what does the *environment*
look like, and what does *guessing* already achieve — because both bound what
any search result can mean.

| file | what it is |
|---|---|
| `07-exact-audit.md` / `.json` | Exact root quantities by exhaustive backward induction: $R(D_i)$, $V^*(C)$, $V^q(C)$, $\rho(C)$, margin, plus the state census and the reward ceiling per terminal length. Reproduces Table 2 of the writeup. |
| `08-random-search.md` / `.json` | Uniform random search from $s_0$ and from $C$, in closed form: $\rho$, $K_{1-\delta}$, $P(\text{exact})$ and $E[R_{\max}(B)]$ across the budget grid, plus the budget-grid design check. |
| `<target>/tree_<target>_depth2.png` / `.json` | **The main picture**, one per target in its own subfolder. Depth-2 tree of the MDP: node = partial expression, fill = exact $V^q$, label carries $V^*$, $V^q$, gap and $\rho$. The root's mean-backup choice and its optimal choice are drawn over the edges, so the inversion is visible directly. Modelled on `Claude-experiments/8-5/tree_sine_seed42_depth2.png`. |
| `<target>/mcts-ucb_<target>_c1_rb1000_rn1_s64.gif` | Pure-MCTS search animated, 64 frames = 64 simulations = 64 terminal evaluations. Each frame decomposes every root action's selection score into exploit + explore *before* the simulation runs, then names the action chosen and the leaf value returned. Modelled on `Claude-experiments/8-6/mcts-ucb_sine_seed42_c15_rb5000.gif`. |
| `<target>/mcts-tree_<target>_c1_rb1000_rn1_s64.gif` | **Companion to the row above — the same search, same seed, same 64 simulations**, but the frame shows *where the tree grew* rather than *why*: the depth-2 derivation tree with edge width = visit count and edge colour = backed-up $Q$, plus a bar panel of exact first-move visit counts. Produced by `04`, which pairs with `05` frame-for-frame. Modelled on `Claude-experiments/8-5/mcts-tree_sine_seed42.gif`. |
| `<target>/mcts-ucb_<target>_c1_rb5000_rn1_s400.gif` | The same UCB decomposition run out to 400 simulations, to show what the allocation converges to rather than how it starts. `rb=5000` again cannot bind (400 sims x 1 rollout x 11 steps = 4,400 < 5,000). |
| `<target>/c-rollout-sweep_<target>_s100.png` / `.json` | Pure-MCTS sweep over $c_{\text{exploration}} \times$ `rollout_n` at 100 simulations/move, 24 paired episodes/cell. Three panels: $R^2$ vs $c$ (mean solid, best dashed), mean-$R^2$ heatmap with the modal expression family, and the exact-recovery fraction — the writeup's primary outcome. Family analogue of `Claude-experiments/8-6/c-sweep_sine_seed42_rb5000.png`. |
| `c-rollout-sweep_family_s100.png` | All four targets stacked, mean $R^2$ on a shared $[-1,1]$ scale and exact-recovery on $[0,1]$, so the targets are directly comparable. |
| `tree_family_depth2.png` | All four targets stacked on a shared colour scale. The tree is structurally identical across targets — only the values change — so this is the family comparison. |
| `08-random-search.png` | Superseded. Recovery curves vs budget; kept because `08-random-search.md` links it, but three of its four curves coincide exactly (see below), so the tree figures carry the message instead. |
| `XX-findings.md` | *Not written yet.* Intended as the running summary of what the numbers mean and what they imply for the search experiments. The figures above are complete for the linear family; the interpretation pass is still outstanding. |
| `cache/` | Memoised `lmfit` $R^2$ per terminal sentence, per target. Shared by `07`/`08`/`09`, so it stays at the top level. Derived data, git-ignored, rebuilt by `07`. |

## Layout

Per-target artifacts live in a per-target subfolder; only genuinely family-wide
artifacts sit at the top level.

The quadratic half of the family was in the end **not** added alongside as
`quad_A/` … `quad_D/`, but given its own folder,
`Claude-experiments/8-20/`, which mirrors this one figure-for-figure with the
same scripts and the same search settings. Keeping the two families in separate
folders keeps each `tree_family_depth2.png` and
`c-rollout-sweep_family_s100.png` a four-row figure rather than an eight-row
one, and keeps the family-wide `07`/`08` reports per family. Read the two
folders side by side; the per-file descriptions below apply unchanged to 8-20.

```
8-17/
  00-index.md               XX-findings.md
  07-exact-audit.{md,json}          family-wide: the exact ground truth
  08-random-search.{md,json,png}    family-wide: the guessing baseline
  tree_family_depth2.png            family-wide: all targets, shared scale
  c-rollout-sweep_family_s100.png   family-wide: c x rollout_n, all targets
  cache/                            shared fit cache (git-ignored)
  lin_A/  lin_B/  lin_C/  lin_D/    per-target artifacts:
      tree_<t>_depth2.{png,json}        exact V*/Vq/rho tree
      mcts-tree_<t>_..._s64.gif         search animated: where the tree grew
      mcts-ucb_<t>_..._s64.gif          search animated: why (UCB split)
      mcts-ucb_<t>_..._s400.gif         the same, run out to 400 sims
      c-rollout-sweep_<t>_s100.{png,json}   c x rollout_n sweep
```

## Scripts

`07`–`10` live in `Claude-scripts/` alongside `01`–`06`, so they can import
each other and reuse the existing helpers; all their output is written here.
`08` and `09` both import `07`, which is the single place the MDP is defined.
`10` imports `03` (for `greedy_episode`) and `06` (for the expression-family
classifier), using both as libraries rather than running their `main()`.

```
.venv/bin/python Claude-scripts/07-family-exact-audit.py
.venv/bin/python Claude-scripts/08-random-search-baseline.py \
    --budgets 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
.venv/bin/python Claude-scripts/09-family-depth2-trees.py
.venv/bin/python Claude-scripts/10-family-c-rollout-sweep.py
```

The two animation scripts are per-target and take the search settings
explicitly, because their defaults are the sine ones. Run once per target, with
`04` and `05` given identical settings so the two GIFs are the same search:

```
for t in lin_A lin_B lin_C lin_D; do
  for s in 04-mcts-tree-animation 05-mcts-ucb-animation; do
    .venv/bin/python Claude-scripts/$s.py --problem $t --sims 64 \
        --c-exploration 1.0 --rollout-budget 1000 --rollout-n 1 \
        --out-dir Claude-experiments/8-17/$t
  done
done
```

`04` and `05` reach the family through the same two-line fix (see
`Claude-milton-experiments/01-transporting-mcts-animations-to-the-family.md`):
a family target cannot be `SymRegConfig`'s `problem` (only `sine` and
`additive_quadratic` are registered), so the target goes in
`cfg.game.kwargs["target"]` with `additive_quadratic` supplying the grammar.
Neither reads the `cache/` above — they memoise from `8-5/cache/`, miss on every
family terminal, and fall back to live `lmfit` solves. Correct, just slower
(~30 s per 64-frame GIF), which is why the sweeps get the cache and the GIFs
do not.

`07` must run before `08` and `09` — it builds the terminal-score cache they
read, and it is what verifies the MDP is the writeup's (`ADDITIVE_GRAMMAR`,
$L = 12$) rather than `SymRegGame`'s sine default (7 productions,
`max_len=15`).

`07`–`10` all default to `--family linear`; pass `--family quadratic` or
`--targets quad_A quad_D` to extend. `09` also takes `--depth` and
`--expansion {all,leftmost}`. `10` takes `--c`, `--rollout-n`, `--sims` and
`--episodes`. The `cache/` already holds `fit_quad_A`…`fit_quad_D`, so the
quadratic half of the family needs no re-fitting.

## Two things to know when reading the figures

**`08-random-search.png` shows two curves, not four.** By Lemma 3.2 of the
writeup, $\rho$ depends on the target only through its support
$P = \{p : c_p \neq 0\}$. `lin_A`, `lin_B` and `lin_C` all have $P = \{0,1\}$,
so their $\rho$ agree to 16 digits ($0.09863155663258756$) and their recovery
curves lie exactly on top of one another. `lin_B` and `lin_C` also coincide in
$E[R_{\max}]$, to within $2\times10^{-13}$. Nothing is wrong with the numbers;
the plot simply cannot separate cells that are genuinely identical.

**But `c-rollout-sweep_family_s100.png` *does* separate `lin_B` from `lin_C`.**
That is not in tension with the paragraph above. $\rho$ and the terminal-reward
*distribution* are properties of the uniform policy $q$, and those coincide;
which *sentence* carries which reward does not, and a tree search navigates
sentences. So random search cannot tell `lin_B` from `lin_C` while MCTS can —
e.g. at $c = 0.25$, `rollout_n` $= 10$, `lin_B` recovers 24/24 and `lin_C`
0/24 on the same seeds. Read the two figures as measuring different things,
not as disagreeing.

**The tree figures use the MDP's branching, not the grammar's.**
`02-grammar-tree-vstar-vrand.py` expands only the leftmost nonterminal, so
`+ S S` shows 4 children there. The real action set is every (position,
production) pair the mask allows, so `+ S S` has 8, and that is what a search
budget is spent against. Pass `--expansion leftmost` to `09` to reproduce
`02`'s convention. A consequence worth noticing: the 8 children come in mirror
pairs (`+ C0 S` and `+ S C0`), which carry identical $V^*$, $V^q$ and $\rho$ —
verified to $10^{-12}$, and a useful self-check on the backward induction, since
addition is commutative.

## Which algorithm the GIFs show

The animations run the searcher this **repo ships**: AlphaZero-style PUCT,

$$\mathrm{UCB}(a) = \tilde Q(a) + c\,P(a)\,\frac{\sqrt{N_{\text{tot}}}}{1 + N(a)},
\qquad c = 1.0,\ P(a) = \tfrac14 .$$

This is **not** the writeup's eq. (17) Mean-UCT, which is
$\bar Q + \sqrt{2\log N(s) / N(s,a)}$ on raw rewards with unvisited actions
scored $+\infty$. Three differences: $\sqrt N$ vs $\sqrt{\log N}$; the
$1 + N(a)$ denominator, so an unvisited action is not forced and can in
principle be skipped indefinitely; and the min–max normalisation of $Q$, which
rescales the constant the writeup deliberately holds fixed. Conclusions from
these GIFs describe the shipped searcher. Reproducing the paper's claims needs
eq. (17) implemented, which does not exist in this codebase.

Settings, and why: $c = 1.0$ and mean backup are the config defaults —
the plainest choice, and unlike the sine case at $c = 0.25$ every root action
does get visited here, so nothing is starved. `rollout_n=1` makes one
simulation cost exactly **one** terminal evaluation, per the writeup's
evaluation protocol; the shipped `pure_mcts` default of 20 would make $B = 64$
simulations cost 1,280 evaluations and break comparability with the budget
grid. `rollout_budget=1000` is provably unable to bind: the longest derivation in
this MDP is 11 actions (nested sums all closed with `C0`), so 64 simulations at
`rollout_n=1` can consume at most $64 \times 11 = 704$ steps. The runs used
18-116. Note the shipped default of 500 is *below* the 704 bound, so it is not
a safe choice either, even though it happens not to bind at these seeds.
Verified: re-rendering at `rb=1000` versus `rb=100000` gives pixel-identical
frames on all four targets, confirming the cap was inert. One persistent tree at $s_0$, no move ever committed.

## Search result at $B = 64$, seed 42

| target | $R_{\max}(64)$ | exact found? | root recommendation | visits to `+ S S` | visits to `* C1 x` |
|---|---|---|---|---|---|
| `lin_A` | $1.000000$ | yes | `* C1 x` — **the trap** | 4 | 56 |
| `lin_B` | $1.000000$ | yes | `+ S S` | 56 | 2 |
| `lin_C` | $1.000000$ | yes | `+ S S` | 56 | 2 |
| `lin_D` | $1.000000$ | yes | `* C1 x` (exact, correct) | 1 | 61 |

Read this against §5.3, which makes the visit-count recommendation *diagnostic*
and the best terminal expression the *primary* output. On that primary metric
pure MCTS **succeeds on all four linear targets** at $B = 64$ — consistent with
`08`, where uniform random guessing also succeeds by $B \approx 29$. The
failure is in the allocation, not the outcome: on `lin_A` the search spends
$56/64$ of its budget inside a branch whose ceiling is $0.821$, and still
*recommends* that branch after having already seen a perfect expression
elsewhere. That is the inversion of eq. (9) acting on a real searcher.
