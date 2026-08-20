# 8-20 — The target family under `ADDITIVE_GRAMMAR`: quadratic members

The quadratic half of the controlled family of `writeup-milton/writeup.pdf`,
built to mirror `Claude-experiments/8-17` figure-for-figure so the two folders
can be read side by side. Same MDP (`ADDITIVE_GRAMMAR`, 4 productions,
$L = 12$, $\tau = 10^{-6}$, grid $x \in [-1,1]$ with $n = 41$), same scripts,
same search settings — only the target varies.

Where 8-17 varies a line's root position, this folder varies the two invariants
a parabola has relative to the domain: the vertex $x_v = -c_1/2c_2$ (inside or
outside $[-1,1]$) and the discriminant $\operatorname{disc} = c_1^2 - 4c_0c_2$
(two real zeros or none). Per `def:target-family` of the writeup.

| target | $y(x)$ | shape on $[-1,1]$ | writeup's label |
|---|---|---|---|
| `quad_A` | $1 - x + 2x^2$ | arch: $x_v = 0.25$ inside, $\operatorname{disc} = -7 < 0$, never crosses zero | — |
| `quad_B` | $6 - 5x + 0.5x^2$ | monotone: $x_v = 5$ far outside, pseudo-linear over the domain | plateau |
| `quad_C` | $0.001 + 0.001x + 1000x^2$ | ill-conditioned: $c_2/c_1 = 10^6$ | solved at once |
| `quad_D` | $-0.48 + 0.4x + 2x^2$ | root crossing: $\operatorname{disc} = 4 > 0$, zeros at $-0.6$ and $0.4$ both inside | **trap** |

## Files

| file | what it is |
|---|---|
| `07-exact-audit.md` / `.json` | Exact root quantities by exhaustive backward induction: $R(D_i)$, $V^*(C)$, $V^q(C)$, $\rho(C)$, margin, the state census and the reward ceiling per terminal length. **Reproduces the four quadratic rows of Table 2 of the writeup exactly** (tolerance $5\times10^{-4}$ on every checked field). |
| `08-random-search.md` / `.json` / `.png` | Uniform random search from $s_0$ and from $C$, in closed form: $\rho$, $K_{1-\delta}$, $P(\text{exact})$ and $E[R_{\max}(B)]$ across the budget grid, plus the budget-grid design check and a Monte-Carlo cross-check of the algebra. |
| `<target>/tree_<target>_depth2.png` / `.json` | Depth-2 tree of the MDP: node = partial expression, fill = exact $V^q$, label carries $V^*$, $V^q$, gap and $\rho$. The root's mean-backup choice and its optimal choice are drawn over the edges, so the inversion is visible directly. |
| `<target>/mcts-ucb_<target>_c1_rb1000_rn1_s64.gif` | Pure-MCTS search animated, 64 frames = 64 simulations = 64 terminal evaluations. Each frame decomposes every root action's selection score into exploit + explore *before* the simulation runs, then names the action chosen and the leaf value returned. |
| `<target>/mcts-tree_<target>_c1_rb1000_rn1_s64.gif` | Companion to the row above — the same search, same seed, same 64 simulations — but showing *where the tree grew* rather than *why*: edge width = visit count, edge colour = backed-up $Q$, plus a bar panel of exact first-move visit counts. |
| `<target>/mcts-ucb_<target>_c1_rb5000_rn1_s400.gif` | The same UCB decomposition run out to 400 simulations, to show what the allocation converges to rather than how it starts. |
| `<target>/c-rollout-sweep_<target>_s100.png` / `.json` | Pure-MCTS sweep over $c_{\text{exploration}} \times$ `rollout_n` at 100 simulations/move, 24 paired episodes/cell. Three panels: $R^2$ vs $c$, mean-$R^2$ heatmap with the modal expression family, and the exact-recovery fraction — the writeup's primary outcome. |
| `tree_family_depth2.png` | All four quadratic targets stacked on a shared colour scale. The tree is structurally identical across targets — only the values change — so this is the family comparison. |
| `c-rollout-sweep_family_s100.png` | All four targets stacked, mean $R^2$ on a shared $[-1,1]$ scale and exact-recovery on $[0,1]$. |
| `cache/` | Memoised `lmfit` $R^2$ per terminal sentence, per target: 247 entries each, one per reachable terminal. Derived data, rebuilt by `07`. |
| `*.log` | Raw stdout of every run, kept so each figure's provenance is checkable. |

## Layout

```
8-20/
  00-index.md
  07-exact-audit.{md,json}          family-wide: the exact ground truth
  08-random-search.{md,json,png}    family-wide: the guessing baseline
  tree_family_depth2.png            family-wide: all targets, shared scale
  c-rollout-sweep_family_s100.png   family-wide: c x rollout_n, all targets
  cache/                            shared fit cache
  quad_A/ quad_B/ quad_C/ quad_D/   per-target artifacts:
      tree_<t>_depth2.{png,json}            exact V*/Vq/rho tree
      mcts-tree_<t>_..._s64.gif             search animated: where the tree grew
      mcts-ucb_<t>_..._s64.gif              search animated: why (UCB split)
      mcts-ucb_<t>_..._s400.gif             the same, out to 400 sims
      c-rollout-sweep_<t>_s100.{png,json}   c x rollout_n sweep
```

## Scripts

Identical to 8-17's, with `--family quadratic` and this folder as `--out-dir`.
No script is specialised to a family; `07` must run first because it builds the
terminal-score cache the others read.

```
.venv/bin/python Claude-scripts/07-family-exact-audit.py \
    --family quadratic --out-dir Claude-experiments/8-20
.venv/bin/python Claude-scripts/08-random-search-baseline.py \
    --family quadratic --out-dir Claude-experiments/8-20 \
    --budgets 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192
.venv/bin/python Claude-scripts/09-family-depth2-trees.py \
    --family quadratic --out-dir Claude-experiments/8-20
.venv/bin/python Claude-scripts/10-family-c-rollout-sweep.py \
    --family quadratic --out-dir Claude-experiments/8-20
```

The animations are per-target and need their search settings given explicitly,
because their defaults are the sine ones. `04` and `05` get identical settings
so the two GIFs are the same search:

```
for t in quad_A quad_B quad_C quad_D; do
  for s in 04-mcts-tree-animation 05-mcts-ucb-animation; do
    .venv/bin/python Claude-scripts/$s.py --problem $t --sims 64 \
        --c-exploration 1.0 --rollout-budget 1000 --rollout-n 1 \
        --out-dir Claude-experiments/8-20/$t
  done
  .venv/bin/python Claude-scripts/05-mcts-ucb-animation.py --problem $t \
      --sims 400 --c-exploration 1.0 --rollout-budget 5000 --rollout-n 1 \
      --out-dir Claude-experiments/8-20/$t
done
```

Two script fixes were needed to produce this folder, both in the *reporting*
path rather than the numerics:

- `09` and `10` built `out_dir` as `Path(args.out_dir)` without `.resolve()`,
  so a **relative** `--out-dir` made the `[saved] {png.relative_to(REPO)}`
  line raise `ValueError` *after* the figure was written — losing every
  later target in the loop. Now `.resolve()`d, matching what `04`/`05`
  already did. 8-17 never hit this because it used the absolute default.
- `07` and `08` hard-coded "linear family" in their markdown titles. The
  title is now derived from the targets actually reported.

## Which algorithm the GIFs show

The animations run the searcher this repo **ships**: AlphaZero-style PUCT,

$$\mathrm{UCB}(a) = \tilde Q(a) + c\,P(a)\,\frac{\sqrt{N_{\text{tot}}}}{1 + N(a)},
\qquad c = 1.0,\ P(a) = \tfrac14 .$$

This is **not** the writeup's eq. (17) Mean-UCT, $\bar Q + \sqrt{2\log N(s)/N(s,a)}$
on raw rewards with unvisited actions scored $+\infty$. Three differences:
$\sqrt N$ vs $\sqrt{\log N}$; the $1 + N(a)$ denominator, so an unvisited action
is not forced; and the min–max normalisation of $Q$, which rescales the constant
the writeup deliberately holds fixed. Conclusions from these GIFs describe the
shipped searcher. Reproducing the paper's claims needs eq. (17), which does not
exist in this codebase.

`rollout_n=1` makes one simulation cost exactly **one** terminal evaluation, per
the writeup's evaluation protocol. `rollout_budget` cannot bind at either
setting: the longest derivation here is 11 actions, so 64 simulations consume at
most $704$ steps against a cap of 1000, and 400 simulations at most $4400$
against 5000. The logs confirm it — every run reports `0 starved`, and actual
consumption is far below both caps: the s64 runs used 23–126 of their 1000
steps, the s400 runs 57–420 of their 5000. (The spread tracks how deep the
episodes go: `quad_C` terminates in one move and barely rolls out at all,
while `quad_B` wanders deepest.) One persistent tree at $s_0$, no
move ever committed.

## Root allocation at $B = 64$, seed 42

Read straight off the s64 logs; $N$ sums to 64. The `mcts-tree` GIF's bar panel
shows the same numbers one lower, because `04` counts the root expansion as a
simulation and `05` does not.

| target | greedy first move | $N$ | $Q$ | correct branch? | $\max_i R(D_i)$ | margin |
|---|---|---|---|---|---|---|
| `quad_A` | `+ S S` | 47 | $+0.374$ | yes — the only branch containing an exact expression | $-0.0000$ | $-0.3540$ |
| `quad_B` | `+ S S` | 56 | $+0.748$ | yes | $+0.0000$ | $-0.1563$ |
| `quad_C` | `* C2 * x x` | 54 | $+1.000$ | yes — exact *to within* $\tau$ (see below) | $+1.0000$ | $+0.5160$ |
| `quad_D` | `* C2 * x x` | 54 | $+0.646$ | **no — this is the trap** | $+0.6461$ | $+0.1355$ |

`quad_D` is the quadratic analogue of `lin_A` in 8-17, and it is the only
quadratic member satisfying the writeup's eq. (9) inversion: the search commits
$54/64$ of its budget to a one-move terminal whose reward is $0.6461$, while the
one branch that contains an exact expression, $V^*(C) = 1$, keeps 6 visits. The
other three rows are the controls — for `quad_A` and `quad_B` the margin is
negative so no trap exists, and for `quad_C` the preferred one-move action is
itself exact at $\tau = 10^{-6}$, so preferring it is correct at that
tolerance. This is the second inequality of
eq. (9) doing the separating, not the margin alone.

A caveat on reading "did it find the exact expression" off these logs: the
`q_max` the logs print is the largest **edge $Q$ after backup** — a mean — not
the largest terminal reward seen. $q_{\max} = 1$ therefore proves an exact
expression was evaluated ( `quad_A`, `quad_B`, `quad_C` ), but $q_{\max} < 1$
does not prove the opposite, because a revisited leaf's running mean can dilute
a $1.0$ away. `quad_D`'s $q_{\max} = 0.8748$ — exactly the $\ell = 7$ ceiling
from `07` — is suggestive, not conclusive. The properly measured recovery
numbers are `10`'s exact-recovery fractions over 24 paired episodes.

## Two things to know when reading the figures

**The quadratic family is about $3.6\times$ rarer than the linear one.** All of
`quad_A`, `quad_B`, `quad_D` share support $P = \{0,1,2\}$, so by Lemma 3.2
their $\rho$ agree exactly: $\rho(C) = 0.1128$, $\rho(s_0) = 0.0282$, giving
$K_{95}(s_0) = 105$ against the linear family's $29$. Their recovery curves in
`08-random-search.png` therefore lie exactly on top of one another — the plot
shows two curves, not four. `quad_C` is the outlier at $\rho(C) = 0.5569$ despite
having the same support, and the reason is round-off, not mathematics
(`rem:conditioning` of the writeup). The single atom `* C2 * x x` returns
$R = 1 - 8.1\times10^{-12}$, which $\tau = 10^{-6}$ reads as **exact** —
whence $R(D_3) = 1$ and $\rho(C) = 0.5569$. Any $\tau < 8.1\times10^{-12}$
reads it as inexact and restores $\rho(C) = 0.1128$, in line with the other
three. Verified directly against this folder's `cache/fit_quad_C.json`, which
stores $R = 0.9999999999918834$, i.e. $1 - R = 8.117\times10^{-12}$. So
`quad_C`'s apparent easiness is a tolerance artifact, and it is the one row of
this folder whose numbers change if $\tau$ is tightened.

**`quad_B`'s difficulty is a plateau, not a root decoy.** Its reward ceiling is
$0.997213$ already at $\ell = 5$ and stays there at $\ell = 7$ and $\ell = 9$,
reaching $1$ only at $\ell = 11$ — three of the five actions of an optimal
derivation buy $0.0028$ of $R^2$ between them. A fitted *line* through a
parabola whose vertex sits at $x_v = 5$ already explains $99.7\%$ of the
variance. So `quad_B` has a negative margin and no root trap, yet the same
inversion reappears one level down *inside* the continue branch. The depth-2
tree cannot show this; it is visible in `07`'s ceiling table and in how the
s400 GIF's allocation settles.

**The tree figures use the MDP's branching, not the grammar's.** The real action
set is every (position, production) pair the mask allows, so `+ S S` has 8
children, not 4, and that is what a search budget is spent against. The 8 come
in mirror pairs (`+ C0 S` and `+ S C0`) carrying identical $V^*$, $V^q$ and
$\rho$ — a useful self-check on the backward induction, since addition is
commutative. Pass `--expansion leftmost` to `09` for `02`'s convention.
