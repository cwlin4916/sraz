# Execution plan — Experiment 1: exact value-semantic audit

Experiment 1 answers two questions with no search and no sampling. Does the
mean random-completion return preserve the ordering induced by best-achievable
reward, and how rare is the exact expression beneath each partial derivation?
Because it is a census, it has no seeds, no confidence intervals and no
significance tests: it either reproduces the closed forms exactly or the
implementation is wrong.

![the six stages and the gate](figures/exp1_pipeline.png)

## 0. Why new code is unavoidable

The controlled environment cannot be reached by parameterizing the existing
game. That game's terminal evaluator converts a token string to an infix
expression and parses it symbolically; the basis functions `ψ₁,…,ψ_M` are
numeric vectors produced by Gram–Schmidt on the sample grid and have no closed
form to parse. Experiment 1 therefore needs a second instance package with a
linear-algebra evaluator, sitting beside the symbolic one and reusing the
engine's game interface. The engine, the search, and the uniform-prior network
are all reused unchanged.

The prize for building it is that every quantity becomes a counting problem:
the reward of a terminal is the total weight of the atoms it selects, so
expectations reduce to marginal inclusion probabilities.

## 1. Target state

Four environments, indexed by the decoy parameter and the vocabulary size.

|  | `M = 4` | `M = 12` |
|---|---|---|
| `λ = 0.25` | no inversion | no inversion |
| `λ = 0.997` | inversion | inversion |

For each, the six root numbers `V*(D), V^q(D), V*(C), V^q(C), ρ₀(C), K₀.₉₅`
and the four plots the writeup names. The values these must hit are already
derived and independently reproduced by enumerating all `M⁴` completions:

| `M` | `V^q(C) = p_M` | `ρ₀(C) = 4!/M⁴` | `K₀.₉₅` |
|---:|---:|---:|---:|
| 4 | 0.683594 | 0.093750 | 31 |
| 12 | 0.293933 | 0.001157 | 2587 |

and `V*(D) = V^q(D) = λ`, `V*(C) = 1` in every cell.

![the controlled design, exactly](figures/exp1_design.png)

## 2. The environment, precisely

**Grid and basis.** `x₁,…,x₄₁` equally spaced on `[-1,1]`. Centre the monomials
`x, x², …, x¹²` and orthonormalize under the empirical inner product
`⟨f,g⟩ₙ = n⁻¹ Σ f(xᵢ)g(xᵢ)`.

**Target.** `y_λ = √λ · ψ₁ + √((1−λ)/3) · (ψ₂ + ψ₃ + ψ₄)`.

**Grammar.** The root has exactly two actions. `D` terminates immediately into
the model class spanned by `{1, ψ₁}`. `C` opens four atom slots filled
**left to right**, each with productions `ψ₁ | … | ψ_M`. Fixing the fill order
is deliberate: it is what makes the completion law factorize into four
independent uniform draws, which is the assumption the closed forms need and
the one the symbolic game breaks.

**Action encoding.** `Discrete(2 + M)`. Actions `0,1` mean `D,C` and are legal
only at the root; actions `2,…,M+1` mean `ψ₁,…,ψ_M` and are legal only at a
slot state. A Boolean mask enforces this, so the engine's masked-selection path
is reused verbatim.

**State encoding.** A length-5 integer buffer `[root, a₁, a₂, a₃, a₄]` with `0`
for unset. Terminal when the root choice is `D`, or when all four slots are
filled. Episode length is 1 or 5, so the state space is tiny and hashable by
the engine's default.

**Reward.** Let `J` be the set of distinct selected atoms and
`w = (λ, (1−λ)/3, (1−λ)/3, (1−λ)/3, 0, …, 0)`. Two evaluators, both required:

- *analytic* — `R = Σ_{m ∈ J} w_m`;
- *least squares* — `R = 1 − ‖y − Πy‖² / ‖y − ȳ·1‖²`, with `Π` the projection
  onto `span({1} ∪ {ψ_j : j ∈ J})` obtained from the design matrix, repeated
  atoms removed before fitting.

No clipping, no penalty, no nonlinear optimizer. All rewards lie in `[0,1]`.

## 3. Modules to write

| Path | Contents |
|---|---|
| `src/sraz/instances/atomsr/basis.py` | `GRID`, `inner(f,g)`, `orthonormal_basis(n_basis=12, grid=GRID, passes=2)`; verifies its own output |
| `src/sraz/instances/atomsr/targets.py` | `y_lambda(lam, psi)`, `atom_weights(lam, M)`, `LAMBDA_WEAK`, `LAMBDA_STRONG`; verifies mean zero and unit norm |
| `src/sraz/instances/atomsr/reward.py` | `reward_analytic(J, lam)`, `reward_lstsq(J, y, psi)`, `p_of_M(M)`, `rho0_of_M(M)`, `k_of(rho, delta)` |
| `src/sraz/instances/atomsr/game.py` | `AtomSelectGame(Game[np.ndarray,int])` per §2, with `stash_state`/`unstash_state` overridden as the symbolic game does |
| `src/sraz/instances/atomsr/census.py` | `Census(M, lam)` exposing `vstar(s)`, `vq(s)`, `rho0(s)`, `a_q(s)`, `g_q(s)`, `states_by_depth()` |
| `src/sraz/instances/atomsr/config.py` | `AtomSRConfig(lam, M, pure_mcts=True)` mirroring the symbolic config so Experiment 2 can reuse it |
| `scripts/run/run_exp1_census.py` | driver: four cells → gates → `census.jsonl` + `summary.json` |
| `scripts/plotting/plot_exp1_census.py` | the four plots |
| `tests/instances/atomsr/` | one test file per module, asserting the gates of §4 |

The Gram–Schmidt currently duplicated inside
`writeup-milton/figures/make_scenario_figures.py` should import from
`basis.py` once it exists, so the writeup's figures and the experiment cannot
drift apart.

## 4. Gates

Each is an assertion in the test suite, not a printed diagnostic. Tolerances
below are what the construction actually achieves, measured on the 41-point
grid with two re-orthogonalization passes.

| Gate | Condition | Achieved | Assert at |
|---|---|---|---|
| G1a | `max_{j,l} |⟨ψ_j,ψ_l⟩ₙ − δ_{jl}|` | `5.6·10⁻¹⁶` | `10⁻¹⁴` |
| G1b | `max_j |⟨ψ_j,1⟩ₙ|` | `1.0·10⁻¹³` | `10⁻¹²` |
| G2 | `\|ȳ_λ\|` and `\|‖y_λ‖ₙ² − 1\|` | `2.2·10⁻¹⁶` | `10⁻¹²` |
| G3 | `max` over all `M⁴` leaves of `\|R_analytic − R_lstsq\|` | expected `≈10⁻¹⁵` | `10⁻¹²` |
| G4 | `V^q(C) = p_M`, `ρ₀(C) = 4!/M⁴`, `V*(C) = 1`, `R(D) = λ` | exact | `10⁻¹²` |
| G5 | every reward in `[0,1]`; `R = 1` iff `J ⊇ {1,2,3,4}` | exact | — |

G1b is the tightest of the six and carries only one order of margin, because
the monomials `x,…,x¹²` are badly conditioned on 41 points. One
re-orthogonalization pass already reaches `2.7·10⁻¹³` on the Gram matrix; two
passes are what buy the `10⁻¹⁶` there. Do not drop to a single pass, and do not
tighten G1b below `10⁻¹²`.

G3 is the gate the writeup names, and it is the one that would catch a wrong
design matrix, a missing constant column, or a failure to deduplicate repeated
atoms.

**Nothing proceeds to Experiment 2 until all six pass in all four cells.**

## 5. The census

For each cell, enumerate the decoy leaf and every continue prefix of depth
`0,1,2,3,4`, then compute at each state `s`:

```
V*(s)   = max over completions of R
V^q(s)  = mean over completions of R, under uniform per-slot draws
ρ₀(s)   = Pr[R = V*(s₀)] under the same law
a_q(s)  = argmax_a V^q(T(s,a))
g_q(s)  = V*(s) − V*(T(s, a_q(s)))
```

by backward recursion over depth, memoized on the state. Because the reward is
additive in the selected atoms, `V^q` at a slot state is the mean of its `M`
children and needs no fitting — that additivity is exactly what the target
normalization buys.

Cost: the largest cell has `1 + 12 + 12² + 12³ + 12⁴ = 22,621` continue-prefix
states and `20,736` leaves. For calibration, the codebase already enumerates
`4,898` sentential forms covering `405,781` terminals under a *nonlinear*
evaluator in seconds. Expect the whole four-cell census in under a second.

**Cross-check against the existing machinery.** Run the same recursion against
the exact-`V*` enumerator already in the repository on the additive-quadratic
instance; the two must agree on `V*` at every shared state. This validates the
new recursion against code that is already tested, independently of the
controlled family.

## 6. Primary outputs

Per cell, the six root numbers, plus:

1. the terminal completion-reward distribution;
2. `V^q(s)` against `V*(s)`, stratified by state depth;
3. oracle sibling regret `g_q(s)` against depth;
4. the exact hit curve `1 − (1 − ρ₀)^K` with `K₀.₉₅` marked.

Plot 2 is where the audit either shows or fails to show the mechanism: points
below the diagonal at shallow depth in the `λ = 0.997` column are the
value-semantic failure, and plot 3 quantifies what following the mean-rollout
oracle costs at each depth.

Written to `experiments/atomsr_exp1/<timestamp>_lam<λ>_M<M>/` as
`census.jsonl`, `summary.json`, and four PNGs, matching the layout the existing
runners use.

## 7. Commands

```bash
# gates first — this is the whole point of Experiment 1
python -m pytest tests/instances/atomsr -q

# the census over all four cells
python scripts/run/run_exp1_census.py --lambdas 0.25 0.997 --M 4 12

# the four plots per cell
python scripts/plotting/plot_exp1_census.py --run experiments/atomsr_exp1/<ts>

# the analytic design preview these numbers must reproduce
python Claude-milton-experiments/figures/make_design_figures.py
```

## 8. Risks

| Risk | Mitigation |
|---|---|
| Gram–Schmidt loses orthogonality at degree 12 | two re-orthogonalization passes, measured above; G1a/G1b assert it. A QR-based construction is a drop-in alternative and reaches the same tolerance |
| repeated atoms make the design matrix singular | deduplicate `J` before building it, or use the pseudoinverse; G3 catches either failure |
| `λ = 0.997` sits close to `1` and invites cancellation | the reward is a sum of exact weights, not a difference of fits; G5 asserts the reward range and the exact-recovery characterization |
| the census silently truncates | assert the enumerated state count equals `1 + Σ_{d=1..4} M^d` and the leaf count equals `M⁴` |
| the new environment drifts from the writeup's figures | `make_scenario_figures.py` imports the basis from `basis.py` rather than duplicating it |

## 9. Done when

- all six gates pass in all four cells;
- the four root tables reproduce `p_M`, `ρ₀`, `K₀.₉₅` and `λ` to `10⁻¹²`;
- the new recursion agrees with the existing exact-`V*` enumerator on the
  additive-quadratic instance;
- the four plots exist per cell;
- the writeup's controlled-cell table is either confirmed or corrected, with
  the correction stated. One is already known: `ρ₀ = 4!/M⁴` is a probability
  only for `M ≥ 4`, since `4!` counts permutations of four distinct atoms.

## 10. What Experiment 2 then needs

Recorded here because Experiment 1's design decisions constrain it, not as part
of this plan's scope. Detail in [01-writeup-vs-codebase.md](01-writeup-vs-codebase.md) §5.

- a selection rule matching the writeup's Mean-UCT, with `+∞` for unvisited
  actions and no min–max normalization of the exploitation term, since that
  normalization is a confound for the mean-versus-maximum contrast;
- a single-persistent-root driver that never commits a move, with the best
  terminal reward recorded across *all* evaluations including rollout leaves;
- a rollout step budget raised well above `B × 5`, because the shared budget is
  spent in steps and silently truncates at its default;
- the four outcome variables `R^max_B`, `S_B`, `T_exact`, `F_B` written at every
  checkpoint of one run to `B_max`.
