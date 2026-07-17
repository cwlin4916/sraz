## Why AlphaDev Works and Our SR Representations Don't

**Date:** 2026-07-13
**Purpose:** Synthesize the underlying reason AlphaZero-style neural-guided combinatorial search succeeded in AlphaDev but has failed twice for symbolic regression in this repo (once for grammar-growth, once for GP-mutation). The failure isn't in the neural architecture or the training loop — it's in a structural property of the state/action space that neither of our representations satisfies.

---

### 1. The framing

MCTS + policy/value networks (the AlphaZero recipe) is powerful when the search domain admits *compositional credit assignment*: the search process is a sequence of small decisions, each with a local, attributable effect on the target, such that credit for a good final outcome can be traced back to specific individual decisions and used to train a policy.

When credit is not compositional — when one action can undo another, when actions have global effects on the target, when sub-goals aren't independent — the value network cannot learn a useful function of partial states, and MCTS collapses into either random search or a bad-local-optimum trap. That's what we observed in v1–v9.

Two properties, together, are what make credit assignment work:

**Property A — Evaluable partial states.** Every intermediate state during search produces a concrete, inspectable observation that can be measured against the target.

**Property B — Compositional structure of state, action, and reward.** Three sub-conditions:
- **B1 — Actions have local scope.** Each action modifies only an identifiable slice of state, not the whole thing.
- **B2 — Sub-goals are independent.** The target decomposes into components that can be attacked by separate action sequences without those sequences colliding.
- **B3 — Progress accumulates.** A good action is a genuine forward step; the search doesn't have to undo it later on a correct trajectory.

**Property A is necessary but not sufficient. Property B is where AlphaDev's leverage actually lives.**

---

### 2. AlphaDev case study — both A and B hold

AlphaDev (Mankowitz et al., Nature 2023) uses single-player AlphaZero to search for short, correct assembly programs for fixed-size sorts. Its state is `S_t = ⟨P_t, Z_t⟩` where `P_t` is the partial program and `Z_t` is the memory + register state after actually executing `P_t` on a bank of test inputs.

**Property A holds by construction.** A partial assembly program can be executed — you just run the instructions that exist. This produces a concrete `Z_t` for every partial state.

**Property B holds because of the domain:**
- **B1:** an assembly instruction like `MOV mem[3], R1` modifies one memory cell and nothing else.
- **B2:** the target ("array is sorted") decomposes across N test cases × array positions. Position 3 of test case 5 can be sorted by one instruction sequence, position 2 of test case 7 by a disjoint one, with no interaction.
- **B3:** an instruction that gets one more position into the correct final value is a monotone forward step; nothing later on a correct path needs to undo it.

The reward function they use is a direct consequence of B, not an independent design choice:

```python
def correctness_reward(self) -> float:
    correct_items = 0
    for output, expected in zip(state.memory, expected_outputs):
      correct_items += output.weight * sum(
          output[i] == expected[i] for i in range(len(output))
      )
    reward = self.task_spec.correctness_reward_weight * (
        correct_items - self.previous_correct_items
    )
    self.previous_correct_items = correct_items
    ...
```

The reward at step `t` is the *increment* in correctly-placed positions across all test cases. Because B holds, this increment is cleanly attributable to the instruction just taken. The value network is asked to predict "how many correctly-placed positions will this partial program accumulate by the end," which is smooth and learnable because the target itself decomposes over independent components.

Dense reward is the *consequence* of compositional structure, not an independent knob you can turn.

---

### 3. Representation 1 — grammar growth (v1–v9). Neither A nor B.

The original setup: state is a partial expression tree with `NONTERMINAL` (`?`) slots. Actions fill one slot with an operator or terminal. Reward is SSE of the final expression after all slots are filled and constants are fit by lmfit.

**Property A fails.** A partial tree like `((?+?)+(?*?))` has no output. You cannot evaluate an expression with unfilled slots. The only ways to attach any number to it are:
1. Marginalize over all completions and take mean/best SSE — the "expected value under a random policy" that AlphaZero's value network is theoretically supposed to predict.
2. Look up the ground-truth best-achievable SSE from that partial state via exhaustive enumeration (v8's approach — worked once, doesn't scale).

Route 1 is what `diagnostic_random_rollout_values.py` measured. Result: on partial states, "average random completion SSE" correlates with "best achievable SSE" at only **0.506**. On the optimal path at depth 6, the true-optimal state ranks 25th/256 by average completion SSE. There is essentially no signal to train a value network on. This is why v1–v9 all plateaued around SSE 76.

**Property B also fails**, even if we somehow patched A:
- **B1 fails:** filling the root position of a depth-3 tree with `ADD` vs `MUL` selects between two entirely different functional classes for the whole expression. One "action" at the root determines the shape of the rest of the computation.
- **B2 fails:** the target is a single function evaluated at N x-points. There's no partition of the target into components that different subtree-fills can independently address.
- **B3 partially holds** in the narrow sense that decisions don't get overwritten (once a slot is filled, it stays filled), but this is defeated in practice by the value-landscape non-smoothness: nearby choices in the search tree can produce 17× SSE differences (VALUE_TARGET_PROBLEM.md), so the reward-per-step signal is dominated by noise.

**Diagnosis:** The grammar-growth representation was structurally hopeless for AlphaZero. The value function is definable only as a marginal over unknowns, that marginal is nearly flat across states in this domain, and the action space doesn't decompose the target. All the tweaks in v1–v9 (bootstrapping, GNN encoders, extreme exploration, oracle features) were rearranging deck chairs.

---

### 4. Representation 2 — GP mutation (July 2026). A holds, B fails.

The reframe: state is a *complete* expression with a definite SSE. Actions are subtree mutations. Explicitly designed to give us Property A.

**Property A now holds.** Every state has a concrete SSE. Every mutation produces a well-defined SSE delta. The "value function is a marginal over unknowns" problem is gone.

`diagnostic_gp_rollout_bootstrap.py` (`GP_ROLLOUT_BOOTSTRAP_ANALYSIS.md`) confirmed that short-K rollout stats predict long-K rollout stats at essentially the noise ceiling for two of three mutation regimes. Interpreted narrowly, that's the "the value target is horizon-stable" property AlphaZero's V is supposed to satisfy.

**But Property B fails on all three sub-conditions, and this is what kills the approach:**

- **B1 fails.** A subtree mutation at position 2 rewrites up to 4 nodes at once, taking `(x+c)*x` (which is `x²+cx`) to `(x*x)*x` (which is `x³`). Different functional class. There is no "small, identifiable slice of state" being modified — subtree replacement is a substantial rewrite of the expression's semantics.
- **B2 fails.** As with representation 1, the target is a single function evaluated at N points. There is no partition of the target that different mutations can independently address. The residual at x=1 depends on the whole expression; changing one subtree changes it via a nonlinear function of everything else in the tree.
- **B3 fails.** A mutation that improves SSE by 30% can be entirely undone by the next mutation, since the next mutation can overwrite the same subtree (or a subtree containing it). Sequences of GP mutations don't compose; they compete. Progress does not accumulate on a search trajectory the way instruction sequences do in AlphaDev.

So even though the SSE-delta signal is dense and immediate — one number per action — that signal doesn't carry the *compositional* information a value network needs to generalize. It tells you "did the last mutation help globally," not "did the last mutation move some independent component of the target closer to correct."

**In hindsight, the bootstrap-ability result was misleading.** With 512 states and ~34-neighbor connectivity, mutation balls cover 50%+ of the space in 3 steps. Under fast mixing, `mean-best-SSE-in-K-steps-from-S` at K=5 and K=40 both largely reflect "which local basin S sits in" rather than "how does the value function traverse across the state space." Horizon-stability is close to tautological at this scale and probably weakens sharply at depth 5. Even if it survived depth 5, it would still be a property of the wrong search algorithm — one whose action space fails B.

---

### 5. Why "add dense per-position reward to GP" would not fix it

An obvious question: SSE is `Σᵢ (yᵢ − f(xᵢ))²`, which is a sum over N data points. That's a per-data-point signal, just like AlphaDev's per-position correctness. Why doesn't that give us B2 for free?

Because B2 requires the sub-goals to be *independently attackable*. In AlphaDev, position 3 of test case 5 can be sorted by an instruction sequence disjoint from the one that sorts position 2 of test case 7. In SR, the residual at x=1 and the residual at x=2 are outputs of the *same function* evaluated at two points. There is no action that changes r(x=1) without also changing r(x=2), because there is only one function.

Per-data-point residual is a decomposable *signal*, but the *actions* don't decompose over it. The signal decomposition is meaningless without a matching action decomposition.

This is the deep reason SR is structurally harder than sorting. The output of a sorting program is an array; different program sub-computations can populate different array elements. The output of an SR expression is a function; there is no meaningful "sub-computation for x=1 vs sub-computation for x=2."

---

### 6. If we want to keep MCTS + value-network: additive fragment composition

Properties A and B were derived from what AlphaZero's specific machinery needs — MCTS growing a search tree by expanding partial states, and a value network bootstrapping estimates *through* those partial states. If we're committed to staying in that framework, the lesson isn't "add better shaping" or "use a bigger network." It's that we need a state × action × reward triple where **actions modify the target-in-progress additively or otherwise compositionally**, so that credit assignment works.

For SR, the most obvious candidate structure is **additive composition of expressions**: `f(x) = g₁(x) + g₂(x) + ... + g_k(x)` where each `gᵢ` is drawn from some library of primitives or fragments. Under this composition:
- **B1 holds:** adding fragment `g_{k+1}` changes the function additively, without disturbing what previous fragments compute.
- **B2 holds:** the residual `r(x) = y(x) − Σᵢ gᵢ(x)` is what remains to be explained, and different fragments can be responsible for different features of `r` (broad shape, oscillations, offsets, tail behavior). Not perfectly independent — fragments still interact through the residual — but far more independent than under mutation.
- **B3 holds:** a well-chosen fragment addition reduces the residual and is not something the next step needs to overwrite.

This is essentially boosting, matching pursuit, or basis expansion for SR. The neural role becomes "which fragment best fits the current residual, conditioned on (X, r)" — a much cleaner supervised signal than "what's the value of this partial tree."

Non-additive fragment composition (fragment-substitution-into-holes) does not obviously satisfy B. Filling one hole with a fragment can radically change what the enclosing expression computes — that's structurally the same failure as GP mutation. So "fragments" alone don't rescue us; it's *additive* fragments (or another cleanly compositional operator) that do.

**Honest caveat about additive composition for SR.** Physics-style SR targets — the ones we ultimately care about — are typically compact and deeply nested (`sin(ωx + φ)`, `1/(1 + exp(−x))`, `mc²`), not additive combinations of many basis functions. Fourier / basis expansions can approximate anything given enough terms, but that's a phenomenological fit, not a structural discovery. Forcing SR into an additive framework to satisfy MCTS's requirements is fighting the target's natural shape. Which is what motivates the next section.

---

### 7. But A and B are specific to MCTS + value-network, not to neural-guided search generally

The A+B framework was derived from what MCTS+value-net specifically needs, because that machinery bootstraps value estimates *through* partial states. Other neural-guided combinatorial-search paradigms have weaker requirements — they don't need partial states to be evaluable or credit to be assignable per-action.

**Policy-gradient generation (DSR-style).** Autoregressive generation of expressions token-by-token (or fragment-by-fragment) following the grammar's production rules, trained with REINFORCE or a risk-seeking policy gradient on the reward at completion. Requirements:
- Well-defined reward at completed states — ✓ (SSE after constant fitting).
- Partial states evaluable — **not required.** Prefixes are just intermediate points in a generation trajectory. They don't need semantic evaluation; they need only to be conditionable inputs for the policy network's next-token distribution.
- Compositional credit assignment — **not required.** REINFORCE assigns credit uniformly across the trajectory that produced a given reward. High-variance but works if you sample enough trajectories and use variance-reduction tricks (baseline, risk-seeking rank truncation).

DSR (Petersen et al. 2020, "Deep Symbolic Regression: Recovering Mathematical Expressions from Data via Risk-Seeking Policy Gradients") is exactly this. It works empirically on physics-style Feynman-benchmark SR problems with deeply-nested compact answers — the domain where additive expansion is unnatural. Hierarchy is generated top-down naturally because the grammar generates hierarchical trees; the policy just learns which productions are likely useful given the data. Fragments enter as multi-token library entries the policy learns to emit as compact units.

**Beam search with a scoring network.** Ranks partial expressions by a learned scorer, prunes low scorers, expands high scorers. Weaker than A+B (the scorer is a heuristic, not a bootstrapped value estimate) but stronger than pure autoregressive (uses the score to prune, not just to sample).

**Latent-space methods (Grammar VAE and successors).** Encode expressions into a continuous latent space, optimize in latent space (BO or gradient descent), decode. No partial-state semantics required at all. Mixed track record because decoder outputs often violate grammar or produce pathological expressions.

**The hierarchical-vs-additive tension dissolves under policy-gradient generation.** With DSR-style generation, the expression is produced hierarchically (following grammar production rules), the target can be a compact deeply-nested expression, and no additive-decomposition constraint is ever imposed. The neural network learns *which productions are likely useful given (X, y)*, which is a direct match for the "extensive menu based on experience" intuition. A grammar with rich hierarchical structure and a growing fragment vocabulary is not a hindrance — it's the search space, and the policy learns to navigate it.

**What the A+B analysis is still doing for us.** It explains why v1–v9 didn't work and why any variant that keeps the AlphaZero training loop won't work — SR under any action space we've considered doesn't satisfy B. That's a useful negative result. It rules out a large class of "just try another AlphaZero variant" proposals cleanly, and points at two disjoint directions forward:
1. Change the representation to satisfy A+B (additive fragments) — makes MCTS work but distorts the target's natural shape toward additive decompositions.
2. Change the algorithm to one that doesn't require A+B (DSR-style policy gradient) — keeps grammar and hierarchy intact, matches physics-SR's compact-nested character.

For a project whose goal is *interpretable equations, not black-box approximation*, direction 2 is the more natural match.

---

### 8. Summary table

| Property | AlphaDev | Grammar growth (v1–v9) | GP mutation | Additive fragments | DSR-style policy gradient |
|---|---|---|---|---|---|
| Algorithm class | MCTS + value | MCTS + value | MCTS + value | MCTS + value | Autoregressive + REINFORCE |
| A: Evaluable partial states | ✓ | ✗ (?-slots have no output) | ✓ (complete expr) | ✓ (partial sum evaluable) | N/A (not required) |
| B1: Local actions | ✓ | ✗ | ✗ (subtree rewrite) | ✓ | N/A |
| B2: Independent sub-goals | ✓ (N×positions) | ✗ (single function) | ✗ (single function) | ~ (fragments → residual features) | N/A |
| B3: Progress accumulates | ✓ | ~ (noise dominates) | ✗ (mutations undo) | ✓ | N/A |
| Requires only reward-at-completion | | | | | ✓ |
| Preserves hierarchical / compact targets | (assembly is compositional, not hierarchical in the SR sense) | ✓ (but doesn't work) | ✓ (but doesn't work) | ✗ (biases toward additive expansions) | ✓ |
| Observed outcome | Discovered faster sorts (LLVM merged) | Plateau at SSE 76 (v1–v9) | Bootstrap-ability positive but algorithm structurally wrong | Untested | Established for physics-SR benchmarks (external) |

---

### 9. Takeaways

1. **The 0.51 rollout diagnostic and the AlphaDev comparison are the same finding.** Both say: partial states in our SR setup do not carry credit-assignable signal, either because they can't be evaluated at all (grammar growth) or because the actions that produced them aren't compositional over the target (GP mutation).

2. **A+B are MCTS+value-net requirements, not universal search requirements.** They apply when you're bootstrapping value estimates through partial states. Policy-gradient methods, beam search with scoring nets, and latent-space methods have weaker requirements.

3. **Evaluable partial states are necessary but not sufficient — for MCTS.** The GP reframe fixed the wrong problem *given that we were still trying to use MCTS*. We had A; we still needed B; and getting B for SR requires a representation distortion (additive expansion) that fights the natural shape of physics-SR targets.

4. **Domain determines algorithm suitability.** AlphaZero works in Go, chess, shogi, sorting because those domains satisfy A+B natively. Symbolic regression does not, under either the growth or mutation action space. Continuing to iterate on AlphaZero training tricks was hitting a structural wall.

5. **Two coherent paths forward, not one:**
   - **(a) Change the representation to satisfy A+B** — additive fragment composition — and keep MCTS+value. Works for additively-decomposable targets, distorts compact nested ones.
   - **(b) Change the algorithm** — DSR-style policy gradient over a hierarchical grammar with a growing fragment vocabulary — and drop the A+B requirement. Preserves grammar, hierarchy, compactness, and matches the project goal of interpretable structural discoveries.

6. **For this project's stated goal (interpretable equations, not black-box approximation), path (b) is the more natural match.** Physics-style targets are usually compact and nested, not additive expansions. Path (a) would work but at the cost of forcing every discovery through a basis-expansion lens.

---

**Related documents:**
- `WHY_ALPHAZERO_MIGHT_BE_FUNDAMENTALLY_BROKEN.md` — earlier framing, now superseded
- `RANDOM_ROLLOUT_ANALYSIS.md` — the 0.51 diagnostic for grammar growth
- `BREAKTHROUGH_PARTIAL_STATES.md` — value fn can be learned under supervision (representation isn't the blocker)
- `GP_ROLLOUT_BOOTSTRAP_ANALYSIS.md` — bootstrap-ability under GP mutation (positive but misleading)
- `RESULTS_V9_ANALYSIS.md` — final AlphaZero-family experiment
