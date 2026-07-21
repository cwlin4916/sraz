# Elaboration — Motivation and scope: the MDP behind $V^*(s)$

Companion note to §1.1 of `../writeup.tex` (eq. `eq:vstar`). We make explicit
the decision process in which the identity
$V^*(s)=\max_{\pi\in\mathcal E(s)}R(\pi)$ holds, and show why the standard
Bellman optimality equation collapses to a maximum over reachable leaves.

---

## 1. The grammar-based construction as an episodic MDP

Grammar-based construction of a mathematical expression is an episodic MDP

$$\mathcal M=\langle\mathcal S,\mathcal A,T,R\rangle,$$

with the following components.

1. **State space $\mathcal S$.** A state $s\in\mathcal S$ is a partial
   derivation — an expression with unfilled slots (e.g. `_ + _`).
2. **Terminal states $\mathcal S_T\subset\mathcal S$.** A state is terminal
   when every slot is filled. Terminal states are denoted $\pi$, matching the
   writeup.
3. **Action space $\mathcal A$.** An action $a\in\mathcal A(s)$ chooses one
   atom (e.g. $\psi_1,\psi_2,\dots$; in a general grammar $x$, $x^2$, $\sin$)
   for the next open slot.
4. **Transition map $T:\{(s,a):s\in\mathcal S\setminus\mathcal S_T,\;a\in\mathcal A(s)\}\to\mathcal S$.**
   Transitions are deterministic: taking $a$ in $s$ leads to a unique
   successor $s'=T(s,a)$. Every state is reached by exactly one action
   sequence from the root, so the state space is a tree (a fortiori a DAG —
   no cycles).
5. **Reward map $R:\mathcal S_T\to[0,1]$.** Rewards are sparse and episodic:
   - $r(s,a,s')=0$ for every nonterminal transition;
   - reaching a terminal state $\pi$ yields the final reward $R(\pi)$ (in the
     writeup, the unpenalized $R^2$ of the fitted expression,
     eq. `eq:r2-reward`).

## 2. The reachable-completion set $\mathcal E(s)$

Because the MDP is a deterministic forward-moving tree, every intermediate
state $s$ (a partial expression) determines a well-defined subset of terminal
states reachable from it. Define

$$\mathcal E:\mathcal S\to 2^{\mathcal S_T},\qquad
\mathcal E(s)=\{\pi\in\mathcal S_T:\text{some valid action sequence leads from }s\text{ to }\pi\}.$$

Equivalently, $\mathcal E(s)$ is the leaf set of the subtree rooted at $s$.
For a terminal state, $\mathcal E(\pi)=\{\pi\}$.

## 3. Why $V^*(s)$ collapses to a maximum over $\mathcal E(s)$

In a general MDP the optimal value function
$V^*:\mathcal S\to\mathbb R$ satisfies the Bellman optimality equation

$$V^*(s)=\max_{a\in\mathcal A(s)}\sum_{s',r}p(s',r\mid s,a)\,[\,r+\gamma V^*(s')\,].$$

Three specializations hold here:

1. transitions are deterministic, so the sum over $(s',r)$ has a single term;
2. the episode is finite and undiscounted, $\gamma=1$;
3. all rewards are zero until the terminal step.

The equation therefore telescopes along any root-to-leaf path:

$$V^*(s)=\max_{a\in\mathcal A(s)}V^*(T(s,a)),\qquad V^*(\pi)=R(\pi)\ \ \text{for }\pi\in\mathcal S_T,$$

and unrolling the recursion to the leaves gives

$$\boxed{\;V^*(s)=\max_{\pi\in\mathcal E(s)}R(\pi)\;}$$

— eq. `eq:vstar` of the writeup. The optimal policy simply traverses the path
to the highest-reward leaf, and $V^*(s)$ is the best-case scenario: the reward
obtained by playing perfectly from $s$.

This is the quantity symbolic-regression search should optimize, and it is
*not* what mean-rollout evaluation estimates: the contrast with
$V^q(s)=\mathbb E_{\pi\sim q(\cdot\mid s)}[R(\pi)]$ (eq. `eq:vroll`, uniform
completion policy $q(a\mid s)=1/|\mathcal A(s)|$) is the central distinction
of the study.
