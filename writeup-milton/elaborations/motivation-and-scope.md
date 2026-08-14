# Elaboration — Motivation and scope: the MDP behind $V^*(s)$

Companion note to §1.2 of `../writeup.tex` (eq. `eq:vstar`). We make explicit
the decision process in which the identity
$V^*(s)=\max_{\pi\in\mathcal E(s)}R(\pi)$ holds, and show why the standard
Bellman optimality equation collapses to a maximum over reachable leaves.
Symbols follow §1.3 (`subsec:notation`) of the write-up throughout.

---

## 1. The grammar-based construction as an episodic MDP

Grammar-based construction of a mathematical expression is an episodic MDP

$$\mathcal M=\langle\mathcal S,\mathcal A,T,R\rangle,$$

with the following components.

1. **State space $\mathcal S$.** A state $s\in\mathcal S$ is a partial
   derivation — a sentential form still holding at least one nonterminal
   (e.g. `+ S S`).
2. **Terminal states $\mathcal S_T\subset\mathcal S$.** A state is terminal
   when no nonterminal remains. Terminal states are denoted $\pi$, matching
   the write-up, and $\Pi=\mathcal E(s_0)$ is the set of all of them.
3. **Action space $\mathcal A$.** An action $a\in\mathcal A(s)$ names a
   *pair*: which buffer cell to rewrite and which production to write there,
   $a=\mathrm{pos}\cdot P+\mathrm{prod}$. It does **not** address "the next
   open slot" — when a form holds several nonterminals the agent chooses
   which one to expand as well as how.
4. **Transition map $T:\{(s,a):s\in\mathcal S\setminus\mathcal S_T,\;a\in\mathcal A(s)\}\to\mathcal S$.**
   Transitions are deterministic: taking $a$ in $s$ leads to a unique
   successor $s'=T(s,a)$. Because an action names a position, a state may be
   reached by *several* distinct action sequences — expanding the two
   nonterminals of `+ S S` in either order lands on the same form — so the
   reachable set is a **directed acyclic graph, not a tree**. It is acyclic
   because every action either lengthens the form or removes a nonterminal.
5. **Reward map $R:\mathcal S_T\to[-1,1]$.** Rewards are sparse and episodic:
   - $r(s,a,s')=0$ for every nonterminal transition;
   - reaching a terminal state $\pi$ yields the final reward $R(\pi)$ (in the
     write-up, the clipped $R^2$ of the fitted expression,
     eq. `eq:code-reward`), with $-1$ for an unparseable expression or a
     failed solve.

The DAG-versus-tree distinction is the reason the two experiments count
different things. Experiment 1 runs a backward induction over the $4{,}898$
reachable **states**, visiting each once. Experiment 2 runs MCTS, which builds
a tree over **action sequences**, so one sentential form may occupy several
nodes of that tree.

## 2. The reachable-completion set $\mathcal E(s)$

Because the MDP is deterministic and acyclic, every state $s$ determines a
well-defined subset of terminal states reachable from it. Define

$$\mathcal E:\mathcal S\to 2^{\mathcal S_T},\qquad
\mathcal E(s)=\{\pi\in\mathcal S_T:\text{some legal action sequence leads from }s\text{ to }\pi\}.$$

Equivalently, $\mathcal E(s)$ is the set of sinks reachable from $s$ in the
DAG. For a terminal state, $\mathcal E(\pi)=\{\pi\}$.

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

— eq. `eq:vstar` of the write-up. Acyclicity is what makes the unrolling
terminate; tree structure is not needed, and a state reachable by two action
sequences contributes the same value either way, since $V^*$ is a function of
the state alone. The optimal policy simply traverses a path to a
highest-reward leaf, and $V^*(s)$ is the best-case scenario: the reward
obtained by playing perfectly from $s$.

This is the quantity symbolic-regression search should optimize, and it is
*not* what mean-rollout evaluation estimates: the contrast with
$V^q(s)=\mathbb E_{\pi\sim q(\cdot\mid s)}[R(\pi)]$ (eq. `eq:vroll`, uniform
completion policy $q(a\mid s)=1/|\mathcal A(s)|$ over the flattened
$(\mathrm{pos},\mathrm{prod})$ mask) is the central distinction of the study.
