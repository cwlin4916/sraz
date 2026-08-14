# Elaboration (superseded) — the $\lambda$-family: why $\mathrm{Var}(y_\lambda)=1$ exactly

> **Status: superseded, retained for reference.** This note documents the
> *synthetic* controlled family — a Gram–Schmidt orthonormal atom basis
> $\psi_1,\dots,\psi_{12}$, a target $y_\lambda$ mixing them by a tunable
> $\lambda$, and a **single** decoy branch $D$ with analytic reward
> $R(D)=\lambda$. The write-up has since replaced that design with the shipped
> `ADDITIVE_GRAMMAR`'s own root structure: **three** decoy states
> $D_1,D_2,D_3$ and one continue state $C$ (`eq:root-decoys`), scored against
> eight chosen polynomial targets on $[-1,1]$ (`def:target-family`). Nothing
> below is referenced by the current `../writeup.tex`, and its equation labels
> (`eq:target-family`, `eq:target-normalization`, `eq:orthonormal-basis`,
> `eq:decoy-reward`, `eq:analytic-terminal-reward`) no longer exist there.
> Figures of the same vintage — `../figures/scenario_strong.png`,
> `scenario_weak.png`, `targets_lambda.png`, produced by
> `../figures/make_scenario_figures.py` — are likewise orphaned.
>
> Read it as a record of the synthetic design, not as a companion to any
> current section. In particular its $D$ is a single branch and is unrelated to
> the states $D_1,D_2,D_3$ of the current write-up.

The claim that the target $y_\lambda$ has empirical variance exactly $1$ relies
entirely on the geometric properties established by the Gram–Schmidt
construction of the same design.

**Constants.** $n=41$ equally spaced points $x_1,\dots,x_n\in[-1,1]$;
empirical inner product $\langle f,g\rangle_n=\frac1n\sum_{i=1}^n f(x_i)g(x_i)$;
orthonormal basis $\psi_1,\dots,\psi_{12}$; mixing weight $\lambda\in(0,1)$.

---

## 1. The three geometric inputs

The Gram–Schmidt construction gives the basis
functions three properties, each used below:

1. **Zero mean** — $\langle\psi_j,1\rangle_n=0$: each $\psi_j$ is centered.
2. **Unit length** — $\langle\psi_j,\psi_j\rangle_n=1$: each $\psi_j$
   individually has empirical variance $1$.
3. **Orthogonality** — $\langle\psi_j,\psi_k\rangle_n=0$ for $j\neq k$: the
   basis functions are entirely uncorrelated in the sample.

The target is

$$y_\lambda=\sqrt{\lambda}\,\psi_1
+\sqrt{\tfrac{1-\lambda}{3}}\,\psi_2
+\sqrt{\tfrac{1-\lambda}{3}}\,\psi_3
+\sqrt{\tfrac{1-\lambda}{3}}\,\psi_4.$$

## 2. Proposition (unit target variance)

**Proposition.** For every $\lambda\in(0,1)$,

$$\boxed{\;\overline y_\lambda=\langle y_\lambda,1\rangle_n=0,
\qquad
\mathrm{Var}_n(y_\lambda)=\langle y_\lambda,y_\lambda\rangle_n=1.\;}$$

*Proof.*

**Step A — the mean of $y_\lambda$ is zero.** The empirical mean is the inner
product with the constant function $1$; by linearity and zero mean of each
$\psi_j$,

$$\overline y_\lambda=\langle y_\lambda,1\rangle_n
=\sqrt{\lambda}\,\langle\psi_1,1\rangle_n
+\sqrt{\tfrac{1-\lambda}{3}}\sum_{m=2}^{4}\langle\psi_m,1\rangle_n
=0+0+0+0=0.$$

**Step B — the variance is the squared norm.** Because the mean is zero, the
empirical variance is the inner product of the function with itself,

$$\mathrm{Var}_n(y_\lambda)=\langle y_\lambda,y_\lambda\rangle_n.$$

Expanding $\langle y_\lambda,y_\lambda\rangle_n$ produces $4\times4=16$
terms. Orthogonality makes every cross-term (such as
$\langle\psi_1,\psi_2\rangle_n$) strictly zero, leaving only the four squared
terms:

$$\langle y_\lambda,y_\lambda\rangle_n
=(\sqrt{\lambda})^2\langle\psi_1,\psi_1\rangle_n
+\left(\sqrt{\tfrac{1-\lambda}{3}}\right)^2\langle\psi_2,\psi_2\rangle_n
+\left(\sqrt{\tfrac{1-\lambda}{3}}\right)^2\langle\psi_3,\psi_3\rangle_n
+\left(\sqrt{\tfrac{1-\lambda}{3}}\right)^2\langle\psi_4,\psi_4\rangle_n.$$

Unit length reduces every $\langle\psi_j,\psi_j\rangle_n$ to $1$, so the
expression collapses to the sum of the squared coefficients:

$$\mathrm{Var}_n(y_\lambda)
=\lambda\cdot1+\tfrac{1-\lambda}{3}\cdot1+\tfrac{1-\lambda}{3}\cdot1+\tfrac{1-\lambda}{3}\cdot1
=\lambda+3\cdot\tfrac{1-\lambda}{3}
=\lambda+1-\lambda=1.\qquad\blacksquare$$

## 3. Remark

The identity is the Pythagorean theorem in the $n$-dimensional geometry of
$\langle\cdot,\cdot\rangle_n$: the squared length of $y_\lambda$ is the sum of
the squared lengths of its orthogonal components. It is exactly this
decomposition that made $\lambda$ the fraction of target variance carried by
$\psi_1$ and $\frac{1-\lambda}{3}$ the share carried by each of
$\psi_2,\psi_3,\psi_4$ — the calibration on which the single decoy's reward
$R(D)=\lambda$ and the analytic terminal reward of that design depended.
