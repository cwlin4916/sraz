# Exact root quantities, linear family

`ADDITIVE_GRAMMAR` (4 productions), $L = 12$, $\tau = 10^{-6}$, grid $x \in [-1,1]$, $n = 41$.

Reachable forms 4898, terminal 247, nonterminal 4651. Terminals by length: $\ell=1$: 1, $\ell=3$: 2, $\ell=5$: 5, $\ell=7$: 14, $\ell=9$: 48, $\ell=11$: 177.

Computed by exhaustive backward induction, then checked field-by-field against Table 2 of the writeup: **all fields agree**.

| target | y(x) | R(D1) | R(D2) | R(D3) | V*(C) | Vq(C) | rho(C) | margin | trap | rho(s0) |
|---|---|---|---|---|---|---|---|---|---|---|
| `lin_A` | $0.5 + 2.0*x$ | +0.0000 | +0.8214 | -0.0793 | 1.000000 | 0.5646 | 0.3945 | +0.2569 | trap | 0.0986 |
| `lin_B` | $5.0 + 1.0*x$ | +0.0000 | -1.0000 | -1.0000 | 1.000000 | 0.1341 | 0.3945 | -0.1341 | --- | 0.0986 |
| `lin_C` | $1000.0 + 0.001*x$ | +0.0000 | -1.0000 | -1.0000 | 1.000000 | 0.1341 | 0.3945 | -0.1341 | --- | 0.0986 |
| `lin_D` | $0.0 + 2.0*x$ | +0.0000 | +1.0000 | -0.0000 | 1.000000 | 0.5924 | 0.5924 | +0.4076 | --- | 0.3981 |

## Reward ceiling by terminal length

| target | $\ell=1$ | $\ell=3$ | $\ell=5$ | $\ell=7$ | $\ell=9$ | $\ell=11$ |
|---|---|---|---|---|---|---|
| `lin_A` | 0.000000 | 0.821429 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `lin_B` | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `lin_C` | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `lin_D` | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
