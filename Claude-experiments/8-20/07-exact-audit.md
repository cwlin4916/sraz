# Exact root quantities, linear family

`ADDITIVE_GRAMMAR` (4 productions), $L = 12$, $\tau = 10^{-6}$, grid $x \in [-1,1]$, $n = 41$.

Reachable forms 4898, terminal 247, nonterminal 4651. Terminals by length: $\ell=1$: 1, $\ell=3$: 2, $\ell=5$: 5, $\ell=7$: 14, $\ell=9$: 48, $\ell=11$: 177.

Computed by exhaustive backward induction, then checked field-by-field against Table 2 of the writeup: **all fields agree**.

| target | y(x) | R(D1) | R(D2) | R(D3) | V*(C) | Vq(C) | rho(C) | margin | trap | rho(s0) |
|---|---|---|---|---|---|---|---|---|---|---|
| `quad_A` | $1.0 + -1.0*x + 2.0*x**2$ | -0.0000 | -1.0000 | -0.0711 | 1.000000 | 0.3540 | 0.1128 | -0.3540 | --- | 0.0282 |
| `quad_B` | $6.0 + -5.0*x + 0.5*x**2$ | +0.0000 | -1.0000 | -1.0000 | 1.000000 | 0.1563 | 0.1128 | -0.1563 | --- | 0.0282 |
| `quad_C` | $0.001 + 0.001*x + 1000.0*x**2$ | +0.0000 | -1.0000 | +1.0000 | 1.000000 | 0.4840 | 0.5569 | +0.5160 | --- | 0.3892 |
| `quad_D` | $-0.48 + 0.4*x + 2.0*x**2$ | +0.0000 | +0.0170 | +0.6461 | 1.000000 | 0.5106 | 0.1128 | +0.1355 | trap | 0.0282 |

## Reward ceiling by terminal length

| target | $\ell=1$ | $\ell=3$ | $\ell=5$ | $\ell=7$ | $\ell=9$ | $\ell=11$ |
|---|---|---|---|---|---|---|
| `quad_A` | -0.000000 | 0.000000 | 0.472144 | 0.527856 | 0.527856 | 1.000000 |
| `quad_B` | 0.000000 | 0.000000 | 0.997213 | 0.997213 | 0.997213 | 1.000000 |
| `quad_C` | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `quad_D` | 0.000000 | 0.016991 | 0.646103 | 0.874804 | 0.874804 | 1.000000 |
