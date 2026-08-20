# Uniform random search on the linear family

`ADDITIVE_GRAMMAR`, $L = 12$, $\tau = 10^{-6}$, grid $x \in [-1,1]$, $n = 41$. Budgets $B \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$.

$R_{\max}(B)$ is the best reward in $B$ i.i.d. draws from the uniform completion policy $q$. The distribution is **exact**: $q$'s law over the 247 terminals is computed by forward propagation, so $P(R_{\max}(B) \le r) = F(r)^B$ in closed form. The `MC` column is an independent check on that algebra, not the source of the numbers; see [Validation](#validation).

## Rarity and the sample complexity of guessing

$K_{1-\delta} = \lceil \log \delta / \log(1-\rho) \rceil$ (writeup eq. 10).

| target | $y(x)$ | trap level | $P(\text{1 draw} > \text{trap})$ | $\rho(s_0)$ | $K_{50}$ | $K_{95}$ | $K_{99}$ | $\rho(C)$ | $K_{95}(C)$ |
|---|---|---|---|---|---|---|---|---|---|
| `lin_A` | $0.5 + 2.0*x$ | +0.8214 | 0.1299 | 0.0986 | 7 | 29 | 45 | 0.3945 | 6 |
| `lin_B` | $5.0 + 1.0*x$ | +0.0000 | 0.0986 | 0.0986 | 7 | 29 | 45 | 0.3945 | 6 |
| `lin_C` | $1000.0 + 0.001*x$ | +0.0000 | 0.0986 | 0.0986 | 7 | 29 | 45 | 0.3945 | 6 |
| `lin_D` | $0.0 + 2.0*x$ | +1.0000 | 0.0000 | 0.3981 | 2 | 6 | 10 | 0.5924 | 4 |

## Exact recovery probability across the budget grid

| target | $B=1$ | $B=2$ | $B=4$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ | $B=128$ | $B=256$ | $B=512$ | $B=1024$ | $B=2048$ | $B=4096$ | $B=8192$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lin_A` | 0.0986 | 0.1875 | 0.3399 | 0.5643 | 0.8101 | 0.9640 | 0.9987 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_B` | 0.0986 | 0.1875 | 0.3399 | 0.5643 | 0.8101 | 0.9640 | 0.9987 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_C` | 0.0986 | 0.1875 | 0.3399 | 0.5643 | 0.8101 | 0.9640 | 0.9987 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_D` | 0.3981 | 0.6377 | 0.8688 | 0.9828 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## $E[R_{\max}(B)]$

| target | $B=1$ | $B=2$ | $B=4$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ | $B=128$ | $B=256$ | $B=512$ | $B=1024$ | $B=2048$ | $B=4096$ | $B=8192$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lin_A` | 0.3267 | 0.5572 | 0.7826 | 0.9187 | 0.9740 | 0.9960 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_B` | -0.4665 | -0.1318 | 0.2379 | 0.5539 | 0.8100 | 0.9640 | 0.9987 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_C` | -0.4665 | -0.1318 | 0.2379 | 0.5539 | 0.8100 | 0.9640 | 0.9987 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `lin_D` | 0.3981 | 0.6377 | 0.8688 | 0.9828 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Design check

Against the writeup's own budget grid $\{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$.

| target | $K_{95}(s_0)$ | grid points $\ge K_{95}$ | $P(\text{exact})$ at $B=16$ | at $B=8192$ | informative $B$ |
|---|---|---|---|---|---|
| `lin_A` | 29 | 9/10 | 0.8101 | 1.000000 | 1--29 |
| `lin_B` | 29 | 9/10 | 0.8101 | 1.000000 | 1--29 |
| `lin_C` | 29 | 9/10 | 0.8101 | 1.000000 | 1--29 |
| `lin_D` | 6 | 10/10 | 0.9997 | 1.000000 | 1--6 |

A grid point at or above $K_{95}$ is one where *random guessing alone* already recovers the target at least 95% of the time, so no search method can be distinguished from any other there. The informative window is the range of $B$ over which $P(\text{exact}) \in [0.05, 0.95]$ -- outside it every method scores the same and $\Delta_{\text{semantic}}(B)$ is squeezed toward zero by the ceiling, not by the search rule.

## Validation

- Closed form vs Monte Carlo (2000 replicates) over 56 $(\text{target}, B)$ cells: 1 falls outside the Wilson 95% interval (about 2.8 expected by chance), and **none** outside the 99.9% interval.
- $q$'s terminal law sums to 1 to within $10^{-12}$ at both roots (asserted in `q_terminal_distribution`).
- $\rho$ here is computed by *forward* propagation; `07` computes it by *backward* induction. The two agree.

| target | walks | TV distance | $\chi^2$ | dof | $p$ | $\hat\rho$ (walked) | $\rho$ (exact) |
|---|---|---|---|---|---|---|---|
| `lin_A` | 100000 | 0.0091 | 225.1 | 232 | 0.616 | 0.0986 | 0.0986 |
| `lin_B` | 100000 | 0.0088 | 228.3 | 232 | 0.556 | 0.0972 | 0.0986 |
| `lin_C` | 100000 | 0.0078 | 246.4 | 232 | 0.247 | 0.0991 | 0.0986 |
| `lin_D` | 100000 | 0.0086 | 222.9 | 232 | 0.655 | 0.3988 | 0.3981 |

End-to-end protocol check (real derivation walks, B=64, 200 seeds):

- `lin_A`: $\hat P(\text{exact}) = 1.000$ (95% CI 0.981--1.000, exact 0.9987)
- `lin_B`: $\hat P(\text{exact}) = 1.000$ (95% CI 0.981--1.000, exact 0.9987)
- `lin_C`: $\hat P(\text{exact}) = 1.000$ (95% CI 0.981--1.000, exact 0.9987)
- `lin_D`: $\hat P(\text{exact}) = 1.000$ (95% CI 0.981--1.000, exact 1.0000)
