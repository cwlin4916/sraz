# Uniform random search on the quadratic family

`ADDITIVE_GRAMMAR`, $L = 12$, $\tau = 10^{-6}$, grid $x \in [-1,1]$, $n = 41$. Budgets $B \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$.

$R_{\max}(B)$ is the best reward in $B$ i.i.d. draws from the uniform completion policy $q$. The distribution is **exact**: $q$'s law over the 247 terminals is computed by forward propagation, so $P(R_{\max}(B) \le r) = F(r)^B$ in closed form. The `MC` column is an independent check on that algebra, not the source of the numbers; see [Validation](#validation).

## Rarity and the sample complexity of guessing

$K_{1-\delta} = \lceil \log \delta / \log(1-\rho) \rceil$ (writeup eq. 10).

| target | $y(x)$ | trap level | $P(\text{1 draw} > \text{trap})$ | $\rho(s_0)$ | $K_{50}$ | $K_{95}$ | $K_{99}$ | $\rho(C)$ | $K_{95}(C)$ |
|---|---|---|---|---|---|---|---|---|---|
| `quad_A` | $1.0 + -1.0*x + 2.0*x**2$ | -0.0000 | 0.1940 | 0.0282 | 25 | 105 | 161 | 0.1128 | 26 |
| `quad_B` | $6.0 + -5.0*x + 0.5*x**2$ | +0.0000 | 0.1628 | 0.0282 | 25 | 105 | 161 | 0.1128 | 26 |
| `quad_C` | $0.001 + 0.001*x + 1000.0*x**2$ | +1.0000 | 0.0000 | 0.3892 | 2 | 7 | 10 | 0.5569 | 4 |
| `quad_D` | $-0.48 + 0.4*x + 2.0*x**2$ | +0.6461 | 0.1236 | 0.0282 | 25 | 105 | 161 | 0.1128 | 26 |

## Exact recovery probability across the budget grid

| target | $B=1$ | $B=2$ | $B=4$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ | $B=128$ | $B=256$ | $B=512$ | $B=1024$ | $B=2048$ | $B=4096$ | $B=8192$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `quad_A` | 0.0282 | 0.0556 | 0.1082 | 0.2046 | 0.3674 | 0.5998 | 0.8398 | 0.9743 | 0.9993 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_B` | 0.0282 | 0.0556 | 0.1082 | 0.2046 | 0.3674 | 0.5998 | 0.8398 | 0.9743 | 0.9993 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_C` | 0.3892 | 0.6269 | 0.8608 | 0.9806 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_D` | 0.0282 | 0.0556 | 0.1082 | 0.2046 | 0.3674 | 0.5998 | 0.8398 | 0.9743 | 0.9993 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## $E[R_{\max}(B)]$

| target | $B=1$ | $B=2$ | $B=4$ | $B=8$ | $B=16$ | $B=32$ | $B=64$ | $B=128$ | $B=256$ | $B=512$ | $B=1024$ | $B=2048$ | $B=4096$ | $B=8192$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `quad_A` | -0.1793 | 0.1108 | 0.3264 | 0.5097 | 0.6726 | 0.8079 | 0.9243 | 0.9879 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_B` | -0.4609 | -0.1257 | 0.2414 | 0.5541 | 0.8092 | 0.9630 | 0.9983 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_C` | 0.1210 | 0.5550 | 0.8557 | 0.9806 | 0.9996 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `quad_D` | 0.2934 | 0.4732 | 0.6624 | 0.7983 | 0.8835 | 0.9434 | 0.9797 | 0.9968 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Design check

Against the writeup's own budget grid $\{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192\}$.

| target | $K_{95}(s_0)$ | grid points $\ge K_{95}$ | $P(\text{exact})$ at $B=16$ | at $B=8192$ | informative $B$ |
|---|---|---|---|---|---|
| `quad_A` | 105 | 7/10 | 0.3674 | 1.000000 | 2--105 |
| `quad_B` | 105 | 7/10 | 0.3674 | 1.000000 | 2--105 |
| `quad_C` | 7 | 10/10 | 0.9996 | 1.000000 | 1--7 |
| `quad_D` | 105 | 7/10 | 0.3674 | 1.000000 | 2--105 |

A grid point at or above $K_{95}$ is one where *random guessing alone* already recovers the target at least 95% of the time, so no search method can be distinguished from any other there. The informative window is the range of $B$ over which $P(\text{exact}) \in [0.05, 0.95]$ -- outside it every method scores the same and $\Delta_{\text{semantic}}(B)$ is squeezed toward zero by the ceiling, not by the search rule.

## Validation

- Closed form vs Monte Carlo (2000 replicates) over 56 $(\text{target}, B)$ cells: 1 falls outside the Wilson 95% interval (about 2.8 expected by chance), and **1** outside the 99.9% interval: quad_D protocol check B=64: exact 0.8398 outside walk CI
- $q$'s terminal law sums to 1 to within $10^{-12}$ at both roots (asserted in `q_terminal_distribution`).
- $\rho$ here is computed by *forward* propagation; `07` computes it by *backward* induction. The two agree.

| target | walks | TV distance | $\chi^2$ | dof | $p$ | $\hat\rho$ (walked) | $\rho$ (exact) |
|---|---|---|---|---|---|---|---|
| `quad_A` | 100000 | 0.0083 | 237.3 | 232 | 0.392 | 0.0286 | 0.0282 |
| `quad_B` | 100000 | 0.0078 | 237.0 | 232 | 0.396 | 0.0275 | 0.0282 |
| `quad_C` | 100000 | 0.0086 | 219.5 | 232 | 0.712 | 0.3881 | 0.3892 |
| `quad_D` | 100000 | 0.0090 | 234.0 | 232 | 0.451 | 0.0283 | 0.0282 |

End-to-end protocol check (real derivation walks, B=64, 200 seeds):

- `quad_A`: $\hat P(\text{exact}) = 0.875$ (95% CI 0.822--0.914, exact 0.8398)
- `quad_B`: $\hat P(\text{exact}) = 0.855$ (95% CI 0.800--0.897, exact 0.8398)
- `quad_C`: $\hat P(\text{exact}) = 1.000$ (95% CI 0.981--1.000, exact 1.0000)
- `quad_D`: $\hat P(\text{exact}) = 0.895$ (95% CI 0.845--0.930, exact 0.8398)
