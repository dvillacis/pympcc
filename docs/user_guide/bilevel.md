# Bilevel KKT-emitter frontend

A bilevel program

$$
\begin{aligned}
\min_{x, y} \quad & F(x, y) \\
\text{s.t.}      \quad & y \in \argmin_y \{\, f(x, y) \;:\; g(x, y) \le 0,\ h(x, y) = 0 \,\}
\end{aligned}
$$

becomes an MPCC by replacing the lower-level $\argmin$ with its KKT system:

$$
\begin{aligned}
& \nabla_y f(x, y) + \sum_i \lambda_i \nabla_y g_i(x, y) + \sum_k \mu_k \nabla_y h_k(x, y) = 0 \\
& h(x, y) = 0 \\
& \lambda \ge 0,\ -g(x, y) \ge 0,\ \lambda \perp -g(x, y).
\end{aligned}
$$

`pympcc.bilevel.from_lower_level` does this rewrite automatically.

## API

```python
from pympcc.bilevel import from_lower_level

problem = from_lower_level(
    n_x=2, n_y=2,
    x0=np.zeros(2), y0=np.zeros(2),
    f_upper=lambda x, y: ...,    # JAX-traceable
    f_lower=lambda x, y: ...,    # JAX-traceable
    n_g_lower=1,
    g_lower=lambda x, y: ...,    # (n_g_lower,)
    n_h_lower=0,
    h_lower=None,
    derivatives="jax",           # or "fd"
    xl=None, xu=None, yl=None, yu=None,
    lambda0=None, mu0=None,
)
result = pympcc.solve(problem, strategy="scholtes")
```

## Variable layout

The emitted MPCC has

$$
z = [\, x_\text{upper}\ (n_x) \;\mid\; y_\text{lower}\ (n_y) \;\mid\; \lambda\ (n_{g,\text{lower}}) \;\mid\; \mu\ (n_{h,\text{lower}}) \,].
$$

The $\lambda$ block is automatically lower-bounded at $0$; $\mu$ stays free.

## When KKT emission is valid

KKT validity assumes lower-level convexity (or at least a stationary lower optimum). For non-convex lower-level problems the resulting MPCC characterises *stationary points* of the lower problem rather than its global $\argmin$. If the lower level lacks inequality constraints there is no complementarity to emit — solve it as a regular NLP instead.
