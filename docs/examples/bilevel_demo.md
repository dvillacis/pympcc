---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Bilevel KKT emission

`pympcc.bilevel.from_lower_level` compiles a bilevel program into an MPCC by replacing the lower-level $\argmin$ with its KKT system. This notebook walks through two small bilevels — one where the lower-level inequality is **inactive** at the upper optimum, and one where it is **active** — to show how the emitter handles each case.

```{code-cell} ipython3
import warnings
import numpy as np
import jax.numpy as jnp
import pympcc
from pympcc.bilevel import from_lower_level
```

## Case A — lower bound inactive

$$
\begin{aligned}
\min_{x, y} \quad & (x - 1)^2 + (y - 1)^2 \\
\text{s.t.} \quad & y \in \argmin_y \{\, (y - x)^2 \;:\; y \ge 0 \,\}.
\end{aligned}
$$

For $x \ge 0$ the lower-level problem is unconstrained at the optimum: $y = x$. The upper objective then reduces to $(x-1)^2 + (x-1)^2$, minimised at $x = 1$. So $(x^*, y^*) = (1, 1)$, $\lambda^* = 0$, $F^* = 0$. The complementarity pair $\lambda \perp -g(x, y) = y$ has $G_0 = \lambda = 0$ and $H_0 = y = 1$ — strictly $G$-active (the multiplier is the side that hits zero).

```{code-cell} ipython3
problem_A = from_lower_level(
    n_x=1, n_y=1,
    x0=np.array([0.5]), y0=np.array([0.5]),
    f_upper=lambda x, y: (x[0] - 1.0) ** 2 + (y[0] - 1.0) ** 2,
    f_lower=lambda x, y: (y[0] - x[0]) ** 2,
    n_g_lower=1,
    g_lower=lambda x, y: jnp.array([-y[0]]),
    derivatives="jax",
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    res_A = pympcc.solve(problem_A, strategy="scholtes")

print(f"x* = {res_A.x[0]:.4f},  y* = {res_A.x[1]:.4f},  λ* = {res_A.x[2]:.4f}")
print(f"F* = {res_A.obj:.4e}    (reference 0)")
print(f"comp pair: G={res_A.G[0]:.3e}, H={res_A.H[0]:.3e}  → {res_A.per_pair_status[0]}")
```

## Case B — lower bound active

Same form, different objective and lower cost. The lower problem now pulls $y$ toward zero, so the lower bound $y \ge 0$ binds at the optimum.

$$
\begin{aligned}
\min_{x, y} \quad & (x - 0.5)^2 + (y - 0.5)^2 \\
\text{s.t.} \quad & y \in \argmin_y \{\, y^2 + 2xy \;:\; y \ge 0 \,\}.
\end{aligned}
$$

For $x \ge 0$ the lower-level optimum is $y = 0$ (interior unconstrained min would be $y = -x \le 0$, infeasible). The upper objective reduces to $(x-0.5)^2 + 0.25$, minimised at $x = 0.5$. So $(x^*, y^*) = (0.5, 0)$, with the lower KKT multiplier $\lambda^* = 1$ on $y \ge 0$. The comp pair has $G_0 = \lambda = 1$ and $H_0 = y = 0$ — strictly $H$-active (the constraint, not the multiplier, is the side that hits zero).

```{code-cell} ipython3
problem_B = from_lower_level(
    n_x=1, n_y=1,
    x0=np.array([0.0]), y0=np.array([0.0]),
    lambda0=np.array([0.5]),
    f_upper=lambda x, y: (x[0] - 0.5) ** 2 + (y[0] - 0.5) ** 2,
    f_lower=lambda x, y: y[0] ** 2 + 2.0 * x[0] * y[0],
    n_g_lower=1,
    g_lower=lambda x, y: jnp.array([-y[0]]),
    derivatives="jax",
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    res_B = pympcc.solve(problem_B, strategy="scholtes")

print(f"x* = {res_B.x[0]:.4f},  y* = {res_B.x[1]:.4f},  λ* = {res_B.x[2]:.4f}")
print(f"F* = {res_B.obj:.4f}    (reference 0.25)")
print(f"comp pair: G={res_B.G[0]:.3e}, H={res_B.H[0]:.3e}  → {res_B.per_pair_status[0]}")
```

## What the emitter actually built

The variable layout of the emitted MPCC is `z = [x | y | λ | μ]`. For Case A, $n_x = n_y = 1$, $n_g = 1$, $n_h = 0$:

```{code-cell} ipython3
print(f"n           = {problem_A.n}")
print(f"n_comp      = {problem_A.n_comp}")
print(f"n_eq        = {problem_A.n_eq}")
print(f"x0 (packed) = {problem_A.x0}      # [x_init, y_init, λ_init]")
print(f"xl          = {problem_A.xl}      # λ ≥ 0 enforced as a bound")
```

The single equality is the lower-level stationarity row $\nabla_y f + \lambda \nabla_y g = 0$, which for Case A reads $2(y - x) - \lambda = 0$. The single complementarity pair is $\lambda \perp -g(x, y) = y$.

```{code-cell} ipython3
print("Equality residual at optimum:",
      problem_A.eq_constraints(res_A.x))
```
