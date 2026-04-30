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

# Differentiable solve with `jax.grad`

`pympcc.solve_jax` is registered as `jax.custom_vjp`, so JAX can differentiate through a converged MPCC. The forward calls IPOPT; the backward solves the KKT saddle once and contracts via `jax.vjp` on the user's parametric callables.

```{code-cell} ipython3
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import pympcc

jax.config.update("jax_enable_x64", True)
```

## A parametric branch-selection MPCC

Two complementarity pairs, $\theta$-shifted objective targets:

$$
\min_x \tfrac12 \lVert x - \theta \rVert^2 \quad \text{s.t.}\quad x_0 \ge 0 \perp x_2 \ge 0,\ x_1 \ge 0 \perp x_3 \ge 0.
$$

The active branch is determined by the sign of each $\theta$ coordinate: if $\theta_0 > 0$ then $x_0 = \theta_0$ and $x_2 = 0$; otherwise $x_0 = 0$ and $x_2 = -\theta_2$ (and similarly for indices 1, 3). The optimum varies *non-smoothly* in $\theta$ at branch boundaries — exactly where MPCC differentiation is interesting.

```{code-cell} ipython3
pmpcc = pympcc.ParametricMPCC(
    n=4, n_comp=2,
    objective=lambda x, theta: 0.5 * jnp.sum((x - theta) ** 2),
    comp_G=lambda x, theta: x[:2],
    comp_H=lambda x, theta: x[2:],
)
```

## Forward solve

```{code-cell} ipython3
theta0 = jnp.array([2.0, 0.0, 0.0, 3.0])
x_star = pympcc.solve_jax(pmpcc, theta0, x0=np.asarray(theta0))
print("x* =", np.asarray(x_star))
```

At $\theta = (2, 0, 0, 3)$ the active branches pin $x_1$ and $x_2$ to 0; the free coordinates take their unconstrained optima.

## Gradient via `jax.grad`

```{code-cell} ipython3
def loss(theta):
    x = pympcc.solve_jax(pmpcc, theta, x0=np.asarray(theta0))
    return 0.5 * jnp.sum(x ** 2)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    g = jax.grad(loss)(theta0)
print("∇_θ L =", np.asarray(g))
```

## Finite-difference verification

```{code-cell} ipython3
def fd_grad(loss_fn, theta, eps=1e-3):
    theta = np.asarray(theta, dtype=float)
    g = np.zeros_like(theta)
    for k in range(theta.size):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lp = float(loss_fn(jnp.asarray(tp)))
            lm = float(loss_fn(jnp.asarray(tm)))
        g[k] = (lp - lm) / (2 * eps)
    return g

g_fd = fd_grad(loss, theta0)
print("∇_θ L (FD) =", g_fd)
print("max abs error:", float(np.max(np.abs(np.asarray(g) - g_fd))))
```

## Full Jacobian via per-row `jax.grad`

`jax.jacrev` is **not** supported in Phase 1 (its internal `vmap` cannot trace the NumPy/IPOPT backward). Build the Jacobian one row at a time instead:

```{code-cell} ipython3
def x_i(theta, i):
    return pympcc.solve_jax(pmpcc, theta, x0=np.asarray(theta0))[i]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    rows = [jax.grad(lambda th, i=i: x_i(th, i))(theta0) for i in range(4)]
J = np.stack([np.asarray(r) for r in rows], axis=0)
print("dx*/dθ =")
print(J)
```

For this θ, the active branches give `x* = (θ_0, 0, 0, θ_3)`, so the Jacobian is `diag(1, 0, 0, 1)`.
