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

# MCP variable-paired form

Many MPCC pairs have the form $x_j \ge 0\ \perp\ F(x) \ge 0$ where the $G$-side is just a variable. Writing `comp_G(x) = x[var_idxs]` by hand is error-prone; `comp_var_pairs` declares the pairing at the variable level and the package fills in the identity Jacobian for $G$ automatically.

## Without `comp_var_pairs`

The traditional way — manually write `comp_G` as a slice and its Jacobian:

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

problem_a = pympcc.MPCCProblem(
    n=3, n_comp=2,
    x0=np.array([0.5, 0.5, 0.5]),
    xl=np.zeros(3),
    objective=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[2] - 0.5) ** 2,
    gradient=lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 1), 2 * (x[2] - 0.5)]),
    # comp pair 1: x[0] >= 0  ⊥  x[1] >= 0
    # comp pair 2: x[2] >= 0  ⊥  (x[0] + x[1]) >= 0
    comp_G=lambda x: np.array([x[0], x[2]]),
    comp_G_jacobian=lambda x: np.array([[1., 0., 0.], [0., 0., 1.]]),
    comp_H=lambda x: np.array([x[1], x[0] + x[1]]),
    comp_H_jacobian=lambda x: np.array([[0., 1., 0.], [1., 1., 0.]]),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r_a = pympcc.solve(problem_a, strategy="scholtes")

print(f"x* = {r_a.x}")
print(f"f* = {r_a.obj:.4f}")
```

## With `comp_var_pairs`

The same problem, declared as a list of `(var_idx, h_fn[, h_jac_fn])` triples:

```{code-cell} ipython3
problem_b = pympcc.MPCCProblem(
    n=3, n_comp=2,
    x0=np.array([0.5, 0.5, 0.5]),
    xl=np.zeros(3),
    objective=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2 + (x[2] - 0.5) ** 2,
    gradient=lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 1), 2 * (x[2] - 0.5)]),
    comp_var_pairs=[
        # x[0] >= 0  ⊥  x[1] >= 0
        (0, lambda x: np.array([x[1]]),
            lambda x: np.array([0., 1., 0.])),
        # x[2] >= 0  ⊥  (x[0] + x[1]) >= 0
        (2, lambda x: np.array([x[0] + x[1]]),
            lambda x: np.array([1., 1., 0.])),
    ],
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r_b = pympcc.solve(problem_b, strategy="scholtes")

print(f"x* = {r_b.x}")
print(f"f* = {r_b.obj:.4f}")
print(f"matches manual form: {np.allclose(r_a.x, r_b.x, atol=1e-5)}")
```

## What the package did for you

- **Identity G-Jacobians** — every $G$-side is an indicator on `var_idx`, so the package builds it as a constant sparse identity instead of calling your code.
- **Bound clamp** — `xl[var_idx]` is silently clamped to `max(xl[var_idx], 0.0)`; you can omit `xl` for var-paired indices and the lower bound is set automatically.
- **FD fallback** — drop the third element of each tuple and the H-side Jacobian falls back to forward finite differences.

## When to use which

| Form | Use when |
|---|---|
| `comp_G` / `comp_H` | Both sides are general functions of `x` (e.g. KKT-style $G(x) \ge 0\ \perp\ H(x) \ge 0$). |
| `comp_var_pairs` | The $G$-side is exactly $x[\text{var\_idx}]$. Common in MCPs and bilevel KKT systems. |
| `comp_var_pairs_bulk` | Same, but $k \gtrsim 10^4$ pairs — see [problem setup guide](../user_guide/problem_setup.md). |

For doubly-bounded $\ell \le x \le u\ \perp\ F(x)$ form, see `comp_box_pairs` in the [problem setup guide](../user_guide/problem_setup.md).
