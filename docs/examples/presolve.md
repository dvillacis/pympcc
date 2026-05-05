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

# Presolve

Before the strategy runs its outer loop, `solve(..., presolve=True)` (or the standalone `pympcc.presolve(problem)`) applies a sequence of structural reductions:

| Pass | What it does |
|---|---|
| **A1** Pinned-variable elimination | Substitutes out every $j$ with `xl[j] == xu[j]`. |
| **A2** FBBT (linear bound tightening) | Iterates `xl[j] := max(xl[j], (b - aᵀx)/a_j)` style updates over linear constraints to fixpoint; pins variables whose bounds collapse. |
| **A4** Empty row/col pruning | Drops constraints with no nonzero Jacobian and variables with no nonzero column. |
| **B1** Dead-pair pruning | Drops comp pairs where one side is structurally zero and trivially satisfied at `x0`. |
| **B2** Forced-pair pruning | Comp pairs where both sides are linear with a forced sign reduce to a single equality. |
| **B3** Linear sign-prefix | Comp pairs whose sign is determined by domain bounds rewrite as equalities. |

When nothing can be eliminated, `presolve` is a no-op and returns an identity `PresolveMap`.

## Demo: pinned-variable elimination

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# x[2] is pinned at 0.5 by xl == xu; presolve will eliminate it.
problem = pympcc.MPCCProblem(
    n=3, n_comp=1,
    x0=np.array([0.5, 0.5, 0.5]),
    xl=np.array([0.0, 0.0, 0.5]),
    xu=np.array([np.inf, np.inf, 0.5]),     # x[2] pinned
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + x[2],
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0), 1.0]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
)

reduced, pmap = pympcc.presolve(problem)
print(f"Original n = {problem.n}, n_comp = {problem.n_comp}")
print(f"Reduced  n = {reduced.n},  n_comp = {reduced.n_comp}")
```

The reduced problem has `n = 2`. The strategy never sees `x[2]`; the `PresolveMap` re-injects it into the result.

## End-to-end with `presolve=True`

`solve(presolve=True)` runs the reduction once at solver construction, lets the strategy operate on the reduced problem, then expands the result back to the original variable space:

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r_with    = pympcc.solve(problem, strategy="scholtes", presolve=True)
    r_without = pympcc.solve(problem, strategy="scholtes", presolve=False)

print(f"with    presolve: x* = {r_with.x},   f* = {r_with.obj:.4f}")
print(f"without presolve: x* = {r_without.x},   f* = {r_without.obj:.4f}")
print(f"agree: {np.allclose(r_with.x, r_without.x, atol=1e-5)}")
```

Both yield identical `result.x` in the **original** variable space — pinned `x[2] = 0.5` shows up automatically.

## Demo: dead-pair pruning

A comp pair where one side is structurally zero (no nonzeros in its Jacobian sparsity pattern) and trivially satisfied at `x0` is dropped:

```{code-cell} ipython3
# Pair 0: x[0] >= 0  ⊥  0 >= 0   — H row 0 has no Jacobian nnz, dead.
# Pair 1: x[1] >= 0  ⊥  x[2] >= 0 — genuine.
problem_dead = pympcc.MPCCProblem(
    n=3, n_comp=2,
    x0=np.array([0.5, 0.5, 0.5]),
    xl=np.zeros(3),
    objective=lambda x: x[0] + x[1] + x[2],
    gradient=lambda x: np.ones(3),
    comp_G=lambda x: np.array([x[0], x[1]]),
    comp_G_jacobian=lambda x: np.array([1.0, 1.0]),
    comp_G_jacobian_sparsity=(np.array([0, 1]), np.array([0, 1])),
    comp_H=lambda x: np.array([0.0, x[2]]),
    # Sparsity declares only one nonzero: row 1, col 2. Row 0 has no nnz → dead.
    comp_H_jacobian=lambda x: np.array([1.0]),
    comp_H_jacobian_sparsity=(np.array([1]), np.array([2])),
)

reduced, pmap = pympcc.presolve(problem_dead)
print(f"Original n_comp = {problem_dead.n_comp}")
print(f"Reduced  n_comp = {reduced.n_comp}")
```

Dead-pair pruning collapses two pairs to one when the structural test detects a row with no nonzero Jacobian entries.

## When presolve helps (and when it doesn't)

- **Helps a lot**: hand-coded MPCCs with redundant pinned variables, AMPL/Pyomo imports that include vacuous bounds, parametric problems where some `theta` collapses pairs.
- **No-op**: clean small MPCCs (the quickstart-style problem). `presolve(...)` returns the identity map and adds zero overhead.
- **Caveat**: `presolve` runs once at solver construction, not at each warm-start `resolve()`. If your parametric sweep mutates structure, reconstruct the solver.

For the full pass-by-pass list, see the [presolve & multistart user guide](../user_guide/presolve_multistart.md).
