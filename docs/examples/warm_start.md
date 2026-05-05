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

# Stateful warm hot-start

When solving a sequence of MPCCs whose **structure** is unchanged but whose **numeric values** vary slightly — typical of MPC rolling-horizon control or parametric sweeps — `MPCCSolver.resolve()` reuses the previous solve's primal-dual state to dramatically cut iteration counts.

## The pattern

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# A parametric MPCC: target = (target_x, target_y) varies; structure is fixed.
def make_problem(target):
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2,
        gradient=lambda x: 2.0 * (x - target),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )

# Initial cold solve at target = (2.0, 1.0)
solver = pympcc.MPCCSolver(make_problem(np.array([2.0, 1.0])), strategy="scholtes")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r0 = solver.solve()

print(f"Cold solve : x* = {r0.x},  iters = {len(r0.history)}")
```

Now slide the target slightly and `resolve()`:

```{code-cell} ipython3
import time

targets = [
    np.array([2.0, 1.0]),
    np.array([2.1, 1.0]),
    np.array([2.2, 1.05]),
    np.array([2.3, 1.10]),
    np.array([2.4, 1.15]),
]

print(f"  {'target':<18}  {'x*':<22}  {'obj':>8}  iters")
print("  " + "-" * 18 + "  " + "-" * 22 + "  " + "-" * 8 + "  -----")
for t in targets:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = solver.resolve(make_problem(t))
    print(f"  {str(t):<18}  {str(r.x):<22}  {r.obj:>8.4f}  {len(r.history):>5d}")
```

Each `resolve()` warm-starts from the previous primal-dual point. For small parameter perturbations the iteration count typically drops to a fraction of the cold-solve cost.

## What `resolve()` does for you

| Step | Action |
|---|---|
| 1 | Verifies the new problem's **structural signature** matches the original (`n`, `n_comp`, `n_eq`, `n_ineq`, every Jacobian sparsity pattern). |
| 2 | If structure changed, emits a `UserWarning` and falls back to a cold rebuild. |
| 3 | Otherwise, hot-swaps the problem's numeric callables in the strategy. |
| 4 | Seeds new `x0` from the previous `result.x` (`warm_x0=True`). |
| 5 | Forwards `mult_g` / `mult_x_L` / `mult_x_U` to IPOPT and toggles `warm_start_init_point=yes` (`warm_dual=True`). |

You can disable either warming by passing `warm_x0=False` or `warm_dual=False`. The savings vs. cold are reported on `result.warmstart_savings_iter`.

## When to use it

- **MPC rolling-horizon control** — every step shifts the horizon by one; structure is fixed, the only changes are reference trajectories and possibly box bounds.
- **Parametric sweeps** — sweeping over a parameter $\theta$ in the objective or RHS while structure is fixed.
- **Sensitivity analysis** — re-solving at perturbed inputs to validate `pympcc.sensitivity` gradients.

## Compared to `pympcc.solve()` again

A second call to `pympcc.solve(...)` on the same problem object **also** warm-starts when used through the same solver instance, but `resolve()` is the intended entry point because it lets you swap the problem in. Constructing a fresh `MPCCSolver` each call discards the warm state.

## Caveats

- **`autoscale` is not re-applied on resolve.** Rescaling the comp pairs would invalidate the warm dual. If you need a fresh autoscale probe, reconstruct the solver.
- **Structural changes force cold restart.** Adding a comp pair, changing `n`, or changing a sparsity pattern triggers a `UserWarning` and a rebuild from scratch. The cold-rebuild's iteration count becomes the new baseline for `warmstart_savings_iter`.
- **`presolve=True`** is re-applied on each resolve only if it was on at construction. The reduced problem may differ across resolves if the parameters change pinned-variable status.
