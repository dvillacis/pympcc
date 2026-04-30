# Presolve and multistart

## Presolve

```python
result = pympcc.solve(problem, strategy="scholtes", presolve=True)
```

Two passes run when `presolve=True`:

1. **Pinned-variable elimination** — variables with `xl[j] == xu[j]` are substituted out of the optimisation; results are auto-expanded back to the original variable space.
2. **Linear FBBT** — feasibility-based bound tightening on rows detected as linear by a single bounded perturbation. Up to 50 fixed-point sweeps; detected infeasibility falls back to the identity map and emits a `UserWarning`.

When nothing can be eliminated, the presolve is a no-op.

For finer control:

```python
from pympcc import presolve

reduced_problem, pmap = presolve(problem)
result = pympcc.solve(reduced_problem)
expanded_x = pmap.expand(result.x)   # back to original n
```

## Multistart

```python
ms = pympcc.multistart(
    problem,
    n_starts=20,
    perturb_scale=0.5,
    seed=0,
    strategy="scholtes",     # forwarded as solve_kwargs
)

ms.best         # MPCCResult — best run by objective among successes
ms.runs         # list[MPCCResult] — every start
```

Useful when the MPCC has multiple stationary points (typical for bilevel and equilibrium models). Each start perturbs `x0` by `perturb_scale * randn(n)`. Currently sequential; parallel `n_jobs` is on the roadmap (§6.1).
