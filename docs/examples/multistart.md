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

# Multistart

MPCCs are non-convex; different starting points may converge to different local optima. `pympcc.multistart` runs `solve()` from several perturbed `x0` values and returns the best result, with diagnostics for the cluster structure across runs.

## Sequential multistart

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# Problem with two local minima:
#   min  (x0 - 2)^2 + (x1 - 1)^2  s.t.  x0 ⊥ x1
# Optima: (2, 0) with f=1   and   (0, 1) with f=4
problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    ms = pympcc.multistart(
        problem,
        n_starts=8,
        perturb_scale=0.5,
        seed=0,
        strategy="scholtes",
    )

print(f"best  obj   = {ms.obj:.4f}    x* = {ms.x}")
print(f"successes  = {ms.n_success} / {len(ms.runs)}")
print(f"failures   = {ms.failures}")
```

`MultiStartResult` proxies the underlying `MPCCResult` for the best run, so you can use `ms.x`, `ms.obj`, `ms.success`, etc. as drop-ins.

## Inspecting the cluster structure

`unique_optima()` clusters runs that converged to the same point (within tolerance) and returns one representative per cluster:

```{code-cell} ipython3
clusters = ms.unique_optima(atol_obj=1e-3, atol_x=1e-3)
print(f"Found {len(clusters)} unique optimum/optima:")
for k, c in enumerate(clusters):
    print(f"  cluster {k}: x* = {c.x}, obj = {c.obj:.4f}")
```

A non-convex MPCC with multiple local minima will produce more than one cluster; if every run converged to the same point, multistart found a unique answer (which may still only be a local min, but at least it's unambiguous from this perturbation set).

## Parallel multistart

`n_jobs > 1` fans out across `concurrent.futures.ProcessPoolExecutor` with the `spawn` start method. **Important caveat**: every callable on `MPCCProblem` must be picklable, which means **no bare lambdas or local closures**. Define top-level functions instead:

```python
# At module level (not shown here because notebooks don't pickle locals well):
def _objective(x):
    return float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2)

def _gradient(x):
    return np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)])

# ... etc

problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=_objective,
    gradient=_gradient,
    # ...
)

ms = pympcc.multistart(problem, n_starts=16, n_jobs=-1)  # n_jobs=-1 → all cores
```

The package guarantees **bit-identical iterates** across runs at the same seed, regardless of `n_jobs`. Per-start seeds are derived deterministically from `numpy.random.SeedSequence(seed).spawn(n_starts)` in the parent process, so completion order in workers cannot perturb the random stream.

## Reading individual runs

```{code-cell} ipython3
print(f"  {'k':>3}  {'obj':>8}  x*")
print("  " + "-" * 3 + "  " + "-" * 8 + "  ----------")
for k, r in enumerate(ms.runs):
    flag = "✓" if r.success else "✗"
    print(f"  {k:>3}  {r.obj:>8.4f}  {r.x}  {flag}")
```

When some starts fail, the indices of the failures are reported in `ms.failures` as `(start_index, exception_type, message)` tuples — useful for diagnosing pathological starting points.
