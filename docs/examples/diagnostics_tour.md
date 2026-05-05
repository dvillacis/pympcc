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

# Diagnostics tour

`solve(..., diagnostics=True)` attaches a battery of post-solve quality checks to the result. This notebook walks through what each one tells you and when to look at it.

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

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
    r = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
```

## Constraint qualification

`result.cq` reports the strongest CQ that holds at the solution:

```{code-cell} ipython3
print(f"cq                  = {r.cq}")
print(f"active set sizes    = {r.cq_active_set_sizes}")
print(f"rank deficit        = {r.cq_rank_deficit}    (0 ⇒ LICQ)")
```

`MPCC-LICQ` is the strongest; `MPCC-MFCQ` is weaker but still enough for KKT to be necessary. `none` means the active gradients are linearly dependent and the strategy may have converged to a non-stationary point. The hierarchy is implemented by `pympcc.classify_cq`:

```{code-cell} ipython3
from pympcc import classify_cq
cq_info = classify_cq(r, problem)
print(cq_info)
```

## B-stationarity certification

`result.b_stationary` runs an LP enumeration over the linearised tangent cone branches at biactive pairs. The non-biactive case is automatic: if there are no biactive pairs, the result is trivially B-stationary.

```{code-cell} ipython3
print(f"b_stationary             = {r.b_stationary}")
print(f"b_stationary_min_descent = {r.b_stationary_min_descent}")
print(f"b_stationary_witness     = {r.b_stationary_witness}")
```

`min_descent ≥ 0` (within tolerance) ⇒ no descent direction along any branch, the point is B-stationary. The witness records the branch assignment that achieved the minimum descent (only set when not B-stationary).

## SOSC: second-order sufficient conditions

`pympcc.sosc_check` builds the reduced Lagrangian Hessian on the MPCC critical cone and tests positive definiteness. Positive ⇒ the point is a strict local minimum:

```{code-cell} ipython3
print(f"sosc                = {r.sosc}")
print(f"sosc_min_eigenvalue = {r.sosc_min_eigenvalue}")
print(f"sosc_skipped_reason = {r.sosc_skipped_reason}")
```

`sosc=None` is common: SOSC is skipped at biactive points (`skipped_reason="biactive_pairs"`) because the critical cone has a wedge that the standard reduced-Hessian test cannot resolve. For non-biactive solutions you'll see `sosc=True` (or `False` with a negative eigenvalue).

## Multi-merit cross-check

`result.merit_cross_check` evaluates three independent MPCC merit functions and reports their disagreement ratio. PATH ships this at termination; close-to-1 ratio ⇒ all three measures agree, which is what you want at a healthy convergence.

```{code-cell} ipython3
import json
print(json.dumps(r.merit_cross_check, indent=2))
```

A `disagreement_ratio` larger than ~10² is a warning sign: one merit thinks complementarity is satisfied, another doesn't, usually because $G$ and $H$ are scaled very differently. Pair scaling (`autoscale=True`) usually fixes it.

## Active-Jacobian degeneracy

`result.degeneracy_report` aggregates row/column norms of the active-constraint Jacobian, the smallest singular value, and the merit disagreement ratio:

```{code-cell} ipython3
print(json.dumps(r.degeneracy_report, indent=2))
```

A `min_singular_value` near zero or `n_zero_rows > 0` indicates rank deficiency — usually a sign that the active set has redundant rows or that a constraint is structurally vacuous.

## Initial-point quality

Set before any solve work runs, `result.initial_point_stats` reports residuals at the user's `x0`:

```{code-cell} ipython3
print(json.dumps(r.initial_point_stats, indent=2))
```

Useful for sanity-checking that your `x0` doesn't catastrophically violate the problem before the solver even starts.

## Cost: when to enable diagnostics

`diagnostics=True` adds:
- one matrix factorisation (CQ classification, ~`O(m_active · n²)`),
- up to $2^{|I_{00}|}$ LPs (B-stationarity — capped by `b_stat_max_biactive=10` in the solver kwargs),
- one eigendecomposition on a reduced Hessian (SOSC).

For interactive analysis: always on. For tight inner loops (parametric sweeps, RL training): off — the convergence info on `result` is enough most of the time.
