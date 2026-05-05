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

# Stationarity hierarchy: S / M / C / W

Because MPCCs generically fail standard LICQ at any solution with a biactive pair, the textbook KKT conditions don't apply directly. The MPCC literature defines a hierarchy of stationarity certificates **specialised to complementarity structure**. Strongest to weakest:

| Level | Sign condition on biactive multipliers $(\mu_G^i, \mu_H^i)$ |
|---|---|
| **S-stationary** | both $\ge 0$ |
| **M-stationary** | both $\ge 0$, **or** one is zero |
| **C-stationary** | $\mu_G^i \cdot \mu_H^i \ge 0$ |
| **W-stationary** | KKT of the relaxed NLP holds — **no sign condition** on biactive multipliers |

For pairs that are *not* biactive (only one side at zero), all four levels collapse: the active side requires $\mu \ge 0$ as in standard NLP. The hierarchy only differs at biactive points.

`pympcc.classify_stationarity(result, problem, mu_G, mu_H)` takes the multipliers and returns the strongest level certified.

## A non-biactive S-stationary point

The quickstart problem ends at $x^\* = (2, 0)$ with `H_active` (only $x_1$ at zero, $x_0 = 2 > 0$). The biactive set is empty, so any converged solution is automatically S-stationary.

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
    r = pympcc.solve(problem, strategy="scholtes", tnlp_refine=True)

print(f"x*              = {r.x}")
print(f"per_pair_status = {r.per_pair_status}")
print(f"stationarity    = {r.stationarity}")
```

`tnlp_refine=True` triggers a second NLP solve with the active set fixed as equalities — that produces clean MPCC multipliers, so the stationarity classification is genuine rather than vacuous.

## A biactive S-stationary point

The MacMPEC `kth1` problem:

$$
\min\ x_0 + x_1 \quad\text{s.t.}\quad x_0 \ge 0\ \perp\ x_1 \ge 0
$$

has its optimum at $x^\* = (0, 0)$ — biactive. The MPCC multipliers $\mu_G = \mu_H = 1$ (from $\nabla f = (1,1)$ being absorbed into the comp gradients) satisfy the S-stationary sign condition.

```{code-cell} ipython3
problem_kth1 = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: x[0] + x[1],
    gradient=lambda x: np.ones(2),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(problem_kth1, strategy="scholtes", tnlp_refine=True)

print(f"x*              = {r.x}")
print(f"per_pair_status = {r.per_pair_status}")
print(f"stationarity    = {r.stationarity}")
if r.tnlp_refined is not None:
    print(f"μ_G (TNLP)      = {r.tnlp_refined.mult_comp_G}")
    print(f"μ_H (TNLP)      = {r.tnlp_refined.mult_comp_H}")
```

Both multipliers are positive at the biactive point — that's the S-stationary signature.

## How `classify_stationarity` makes its decision

The classifier reads the converged multipliers off the result object — specifically the slice of `result.mult_g` that corresponds to the $G$ and $H$ rows — and applies the sign tests in the hierarchy table at the top of this page. You can call it directly:

```{code-cell} ipython3
from pympcc import classify_stationarity

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(problem_kth1, strategy="scholtes", tnlp_refine=True)

print(f"classify_stationarity → {classify_stationarity(r, problem_kth1)}")
print(f"result.stationarity   → {r.stationarity}")
```

Both routes return the same label; `result.stationarity` is just `classify_stationarity(result, problem)` cached on the result for convenience.

In practice, IPOPT's interior-point bias forces $\mu_G, \mu_H \ge 0$ at convergence on all active lower bounds — so without the TNLP refinement step the classifier almost always returns S-stationary, regardless of whether the point is genuinely S- or only W-/C-stationary in MPCC terms. The [TNLP refinement notebook](tnlp_refinement.md) explains how to recover the genuine MPCC multipliers.
