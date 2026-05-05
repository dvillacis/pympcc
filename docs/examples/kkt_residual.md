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

# KKT residual: how to read it

Every successful solve populates `result.kkt_residual` — the $\ell_\infty$ norm of the MPCC-KKT stationarity residual

$$
\bigl\| \nabla f + J_g^\top \lambda_g + J_h^\top \lambda_h + J_G^\top \mu_G + J_H^\top \mu_H - z_L + z_U \bigr\|_\infty
$$

where $\mu_G, \mu_H$ are the **MPCC multipliers** (not the relaxed-NLP multipliers IPOPT returns directly). Each strategy applies its own correction to convert the IPOPT multipliers back to the MPCC ones — see `pympcc.compute_kkt_residual`.

A small `kkt_residual` is the **primary** indicator of solution quality. Specifically:

| Threshold | Interpretation |
|---|---|
| `< 1e-6` | Converged to machine-precision-class stationarity. |
| `1e-6` to `1e-4` | Acceptable; IPOPT's default tolerance is `1e-6` on the relaxed problem. |
| `1e-4` to `1e-2` | Suspect. Likely premature termination or scaling issue. |
| `> 1e-2` | The point is **not** MPCC-stationary; the solver may have stalled. |

## Reading it on a clean problem

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
    r = pympcc.solve(problem, strategy="scholtes")

print(f"obj            = {r.obj:.6f}")
print(f"comp_residual  = {r.comp_residual:.2e}    # G·H ≈ 0 ?")
print(f"kkt_residual   = {r.kkt_residual:.2e}     # ∇L ≈ 0 ?")
print(f"stationarity   = {r.stationarity}")
```

## How it changes across strategies

Each strategy reaches the same optimum but with slightly different residual magnitudes — the $\varepsilon$-relaxed strategies leave a residue proportional to the final $\varepsilon$:

```{code-cell} ipython3
def make_problem():
    return pympcc.MPCCProblem(
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

print(f"  {'strategy':<22}  {'comp_res':>10}  {'kkt_res':>10}")
print("  " + "-" * 22 + "  " + "-" * 10 + "  " + "-" * 10)
for s in ["direct", "scholtes", "smoothing", "lin_fukushima",
          "augmented_lagrangian", "slack"]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = pympcc.solve(make_problem(), strategy=s)
    print(f"  {s:<22}  {r.comp_residual:>10.2e}  {r.kkt_residual:>10.2e}")
```

## Computing it manually

`pympcc.compute_kkt_residual` is exposed publicly so you can recompute the residual with your own MPCC multipliers (e.g. for a strategy not in the package, or for sensitivity-analysis post-processing). The cleanest way to get clean MPCC multipliers is via `tnlp_refine=True`, which populates `result.mult_comp_G_mpcc` and `result.mult_comp_H_mpcc`:

```{code-cell} ipython3
from pympcc import compute_kkt_residual

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(make_problem(), strategy="scholtes", tnlp_refine=True)

manual = compute_kkt_residual(
    r, make_problem(),
    mpcc_mult_G=r.mult_comp_G_mpcc,
    mpcc_mult_H=r.mult_comp_H_mpcc,
)
print(f"result.kkt_residual : {r.kkt_residual:.2e}")
print(f"manual recomputed   : {manual:.2e}")
```

The two values will differ slightly: `result.kkt_residual` was computed at the end of the strategy's outer loop using strategy-specific multiplier corrections, while the manual recompute uses the TNLP-refined multipliers. Both are valid MPCC stationarity residuals; the TNLP one is generally sharper at biactive points.

## When the residual is large

- **Check `result.success`** first — IPOPT may have terminated on `acceptable_iter` rather than full convergence.
- **Inspect `result.history`** for the iterative strategies: a residual that *grew* in the last outer iteration is a tell-tale sign of ε-driven ill-conditioning (try a slower `reduction` factor).
- **Pair scaling** — `result.comp_G_scale` / `result.comp_H_scale` show what auto-scaling did. Wildly different magnitudes between $G$ and $H$ inflate `kkt_residual`.
- **Use `tnlp_refine=True`** — recovers MPCC multipliers from a second NLP solve. The reported `kkt_residual` after refinement is much sharper. See the [TNLP refinement notebook](tnlp_refinement.md).

For the full diagnostic story (CQ, SOSC, B-stationarity, merit cross-check), see the [diagnostics tour](diagnostics_tour.md).
