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

# TNLP refinement

The MPCC stationarity classification depends on the **MPCC** multipliers $\mu_G, \mu_H$, but IPOPT's interior-point bias forces those multipliers to be non-negative on every active lower-bound constraint. As a result, naïvely reading multipliers off the converged result almost always reports `S-stationary` regardless of whether the point genuinely satisfies the S-stationary sign condition.

`solve(..., tnlp_refine=True)` re-solves a **tightened NLP** with the active set fixed as equalities, extracts clean MPCC multipliers, and uses those to classify stationarity. The result is a much sharper certificate.

## Without refinement

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# kth1: a biactive optimum at (0, 0).
problem = pympcc.MPCCProblem(
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
    r_plain = pympcc.solve(problem, strategy="scholtes")

print(f"x*               = {r_plain.x}")
print(f"per_pair_status  = {r_plain.per_pair_status}")
print(f"stationarity     = {r_plain.stationarity}")
print(f"mult_comp_G_mpcc = {r_plain.mult_comp_G_mpcc}")
print(f"mult_comp_H_mpcc = {r_plain.mult_comp_H_mpcc}")
```

`mult_comp_G_mpcc` is `None` — without TNLP refinement we don't have certified multipliers.

## With refinement

```{code-cell} ipython3
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(problem, strategy="scholtes", tnlp_refine=True)

print(f"x*               = {r.x}")
print(f"per_pair_status  = {r.per_pair_status}")
print(f"stationarity     = {r.stationarity}")
print(f"mult_comp_G_mpcc = {r.mult_comp_G_mpcc}")
print(f"mult_comp_H_mpcc = {r.mult_comp_H_mpcc}")
```

After refinement, the MPCC multipliers are populated. For `kth1` we expect $\mu_G = \mu_H = 1$ (both positive) — the S-stationary signature on a biactive point.

## What the refinement actually does

`result.tnlp_refined` carries a `TNLPResult` dataclass with the diagnostic info from the second NLP solve. It is `None` when refinement was skipped (see "When TNLP refinement is skipped" below).

```{code-cell} ipython3
tnlp = r.tnlp_refined
if tnlp is None:
    print("TNLP refinement was skipped on this solve.")
else:
    kkt_str = f"{tnlp.kkt_residual:.2e}" if tnlp.kkt_residual is not None else "None"
    print(f"success         = {tnlp.success}")
    print(f"obj             = {tnlp.obj:.6f}")
    print(f"kkt_residual    = {kkt_str}")
    print(f"active_set      = {tnlp.active_set}")
    print(f"n_violations    = {tnlp.n_violations}")
    print(f"stationarity    = {tnlp.stationarity}")
```

The active set `(I_G, I_H)` records which side of each pair was pinned to zero. For our biactive solution, both indices appear because both sides are at zero.

`n_violations` records the count of MPCC multipliers that came back with the wrong sign. When this is small (≤ 20% of pairs), the package automatically **flips** the offending pair assignments and re-solves once — that's the "flip-and-retry" guard built into TNLP refinement.

## When TNLP refinement is skipped

The package skips TNLP refinement (and falls back to the unrefined classification) when:

- More than 10% of pairs are biactive at the converged point — the tightened NLP is too constrained and IPOPT enters restoration immediately.
- The forward solve did not converge (`result.success == False`).
- The user disabled it (`tnlp_refine=False`, the default).

A skipped refinement looks like the "without refinement" cell above: `mult_comp_G_mpcc` is `None` and stationarity classification falls back to the (interior-point-biased) heuristic.

## Cost

The TNLP solve is one additional IPOPT call with `n_eq` constraints expanded by `|I_G| + |I_H|`. For small problems this is essentially free; for large problems with many active comp pairs it can cost as much as the original outer loop. Enable selectively in production.

## Together with `diagnostics=True`

`tnlp_refine=True` and `diagnostics=True` compose: the diagnostic CQ / SOSC / B-stat checks consume the TNLP-refined multipliers when present, so all the certificates are sharper.

```python
r = pympcc.solve(
    problem,
    strategy="scholtes",
    tnlp_refine=True,
    diagnostics=True,
)
```
