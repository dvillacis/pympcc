# Diagnostics

Pass `diagnostics=True` to `pympcc.solve` to populate the constraint qualification, B-stationarity, and SOSC fields on `MPCCResult`. Pass `tnlp_refine=True` to extract certified MPCC multipliers.

## CQ, B-stationarity, SOSC

```python
result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)

print(result.cq)                  # "MPCC-LICQ"
print(result.b_stationary)        # "B-stationary"
print(result.sosc)                # True — strict local minimiser
print(result.sosc_min_eigenvalue) # > 0
```

**SOSC** checks positive definiteness of the reduced Lagrangian Hessian on the critical cone:

- `True` — $x^*$ is a certified strict local minimiser.
- `False` — saddle point, or only a local max in some direction.
- `None` — skipped (biactive pairs present, or solve did not converge); check `result.sosc_skipped_reason`.

## TNLP-certified multipliers

```python
result = pympcc.solve(
    problem, strategy="scholtes",
    diagnostics=True, tnlp_refine=True,
)

print(result.mult_comp_G_mpcc)  # certified μ_G (≥ 0 at S-stationary points)
print(result.mult_comp_H_mpcc)  # certified μ_H
print(result.stationarity)      # "S-stationary" or "W-stationary"
```

TNLP refinement re-solves a tightened NLP with the active set fixed as equalities. This extracts MPCC-clean multipliers and certifies S- or W-stationarity. The FD-based SOSC check uses TNLP multipliers when available (most accurate path).

## Multi-merit cross-check

After every solve, three independent MPCC merit functions are evaluated and reported on the result:

- Fischer-Burmeister: $\lvert G + H - \sqrt{G^2 + H^2} \rvert$
- min-map: $\lvert \min(G, H) \rvert$
- inner-product: $\lvert G \cdot H \rvert$

```python
print(result.merit_cross_check)   # {"fb_max": ..., "minmap_max": ..., "ip_max": ..., "disagreement_ratio": ...}
```

A `disagreement_ratio` close to 1 means the merits agree (healthy convergence); large values flag scaling mismatch or near-degeneracy.

## Active-Jacobian diagnostics

```python
print(result.jac_row_norms)        # {"max": ..., "min": ..., "near_zero": int}
print(result.jac_col_norms)        # idem
print(result.degeneracy_report)    # {"n_biactive": ..., "n_zero_rows": ..., "min_singular_value": ...}
```

## Condition numbers

```python
print(result.jac_condition)                # κ(J_active)
print(result.hessian_condition_estimate)   # diagonal-scaling estimate from MA57/MA27
```

## Initial-point statistics

```python
from pympcc import initial_point_statistics

stats = initial_point_statistics(problem)
# bound violations, complementarity violation at x0, constraint residuals
```
