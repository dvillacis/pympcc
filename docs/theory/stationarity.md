# Stationarity hierarchy

For an NLP, "stationary point" means *KKT point*: $\nabla f + \sum \lambda_i \nabla g_i + \sum \mu_k \nabla h_k = 0$ with $\lambda \ge 0$ on active inequalities. There is one notion.

For an MPCC, the disjunctive structure of $G \perp H$ produces a **lattice of stationarity concepts** parameterised by the sign requirements imposed on the multipliers at biactive pairs ($i \in \mathcal I_{00}$, where $G_i(x) = H_i(x) = 0$). All of them require the basic Lagrangian condition

$$
\nabla f(x) + \sum_k \mu_k\,\nabla h_k(x) + \sum_{i \in \mathcal I_g} \lambda_i\,\nabla g_i(x)
- \sum_i \nu^G_i\,\nabla G_i(x) - \sum_i \nu^H_i\,\nabla H_i(x) = 0
$$

with $\lambda \ge 0$, $\nu^G_i = 0$ for $i \in \mathcal I_H$, and $\nu^H_i = 0$ for $i \in \mathcal I_G$. The differences are entirely in what is required of $(\nu^G_i, \nu^H_i)$ at biactive indices.

## The hierarchy

Listed strongest to weakest. **Stronger ⇒ weaker.**

| Name | Condition at $i \in \mathcal I_{00}$ | When it occurs |
|---|---|---|
| **S-stationary** (strong) | $\nu^G_i \ge 0 \text{ and } \nu^H_i \ge 0$ | Local min when MPCC-LICQ holds. |
| **M-stationary** | $\nu^G_i \nu^H_i = 0 \text{ or } \nu^G_i, \nu^H_i \ge 0$ | Limit of regularisation methods (Scholtes, smoothing) under MPCC-MFCQ. |
| **C-stationary** (Clarke) | $\nu^G_i \nu^H_i \ge 0$ | A weaker convergence guarantee; insufficient for local optimality on its own. |
| **A-stationary** (alternative) | $\nu^G_i \ge 0 \text{ or } \nu^H_i \ge 0$ | Theoretical mid-tier; rarely surfaced by solvers. |
| **W-stationary** (weak) | (no sign requirement) | The basic KKT condition. |
| **B-stationary** (Bouligand) | $0 \in \nabla f(x) + N_{T(x)}(x)$ | Geometric: $0$ is a stationary direction over the *tangent cone* — the strongest local-optimality necessary condition. |

S-stationarity is the gold standard: it's the *necessary* condition for local optimality under MPCC-LICQ, and equivalent to B-stationarity under MPCC-LICQ.

## What pympcc gives you

After `pympcc.solve(...)`, `result.stationarity` reports one of `"S-stationary"`, `"W-stationary"`, or `"unknown"`. The classification is done by inspecting the IPOPT multipliers; with `tnlp_refine=True` it uses certified MPCC multipliers from a TNLP re-solve and the result is much more reliable.

```python
result = pympcc.solve(problem, strategy="scholtes",
                      diagnostics=True, tnlp_refine=True)
result.stationarity        # "S-stationary"
result.mult_comp_G_mpcc    # ν^G  (≥ 0 at S-stationary points)
result.mult_comp_H_mpcc    # ν^H
```

For a finite-step **B-stationarity** certificate (`b_stat_max_biactive` controls when this is feasible), enumerate active branches:

```python
result = pympcc.solve(problem, strategy="scholtes",
                      diagnostics=True, b_stat_max_biactive=10)
result.b_stationary        # "B-stationary" / "not B-stationary" / "intractable"
```

## Why TNLP refinement matters

The IPOPT multipliers at the converged regularised NLP are *not* the MPCC multipliers — they pick up shifts from the regularisation parameter $\varepsilon$. To get clean MPCC multipliers and a trustworthy stationarity classification, re-solve a **tightened NLP** with the active set from the regularised solution fixed as equalities:

* $G_i = 0$ for $i \in \mathcal I_G \cup \mathcal I_{00}$
* $H_i = 0$ for $i \in \mathcal I_H \cup \mathcal I_{00}$
* $g_i = 0$ for $i \in \mathcal I_g$

Solving this TNLP returns multipliers in the original MPCC's space. `pympcc.solve(..., tnlp_refine=True)` does this automatically and stores the result under `result.tnlp_refined`. These are also the multipliers `pympcc.sensitivity` and `pympcc.solve_jax` consume.

## Practical reading order

1. Look at `result.success` first — without convergence, no stationarity claim is meaningful.
2. Then `result.stationarity`: S beats W. If `"unknown"`, suspect biactive indices or CQ failure.
3. Cross-check with `result.cq` (MPCC-LICQ ⇒ S ⇔ B) and `result.merit_cross_check` (large `disagreement_ratio` flags numerical trouble).
4. For local-optimality certification, also pass `diagnostics=True` and inspect `result.sosc`.

## References

* Scheel & Scholtes (2000). *Math. Oper. Res.* — S/M/C/A/W definitions.
* Pang & Fukushima (1999). Complementarity constraint qualifications and simplified B-stationarity conditions.
* Flegel & Kanzow (2005). On M-stationary points. *J. Math. Anal. Appl.*
