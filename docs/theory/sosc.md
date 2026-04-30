# Second-order sufficient conditions

A first-order stationary point may be a local **min**, a local **max**, or a saddle. Distinguishing them requires second-order information.

For a standard NLP, the SOSC reads: $x^*$ is a strict local minimum if for every nonzero direction $d$ in the **critical cone**, $d^\top \nabla^2_x L(x^*, \lambda^*) \, d > 0$.

For an MPCC the same shape works, but two things change:

1. The **critical cone** has to account for the disjunctive structure at biactive indices.
2. The **Hessian** is the MPCC Lagrangian Hessian, evaluated with MPCC multipliers (not the regularised NLP's IPOPT multipliers).

## The critical cone

For a strongly stationary point $x^*$ with multipliers $(\lambda, \mu, \nu^G, \nu^H)$, the critical cone $\mathcal C(x^*)$ is the set of $d \in \RR^n$ satisfying:

$$
\begin{aligned}
\nabla h_k(x^*)^\top d &= 0,                 & & \forall\, k, \\
\nabla g_i(x^*)^\top d &\le 0,               & & i \in \mathcal I_g \text{ with } \lambda_i = 0, \\
\nabla g_i(x^*)^\top d &= 0,                 & & i \in \mathcal I_g \text{ with } \lambda_i > 0, \\
\nabla G_i(x^*)^\top d &= 0,                 & & i \in \mathcal I_G \cup (\mathcal I_{00} \cap \{\nu^G_i > 0\}), \\
\nabla H_i(x^*)^\top d &= 0,                 & & i \in \mathcal I_H \cup (\mathcal I_{00} \cap \{\nu^H_i > 0\}), \\
\nabla G_i(x^*)^\top d &\ge 0,\ \nabla H_i(x^*)^\top d \ge 0, & & i \in \mathcal I_{00} \cap \{\nu^G_i = \nu^H_i = 0\}, \\
\nabla f(x^*)^\top d &\le 0.
\end{aligned}
$$

The biactive indices with non-zero multipliers contribute equality rows; the remaining biactive indices contribute a "wedge" condition that lives outside any single linear subspace — this is what makes the MPCC critical cone non-trivial to characterise.

## MPCC-SOSC

**Theorem.** If $x^*$ is S-stationary with MPCC multipliers $(\lambda, \mu, \nu^G, \nu^H)$ and

$$
d^\top \nabla^2_x L_{\text{MPCC}}(x^*, \lambda, \mu, \nu^G, \nu^H)\,d > 0
\qquad \forall\, d \in \mathcal C(x^*) \setminus \{0\},
$$

then $x^*$ is a **strict local minimiser** of the MPCC.

The Hessian is the standard MPCC Lagrangian Hessian:

$$
\nabla^2_x L_{\text{MPCC}} = \nabla^2 f
+ \sum_k \mu_k \nabla^2 h_k
+ \sum_i \lambda_i \nabla^2 g_i
- \sum_i \nu^G_i \nabla^2 G_i
- \sum_i \nu^H_i \nabla^2 H_i.
$$

(Sign convention: pympcc's `lagrangian_hessian` callable consumes IPOPT-convention multipliers; the package's internal SOSC routine handles sign bookkeeping for you.)

## What pympcc returns

```python
result = pympcc.solve(problem, strategy="scholtes",
                      diagnostics=True, tnlp_refine=True)

result.sosc                   # True / False / None
result.sosc_min_eigenvalue    # min eigenvalue of the reduced Hessian W = Z^T ∇²L Z
result.sosc_skipped_reason    # "biactive_pairs" | "not_converged" | None
```

Internally pympcc:

1. Extracts MPCC multipliers via TNLP refinement (when `tnlp_refine=True`).
2. Builds the reduced Hessian $W = Z^\top \nabla^2_x L \, Z$, where the columns of $Z$ form an orthonormal basis of the *equality-active* part of the critical cone (biactive wedges are handled by branch enumeration when there are ≤ `b_stat_max_biactive` of them).
3. Returns `True` if $\lambda_{\min}(W) > 0$, `False` if $\le 0$, `None` (with `skipped_reason`) when the wedge structure makes the test ambiguous.

If you supply an analytic `lagrangian_hessian` callable, the SOSC builder uses it directly — much more accurate than the FD fallback, especially at problems where $\nabla^2 f$ is nearly singular on the critical cone.

## Edge cases

* **Biactive pairs without analytic Hessian** — the FD reduced Hessian on a thin critical cone is fragile. Either supply an analytic Hessian or interpret a `None` SOSC verdict as "test inconclusive", not "not a local min".
* **Degenerate critical cone** — when the cone is just $\{0\}$, every point is trivially SOSC. pympcc returns `True` in this case.
* **W-stationary points** — SOSC is *only* a sufficient condition at *strongly* stationary points. At a W-stationary point that's not S-stationary, SOSC does not certify local optimality even when the reduced Hessian is positive definite.

## References

* Scheel & Scholtes (2000). *Math. Oper. Res.* — MPCC-SOSC theorem.
* Bonnans & Shapiro (2000). *Perturbation Analysis of Optimization Problems*. Springer — second-order theory at large.
* Anitescu (2005). On using the elastic mode in nonlinear programming approaches to mathematical programs with complementarity constraints. *SIAM J. Optim.*
