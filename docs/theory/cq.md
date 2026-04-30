# Constraint qualifications

For an NLP $\min f(x)$ s.t. $g(x) \le 0$, $h(x) = 0$, the **active set** at a feasible point $x$ is

$$
\mathcal A(x) = \{\, i : g_i(x) = 0 \,\}.
$$

For an MPCC, the active set splits across the three constraint types and additionally tracks each complementarity pair as $G$-active, $H$-active, or **biactive** (both zero):

$$
\begin{aligned}
\mathcal I_g(x)  &= \{\, i : g_i(x) = 0 \,\}, \\
\mathcal I_G(x)  &= \{\, i : G_i(x) = 0 < H_i(x) \,\}, \\
\mathcal I_H(x)  &= \{\, i : H_i(x) = 0 < G_i(x) \,\}, \\
\mathcal I_{00}(x) &= \{\, i : G_i(x) = H_i(x) = 0 \,\}\quad \text{(biactive)}.
\end{aligned}
$$

`pympcc.active_sets(result, problem)` returns these four index sets.

## Standard LICQ (and why it fails)

**Linear Independence Constraint Qualification:** the gradients of active inequality and equality constraints are linearly independent.

For an MPCC with the orthogonality written as $G \cdot H = 0$, suppose pair $i$ is $G$-active ($G_i = 0$, $H_i > 0$). The two corresponding active rows are

$$
\nabla G_i(x), \qquad \nabla (G \cdot H)_i = H_i(x)\,\nabla G_i(x) + G_i(x)\,\nabla H_i(x) = H_i(x)\,\nabla G_i(x).
$$

These are positively proportional. Standard LICQ **fails generically** at every MPCC feasible point with at least one active pair. Standard MFCQ fails for the same reason.

## MPCC-LICQ

The fix: treat $G$ and $H$ as separate constraints and *drop* the redundant orthogonality row from the CQ. **MPCC-LICQ** at $x$ requires linear independence of

$$
\{\, \nabla h_k(x) \,\}_{k} \;\cup\;
\{\, \nabla g_i(x) \,\}_{i \in \mathcal I_g} \;\cup\;
\{\, \nabla G_i(x) \,\}_{i \in \mathcal I_G \cup \mathcal I_{00}} \;\cup\;
\{\, \nabla H_i(x) \,\}_{i \in \mathcal I_H \cup \mathcal I_{00}}.
$$

That is: at biactive indices, *both* $\nabla G_i$ and $\nabla H_i$ enter the active-gradient set.

`pympcc.classify_cq(result, problem)` (also surfaced as `result.cq` when `diagnostics=True`) returns one of `"MPCC-LICQ"`, `"MPCC-MFCQ"`, `"none"`, `"unknown"`. The companion `result.cq_rank_deficit` is the number of linearly dependent rows; zero ⇔ MPCC-LICQ.

## MPCC-MFCQ

**Mangasarian-Fromovitz Constraint Qualification (MPCC variant):** there exists a direction $d$ such that

$$
\begin{aligned}
\nabla h_k(x)^\top d &= 0 && \text{for all } k, \\
\nabla g_i(x)^\top d &< 0 && \text{for all } i \in \mathcal I_g, \\
\nabla G_i(x)^\top d &= 0 && \text{for all } i \in \mathcal I_G, \\
\nabla H_i(x)^\top d &= 0 && \text{for all } i \in \mathcal I_H, \\
\nabla G_i(x)^\top d &> 0 \text{ and } \nabla H_i(x)^\top d > 0 && \text{for all } i \in \mathcal I_{00},
\end{aligned}
$$

and the equality and active-pair gradients are linearly independent. MPCC-MFCQ is strictly weaker than MPCC-LICQ (the latter implies the former). It guarantees existence — but not uniqueness — of MPCC multipliers.

## Why this matters in practice

* **Multiplier uniqueness** — needed for sensitivity analysis (§5.1) and for `solve_jax` (§5.6). Both prerequisite MPCC-LICQ at the converged point. `pympcc.sensitivity` skips with `skipped_reason="biactive_pairs"` when biactive indices break the IFT prerequisites.
* **Strategy choice** — `lin_fukushima` is designed so that the regularised problem retains MPCC-MFCQ throughout the outer loop, even when other strategies lose CQ.
* **Diagnostics signal** — large `result.cq_rank_deficit` or many near-zero rows in `result.jac_row_norms` flag degeneracy that other diagnostics (multi-merit cross-check, condition numbers) can corroborate.

## References

* Scheel & Scholtes (2000). *Math. Oper. Res.* — definitions of MPCC-LICQ, MPCC-MFCQ.
* Flegel & Kanzow (2005). On M-stationary points for mathematical programs with equilibrium constraints. *J. Math. Anal. Appl.*
