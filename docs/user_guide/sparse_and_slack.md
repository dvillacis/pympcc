# Sparse Jacobians and the slack strategy

## Sparse Jacobians (large-$n$ problems)

When `n_comp` Jacobian rows each have only a few non-zero columns, pass sparsity patterns in COO format. Jacobian callables then return 1-D arrays of `nnz` values instead of dense `(n_comp, n)` matrices.

```python
# Example: n=1000, comp_G only depends on variables 0 and 1
rows = np.array([0, 0])   # row indices
cols = np.array([0, 1])   # column indices

problem = pympcc.MPCCProblem(
    n=1000, n_comp=1, ...,
    comp_G_jacobian=lambda x: np.array([2*x[0], 3*x[1]]),  # returns nnz values
    comp_G_jacobian_sparsity=(rows, cols),
)
```

All six strategies dispatch to the sparse NLP adapter automatically when any sparsity field is set.

## Slack strategy for large-$n$ problems

The `"slack"` strategy introduces explicit slack variables $s_G = G(x)$, $s_H = H(x)$ so the complementarity Jacobian rows have **zero entries in $x$** — only $2 \cdot n_\text{comp}$ nonzeros regardless of $n$.

```python
result = pympcc.solve(problem, strategy="slack")
```

For an imaging application with $n=10{,}000$ and $n_\text{comp}=50$:

| Strategy | comp-row nnz | total Jacobian entries |
|---|---|---|
| Scholtes / smoothing | 10,000 / row | 50 × 10,000 = 500,000 |
| Slack | 2 / row | 50 × 2 = 100 (+ pinning) |

## Analytical Lagrangian Hessian

By default IPOPT uses L-BFGS. Supplying the exact Hessian improves inner-NLP convergence:

```python
def my_hessian(x, lagrange, obj_factor):
    # Multiplier ordering (direct / scholtes / lin_fukushima):
    #   lagrange = [λ_g (n_ineq), λ_h (n_eq), λ_G (n_comp), λ_H (n_comp)]
    ...
    return nnz_values  # (nnz,), lower triangle

problem = pympcc.MPCCProblem(
    ...,
    lagrangian_hessian=my_hessian,
    lagrangian_hessian_sparsity=(hess_rows, hess_cols),  # row ≥ col
)
```

The `slack` strategy uses `lagrangian_hessian_slack` with $z = [x, s_G, s_H]$.

**Compatibility:**

- Compatible: `direct`, `scholtes`, `lin_fukushima` (via `lagrangian_hessian`); `slack` (via `lagrangian_hessian_slack`).
- Not compatible: `augmented_lagrangian`, `smoothing` — their objective perturbations add second-derivative terms not captured by a static callable.
- The SOSC check uses the user-supplied Hessian when present, eliminating FD cost.

## Auto pair-scaling

`pympcc.autoscale_comp_pairs(problem)` returns a copy of the problem with each complementarity pair re-scaled to balance $\lvert G_i \rvert$ and $\lvert H_i \rvert$ at $x_0$. Useful when the two sides of a pair have very different magnitudes.
