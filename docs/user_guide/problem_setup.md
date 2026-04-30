# Problem setup

## `MPCCProblem`

The canonical numeric problem definition. All callables are validated at construction by evaluating them at `x0`.

```python
pympcc.MPCCProblem(
    n,                         # int — number of decision variables
    n_comp,                    # int — number of complementarity pairs
    x0,                        # (n,) — initial guess
    objective,                 # f(x)  → float
    gradient,                  # ∇f(x) → (n,)   or "fd" / "jax"

    # Complementarity — standard form:
    comp_G=None,               # G(x) → (n_comp,)   must be ≥ 0
    comp_G_jacobian=None,      # ∇G(x) → (n_comp, n)  or 1-D nnz values if sparse
    comp_H=None,               # H(x) → (n_comp,)   must be ≥ 0
    comp_H_jacobian=None,      # ∇H(x) → (n_comp, n)  or 1-D nnz values if sparse

    # Complementarity — MCP variable-paired form (alternative to comp_G / comp_H):
    comp_var_pairs=None,       # list of (var_idx, h_fn) or (var_idx, h_fn, h_jac_fn)

    xl=None, xu=None,          # bounds on x  (default: ±∞)

    n_ineq=0, ineq_constraints=None, ineq_jacobian=None,
    n_eq=0,   eq_constraints=None,   eq_jacobian=None,

    # Derivative shorthand — fills all unset derivative fields at once:
    derivatives=None,          # "fd" or "jax"

    # Sparse Jacobian support (COO format):
    comp_G_jacobian_sparsity=None,
    comp_H_jacobian_sparsity=None,
    ineq_jacobian_sparsity=None,
    eq_jacobian_sparsity=None,

    # Analytical Lagrangian Hessian (optional):
    lagrangian_hessian=None,
    lagrangian_hessian_sparsity=None,
    lagrangian_hessian_slack=None,
    lagrangian_hessian_slack_sparsity=None,

    fd_h=1.49e-8,
    fd_mode="forward",
)
```

## Variable-paired complementarity

For `x[j] ≥ 0 ⊥ F(x) ≥ 0` pairs, declare them at the variable level instead of writing `comp_G` manually:

```python
problem = pympcc.MPCCProblem(
    n=3, n_comp=2, x0=np.array([0.5, 0.5, 0.5]),
    objective=..., gradient=...,
    comp_var_pairs=[
        (0, lambda x: np.array([x[1]])),                                     # FD Jacobian
        (1, lambda x: np.array([x[2]]), lambda x: np.array([0., 0., 1.])),   # exact
    ],
)
```

`xl[var_idx]` is silently clamped to `max(xl[var_idx], 0.0)`. **Mixed mode**: provide both `comp_G`/`comp_H` and `comp_var_pairs` — `n_comp` must equal the sum.

## `StructuredMPCC`

Higher-level interface accepting linear constraints as matrices alongside nonlinear callables. Converted to `MPCCProblem` automatically by `solve()`.

```python
model = pympcc.StructuredMPCC(
    n=5, n_comp=2, x0=np.ones(5),
    objective=..., gradient=...,
    comp_G=..., comp_G_jacobian=...,
    comp_H=..., comp_H_jacobian=...,

    A_eq=np.array([[1, 1, 0, 0, 0]]),
    b_eq=np.array([1.0]),

    n_nl_ineq=1,
    ineq_nl=lambda x: np.array([x[0]**2 + x[1] - 2]),
    jac_ineq_nl=lambda x: np.array([[2*x[0], 1, 0, 0, 0]]),
)
result = pympcc.solve(model)
```

## Derivative shorthand

`derivatives="fd"` uses forward finite differences for every Jacobian + the gradient. `derivatives="jax"` uses `jax.jacfwd`/`jax.grad` and requires every callable to be `jax.numpy`-traceable. Per-field overrides (e.g. `comp_G_jacobian="jax"`) are also allowed.

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    problem = pympcc.MPCCProblem(
        n=2, n_comp=1, x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0]-1)**2 + (x[1]-1)**2,
        comp_var_pairs=[(0, lambda x: np.array([x[1]]))],
        derivatives="fd",
    )
```
