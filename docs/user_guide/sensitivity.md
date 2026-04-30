# Parametric sensitivity

`pympcc.sensitivity` returns $\partial x^* / \partial p$ and $\partial \lambda^* / \partial p$ at a converged MPCC solution by implicit differentiation through the KKT system. No re-solve required.

## When to use it

Whenever you have a parameter $p$ entering the problem (cost weights, RHS values, regulariser strengths) and want sensitivities of the optimum to $p$ — for parameter studies, variance estimation, gradient-based outer optimisation, or `dx*/dp` Jacobians for downstream code.

For end-to-end JAX gradients, prefer the higher-level [`solve_jax`](autodiff.md), which wraps this primitive in `jax.custom_vjp`.

## API

```python
import pympcc

result = pympcc.solve(problem, strategy="scholtes", tnlp_refine=True)

sens = pympcc.sensitivity(
    result, problem,
    dgrad_L_dp=...,   # (n, n_p)         — ∂(∇_xL)/∂p at x*
    dc_dp=...,        # (m_active, n_p)  — ∂c_active/∂p
)

sens.dx_dp           # (n, n_p)
sens.dlam_dp         # (m_active, n_p)
sens.skipped_reason  # "not_converged" | "biactive_pairs" | None
sens.rank_deficit    # int
sens.used_pseudoinverse  # bool — Tikhonov fallback fired
```

## Active-row ordering for `dc_dp`

The active-constraint block stacks rows in the order

$$
[\,h \ \mid\ G_{I_G} \ \mid\ H_{I_H} \ \mid\ g_{I_g} \ \mid\ \text{active bounds}\,].
$$

Use the helper to read the exact labels off a result:

```python
labels = pympcc.active_row_labels(result, problem)
# e.g. [("h", 0), ("G", 1), ("H", 0), ("g", 2), ("xL", 4)]
```

Build `dc_dp` with rows in that order.

## Skip conditions

`sensitivity` returns zero `dx_dp` / `dlam_dp` and sets `skipped_reason` when MPCC-LICQ prerequisites fail:

- `"not_converged"` — `result.success` is `False`.
- `"biactive_pairs"` — IFT invalid at biactive points.
- `"no_hessian_callable_and_fd_failed"` — finite-difference Hessian fallback raised.

## Multipliers

When `tnlp_refine=True` was passed to `solve`, `sensitivity` automatically picks up the certified TNLP multipliers. Without TNLP it falls back to zero multipliers and emits a `UserWarning` — the result is exact only when constraints are linear in $x$.
