# Examples

Each page below is an executable notebook — built and run by Sphinx on every docs build, so the outputs you see are guaranteed to match the current code.

```{toctree}
:maxdepth: 1
bilevel_demo
solve_jax_demo
parametric_sweep
```

| Notebook | Demonstrates |
|---|---|
| [Bilevel KKT emission](bilevel_demo.md) | `pympcc.bilevel.from_lower_level` on two small bilevels — one with the lower bound inactive, one with it active. |
| [Differentiable solve](solve_jax_demo.md) | `jax.grad` through a converged MPCC; finite-difference verification. |
| [Parametric sweep](parametric_sweep.md) | Sensitivity-based prediction `x*(p+δ) ≈ x*(p) + (dx*/dp)·δ` validated against a re-solve. |
