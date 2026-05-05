# Examples

Each page below is an executable notebook — built and run by Sphinx on every docs build, so the outputs you see are guaranteed to match the current code.

## Getting started

```{toctree}
:maxdepth: 1
simple_qpcc
strategies_tour
mcp_form
sparse_jacobians
stationarity_hierarchy
kkt_residual
```

| Notebook | Demonstrates |
|---|---|
| [A first MPCC](simple_qpcc.md) | The 15-line quickstart, walked through with full output explained. |
| [Tour of strategies](strategies_tour.md) | Same problem solved by all six canonical reformulations. |
| [MCP variable-paired form](mcp_form.md) | `comp_var_pairs` for $x_j \ge 0\ \perp\ F(x) \ge 0$ pairs. |
| [Sparse Jacobians](sparse_jacobians.md) | COO sparsity for `comp_G_jacobian`, etc.; dense vs. sparse parity. |
| [Stationarity hierarchy](stationarity_hierarchy.md) | S / M / C / W stationarity in one notebook. |
| [KKT residual](kkt_residual.md) | Reading and recomputing `result.kkt_residual`. |

## Strategies & solvers

```{toctree}
:maxdepth: 1
slack_strategy
ncp_variants_tour
multistart
warm_start
```

| Notebook | Demonstrates |
|---|---|
| [Slack-lifting strategy](slack_strategy.md) | `strategy="slack"` for large-`n` MPCCs with dense $G$, $H$. |
| [NCP variants](ncp_variants_tour.md) | The seven smooth-NCP-function reformulations side-by-side. |
| [Multistart](multistart.md) | `pympcc.multistart` with `n_starts`, `unique_optima()`, parallel notes. |
| [Stateful warm hot-start](warm_start.md) | `MPCCSolver.resolve()` for MPC and parametric sweeps. |

## Diagnostics & certification

```{toctree}
:maxdepth: 1
presolve
diagnostics_tour
tnlp_refinement
```

| Notebook | Demonstrates |
|---|---|
| [Presolve](presolve.md) | Pinned-variable elimination, FBBT, dead-pair pruning. |
| [Diagnostics tour](diagnostics_tour.md) | CQ, B-stationarity, SOSC, merit cross-check, degeneracy report. |
| [TNLP refinement](tnlp_refinement.md) | Certified MPCC multipliers via `tnlp_refine=True`. |

## Bilevel & differentiable

```{toctree}
:maxdepth: 1
bilevel_demo
solve_jax_demo
parametric_sweep
```

| Notebook | Demonstrates |
|---|---|
| [Bilevel KKT emission](bilevel_demo.md) | `pympcc.bilevel.from_lower_level` on two small bilevels. |
| [Differentiable solve](solve_jax_demo.md) | `jax.grad` through a converged MPCC; FD verification. |
| [Parametric sweep](parametric_sweep.md) | `pympcc.sensitivity` validated against a re-solve. |
