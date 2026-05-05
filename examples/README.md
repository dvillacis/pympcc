# pympcc examples

Runnable scripts under this directory.  Each can be executed as
`uv run python examples/<script>.py` (or `python examples/...` once
pympcc is installed in your active environment).

## Quick reference

| Script | Topic | Key API surface |
|---|---|---|
| [`01_stationarity_hierarchy.py`](01_stationarity_hierarchy.py) | The four MPCC stationarity levels (S / M / C / W) demoed with hand-crafted multipliers | `pympcc.classify_stationarity` |
| [`02_s_stationary_nonbiactive.py`](02_s_stationary_nonbiactive.py) | S-stationarity verified at a non-biactive solution (MPCC-LICQ holds) | `pympcc.classify_cq`, `result.b_stationary` |
| [`03_s_stationary_biactive.py`](03_s_stationary_biactive.py) | S-stationarity verified at a biactive solution (MPCC-LICQ fails, MPCC-MFCQ holds) | `pympcc.classify_cq`, `tnlp_refine=True` |
| [`04_c_stationary.py`](04_c_stationary.py) | C-stationary point (multiplier signs admit "both negative" on a biactive pair) | Stationarity hierarchy |
| [`05_w_stationary.py`](05_w_stationary.py) | W-stationary point (KKT of the relaxed NLP holds without a stronger sign) | Stationarity hierarchy |
| [`06_strategies_comparison.py`](06_strategies_comparison.py) | Run several strategies on a small MacMPEC-style benchmark and compare | `pympcc.solve(strategy=...)` |
| [`07_sparse_api.py`](07_sparse_api.py) | Supplying COO sparsity for the comp / equality / inequality Jacobians | `comp_G_jacobian_sparsity`, etc. |
| [`08_sparse_parallel.py`](08_sparse_parallel.py) | Sparse Jacobians + multistart parallelism (`n_jobs=-1`) | `pympcc.multistart`, `n_jobs` |
| [`09_slack_strategy.py`](09_slack_strategy.py) | The slack-lifting strategy on a large-n problem | `strategy="slack"` |
| [`10_jax_backend.py`](10_jax_backend.py) | JAX autodiff backend: gradient + Jacobians + Hessian | `derivatives="jax"`, `use_jax_hessian=True` |
| [`11_kkt_residual.py`](11_kkt_residual.py) | Reading the MPCC-KKT residual to gauge stationarity quality | `result.kkt_residual`, `compute_kkt_residual` |
| [`perf_profile.py`](perf_profile.py) | Wall-clock + cProfile performance scan across problem sizes | profiling harness |

## Executable notebooks on the docs site

The bulk of the package's runnable demos live as myst-nb notebooks
under [`docs/examples/`](../docs/examples/) — they are executed by
Sphinx on every docs build, so the outputs cannot drift from the
code.  Browse them on the rendered site, or open the markdown
directly:

**Getting started:**
* [A first MPCC](../docs/examples/simple_qpcc.md)
* [Tour of strategies](../docs/examples/strategies_tour.md)
* [MCP variable-paired form](../docs/examples/mcp_form.md)
* [Sparse Jacobians](../docs/examples/sparse_jacobians.md)
* [Stationarity hierarchy](../docs/examples/stationarity_hierarchy.md)
* [KKT residual](../docs/examples/kkt_residual.md)

**Strategies & solvers:**
* [Slack-lifting strategy](../docs/examples/slack_strategy.md)
* [NCP variants](../docs/examples/ncp_variants_tour.md)
* [Multistart](../docs/examples/multistart.md)
* [Stateful warm hot-start](../docs/examples/warm_start.md)

**Diagnostics & certification:**
* [Presolve](../docs/examples/presolve.md)
* [Diagnostics tour](../docs/examples/diagnostics_tour.md)
* [TNLP refinement](../docs/examples/tnlp_refinement.md)

**Bilevel & differentiable:**
* [Bilevel KKT emission](../docs/examples/bilevel_demo.md)
* [Differentiable solve via custom_vjp](../docs/examples/solve_jax_demo.md)
* [Parametric sweep](../docs/examples/parametric_sweep.md)

**Topics covered elsewhere:**
* **EPEC multi-leader emission** — [docs/user_guide/epec.md](../docs/user_guide/epec.md)
  contains a Cournot-game quickstart for `pympcc.from_epec`.
* **AMPL `.nl` reader** — [docs/user_guide/ampl_io.md](../docs/user_guide/ampl_io.md).
* **Pyomo frontend** — [docs/user_guide/problem_setup.md](../docs/user_guide/problem_setup.md)
  has a `from_pyomo` snippet.

## Running

```bash
uv run python examples/06_strategies_comparison.py
uv run python examples/perf_profile.py --kmax 200 --strategy scholtes
```

The performance profiler accepts CLI flags; pass `--help` for
options.  All other scripts are runnable as-is and print to stdout.
