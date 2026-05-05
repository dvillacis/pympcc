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

## Topics covered elsewhere

These features have user-guide pages but no dedicated `examples/*.py`
script.  See the linked pages for runnable snippets:

* **Bilevel emission** — [docs/examples/bilevel_demo.md](../docs/examples/bilevel_demo.md)
  walks through `pympcc.from_lower_level` on two small bilevels.
* **EPEC multi-leader emission** — [docs/user_guide/epec.md](../docs/user_guide/epec.md)
  contains a Cournot-game quickstart for `pympcc.from_epec`.
* **Differentiable solve** — [docs/examples/solve_jax_demo.md](../docs/examples/solve_jax_demo.md)
  uses `solve_jax` and `jax.grad` end-to-end.
* **Parametric sensitivity** — [docs/examples/parametric_sweep.md](../docs/examples/parametric_sweep.md)
  validates `pympcc.sensitivity(...)` against a re-solve.
* **AMPL `.nl` reader** — [docs/user_guide/ampl_io.md](../docs/user_guide/ampl_io.md).
* **Pyomo frontend** — [docs/user_guide/problem_setup.md](../docs/user_guide/problem_setup.md)
  has a `from_pyomo` snippet.
* **Presolve / multistart** — [docs/user_guide/presolve_multistart.md](../docs/user_guide/presolve_multistart.md).
* **NCP-function variants** — [docs/user_guide/strategies.md](../docs/user_guide/strategies.md)
  documents the seven smooth-NCP entry points; pick one and pass to
  `pympcc.solve(strategy="...")`.

## Running

```bash
uv run python examples/06_strategies_comparison.py
uv run python examples/perf_profile.py --kmax 200 --strategy scholtes
```

The performance profiler accepts CLI flags; pass `--help` for
options.  All other scripts are runnable as-is and print to stdout.
