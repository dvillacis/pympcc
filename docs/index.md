# pympcc

A Python solver for **Mathematical Programs with Complementarity Constraints (MPCC)**, built on [IPOPT](https://github.com/coin-or/Ipopt) via [cyipopt](https://github.com/mechmotum/cyipopt).

## Problem form

$$
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g(x) \le 0 \\
                  & h(x) = 0  \\
                  & G(x) \ge 0,\ H(x) \ge 0,\ G(x)^\top H(x) = 0.
\end{aligned}
$$

```{toctree}
:maxdepth: 2
:caption: Getting started
installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: User guide
user_guide/problem_setup
user_guide/strategies
user_guide/diagnostics
user_guide/sparse_and_slack
user_guide/sensitivity
user_guide/autodiff
user_guide/bilevel
user_guide/presolve_multistart
user_guide/ampl_io
```

```{toctree}
:maxdepth: 1
:caption: Reference
api/index
changelog
```

## Why pympcc

- **Six reformulation strategies** — direct, Scholtes, smoothing (Fischer-Burmeister), Lin-Fukushima, augmented Lagrangian, slack lifting.
- **Certified stationarity** — TNLP refinement extracts MPCC-clean multipliers and classifies S- / W- / C-stationarity.
- **Diagnostics** — MPCC-LICQ / MPCC-MFCQ check, B-stationarity certification, MPCC-SOSC, multi-merit cross-check.
- **Differentiable** — `pympcc.solve_jax` registers the solve as `jax.custom_vjp` for end-to-end gradients through the converged optimum.
- **Bilevel-ready** — `pympcc.bilevel.from_lower_level` compiles bilevel programs to MPCCs by emitting lower-level KKT.

## Indices

- {ref}`genindex`
- {ref}`modindex`
