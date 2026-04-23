# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-04-23

### Added

**KKT stationarity residual**
- `compute_kkt_residual(result, problem, *, mpcc_mult_G, mpcc_mult_H, mult_x_L, mult_x_U)`
  — computes `‖∇f + Jgᵀλ_g + Jhᵀλ_h + JGᵀμ_G + JHᵀμ_H − z_L + z_U‖_∞`;
  requires explicit MPCC multipliers to account for strategy-specific reformulation terms
- `MPCCResult.kkt_residual` — populated automatically by all six strategies using the
  correct per-strategy MPCC multiplier corrections:
  Scholtes/Direct: `μ_G = λ_G + H⊙λ_GH`, `μ_H = λ_H + G⊙λ_GH`;
  Lin-Fukushima: additionally `+λ_GPH` for both;
  Smoothing: `μ_G = λ_G + (1−G/r)⊙λ_φ`, `μ_H = λ_H + (1−H/r)⊙λ_φ`, `r=√(G²+H²+ε²)`;
  Augmented Lagrangian: `μ_G = λ_G + μ_AL⊙H`, `μ_H = λ_H + μ_AL⊙G`;
  Slack: direct z-space multipliers, variable bound multipliers truncated to `[:n]`
- `compute_kkt_residual` exported in the top-level `pympcc` namespace

**Bound validation**
- `MPCCProblem._validate()` now emits a `UserWarning` when `x0` violates any finite
  variable bound; IPOPT projects `x0` internally, so this is a warning not an error

**Stationarity documentation**
- Interior-point bias caveat added to `pympcc._stationarity` module docstring:
  IPOPT's barrier forces μ_G, μ_H ≥ 0 for all active lower-bound constraints at
  convergence, making `classify_stationarity` almost always return `"S-stationary"`;
  `kkt_residual` is the primary quality metric

**Performance kernels (`pympcc._kernels`)**
- `eval_phi_eps_weighted_union` — fused Fischer-Burmeister weighted union kernel (Numba JIT)
- `coo_to_dense` — COO-to-dense conversion kernel (Numba JIT)
- Pre-allocated Jacobian output buffers: `_union_buf` aliased as a view into `_jac_flat_buf`,
  eliminating the last per-call allocation on the hot path

**MacMPEC benchmark suite**
- Expanded to 13 problems (was 8); added `kth2`, `outrata32`, `simple_ineq`, `chain2`, `bilevel1`

**CI/CD**
- New `test-numba` job in `.github/workflows/tests.yml` verifies JIT kernels compile and
  produce correct results on Python 3.11 + numba≥0.57; pre-warms the Numba cache before
  the test run

**Analytical Lagrangian Hessian support**
- Four new optional fields on `MPCCProblem`:
  - `lagrangian_hessian` — callable `(x, lagrange, obj_factor) → nnz_values` for the strategies that operate in the original x-space (`direct`, `scholtes`, `lin_fukushima`)
  - `lagrangian_hessian_sparsity` — COO lower-triangle sparsity pattern `(row_indices, col_indices)` for the above
  - `lagrangian_hessian_slack` — same callable signature but evaluated in the lifted z-space `z = [x, s_G, s_H]`; used by the `slack` strategy
  - `lagrangian_hessian_slack_sparsity` — COO lower-triangle sparsity pattern for the lifted Hessian
- Strategies `direct`, `scholtes`, and `lin_fukushima` query `lagrangian_hessian` and fall back to JAX autodiff (if available) or L-BFGS
- Strategy `slack` queries `lagrangian_hessian_slack` and falls back to JAX autodiff or L-BFGS
- `_has_manual_hessian()` helper on `BaseStrategy` checks whether `lagrangian_hessian` is populated
- Constraint multiplier ordering for `lagrangian_hessian`: `[g (n_ineq), h (n_eq), G (n_comp), H (n_comp), G·H (n_comp)]`; `lin_fukushima` appends an additional `G+H (n_comp)` block (which has zero Hessian, so the same callable works for all three strategies)
- Constraint multiplier ordering for `lagrangian_hessian_slack`: `[h (n_eq), G−s_G (n_comp), H−s_H (n_comp), s_G·s_H (n_comp)]` where `z = [x, s_G, s_H]`
- `augmented_lagrangian` and `smoothing` are intentionally excluded: the PHR penalty and φ_ε smoothing introduce Hessian terms that cannot be expressed as a static callable independent of the strategy internals
- JAX autodiff Hessian fallback remains available via `pip install "pympcc[jax]"` for all four supported strategies

---

## [0.1.0] — 2026-04-21

Initial release.

### Added

**Core problem interfaces**
- `MPCCProblem` — numeric problem definition with dense or sparse Jacobians; all callables validated at construction by evaluating at `x0`
- `StructuredMPCC` — higher-level interface accepting linear constraints as matrices (`A_eq`, `b_eq`, `A_ineq`, `b_ineq`) alongside nonlinear callables; converts to `MPCCProblem` via `.to_mpcc_problem()`
- Finite-difference Jacobian fallback: pass `"fd"` as any Jacobian argument in `StructuredMPCC` (`gradient`, `comp_G_jacobian`, `comp_H_jacobian`, `jac_eq_nl`, `jac_ineq_nl`); supports forward and central differences; warns at construction

**Strategies — six NLP-based reformulations**
- `"direct"` — single IPOPT solve with `G·H ≤ 0`
- `"scholtes"` — sequential relaxation `G·H ≤ ε`, `ε → 0` (Scholtes 2001)
- `"smoothing"` — sequential Fischer-Burmeister smoothing `φ_ε(G,H) = 0`, `ε → 0`
- `"lin_fukushima"` — sequential `G·H ≤ ε` and `G+H ≥ ε`; guarantees MPCC-MFCQ (Lin & Fukushima 2003)
- `"augmented_lagrangian"` — PHR penalty in the objective; complementarity never enters the NLP constraints
- `"slack"` — lifts `G(x)`, `H(x)` to slack variables `s_G`, `s_H`; complementarity rows have zero x-entries (`2·n_comp` nonzeros total, independent of `n`)

All iterative strategies share: dual warm-starting between outer iterations (`dual_warmstart=True`), configurable `epsilon_0`, `reduction`, `max_iter`, `epsilon_min`.

**Sparse Jacobian support**
- Four optional `*_jacobian_sparsity` fields on `MPCCProblem` (COO format: `(row_indices, col_indices)`)
- When set, Jacobian callables return 1-D nnz-value arrays; `_SparseNLP` adapter passes the structure to IPOPT's `jacobianstructure()`
- All strategies auto-detect sparsity and dispatch to the sparse NLP path
- Derived blocks (`G·H`, `G+H`, `φ_ε`) computed from union of G and H sparsity patterns — no dense `(m, n)` matrix allocated on hot-path callbacks
- Union index maps (`map1`, `map2`) precomputed once per solve in `_make_union_maps`

**Performance kernels (`pympcc._kernels`)**
- `eval_weighted_union` — fills derived-block values from union sparsity without allocation; optional pre-allocated output buffer
- `weighted_row_sum` — `out[i,j] = α[i]·A[i,j] + β[i]·B[i,j]` in-place, single pass
- `scatter_add` — equivalent to `np.add.at` for gradient accumulation
- Optional Numba JIT compilation (`pip install "pympcc[numba]"`); pure-NumPy fallbacks when Numba is absent; `HAS_NUMBA` flag

**Result and diagnostics**
- `MPCCResult` — solution, objective, complementarity values, success flag, IPOPT status, per-iteration history, constraint multipliers, stationarity classification
- `IterationInfo` — per-outer-iteration snapshot: `epsilon`, `x`, `obj`, `status`, `message`, `comp_residual`
- `classify_stationarity(result, problem, tol)` — classifies converged solutions as `"S-stationary"`, `"not stationary"`, or `"unknown"` based on biactive set and multiplier signs

**Benchmark suite**
- 8 MacMPEC problems (`tests/macmpec_problems.py`) with exact Jacobians and verified optimal values
- Parametrized benchmark tests across all strategies: convergence, objective accuracy (≤ 1% relative), complementarity feasibility (< 1e-4)

**Examples** (`examples/`)
- `01` – `05`: stationarity hierarchy demonstrations
- `06`: strategy comparison on the same problem
- `07`: sparse Jacobian API walkthrough
- `08`: parallel sparse solve
- `09`: slack strategy — nnz reduction table, timing, stationarity on an imaging-proxy problem
