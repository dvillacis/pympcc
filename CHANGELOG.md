# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-04-26

### Added

**Presolve layer (`pympcc.presolve`, `pympcc.PresolveMap`)**
- Opt-in via `solve(problem, presolve=True)` or `MPCCSolver(problem, presolve=True)`;
  runs once at solver construction, returns a reduced problem to the strategy, then
  expands the result back to the original variable space
- A1 — pinned-variable elimination: substitutes out every `j` with `xl[j] == xu[j]`
- A2 — feasibility-based bound tightening (FBBT) over linear constraints; iterates to
  fixpoint and pins variables whose bounds collapse
- A4 — empty Jacobian row / column pruning: drops constraints with no nonzero
  Jacobian entries and variables with no nonzero column
- B1 — dead complementarity-pair pruning: drops pairs where one side is structurally
  zero and trivially satisfied at `x0`
- B2 — forced complementarity-pair detection: pairs where both sides are linear
  with a forced sign get reduced to a single equality
- B3 — prefix-equality pass: identifies linear comp pairs whose sign is determined
  by domain bounds and rewrites them as equalities; emits an infeasibility warning
  when both sides are forced strictly positive
- `PresolveMap.expand_result` re-evaluates `comp_G`/`comp_H` on the original problem,
  scatters `result.x` and per-iteration `history[k].x` back to the original size, and
  zero-pads multipliers / unit-pads pair scales for pruned indices

**Constraint qualification diagnostics (`pympcc.classify_cq`, `pympcc.active_sets`)**
- Opt-in via `solve(problem, diagnostics=True)`; the result gains
  `result.cq` ∈ {`"MPCC-LICQ"`, `"MPCC-MFCQ"`, `"none"`},
  `result.cq_active_set_sizes`, and `result.cq_rank_deficit`
- `active_sets(result, problem, tol)` returns the index partition
  `{I_g, I_G, I_H, I_00, I_xL, I_xU}`; LICQ tested via SVD rank of the stacked
  active-gradient matrix; MFCQ tested via an LP direction-finding subproblem
  (HiGHS) that maximises strict-descent slack subject to the active eq-block
  and bound-respecting branches

**B-stationarity auto-attached**
- With `diagnostics=True`, `verify_b_stationarity` runs after the inner solve
  and populates `result.b_stationary`, `result.b_stationary_witness`, and
  `result.b_stationary_min_descent`
- `MPCCSolver(..., b_stat_max_biactive=10)` caps the biactive-branch enumeration
  to keep the verification cheap on large problems

**`MPCCResult` extensions**
- Six new fields: `cq`, `cq_active_set_sizes`, `cq_rank_deficit`,
  `b_stationary`, `b_stationary_witness`, `b_stationary_min_descent`. All default
  to `None`; populated only when `diagnostics=True`

### Fixed

**B-stationarity tangent cone at I_p0 / I_0p (behaviour change)**
- `verify_b_stationarity` previously linearised the strict-positive sides of the
  biactive branches as inequalities (`∇G_i·d ≥ 0` on I_0p, `∇H_i·d ≥ 0` on I_p0).
  The MPCC linearised tangent cone forces these to *equalities*: a strictly
  positive component locally pins the zero-side gradient direction. The
  inequality form admitted spurious descent directions and could falsely flag
  some genuine global optima as not-B-stationary. Fixed by appending `JG[I_0p]`
  and `JH[I_p0]` to the equality block instead of the inequality block in
  `_stationarity.py`. All seven existing B-stat tests continue to pass

### Tests

- 635 passed, 1 skipped, 54 xfailed (was 569 in 0.2.0)
- New: `test_presolve.py`, `test_fbbt.py`, `test_empty_rows_cols.py`,
  `test_forced_pair.py`, `test_prefix_eq.py`, `test_diagnostics.py`,
  `test_b_stationarity.py`, `test_auto_epsilon_0.py`, `test_problem_scaling.py`,
  `test_restoration_awareness.py`, `test_rollback_backoff.py`

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
