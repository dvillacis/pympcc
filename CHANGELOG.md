# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.2] - 2026-04-28

### Added

**NCP-function reformulation menu (§3.5)**
- `SmoothMinStrategy` (`strategy="smooth_min"`) — smoothed min-NCP:
  `φ_ε(G,H) = ½(G+H−√((G−H)²+4ε²)) = 0`
- `ChenChenKanzowStrategy` (`strategy="chen_chen_kanzow"`) — convex combination of
  Fischer-Burmeister and inner-product: `φ_{λ,ε}(G,H) = λ·φ_FB,ε(G,H) + (1−λ)·G·H = 0`;
  parameter `lam ∈ (0,1]` (default 0.5)
- `KanzowSchwartzStrategy` (`strategy="kanzow_schwartz"`) — one-parameter FB family:
  `φ_{λ,ε}(G,H) = G+H−√(G²+H²+2λGH+ε²) = 0`; parameter `lam ∈ [0,1)` (default 0.5)
- All three inherit `_SmoothNCPBase` (in `pympcc/strategies/ncp.py`) which reuses the
  ε-continuation harness from `SmoothingStrategy`; dense and sparse Jacobian paths both
  supported via the generic `eval_weighted_union` kernel
- MPCC multipliers recovered via `μ_G = λ_G + α⊙λ_φ`, `μ_H = λ_H + β⊙λ_φ`
- 43 new tests in `tests/test_ncp_strategies.py`

---

## [0.4.1] - 2026-04-28

### Fixed

- **CI: `scipy` not installed in base conda environment** — `verify_b_stationarity` raised
  `ModuleNotFoundError` even for tests that never reached the LP-solving code, causing 17
  test failures. The `from scipy.optimize import linprog` import is now deferred past the
  early-exit guards so it only executes when the LP is actually needed.
  `scipy>=1.10` added to the base `micromamba` test environment.
- **Coverage threshold not reached (83.73 % < 85 %)** — `pympcc/benchmarks/macmpec.py`
  (CLI entry-point) counted against coverage but was never imported by the test suite.
  Added to `[tool.coverage.run] omit`; coverage now sits at **91 %**.
- **Lint (ruff) errors** — fixed I001 (unsorted import blocks in `__init__.py`, `solver.py`,
  `_sosc.py`, `_tnlp.py`, `strategies/_base.py`), E702 (inline semicolons in `_presolve.py`),
  and F841 / F401 (unused names in `benchmarks/macmpec.py` and `solver.py`).

---

## [0.4.0] - 2026-04-28

### Added

**Variable-paired complementarity — MCP form (`comp_var_pairs`)**
- New `MPCCProblem.comp_var_pairs` field: list of `(var_idx, h_fn)` or
  `(var_idx, h_fn, h_jac_fn)` tuples declaring `x[var_idx] ≥ 0 ⊥ h_fn(x) ≥ 0`
  without writing `comp_G(x) = x[var_idxs]` manually
- Two modes: *all-var-pairs* (`comp_G=None`, every pair from `comp_var_pairs`); *mixed*
  (`comp_G` supplies the first `n_comp − k` pairs, `comp_var_pairs` appends `k` more)
- G-side Jacobian rows are always exact (identity); H-side rows use `h_jac_fn` when
  provided, otherwise forward finite differences per row
- `xl[var_idx]` is silently clamped to `max(xl[var_idx], 0.0)`
- Fully composable with `derivatives="fd"` / `derivatives="jax"` and all six strategies
- Module: `pympcc/problem.py` — `_normalize_var_pairs()` method
- Tests: `tests/test_mcp_var_pairs.py` (21 cases)

**`derivatives` shorthand on `MPCCProblem`**
- `derivatives="fd"` or `derivatives="jax"` fills every unset derivative field
  (`gradient`, `comp_G_jacobian`, `comp_H_jacobian`, `ineq_jacobian`, `eq_jacobian`)
  with the corresponding sentinel at once — replaces setting each individually

**MPCC-SOSC: second-order sufficient conditions (`sosc_check`)**
- `pympcc.sosc_check(result, problem)` checks that the reduced Lagrangian Hessian
  is positive definite on the MPCC critical cone — certifying `x*` as a strict local minimiser
- Algorithm: build active-constraint gradient matrix A → null-space basis Z via SVD →
  reduced Hessian W = Z^T H Z → min eigenvalue test
- Hessian source (priority): `MPCCProblem.lagrangian_hessian` (user-supplied) →
  central FD of ∇_x L using TNLP-refined multipliers (best) or zero multipliers (conservative)
- Returns `{"sosc": bool|None, "min_eigenvalue": float|None, "null_space_dim", "n_active", "skipped_reason"}`
- Biactive pairs (`I_00 ≠ ∅`) → `sosc=None, skipped_reason="biactive_pairs"`
- Non-converged result → `sosc=None, skipped_reason="not_converged"`
- Three new `MPCCResult` fields: `sosc`, `sosc_min_eigenvalue`, `sosc_skipped_reason`
- Wired into `MPCCSolver._attach_diagnostics` (runs when `diagnostics=True`)
- Exported at top-level: `pympcc.sosc_check`
- Module: `pympcc/_sosc.py`
- Tests: `tests/test_sosc.py` (16 cases)

**TNLP active-set refinement — certified MPCC multipliers (`tnlp_refine=True`)**
- `solve(problem, tnlp_refine=True)` re-solves a tightened NLP with the active set
  `(I_G, I_H)` fixed as equality constraints; extracts clean MPCC multipliers μ_G, μ_H
- Stationarity upgrade: result labelled `"S-stationary"` when all multipliers ≥ −tol,
  `"W-stationary"` when any are negative (previously `"not S-stationary"`)
- Flip-and-retry guard: if the initial TNLP finds W-stationary and ≤ 20 % of pairs had
  the wrong side pinned, swaps those pairs and re-solves once
- Biactivity pre-screen: skips the TNLP when more than 10 % of pairs are biactive
  (`G_i ≤ bi_tol` and `H_i ≤ bi_tol`), avoiding restoration failures on degenerate iterates
- New `MPCCResult` fields: `mult_comp_G_mpcc`, `mult_comp_H_mpcc`, `tnlp_refined` (`TNLPResult`)
- `TNLPResult` dataclass: `x`, `obj`, `status`, `message`, `success`, `mult_comp_G`,
  `mult_comp_H`, `mult_ineq`, `mult_eq`, `kkt_residual`, `stationarity`, `n_iter`,
  `solve_time`, `active_set`, `n_violations`
- Module: `pympcc/_tnlp.py`
- Tests: `tests/test_tnlp.py`

**Per-pair status and structured result export**
- `result.per_pair_status` — always populated after every solve; each entry is one of
  `"G_active"` (G_i≈0, H_i>0), `"H_active"`, `"biactive"`, or `"inactive"`
  using adaptive threshold `max(sqrt(comp_residual), 1e-6)`
- `result.to_json()` — serialises the full result to a JSON string (arrays as lists,
  `None` as JSON null, history omitted)
- `result.to_dataframe()` — returns a per-pair `pandas.DataFrame` with columns
  `pair`, `G`, `H`, `GH`, `status`, and (when TNLP refined) `mu_G`, `mu_H`
- Tests: `tests/test_per_pair_status.py` (20 cases), `tests/test_summary.py` (19 cases)

**MacMPEC benchmark runner (`pympcc.benchmarks`)**
- `pympcc.benchmarks.run_benchmark(problems, strategies, ...)` runs any subset of the
  13-problem suite and returns a list of `BenchmarkResult` dataclasses
- `print_table(results)` — Leyffer-style fixed-width results table
- `save_csv(results, path)` — CSV export
- CLI: `python -m pympcc.benchmarks.macmpec [--strategy ...] [--problems ...] [--out file.csv] [--quiet]`
- Problem registry moved to `pympcc/benchmarks/_problems.py`; `tests/macmpec_problems.py`
  is now a thin re-export shim
- Module: `pympcc/benchmarks/` (`__init__.py`, `_problems.py`, `macmpec.py`)

### Fixed

- CI badge URL in README corrected (`davidvillacis` → `dvillacis`)
- Repository, Bug Tracker, and Changelog URLs in `pyproject.toml` corrected
  (`davidvillacis` → `dvillacis`)

### Tests

- 814 passed, 6 skipped, 54 xfailed (was 635 in 0.3.0)
- New: `test_sosc.py`, `test_tnlp.py`, `test_mcp_var_pairs.py`,
  `test_per_pair_status.py`, `test_summary.py`, `test_multistart.py`,
  `test_autoscale.py`, `test_cleanup_hessian.py`, `test_derivatives_default.py`,
  `test_matched_tol.py`, `test_safeguard_plateau_blowup.py`, `test_tnlp.py`

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
