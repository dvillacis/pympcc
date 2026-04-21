# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
