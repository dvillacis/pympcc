# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.6.0] - 2026-05-05

The Phase-3 hardening release.  Closes every audit-flagged debt item
in `ROADMAP.md` §7 ahead of a public 1.0; no API breakage.

### Added

**filterSQP backend (§3.2)**
- New `pympcc/_filtersqp_adapter.py` plus `BackendName` literal
  expansion to `Literal["ipopt", "filterSQP", "scipy"]`.  Strategies
  with dense Jacobians dispatch through the optional
  `pyfiltersqp` extra.

**Documentation (§7.4.4)**
- New `docs/user_guide/strategy_selection.md` — TL;DR rule of thumb,
  decision tree keyed on observable problem properties, FB-stalling
  fallbacks, and warm-start guidance.
- New `docs/user_guide/troubleshooting.md` — symptom-to-action map
  for IPOPT divergence, restoration loops, biactive pairs, when to
  use `tnlp_refine`, and a glossary of every diagnostic field on
  `MPCCResult`.
- New `docs/user_guide/epec.md` — the §5.5 EPEC v1 docs gap;
  Cournot-game quickstart, `result.x` slicing recipe, derivative
  requirements (JAX-only in v1), v1 limitations.
- New `examples/README.md` — index of every `examples/*.py` script
  with cross-links to user-guide pages for features documented
  there.
- `docs/user_guide/strategies.md` — added a full NCP-variants
  section (`smooth_min`, `chen_chen_kanzow`, `kanzow_schwartz`,
  `chen_mangasarian`, `billups`, `veelken_ulbrich_pow`,
  `veelken_ulbrich_sin`) with formulas, parameter ranges, and
  when-to-reach-for-it blurbs.

**Tooling (§7.5)**
- New `.pre-commit-config.yaml` mirroring the CI lint + type-check
  job: `ruff check`, `ruff format`, `mypy --ignore-missing-imports`,
  trailing-whitespace, EOF-fixer, `check-yaml`, `check-toml`.
- New `pympcc[all]` optional-dependencies group aggregating
  numba + jax + scipy + pyomo + pyfiltersqp.
- New `test-all-extras` CI job exercising every optional extra in
  one environment to catch interaction bugs.
- Python 3.13 added to the `pyproject.toml` classifier list (the CI
  matrix already had it).

**Public API tidying (§7.4.2)**
- `pympcc.frontend.from_nl`, `pympcc.frontend.from_pyomo`,
  `pympcc.frontend.apply_solution`, `pympcc.frontend.PyomoMPCC`
  re-exported at the `pympcc.frontend` namespace.
- `pympcc.from_lower_level`, `pympcc.from_epec`, `pympcc.Leader`,
  `pympcc.LowerLevel` re-exported at the top level.
- `pympcc.jac_condition_number` lifted to the public surface for
  consistency with the six other diagnostic exports.

**Typing (§7.4.3)**
- New `pympcc/_typing.py` centralising the `Literal` aliases:
  `StrategyName`, `BackendName`, `FDMode`, `Derivatives`,
  `InnerTolMode`.  Strategy / backend kwargs across the public API
  now narrow to these literals instead of bare `str`.
- Return-type annotations tightened on
  `_diagnostics.active_sets/classify_cq/jac_norms/merit_cross_check/
  initial_point_statistics/degeneracy_report` and
  `sensitivity.active_row_labels`.

### Changed

**Multistart determinism (§7.3.2)**
- Every parallel-path test was added: `problem.x0` immutability
  across spawn-mode workers, bit-identical iterates across runs at
  the same seed, identical iterates between `n_jobs=1` and
  `n_jobs=2`.  The seed propagation via
  `SeedSequence.spawn(n_starts)` was already shipped in 0.5.0; this
  release locks it in with regression tests.
- `multistart.py` documents the broad-by-design `except Exception`
  catches; `MultiStartResult.failures` records type + message per
  failed start.

**Hot-path callbacks (§7.1)**
- `ncp.py` dense-Jacobian path: pre-allocated `(m, n)` output buffer
  at strategy construction; per-call writes block-rows in-place.
  Replaces a `np.vstack` allocation per IPOPT iteration.
- `scholtes.py` and `lin_fukushima.py` cleanup-Hessian wrappers:
  pre-allocate the zero-padded Lagrangian buffer; per-call slice-write
  `lam_cu` into it.  Replaces a `np.concatenate` per cleanup IPOPT
  iteration.
- `_tnlp.py` multiplier rescaling: dropped redundant
  `np.asarray(p.comp_G_scale, dtype=float)` rewrap — scale arrays
  are canonicalised at `MPCCProblem.__post_init__`.

**Internal exception handling (§7.2.7)**
- `sensitivity.py:_build_hessian` failure path: tightened from bare
  `except Exception:` to a narrow tuple
  `(ArithmeticError, AttributeError, TypeError, ValueError,
  RuntimeError, np.linalg.LinAlgError)` plus a `logger.debug`
  emit so previously-silent fallbacks now surface in debug logs.
- Same tightening in `_base.py:_maybe_run_cleanup`.

**README + docs/index.md**
- "Six reformulation strategies" → "Thirteen reformulation
  strategies" (six canonical + seven NCP variants), with a link to
  the new selection guide.

**`mypy` configuration (§7.4.3)**
- Replaced the global `ignore_missing_imports = true` with
  module-scoped overrides for cyipopt / scipy / pyomo / jax / jaxlib
  / numba / pyfiltersqp / pandas / `pympcc.cython.*`.  User code
  importing pympcc now benefits from full strict-typing visibility.

### Internal

**Centralised constants (§7.2.5)**
- `pympcc/_constants.py` already shipped in 0.5.0; this release adds
  `SPARSITY_TOL` and migrates ~20 remaining call sites
  (`_diagnostics`, `_stationarity`, `_sosc`, `sensitivity`,
  `_tnlp`, `_autodiff`, `slack`, `augmented_lagrangian`,
  `benchmarks/macmpec`, `models`, `problem`, `_jax`).
- Empirical-default values (`cleanup_obj_worsen_tol=1e-3`,
  `comp_eps_ratio_theta=10.0`, `eps_hold_factor=3.0`) now have
  one-line justifications in `_base.py`'s `SAFEGUARD_DEFAULTS` /
  `CLEANUP_DEFAULTS`.

**Strategy `__init__` deduplication (§7.2.1)**
- `BaseStrategy._init_continuation_options(problem, ipopt_options,
  defaults, kwargs)` is the single source of truth for the
  merge / validate / set-attrs / safeguard / cleanup boilerplate.
  Five strategy classes (`scholtes`, `smoothing`, `lin_fukushima`,
  `slack`, `_SmoothNCPBase`) collapsed from a 23-line `__init__`
  to a single line.

**NCP variant deduplication (§7.2.2)**
- New `_RegistryShim(_SmoothNCPBase)` base binds a `(phi, grad)`
  pair from `pympcc._reformulation.NCP_REGISTRY` to the
  `_SmoothNCPBase` ε-continuation harness via `partial(...)`.
- The seven dedicated NCP variant classes
  (`SmoothMinStrategy`, `ChenChenKanzowStrategy`, ...,
  `VeelkenUlbrichSinStrategy`) demoted to ~15-line shims (down
  from ~40-50 each).  Public string names in `_STRATEGIES` are
  unchanged; subclass overrides + `tests/test_ncp_strategies.py`
  unaffected.
- Net: −139 lines in `pympcc/strategies/`.

**God-module splits (§7.2.3)**
- `pympcc/_presolve.py` (1247 → 379 LOC) split into
  `_presolve_detect.py` (402 LOC), `_presolve_fbbt.py` (158 LOC),
  `_presolve_reduce.py` (388 LOC).  `PresolveMap` and the public
  `presolve()` entry point stay in `_presolve.py`; private
  helpers re-exported for backward compat with test imports.
- `pympcc/problem.py` (1543 → 449 LOC) split into
  `_problem_derivatives.py` (206 LOC) and
  `_problem_reduction.py` (972 LOC).  `MPCCProblem` dataclass +
  `__post_init__` orchestrator + `_validate` + `_check_shape` stay.
- `pympcc/strategies/_base.py` (1636 → 778 LOC) refactored with a
  three-mixin composition: `_safeguards_mixin.py` /
  `_cleanup_mixin.py` / `_continuation_mixin.py`.
  `class BaseStrategy(SafeguardsMixin, CleanupMixin,
  ContinuationMixin, ABC)` — every existing `self._method(...)`
  call resolves through normal MRO; subclass overrides keep
  working unchanged.

**Time-limit enforcement (§7.3.1)**
- Already shipped in 0.5.0; verified in this release that every
  inner IPOPT solve receives `max_cpu_time` from the remaining
  wall-clock budget (covers ε-continuation main loop, cleanup
  phase, and the augmented Lagrangian outer loop).

### Community files (§7.4.1)

- `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `AUTHORS.md`
  shipped in 0.5.0; `CONTRIBUTING.md` updated this release with the
  pre-commit install + run flow.

### Deferred (P2 polish, post-1.0)

- Test directory split into `tests/unit/`, `tests/integration/`,
  `tests/bench/` (mechanical move; would touch every test path).
- Dedicated `examples/*.py` scripts for `smoothing`,
  `lin_fukushima`, `augmented_lagrangian`, NCP variants, the Pyomo
  frontend, presolve, and multistart.  The new
  `examples/README.md` cross-links to user-guide pages where these
  features have runnable snippets; standalone scripts are an
  incremental win.

---

## [0.5.0] - 2026-05-01

### Added

**Sphinx documentation site**
- New `docs/` tree with Furo theme, autodoc + autosummary API reference,
  numpy-style docstring rendering, and intersphinx links to NumPy / JAX /
  SciPy.  Built on every push by ReadTheDocs (`.readthedocs.yaml`).
- Three executable notebooks under `docs/examples/` powered by
  `myst-nb` (`bilevel_demo`, `solve_jax_demo`, `parametric_sweep`) — the
  output blocks are produced from real solves on every build, so they
  cannot drift from the code.
- Theory primer under `docs/theory/` covering MPCC fundamentals, why
  generic LICQ fails, MPCC-LICQ / MPCC-MFCQ, the S/M/C/A/W/B-stationarity
  hierarchy, and MPCC-SOSC with the biactive critical-cone wedge.
- User guide pages under `docs/user_guide/` covering problem setup,
  strategies, diagnostics, sparse / slack form, sensitivity, autodiff,
  bilevel, presolve / multistart, and AMPL I/O.
- README slimmed to a one-page pitch with deep-links into the docs;
  per-feature examples moved into the user guide.
- `pyproject.toml` gains a `[docs]` extra (`sphinx`, `myst-nb`, `furo`,
  `sphinx-copybutton`, `sphinx-autodoc-typehints`, `matplotlib`).

**JAX-differentiable solve (§5.6)**
- New :class:`pympcc.ParametricMPCC` dataclass — a parametric problem
  description whose ``objective``, ``eq_constraints``,
  ``ineq_constraints``, ``comp_G``, ``comp_H`` callables take
  ``(x, theta)`` with JAX-traceable bodies.  ``materialise(theta, x0)``
  closes ``theta`` and returns an :class:`MPCCProblem` with
  ``derivatives="jax"``.
- New :func:`pympcc.solve_jax(parametric, theta, *, x0, strategy=...,
  **solve_kwargs)` registered as ``jax.custom_vjp``: forward calls
  :func:`pympcc.solve` with ``tnlp_refine=True`` (§2.6); backward
  performs an sIPOPT-style adjoint solve of the KKT saddle system
  ``K·[u; w] = [v; 0]`` once and returns ``θ̄ = −(uᵀ ∂(∇_xL)/∂θ +
  wᵀ ∂c/∂θ)`` via :func:`jax.vjp` on the parametric callables, so
  pytree θ shapes pass through.
- Skip behaviour mirrors :func:`pympcc.sensitivity`: returns a
  zero θ-cotangent with a ``UserWarning`` when the forward solve
  fails to converge or the optimum is biactive (IFT prerequisites
  invalid).
- Public exports: ``pympcc.ParametricMPCC``, ``pympcc.solve_jax``.
- 11 new tests in ``tests/test_autodiff.py`` covering forward-only
  parity with :func:`pympcc.solve`, closed-form gradient on a
  parametric equality NLP, FD verification of ``jax.grad`` and the
  full ``dx*/dθ`` Jacobian (assembled per-row via :func:`jax.grad`),
  the biactive skip path, and the documented incompatibility with
  :func:`jax.jit`.

**Parametric sensitivity analysis (§5.1)**
- New module :mod:`pympcc.sensitivity` exposing
  ``pympcc.sensitivity(result, problem, *, dgrad_L_dp, dc_dp, ...)`` —
  a low-level KKT-linear-solve primitive that returns ``dx*/dp`` and
  ``dλ*/dp`` at a converged MPCC solution by implicit differentiation.
- Solves the saddle-point system ``[[H, J_cᵀ], [J_c, 0]] · [dx; dλ] =
  −[dgrad_L_dp; dc_dp]`` where ``H`` is the Lagrangian Hessian (reusing
  the §2.3 Hessian builder) and ``J_c`` is the active-constraint
  Jacobian (reusing the §2.1 active-set machinery).
- Skips with ``skipped_reason`` set when MPCC-LICQ prerequisites fail:
  ``"not_converged"`` (failed result), ``"biactive_pairs"`` (IFT
  invalid at biactive points), or
  ``"no_hessian_callable_and_fd_failed"`` (FD fallback raised).
- Tikhonov-regularised ``lstsq`` fallback when the dense KKT solve
  fails; ``rank_deficit`` and ``used_pseudoinverse`` flags surface the
  fallback path.
- Companion helper ``pympcc.active_row_labels(result, problem)`` returns
  the per-row labels (``("h", k)`` / ``("G", i)`` / ``("H", i)`` /
  ``("g", j)`` / ``("xL"|"xU", j)``) the caller's ``dc_dp`` rows must
  follow, so users can assemble the RHS without first running a
  sensitivity solve.
- Picks up TNLP-refined multipliers (§2.6) automatically when
  ``tnlp_refine=True`` was passed to :func:`pympcc.solve`; falls back
  to zero multipliers with a ``UserWarning`` when no analytic Hessian
  is available (the fallback is exact only when constraints are linear
  in ``x``).
- Public exports: ``pympcc.sensitivity``, ``pympcc.SensitivityResult``,
  ``pympcc.active_row_labels``.
- 17 new tests in ``tests/test_sensitivity.py`` covering closed-form
  IFT on a parametric equality NLP, end-to-end FD verification on a
  branch-selection MPCC (single- and multi-parameter), shape validation,
  TNLP-vs-zero-multiplier paths, and skipped-result paths.

**PATH-style multi-merit & degeneracy diagnostics (§2.7)**
- New `result.merit_cross_check` (dict) cross-checks three independent
  MPCC merit functions at the converged point: Fischer-Burmeister
  `|G + H − √(G² + H²)|`, min-map `|min(G, H)|`, and inner-product
  `|G · H|`.  Reports per-merit `*_max` and `*_mean`, plus a
  `disagreement_ratio = max / min(merit_maxes)` — close to 1 when
  merits agree (healthy convergence), large when they disagree
  (scaling mismatch or near-degeneracy).
- New `result.jac_row_norms` / `result.jac_col_norms` (dicts) report
  `max`, `min`, and near-zero count over the active-constraint
  Jacobian (rows: ∇h, ∇G_{I_G}, ∇H_{I_H}, ∇g_{I_g}, active bounds).
  Near-zero rows/cols are degeneracy signals.
- New `result.degeneracy_report` (dict) aggregates `n_biactive`,
  `n_zero_rows`, `n_zero_cols`, `min_singular_value` of the active
  Jacobian, and the merit disagreement ratio for one-glance assessment.
- New `result.initial_point_stats` (dict) — PATH's
  `output_initial_point_statistics` parity: reports `comp_residual`,
  `min_map_residual`, `max_bound_violation`, `ineq_residual`, and
  `eq_residual` evaluated at the user's `x0` *before* any solve work.
- All five fields are populated only when the solver is invoked with
  `diagnostics=True`; default solves are unchanged (no overhead).
- `result.summary(verbosity=1)` now renders a "Merit cross-check" line
  and a "Degeneracy" line whenever those diagnostics are populated.
- `result.to_json()` includes all five new fields.
- Public functions exported at the package root:
  `pympcc.merit_cross_check`, `pympcc.jac_norms`,
  `pympcc.degeneracy_report`, `pympcc.initial_point_statistics`.
- 18 new tests in `tests/test_path_diagnostics.py` covering each merit's
  formula, empty-problem handling, biactive detection, end-to-end solve
  with diagnostics on/off, JSON roundtrip, and summary rendering.

**Box-MCP / doubly-bounded canonical form (§4.9 — Phase 1)**
- New `comp_box_pairs` field on `MPCCProblem` declares
  `xl[var_idx] <= x[var_idx] <= xu[var_idx]  ⊥  F_fn(x)` and dispatches
  by bound finiteness:
  - **Lower-only finite** → comp pair `(x[var_idx] - xl) >= 0  ⊥  F(x) >= 0`.
  - **Upper-only finite** → comp pair `(xu - x[var_idx]) >= 0  ⊥  -F(x) >= 0`.
  - **Free** (both infinite) → equality `F(x) = 0` appended to `eq_constraints`.
  - **Doubly-bounded** (both finite) → `NotImplementedError` pointing to
    §4.9 Phase 2 / median NCP (§3.5 ext).
- `n_comp` and `n_eq` are auto-bumped to count synthesized rows; users
  supply `n_comp` / `n_eq` for the *base* problem only.
- 2-tuple `(var_idx, F_fn)` form falls back to forward fd for the F-row
  Jacobian; 3-tuple `(var_idx, F_fn, F_jac_fn)` uses the user callable.
- Mutually exclusive with `comp_var_pairs` and `comp_var_pairs_bulk` in
  this release.  Sparse base `comp_G/H` and sparse `eq_jacobian` are
  rejected (deferred).
- 21 new tests in `tests/test_box_mcp.py` covering all three categories,
  mixed combination, fd fallback, and every error path.

**AMPL `.nl` reader (§6.4)**
- New `pympcc.frontend.ampl` module — a self-contained text-format `.nl`
  reader that produces an `MPCCProblem` directly.  No external AMPL
  runtime or solver SDK required.
- `from_nl(path)` — read an AMPL text-format `.nl` file and return an
  `MPCCProblem`.  Complementarity constraints (encoded via bound-type-5
  in the `b` segment, the AMPL/MacMPEC convention) route into the bulk
  MCP form (`comp_var_pairs_bulk`).  Equality and inequality constraints
  are split based on the `r` segment bound types.
- Header parser (`parse_header`) covers the 10 fixed numeric lines plus
  the optional suffix-counts line; binary `.nl` is rejected with a clear
  error.
- Body-segment parser (`parse_body`) handles `C`, `O`, `b`, `r`, `k`,
  `J`, `G`, `S`, `x`, `d` segments; defined variables (`V`), logical
  constraints (`L`), and imported functions (`F`) raise a clear error.
- Op-tree reader (`_read_optree`) handles the ~30 AMPL operators
  MacMPEC uses (binary arithmetic `o0`–`o5`, `o27` atan2, transcendentals
  `o37`–`o52`, `o53` sumlist, `o76` square); unsupported ops raise
  `NLParseError` with the op code in the message.
- Op-tree evaluation (`eval_value`) and forward-mode AD (`eval_grad`)
  walk the parsed tree; gradients verified against central-finite-
  differences on randomized expressions.
- 55 new tests in `tests/test_ampl_reader.py` covering header parsing,
  body segments, op-tree recursion, evaluator correctness, gradient
  matches against finite differences, and an end-to-end round-trip
  (hand-authored 2-variable MPCC → `pympcc.solve` → unique optimum
  recovered).

**MacMPEC `.nl` fixtures + benchmark CLI**
- Hand-authored `.nl` fixtures for `simple`, `kth1`, `ralph1` under
  `tests/fixtures/nl/`; each is parsed via `from_nl` and parity-tested
  against its analytical `ProblemSpec` counterpart in
  `pympcc.benchmarks._problems`.
- New helper `pympcc.benchmarks._nl_loader.load_nl_directory()` scans a
  directory of `.nl` files and produces `ProblemSpec` records by joining
  on stem against the registry (`f_opt`/tolerances are reused).
- `python -m pympcc.benchmarks.macmpec --from-nl <dir>` runs the
  benchmark suite against an `.nl` directory instead of the built-in
  Python registry; combinable with `--problems` to filter by name.
- `--skip <names>` flag on the benchmark CLI to exclude specific
  problems (useful for skipping large MCPs whose per-row Python-callable
  dispatch makes them currently impractical).

### Fixed

- `from_nl()` now correctly populates `n_eq` and `n_ineq` on the
  returned `MPCCProblem`.  Previously the equality and inequality
  callables were built but the row counts defaulted to 0, so all general
  constraints were silently dropped from the NLP — affected ~12
  MacMPEC problems (`bard3`, `ex9.1.x`, `bilevel3`, etc.) that appeared
  to "solve" to unbounded objectives.

### Tests

- 1068 passed, 6 skipped, 54 xfailed (was 814 in 0.4.0).

---

## [0.4.3] - 2026-04-29

### Added

**Bilevel KKT-emitter frontend (§5.4)**
- New module `pympcc.bilevel` exposing `from_lower_level(...)`, which rewrites a
  bilevel program `min_{x,y} F(x,y) s.t. y ∈ argmin_y{f(x,y): g(x,y)≤0, h(x,y)=0}`
  into an `MPCCProblem` by emitting the lower-level KKT system (stationarity,
  feasibility, and `λ ≥ 0 ⊥ −g(x,y) ≥ 0`).
- Variable layout `z = [x_upper, y_lower, λ, μ]`; the λ block is automatically
  lower-bounded at zero, and the μ block is left free.
- `derivatives="jax"` (default) builds the stationarity rows via `jax.grad` and
  fills every Jacobian via JAX autodiff; `derivatives="fd"` falls back to nested
  finite differences.  No new dependencies.
- 23 new tests in `tests/test_bilevel.py` covering construction, KKT
  residuals at analytical optima, end-to-end solves under both backends,
  and input validation.

**Bulk MCP form (`comp_var_pairs_bulk`)**
- New `MPCCProblem.comp_var_pairs_bulk` field for large-scale MCPs; takes a
  4-tuple `(var_idxs, h_bulk_fn, h_bulk_jac_fn, h_bulk_jac_sparsity)` where
  `h_bulk_fn(x) → ndarray (k,)` and `h_bulk_jac_fn(x) → flat nnz values` are
  evaluated **once per callback** instead of per row.
- `comp_G(x) = x[var_idxs]` (view, no copy); `comp_G_jacobian` is a constant
  `ones(k)` callable with sparsity `(arange(k), var_idxs)`; H sparsity is
  forwarded through unchanged.
- Mutually exclusive with the per-row `comp_var_pairs`; raises `ValueError`
  when both are set.
- Designed for `n ≳ 10⁵`, `k ≳ 10⁴`. Measured ~2900× speedup for `comp_H`
  and ~7700× for `comp_H_jacobian` at `k=1000` versus the per-row form.

### Changed

- **MCP per-row form: vectorised value/Jacobian closures.** `_normalize_var_pairs`
  now writes into a pre-allocated `out` buffer instead of building a Python list
  per row, removing the `np.array([...])` allocation on every `comp_H` call.
- **MCP per-row Jacobian: hoisted size validation to construction.** Each
  `h_jac_fn` is evaluated once at `x0` to verify it matches `h_cols`; the
  per-callback `if vals.size != ...` branch is gone.
- **MCP fd-fallback guard.** Building a 2-tuple MCP whose finite-difference
  Jacobian would evaluate `h_fn` more than 10⁷ times per call now raises
  `ValueError` pointing the user to `h_jac_fn` or `comp_var_pairs_bulk`.

### Fixed

- `BaseStrategy._build_safeguard_mode`: `safeguards="all"` no longer silently
  upgrades `inner_tol_mode` from `"linear"` to `"quadratic"`. The default mode
  is preserved; users who want quadratic must pass it explicitly.

### Tests

- 11 new tests in `tests/test_mcp_var_pairs.py` covering `TestBulkForm`
  (construction, G/H values, constant ones G-Jacobian, lower-bound clamp,
  parity with the per-row form, mutual exclusion, validation errors) and
  `TestFdGuard` (large-fd rejection, small-fd allowed).

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
