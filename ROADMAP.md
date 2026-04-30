# pympcc Roadmap

This roadmap tracks parity with commercial MPCC and MCP solvers
(KNITRO MPEC, GAMS-NLPEC, GAMS-PATH, BARON-MPCC, FilterMPEC, LINDO MPEC,
Pyomo.MPEC, Complementarity.jl/PATHSolver.jl) across presolve,
diagnostics, solution methods, and modeling UX.

Items are tagged ✅ *shipped*, 🔄 *in progress*, or *planned*.  Scope
estimates: **S** (≲200 lines + tests), **M** (~500–1500), **L** (multi-week).

---

## Near-term priority queue (gap analysis vs. commercial solvers)

Ordered by impact-to-effort ratio derived from the commercial-grade gap
analysis.  Complete each tier before starting the next.

### Phase 1 — shipped

| # | Item | Section | Scope | Status |
|---|------|---------|-------|--------|
| 1 | Per-pair status + `to_json` / `to_dataframe` | §4.6 | S | ✅ |
| 2 | TNLP refinement (certified MPCC multipliers) | §2.6 | M | ✅ |
| 3 | SOSC — second-order sufficient conditions | §2.3 | M | ✅ |
| 4 | Variable-paired complementarity (MCP form) | §4.5 | M | ✅ |
| 5 | Bilevel KKT-emitter frontend | §5.4 | M | ✅ |
| 6 | NCP-function reformulation menu | §3.5 | M | ✅ |
| 7 | MacMPEC full benchmark runner (150 problems) | §4.7 | S | ✅ |
| 8 | Branch-and-bound (global MPCC) | §3.1 | L | deferred |

### Phase 2 — surfaced by commercial gap analysis (planned)

Ordered by impact-to-effort.  Items 9 and 10 are research-positioning
plays (no GAMS-tier solver ships them today); 11–13 close the largest
PATH/KNITRO parity gaps; 14–17 are smaller polish items.

| # | Item | Section | Scope | Status |
|---|------|---------|-------|--------|
| 9  | JAX-differentiable solve (`custom_vjp` through TNLP)         | §5.6 | M | ✅ shipped |
| 10 | Pyomo / `mpec.complementarity` frontend                      | §4.1 | M | ✅ shipped |
| 11 | Box-MCP / doubly-bounded canonical form `ℓ ≤ x ≤ u ⊥ F(x)`   | §4.9 | M | ✅ shipped |
| 12 | Stateful warm hot-start solver object (MPC rolling-horizon)  | §6.5 | M | planned |
| 13 | PATH-style multi-merit & degeneracy termination diagnostics  | §2.7 | S | ✅ |
| 14 | Parallel multistart (`n_jobs`)                               | §6.1 | S | planned |
| 15 | Additional NCPs (Chen-Mangasarian, Billups, Veelken-Ulbrich, median) | §3.5 ext | S | planned |
| 16 | EPEC multi-leader KKT emitter                                | §5.5 | M | planned |
| 17 | NLPEC modifier matrix (slack × constraint × aggregate × NCPBounds) | §6.6 | M | planned |
| 18 | MPECopt-style piecewise-SQP (finite-step B-stationarity)     | §3.8 | L | research |
| 19 | Raghunathan-Biegler IPOPT-C (modified barrier interior point) | §3.9 | L | deferred |

---

## 1. Presolve & problem reduction

Goal: every downstream strategy sees a maximally-reduced problem.  All
passes ship behind the opt-in flag `MPCCSolver(..., presolve=True)`
(equivalently `pympcc.solve(..., presolve=True)`).  When nothing can
be eliminated, every pass is a no-op and returns an identity
`PresolveMap`.

### Tier A — generic NLP presolve

#### A1. Pinned-variable elimination ✅ *(shipped)*

Detect `xl[j] == xu[j]` (finite, equal); substitute the variable out of
the optimisation, wrap callbacks to reinject the fixed value before
evaluation.  Reduces `n`, every Jacobian sparsity column index, and the
result is auto-expanded back to the original `n` on return.

Module: `pympcc/_presolve.py` — `PresolveMap`, `presolve()`.

#### A2. Linear FBBT (Feasibility-Based Bound Tightening) ✅ *(shipped)*

Linearity is detected per-row by comparing `g(x0+δ) − g(x0)` against
`J(x0)·δ` on a single bounded perturbation; rows passing the test
contribute to a fixed-point bound-tightening sweep up to
`_FBBT_BUDGET = 50` iterations.  Detected infeasibility (a tightening
that crosses a bound, or a constant violation on an empty row) emits a
`UserWarning` and falls back to identity so the strategy can surface
the issue through its normal infeasibility path.

Composes with A1: variables that FBBT collapses to `xl[j] == xu[j]` are
picked up by the pinned-var pass on the same presolve call (FBBT runs
first).  Equality rows whose only column is then pinned are dropped via
`drop_empty_rows=True` in `reduce_jac`, with `surviving_orig_rows`
tracked on the `PresolveMap` so multipliers are zero-padded at the
pruned indices on `expand_result`.

Module: `pympcc/_presolve.py` — `_identify_linear_rows`, `_fbbt`.
Tests: `tests/test_fbbt.py` (11 cases).

#### A3. Singleton equality substitution — *subsumed*

Equivalent to FBBT collapsing `xl[j] = xu[j]` on a single-column
equality.  Verified by `tests/test_fbbt.py::TestComposition`.  No
dedicated pass needed unless we hit a counter-example in the wild.

#### A4. Empty-row / empty-column removal ✅ *(shipped)*

Two complementary passes:

1. **Empty-row removal** (`_detect_empty_rows`): ineq/eq rows with
   structurally empty Jacobian sparsity are evaluated once at `x0`;
   feasible rows (`≤ tol` for ineq, `|·| ≤ tol` for eq) are dropped via
   the same `keep_ineq`/`keep_eq` machinery FBBT introduced.  Violated
   constants emit a `UserWarning` and fall back to identity.
2. **Empty-column removal** (`_detect_empty_cols`): variables absent
   from every Jacobian sparsity *and* with `|∇f|` below tolerance at
   two random probe points are pinned to `clip(0, xl[j], xu[j])`.
   A1 then eliminates them.

Conservative: empty-col detection refuses to act when any present
constraint exposes a dense Jacobian (no sparsity to inspect).

Module: `pympcc/_presolve.py` — `_detect_empty_rows`,
`_detect_empty_cols`.  Tests: `tests/test_empty_rows_cols.py` (12).

#### A5. Forcing-constraint detection — *partially subsumed*

If `Σ aᵢⱼ xⱼ ≤ b` saturates at `Σ aᵢⱼ uⱼ = b`, FBBT already pins every
variable in the row to its bound (the `rest_min` term equals `−b` for
each `j`).  Worth re-checking once a real combinatorial-MPCC test case
exists; a dedicated detector might still beat FBBT on cost.

### Tier B — MPCC-specific presolve

#### B1. Dead-pair pruning ✅ *(shipped)*

Comp pair `i` whose `comp_G_jacobian_sparsity` row is empty AND whose
value at `x0` is strictly positive is trivially satisfied; drop the
pair.  Symmetric for `comp_H`.  Restricted to the COO-sparsity case.

#### B2. Forced-pair pruning ✅ *(shipped)*

Symmetric to B1 from the other side.  If `H_i` has a structurally
empty Jacobian row *and* `|H_i(x0)| ≤ tol`, complementarity degenerates
(`H_i = 0` already satisfied) and `G_i ≥ 0` is promoted to a regular
inequality.  Symmetric for `G_i ≡ 0`.  Pairs with *both* rows constant
zero are added to `extra_dead` and dropped entirely.

Implementation augments `iJ_sp` / `ineq_constraints` / `ineq_jacobian`
in `_build_reduced` with the promoted rows: sparsity rows are
extracted from `comp_G_jacobian_sparsity` (resp. `comp_H_…`), pinned
columns are dropped, values are negated (`G_i ≥ 0` ⇔ `−G_i ≤ 0`).
Multipliers on promoted rows are dropped on `expand_result` (info
loss; the original-space comp multipliers at the promoted indices are
zero-padded).

Restricted to the all-COO case (refuses if `ineq_jacobian_sparsity` is
dense — would force mixing dense + sparse blocks).

Module: `pympcc/_presolve.py` — `_detect_forced`, augmentation block
in `_build_reduced`.  Tests: `tests/test_forced_pair.py` (8).

#### B3. Pre-fixing on linear sign analysis ✅ *(shipped)*

For each comp pair not already dropped by B1/B2, run the FBBT linearity
probe on `comp_G` and `comp_H`.  When a row passes, compute its
interval over the FBBT-tightened box `[xl, xu]` via
`_row_interval` (inf-safe).  If `G_min > _DEAD_VAL_TOL` then
complementarity forces `H_i = 0`: pair index goes to `prefix_H_eq`,
the pair is dropped, and the row of `comp_H_jacobian_sparsity` for
that pair is appended to the eq block as a regular equality.
Symmetric for `H_min > tol` → `prefix_G_eq` and `G_i = 0` appended.
If *both* sides certify positive on the same pair the feasible set is
empty: emit `UserWarning` and fall back to identity.

Augmentation lives in `_build_reduced` mirroring B2's ineq path
(values are *not* negated since we're producing equalities).
Multipliers on the prefix-eq slot are dropped on `expand_result`
(info loss; the comp multipliers at the dropped indices are
zero-padded).

Module: `pympcc/_presolve.py` — `_row_interval`, `_detect_prefix_eq`,
B3 augmentation block in `_build_reduced`.
Tests: `tests/test_prefix_eq.py` (11).

#### B4. Linear-comp-pair detection — *deferred*

When both `G_i` and `H_i` are linear in `x`, that pair is an LCP-style
row.  Detection lives in presolve; the strategy layer would route
those rows to a specialised LP-based branch.  Deferred until the
LP-branch consumer (§3.1) lands — shipping a detector with no consumer
is dead code.

### Tier C — deferred presolve passes

* C1. Duplicate-row / parallel-column detection — useful in MILP, marginal in MPCC.
* C2. Implied free variable substitution — fragile without symbolic Jacobians.
* C3. Redundant-row / IIS extraction — see §2.4 (diagnostics).
* C4. Coefficient strengthening — primarily a MILP technique.
* C5. Dual fixing — requires reduced-cost information from a previous solve.

---

## 2. Diagnostics & stationarity certificates

pympcc currently classifies S/M/C-stationarity in `_stationarity.py`
but does not certify constraint qualifications, second-order
conditions, B-stationarity, or extract infeasibility certificates.

### 2.1. Constraint-qualification check (MPCC-LICQ / MPCC-MFCQ) ✅ *(shipped)*

At a candidate point `x*` with active sets
`I_g(x*) = {j : g_j = 0}`, `I_G = {i : G_i = 0}`, `I_H = {i : H_i = 0}`,
`I_GH = I_G ∩ I_H` (biactive set):

* **MPCC-LICQ**: gradients of all active `g_j`, all `h_k`, all
  `∇G_i (i ∈ I_G)`, `∇H_i (i ∈ I_H)`, and all active variable-bound
  rows are linearly independent.  Test by computing the rank of the
  stacked active-gradient matrix (numpy SVD / `matrix_rank`).
* **MPCC-MFCQ**: equality-style block (h, G_{I_G}, H_{I_H}) linearly
  independent, plus a Mangasarian–Fromovitz direction `d` satisfying
  `∇h·d = 0`, `∇G_i·d = 0 (i ∈ I_G)`, `∇H_i·d = 0 (i ∈ I_H)`,
  `∇g_j·d < 0 (j ∈ I_g)`, and bound-compatible signs at active xL/xU.
  Solve as a single LP: maximise `t` s.t. `∇g_j·d + t ≤ 0`, etc.
  MPCC-MFCQ holds iff `t > 0` is achievable.

Surfaces as `result.cq` ∈ {`"MPCC-LICQ"`, `"MPCC-MFCQ"`, `"none"`,
`"unknown"`} plus `result.cq_active_set_sizes` (dict) and
`result.cq_rank_deficit` (int — slack from full rank).
Active-set partition exposed via `pympcc.active_sets(result, problem)`.

Hook: `MPCCSolver.solve()` runs the diagnostics block (CQ + B-stat)
after `expand_result`, behind opt-in `diagnostics=True`.  Default is
off so the no-overhead fast path is preserved.

Module: `pympcc/_diagnostics.py` — `classify_cq`, `active_sets`.
Tests: `tests/test_diagnostics.py` (11).

### 2.2. B-stationarity certification ✅ *(shipped)*

`pympcc._stationarity.verify_b_stationarity` solves the LPCC

    min_d  ∇f(x*)·d
    s.t.   linearised tangent cone of (g, h, G, H) at x*

by enumerating `2^|I_00|` branches over the biactive set.  Each branch
fixes a (G-active, H-≥0) or (H-active, G-≥0) assignment per biactive
pair and solves a dense LP via `scipy.optimize.linprog`.  Returns a
status dict with `n_biactive`, `n_branches_checked`, `min_descent`,
`witness_branch`, `witness_d`.  Skips when `|I_00| > max_biactive`
(default 10 → up to 1024 LPs).

Wired into the §2.1 diagnostics block: when `diagnostics=True`,
`MPCCSolver.solve()` populates `result.b_stationary`,
`result.b_stationary_witness`, and `result.b_stationary_min_descent`.

Bug fixed alongside §2.1: at points with `G_i > 0, H_i = 0` (resp.
`G_i = 0, H_i > 0`), the linearised MPCC tangent cone enforces
`∇H_i·d = 0` (resp. `∇G_i·d = 0`) since the strict-positive side
forces the zero side to stay zero locally.  Previous code used the
looser `≥ 0` form, which could falsely flag global optima as not
B-stationary.

Module: `pympcc/_stationarity.py` — `verify_b_stationarity`.
Tests: `tests/test_b_stationarity.py`, `tests/test_diagnostics.py`.

### 2.3. Second-order condition (MPCC-SOSC) ✅ *(shipped)*

Checks the reduced Lagrangian Hessian is positive definite on the
MPCC critical cone — verifies `x*` is a strict local minimiser, not
a saddle.

**Algorithm:**

1. Build the active constraint gradient matrix A (rows from ∇h, ∇G_{I_G},
   ∇H_{I_H}, ∇g_{I_g}, and unit variable-bound rows).
2. Compute null-space basis Z of A via SVD.
3. Form reduced Hessian W = Z^T H Z, where H = ∇²_xx L(x*, λ).
4. SOSC holds iff min_eigenvalue(W) > 0.

**Hessian sources (priority order):**
* `MPCCProblem.lagrangian_hessian` (user-supplied, lower-triangle COO or dense).
* Central FD of ∇_x L using multipliers from TNLP refinement (best) or
  zeros (conservative fallback when no TNLP is available).

**Result fields:**
* `result.sosc` ∈ {`True`, `False`, `None`}
* `result.sosc_min_eigenvalue` — minimum eigenvalue of W (None when skipped)
* `result.sosc_skipped_reason` ∈ {`None`, `"not_converged"`,
  `"biactive_pairs"`, `"no_hessian_callable_and_fd_failed"`}

Biactive pairs (I_00 ≠ ∅) make the critical cone non-convex;
the check returns `None` in that case.

Module: `pympcc/_sosc.py` — `sosc_check`.
Hook: `MPCCSolver._attach_diagnostics` (runs when `diagnostics=True`).
Tests: `tests/test_sosc.py` (16 cases).

### 2.4. IIS / minimal infeasible subsystem — (L)

Replaces roadmap C3.  When a strategy returns infeasible, deletion-
filter or Chinneck-style elimination identifies a minimal subset of
constraints whose joint infeasibility certifies the original.  Useful
for debugging large bilevel formulations where infeasibility is hard
to localise.

### 2.5. Solver telemetry ✅ *(shipped)*

Per-iteration log of ε, complementarity residual, KKT residual,
biactive-set size, CQ rank deficit.  Exposed via `result.summary(verbosity=2)`
and `result.history`; structured export covered by §4.6.

### 2.6. TNLP refinement / B-stationarity by solve ✅ *(shipped)*

After the relaxation converges, fix the active set
`(I_G, I_H, I_GH)` at the final iterate and re-solve the resulting
**tightened NLP** as a regular equality-constrained problem.  This is
the postprocessing step KNITRO's `mpec_finalize` and FilterMPEC use
to (a) clean up multipliers, (b) extract MPCC-stationarity multipliers
`(λ^G, λ^H)` with correct signs by *solving* rather than enumerating,
and (c) certify B-stationarity when the biactive set is too large for
the §2.2 LP enumeration (`|I_00| > max_biactive`).

Surface as `result.tnlp_refined` (sub-result) plus refined
`mult_comp_G` / `mult_comp_H`.  Skipped silently when the relaxation
already returned MPCC-LICQ S-stationary multipliers.

Module: `pympcc/_tnlp.py` (new); hook into `MPCCSolver.solve()`
behind `tnlp_refine=True` (or auto-on when `diagnostics=True`).

### 2.7. PATH-style multi-merit & degeneracy diagnostics ✅ *(shipped)*

PATH is the de-facto MCP reference and ships ~12 termination-time
diagnostics; pympcc now ships the load-bearing subset:

* **Multi-merit cross-check at `x*`** ✅ — Fischer-Burmeister
  `|G + H − √(G² + H²)|`, min-map `|min(G, H)|`, and inner-product
  `|G · H|`.  Disagreement between merits localises numerical trouble
  (one merit can mask issues another exposes).  Reported max/mean for
  each, plus a single `disagreement_ratio` summary.
* **Jacobian row/col norms** ✅ — max / min / near-zero count over the
  active-constraint Jacobian (the same stack `classify_cq` builds for
  the LICQ test, so cost is negligible).  Near-zero rows or columns
  are degeneracy signals.
* **Zero-row / zero-column counts** ✅ — surfaced inside
  `result.jac_row_norms["n_zero"]` and the analogous column field.
* **Initial-point statistics** ✅ — PATH's
  `output_initial_point_statistics` parity: comp residual, min-map
  residual, bound violation, ineq/eq residual at `x0`.
* **Proximal perturbation usage** — deferred (PATH-specific to its
  pivoting linear-solve path; pympcc's IPOPT backend exposes no
  equivalent state).

Surfaces as `result.merit_cross_check`, `result.jac_row_norms`,
`result.jac_col_norms`, `result.degeneracy_report`,
`result.initial_point_stats`.  All five are populated only when the
solver is invoked with `diagnostics=True`; default solves are
overhead-free.

Module: `pympcc/_diagnostics.py` — `merit_cross_check`, `jac_norms`,
`initial_point_statistics`, `degeneracy_report`.
Hook: `MPCCSolver._attach_diagnostics`.
Tests: `tests/test_path_diagnostics.py` (18 cases).

---

## 3. Solution methods

### 3.1. Branch-and-bound on disjunctions — (L) · *priority 8 (deferred)*

For each pair branch `G_i = 0` ∨ `H_i = 0` and solve each leaf as a
regular NLP.  Globalises the local NLP relaxation; finds non-S
points the relax-and-drive methods miss.  Best-first or depth-first
enumeration with bound pruning from the parent NLP.

Consumer for B4 linear-pair detection: linear pairs route to an LP
sub-solver instead of an NLP.

Module: new `pympcc/strategies/branch_and_bound.py`.

### 3.2. Active-set SQP for MPCC — (L)

Alternative to the relax-and-solve family: enumerate trial active sets
on the comp pairs and solve the resulting equality-constrained QP at
each iterate.  Following Fletcher–Leyffer FilterMPEC.  Faster on
problems with a clear active set but more brittle near degeneracy.

### 3.3. Multi-start wrapper ✅ *(shipped)*

`pympcc.multistart(problem, *, n_starts=16, perturb_scale=0.1, seed=0,
**solve_kwargs)` runs :func:`pympcc.solve` from ``n_starts`` perturbed
starting points and returns a :class:`MultiStartResult` exposing
``.best``, ``.runs``, ``.n_success``, and ``.unique_optima(...)`` for
basin clustering.  The first start uses ``problem.x0`` verbatim;
subsequent starts perturb each coordinate by Gaussian noise with
standard deviation ``perturb_scale * max(|x0|, 1)`` and clip to
``[xl, xu]``.  ``problem.x0`` is restored on return.

`pympcc.solve(problem, ..., n_starts=N, perturb_scale=..., multistart_seed=...)`
dispatches to the multistart wrapper when ``N > 1``.

Module: `pympcc/multistart.py`.
Tests: `tests/test_multistart.py` (18 cases).

### 3.4. Elastic-mode penalty — (M)

Anitescu / Leyffer ℓ₁-elastic penalty as a fallback when MFCQ collapses.
Replaces hard equality `G·H = 0` by penalised slacks; complementary
to Scholtes / smoothing when those stall on highly degenerate problems.

Module: `pympcc/strategies/elastic.py`.

### 3.5. NCP-function reformulation menu — (M) · *priority 6* ✅

GAMS-NLPEC ships ~12 reformulations as switches.  pympcc has 6;
adding the most-cited remaining NCP functions makes the package a
direct benchmarking platform for the reformulation literature.
Targets:

* **min-NCP** — `min(G, H) = 0` smoothed via
  `½(G + H − √((G−H)² + 4ε²))`.
* **Chen-Chen-Kanzow** `φ_λ(a,b) = λ·φ_FB(a,b) + (1−λ)·a₊·b₊`
  (interpolates Fischer-Burmeister and penalised inner-product).
* **Kanzow-Schwartz** `(G + H) − √(G² + H² + 2λGH)` for
  `λ ∈ [0, 1)`.

Each lands as its own thin strategy class reusing the smoothing
ε-continuation harness.  No new infrastructure.

Module: `pympcc/strategies/ncp.py` — `SmoothMinStrategy`, `ChenChenKanzowStrategy`, `KanzowSchwartzStrategy`. ✅ Shipped in 0.4.2.

**Phase-2 additions (§3.5 ext, priority 15) — planned:**

GAMS-NLPEC's `equreform` table currently has 33 rows; pympcc has 9
(direct, Scholtes, smoothing/FB, Lin-Fukushima, slack, augLag, plus the
three NCPs above).  Closing the most-cited remaining gaps:

* **Chen-Mangasarian asymmetric** — `φ_α(a, b) = α·(a + b) − √(a² + b² + (1−α)·2ab)`
  for `α ∈ (0, 1]`; the FB ↔ inner-product interpolation that NLPEC
  exposes as `CMxf` / `CMfx`.
* **Billups composite** — `φ(a, b) = φ_FB(a, b) − γ·a₊·b₊` with
  γ-schedule; `Bill` / `fBill` in NLPEC.
* **Veelken-Ulbrich smoothings** — `fVUsin`, `fVUpow` smooth perturbations
  of the min-NCP that retain MFCQ near degeneracy.
* **Median NCP for doubly-bounded variables** — `median(x − ℓ, x − u, F(x)) = 0`,
  the alternative reformulation for box-MCP (§4.9).  Not strictly
  required: box-MCP §4.9 already ships using a universal Billups slack
  lift that any existing strategy can solve.  A median-NCP strategy
  would skip the slack variables and reformulate complementarity in the
  inequality block instead — useful for benchmarking against NLPEC's
  `med` row, otherwise optional.

Each lands as a thin strategy class (~80 LOC) reusing the smoothing
ε-continuation harness.

### 3.8. Piecewise-SQP for finite-step B-stationarity — (L) · *priority 18 (research)*

MPECopt (Nurkanović et al., 2024) achieves *finite-step* B-stationarity
by enumerating active-set branches over the biactive set and solving
each branch as an equality-constrained QP, with explicit M- and
B-stationarity certificates per branch.  This is currently the
strongest published convergence guarantee for MPCCs, and no commercial
solver ships it.  pympcc has a B-stationarity *certifier* (§2.2) but
no solver that targets B-stationarity by design.

Sketch:

1. Solve a relaxation strategy to obtain a candidate `x*` and biactive
   estimate `Î_00`.
2. Enumerate `2^|Î_00|` leaf NLPs, each fixing a (G-active, H-≥0) or
   (H-active, G-≥0) assignment per pair.
3. Solve each leaf via SQP / IPOPT; the minimum-objective leaf with a
   feasible KKT system is B-stationary.
4. Reuse the §2.2 LP machinery for branch enumeration (already
   implemented and tested).

Synergy with the planned **pyfiltersqp** project (memory: SQP solver at
`../pyfiltersqp`): once that ships, MPECopt becomes natural to host as
a pympcc strategy backed by pyfiltersqp leaves.

Module: `pympcc/strategies/piecewise_sqp.py`.

### 3.9. Raghunathan-Biegler IPOPT-C — (L) · *priority 19 (deferred)*

An interior-point method that modifies IPOPT's barrier loop to handle
complementarity *natively* (simultaneous σ-τ updates inside the IP
iteration) rather than as an outer ε-continuation.  Faster on highly
degenerate problems where Scholtes / FB stall.

Requires a fork of cyipopt's barrier callbacks or a custom barrier
loop.  High effort with uncertain practical payoff vs the existing
strategies; deferred until a real degenerate test case demands it.

Module: hypothetical `pympcc/strategies/ipopt_c.py`.

### 3.6. Adaptive penalty escalation — (S)

Extends §3.4: instead of a single global penalty parameter, escalate
the penalty *only on violated complementarity pairs* (Leyffer-López-
Calva-Nocedal, SIAM 2006).  Tracks a per-pair penalty vector `ρ_i`
that doubles whenever pair `i` exceeds the target residual.

Diagnostic: report number of pairs that needed escalation (`result.
n_pairs_escalated`).  High counts flag genuinely degenerate pairs.

Wires into `pympcc/strategies/elastic.py` and the existing
`augmented_lagrangian.py`.

### 3.7. LPCC / QPCC fast-path subsolver — (M)

When `f` is linear or quadratic and `g, h, G, H` are linear, the
MPCC is an LPCC / QPCC.  Specialised pivoting (Fletcher-Leyffer
piecewise-linear active-set) is dramatically faster than the
nonlinear NLP path and gives an exact reference solution for
benchmarking.  Detection extends presolve B4
(linear-comp-pair detection); the subsolver consumes it.

Module: `pympcc/strategies/lpcc.py`.  Auto-dispatch when every
constraint passes the linearity probe; falls back to the user's
chosen nonlinear strategy otherwise.

---

## 4. Modeling & user experience

### 4.1. Pyomo / mpec.complementarity bridge ✅ *(shipped)*

`pympcc.frontend.pyomo.from_pyomo(model)` accepts a Pyomo
`ConcreteModel` containing one or more
`Complementarity(expr=complements(...))` blocks and returns a
`PyomoMPCC` carrying the numeric `MPCCProblem` plus name-to-index
maps for variables and constraints.  Implementation: clone the model,
apply `TransformationFactory("mpec.nl")` (which encodes each comp
block via AMPL bound-type-5 ``cvar`` mapping), pipe through Pyomo's
NL writer, then parse with the existing `pympcc.frontend.ampl.from_nl`.

Companion helper `apply_solution(model, x, var_index)` writes a
solution back into the original model by qualified name.  Pure-NLP
models (no `Complementarity` blocks) raise `ValueError` with a
pointer to IPOPT/cyipopt; pympcc's scope is MPCC only.

Pyomo is an opt-in dependency: install with `pip install pympcc[pyomo]`.

Module: `pympcc/frontend/pyomo.py`.
Tests: `tests/test_pyomo_frontend.py` (8 cases).

### 4.2. Default JAX-AD path ✅ *(shipped)*

`MPCCProblem` and `StructuredMPCC` accept a top-level
``derivatives`` keyword.  Setting ``derivatives="jax"`` (or
``"fd"``) auto-fills every unset derivative field — ``gradient``,
every Jacobian — with the matching sentinel before resolution.
Users supplying JAX-traceable ``objective`` / ``comp_G`` / ``comp_H``
no longer need to spell out the gradient or any Jacobian, and the
existing per-field sentinel API (``comp_G_jacobian="jax"``, …)
remains supported for partial opt-in.

A clear error is raised when a required derivative is left
unresolved (no callable, no sentinel, no ``derivatives`` keyword).

Module: `pympcc/problem.py` and `pympcc/models.py` —
`_apply_derivatives_default`, `_check_derivatives_resolved`.
Tests: `tests/test_derivatives_default.py` (14 cases).

### 4.3. Auto pair-scaling ✅ *(shipped)*

`pympcc.autoscale_comp_pairs(problem, *, threshold=1e3, n_probes=5,
seed=0, ...)` probes `|G_i|, |H_i|` at `x0` and a handful of bounded
perturbations, takes per-pair medians, and returns diagonal scales
`(s_G, s_H)` that equilibrate pairs whose `max/min` magnitude ratio
exceeds `threshold`.  Well-conditioned pairs are left at unit scale.

`pympcc.solve(..., autoscale=True)` runs the detector after presolve,
populates `problem.comp_G_scale` / `comp_H_scale`, and emits a single
`UserWarning` summarising how many pairs were rescaled.  User-supplied
scales always win over the detector.

Module: `pympcc/_autoscale.py`.
Solver hook: `MPCCSolver._apply_autoscale` in `pympcc/solver.py`.
Tests: `tests/test_autoscale.py` (14 cases).

### 4.4. Result repr / summary formatter ✅ *(shipped)*

`result.summary(verbosity=0|1|2)` prints obj, comp residual, CQ class,
stationarity class, B-stat verdict, biactive set size, and full per-iteration
history.  Structured machine-readable export covered by §4.6.

### 4.5. Variable-paired complementarity (MCP form) ✅ *(shipped)*

`MPCCProblem` now accepts a `comp_var_pairs` field — a list of
`(var_idx, h_fn)` or `(var_idx, h_fn, h_jac_fn)` tuples that declare
`x[var_idx] >= 0 ⊥ h_fn(x) >= 0` directly at the variable level,
without needing a manual `G(x) = x[var_idx]` row in `comp_G`.

Two modes:
* **All-var-pairs** (`comp_G=None`): every pair is declared via
  `comp_var_pairs`; `len(comp_var_pairs)` must equal `n_comp`.
* **Mixed**: `comp_G`/`comp_H` supply the first `n_comp − k` pairs;
  `comp_var_pairs` appends the remaining `k` pairs.

`xl[var_idx]` is silently clamped to `max(xl[var_idx], 0.0)`.
G-side Jacobian rows are built exactly (identity matrix rows); H-side
rows use the supplied `h_jac_fn` or forward fd when omitted.
`derivatives="fd"` or `derivatives="jax"` compose naturally with this
field — the merged callables are standard `comp_G`/`comp_H` from each
strategy's perspective.

Module: `pympcc/problem.py` (`comp_var_pairs` field + `_normalize_var_pairs()`).
Tests: `tests/test_mcp_var_pairs.py` (21 cases).

### 4.6. Per-pair status & structured result export ✅ *(shipped)*

`result.per_pair_status` (list of `"G_active"`, `"H_active"`,
`"biactive"`, `"inactive"`) is populated by `MPCCSolver._attach_per_pair_status`
after every solve using an adaptive threshold `max(sqrt(comp_residual), 1e-6)`
to distinguish near-biactive from cleanly active pairs.

`result.mult_comp_G_mpcc` / `mult_comp_H_mpcc` are populated from
the TNLP active-set refinement (§2.6) when `tnlp_refine=True` and the
refinement succeeds; `None` otherwise.

`result.to_json()` serialises the full result (arrays as lists, `None`
as JSON `null`, history omitted) including `per_pair_status` and the
TNLP sub-result when present.  `result.to_dataframe()` returns a
per-pair `pandas.DataFrame` with columns `pair`, `G`, `H`, `GH`,
`status`, and (when available) `mu_G`, `mu_H`.

Module: `pympcc/result.py`.
Tests: `tests/test_per_pair_status.py` (20 cases), `tests/test_summary.py` (19 cases).

### 4.7. MacMPEC benchmark runner ✅ *(shipped — 13 problems)*

`pympcc.benchmarks.macmpec` runs any subset of the current 13-problem
suite and emits a Leyffer-style results table.  The problem registry
(`pympcc/benchmarks/_problems.py`) is the authoritative source;
`tests/macmpec_problems.py` is now a thin re-export shim.

CLI: `python -m pympcc.benchmarks.macmpec [--strategy ...] [--problems ...] [--out file.csv] [--quiet]`

Programmatic: `from pympcc.benchmarks import run_benchmark, print_table, save_csv`.

Module: `pympcc/benchmarks/` (`__init__.py`, `_problems.py`, `macmpec.py`).

**Gap vs. full MacMPEC (≈150 problems):** the actual Leyffer collection is
distributed as AMPL `.mod`/`.dat` files.  Reaching ~150 problems requires
one of the approaches in §6.4:

* **Hand-coding** — exact Jacobians, no new dependencies; ~weeks of work
  for the full set.
* **AMPL Python API** — parse `.nl` files directly; adds AMPL SDK
  dependency, Jacobians via finite differences.
* **pycutest** — ~50 MPCC problems with gradient/Jacobian support;
  most practical near-term path to a larger suite without hand-coding.

Expanding the suite is tracked under §6.4.

### 4.8. Inner-iteration callback hook ✅ *(shipped)*

`pympcc.solve(..., inner_callback=cb)` — and the equivalent
`MPCCSolver(..., inner_callback=cb)` — forwards IPOPT's per-NLP-iter
`intermediate` hook to the user as
`cb(iter_count: int, info: dict) -> bool`.  Returning ``False`` stops
the current inner solve (the outer loop then continues with the
partial iterate as a warm-start).  ``info`` carries IPOPT's full
``intermediate`` payload: ``alg_mod``, ``obj_value``, ``inf_pr``,
``inf_du``, ``mu``, ``d_norm``, ``regularization_size``, ``alpha_du``,
``alpha_pr``, ``ls_trials``.

Wired through both `_DenseNLP` and `_SparseNLP` so problems with sparse
Jacobians (every `from_nl()` build, and the `slack` strategy's lifted
NLP) get the hook too.  filterSQP / scipy backends ignore it (no
equivalent intermediate hook).

Module: `pympcc/_nlp.py` (`_invoke_inner_callback`,
`_DenseNLP.intermediate`, `_SparseNLP.intermediate`); thread-through in
`pympcc/strategies/_base.py` and each iterative strategy's
`__init__`; surface in `pympcc/solver.py`.
Tests: `tests/test_inner_callback.py` (10 cases).

### 4.9. Box-MCP / doubly-bounded canonical form — (M) · *priority 11* · ✅ *(shipped)*

PATH's canonical form is

    ℓ ≤ x ≤ u  ⊥  F(x)

which generalises the AMPL `complements` operator's three-way sign
convention: `F(x)` may be `≥ 0` (when `x = ℓ`), `= 0` (when
`ℓ < x < u`), or `≤ 0` (when `x = u`).  This is the standard MCP form
used in economic-equilibrium and game-theory models.

`MPCCProblem` accepts a `comp_box_pairs` field: list of
`(var_idx, F_fn)` or `(var_idx, F_fn, F_jac_fn)` tuples interpreted
with respect to the variable's `[xl, xu]` box.  Behaviour by bound
finiteness:

* **Lower-only** (`xl[j]` finite, `xu[j] = +∞`) ✅ shipped — appends
  `(x − ℓ) ≥ 0 ⊥ F(x)` as a standard `comp_G` / `comp_H` row;
  `n_comp` auto-bumps.
* **Upper-only** (`xl[j] = −∞`, `xu[j]` finite) ✅ shipped — appends
  `(u − x) ≥ 0 ⊥ −F(x)` (sign-flipped).
* **Free** (both infinite) ✅ shipped — appends `F(x) = 0` to the eq
  block (constrained nonlinear system); `n_eq` auto-bumps.
* **Doubly-bounded** (both finite) ✅ shipped — universal Billups /
  Mangasarian slack lift.  For each entry, two new variables
  `s₋, s₊ ≥ 0` are appended to `x`, an equality
  `F(x) − s₋ + s₊ = 0` is appended to the eq block, and two
  complementarity pairs `(x[j] − ℓ) ⊥ s₋`, `(u − x[j]) ⊥ s₊` are
  appended to `comp_G` / `comp_H`.  Reproduces PATH's three-way sign
  convention: when `F(x*) > 0` the second pair forces `s₊ = 0` and the
  active first pair drives `x = ℓ`; symmetric at the upper bound; in
  the box interior both slacks vanish and `F = 0`.  Original variables
  remain at indices `0..n_orig` of the lifted `result.x`; recover
  `n_orig` via `problem.n_orig_doubly_bounded`.  The lift is universal
  — every existing strategy (Direct, Scholtes, Smoothing,
  Lin–Fukushima, AugLag, Slack) sees a standard `MPCCProblem` and
  needs no per-strategy change.

Limitations: `comp_box_pairs` is mutually exclusive with
`comp_var_pairs` / `comp_var_pairs_bulk`; sparse base
`comp_G_jacobian_sparsity` / `eq_jacobian_sparsity` are rejected (the
helper builds dense rows; doubly-bounded entries also reject sparse
base Jacobians).

Composes with `derivatives="jax"` / `"fd"` and with all existing
strategies — each strategy sees the standard `comp_G` / `comp_H` /
eq rows after expansion; box-handling lives in
`MPCCProblem._normalize_box_pairs`.

Module: `pympcc/problem.py` — `comp_box_pairs` field +
`_normalize_box_pairs`, `_extend_comp_with_box`, `_extend_eq_with_free`,
`_extend_with_doubly_bounded`.
Tests: `tests/test_box_mcp.py` (26 cases).

---

## 5. Bilevel / parametric extensions

### 5.1. Parametric sensitivity (sIPOPT-style) ✅ *(shipped)*

`pympcc.sensitivity(result, problem, *, dgrad_L_dp, dc_dp, ...)` — a
low-level KKT-linear-solve primitive returning `dx*/dp` and `dλ*/dp` at
a converged MPCC solution by implicit differentiation through the TNLP
active set.  Solves the saddle-point system

    ┌ ∇²_xx L   J_cᵀ ┐ ┌ dx*/dp ┐     ┌ ∂(∇_x L)/∂p ┐
    │                │ │        │ = − │              │
    └ J_c       0    ┘ └ dλ*/dp ┘     └ ∂c/∂p       ┘

reusing the §2.3 Hessian builder for `∇²_xx L` and the §2.1 active-set
machinery for `J_c`.  Picks up TNLP-refined multipliers (§2.6)
automatically when `tnlp_refine=True` was passed to `solve()`; falls
back to zero multipliers with a `UserWarning` otherwise (exact only
when constraints are linear in `x`).  Skips cleanly with
`skipped_reason` populated when the result is non-converged or has
biactive pairs (IFT requires LICQ).

Companion helper `pympcc.active_row_labels(result, problem)` returns
the per-row labels (`("h", k)` / `("G", i)` / `("H", i)` / `("g", j)`
/ `("xL"|"xU", j)`) the caller's `dc_dp` rows must follow.

Designed as the substrate §5.6 will hook into via `jax.custom_vjp`.
The high-level convenience that builds `dgrad_L_dp` / `dc_dp`
automatically from JAX-traceable parametric callables is deferred to
§5.6.

Module: `pympcc/sensitivity.py`.
Tests: `tests/test_sensitivity.py` (17 cases).
API: `pympcc.sensitivity`, `pympcc.SensitivityResult`,
`pympcc.active_row_labels`.

### 5.2. Multiplier warm-start ✅ *(shipped)*

All five iterative strategies (`scholtes`, `smoothing`, `lin_fukushima`,
`slack`, `augmented_lagrangian`) carry `mult_g`, `mult_x_L`, `mult_x_U`
between outer iterations via the `warm_dual` dict in
`_run_epsilon_continuation`, and toggle IPOPT's
`warm_start_init_point=yes` after the first NLP solve.  On by default
(`dual_warmstart=True`); pass `dual_warmstart=False` to disable.
Empirical impact: ~25–30% fewer IPOPT iterations across the MacMPEC
benchmark suite.

Module: `pympcc/strategies/_base.py` — `_run_epsilon_continuation`,
`_timed_solve` (passes `lagrange`/`zl`/`zu` kwargs to cyipopt).

### 5.3. EPEC / VI extension — (L)

Equilibrium problems with equilibrium constraints; out of scope for
the SIAM imaging paper but a natural follow-up.  Concrete first step
tracked under §5.5.

### 5.4. Bilevel KKT-emitter frontend ✅ *(shipped)*

`pympcc.bilevel.from_lower_level(...)` rewrites a bilevel program

```
min_{x, y}  F(x, y)
s.t.        y ∈ argmin_y { f(x, y) : g(x, y) ≤ 0,  h(x, y) = 0 }
```

into an `MPCCProblem` by emitting the lower-level KKT system:
stationarity (`∇_y f + Σλ ∇_y g + Σμ ∇_y h = 0`), lower-level
equality (`h(x, y) = 0`), and the complementarity pair
`λ ≥ 0 ⊥ −g(x, y) ≥ 0`.  Variable layout
`z = [x_upper, y_lower, λ, μ]`; the λ block is automatically lower-
bounded at zero.

API:

```python
mpcc = pympcc.bilevel.from_lower_level(
    n_x=..., n_y=..., x0=..., y0=...,
    f_upper=...,
    f_lower=...,
    n_g_lower=..., g_lower=...,
    n_h_lower=0, h_lower=None,
    derivatives="jax",   # or "fd"
)
result = pympcc.solve(mpcc)
```

`derivatives="jax"` produces single-level autodiff for stationarity
and every Jacobian; `derivatives="fd"` falls back to nested finite
differences (acceptable for prototyping).  No new dependencies.

Module: `pympcc/bilevel.py`.
Tests: `tests/test_bilevel.py` (23 cases).

### 5.5. EPEC multi-leader emitter — (M) · *priority 16*

Generalises §5.4 to *multiple* leaders sharing variables.  An EPEC

```
For each leader i:
    min_{x_i, y}  F_i(x_i, x_{-i}, y)
    s.t.          y ∈ argmin_y { f(x, y) : g(x, y) ≤ 0 }
```

is reformulated by emitting each leader's MPCC KKT conditions and
concatenating, producing a single large MPCC over
`(x_1, …, x_n, y, λ_lower, μ_leaders)`.  This is the GAMS EMP
"equilibrium" pattern; commercial coverage is thin (only EMP/NLPEC
ships it directly).

API:

```python
mpcc = pympcc.bilevel.from_epec(
    leaders=[
        pympcc.bilevel.Leader(F=..., g=..., h=...),
        pympcc.bilevel.Leader(F=..., g=..., h=...),
    ],
    common_lower=pympcc.bilevel.LowerLevel(f=..., g=..., h=...),
    derivatives="jax",
)
```

Reuses §5.4 KKT-emitter machinery; the new code is mostly bookkeeping
for stitching leader blocks.

Module: extension to `pympcc/bilevel.py` (new `Leader` /
`from_epec` API).  Test set: standard MacEPEC instances.

### 5.6. JAX-differentiable solve via custom_vjp — (M) · ✅ shipped

Distinct from §5.1: §5.1 is *parametric sensitivity* returning `dx*/dp`
on demand.  This item is the **autograd integration**: register
`pympcc.solve` (or `pympcc.solve_jvp`) with `jax.custom_vjp` so that
JAX can differentiate *through* a converged MPCC solve transparently.

Implementation reuses the §5.1 KKT linear solve, but exposes it as
`jax.numpy`-compatible primitives so an MPCC sits inside a JAX
neural-network pipeline.  Differentiation goes through the **TNLP-
refined active set** (§2.6) so the LICQ assumption holds; degenerate
cases fall back to a Tikhonov-regularised pseudo-inverse with a
documented warning.

Why this matters: every commercial MPCC solver (KNITRO, PATH, BARON,
LINDO) treats the solve as a black box.  Among open-source codes, only
DiffMPC / mpc.pytorch differentiate through MPC, and they require
*linear-quadratic* problems.  Differentiable *general* MPCC is a niche
no other solver currently owns — and pympcc already has JAX as a first-
class dependency, making this a uniquely cheap differentiator.

API sketch:

```python
@jax.jit
def loss(theta):
    problem = build_mpcc(theta)            # JAX-traceable problem
    result = pympcc.solve(problem)         # custom_vjp-registered
    return objective(result.x, theta)

grad = jax.grad(loss)(theta)               # works
```

Module: `pympcc/_autodiff.py` (new); registration in
`pympcc/__init__.py`.  Depends on §2.6 TNLP refinement and §5.1
sensitivity primitives.

**Phase-1 scope shipped.** `pympcc.ParametricMPCC` carries the
parametric `(x, theta)` callables; `pympcc.solve_jax(parametric,
theta, *, x0, strategy=...)` is registered as `jax.custom_vjp`.
Forward materialises an `MPCCProblem` (closing `theta` into the
callables) and runs `pympcc.solve(..., tnlp_refine=True)`; backward
solves the §5.1 KKT saddle system once and contracts via
`jax.vjp` on the parametric callables, so pytree θ shapes pass
through unchanged.  `jax.grad` works; `jax.jacrev` is **not**
supported in Phase-1 (its internal `vmap` traces the NumPy/IPOPT
bwd) — assemble Jacobians per-row via `jax.grad` instead.
Phase-2 (`custom_jvp` for forward-mode, θ-dependent bounds, jit-
compatibility) deferred.

---

## 6. Robustness & production features

Items surfaced by the commercial-grade gap analysis that were not on the
original roadmap.  Lower priority than §2–5 but relevant before a 1.0 release.

### 6.1. Parallel multistart — (S)

`pympcc.multistart` currently runs starts sequentially.  Wrap the inner
loop with `concurrent.futures.ProcessPoolExecutor` behind a
`n_jobs` parameter (default ``1`` = sequential, ``-1`` = all CPUs).
Each worker receives a deep-copied problem and a perturbed `x0`.  The
`MultiStartResult` aggregates results as futures complete.

Caveat: cyipopt / IPOPT must be fork-safe or use "spawn" start method;
test on macOS where fork is restricted.

Module: extend `pympcc/multistart.py`.  Tests: `tests/test_multistart.py`.

### 6.2. Condition-number diagnostics at x* — (S) · ✅ shipped

After a solve, report:

* **Constraint Jacobian condition number** `κ(J_active)` — rank + condition
  of the active-constraint Jacobian (already assembled for §2.1 LICQ check).
* **Reduced Hessian condition estimate** — diagonal scaling from IPOPT's
  linear solver (MA57/MA27 pivot sizes) when available.

Surface as `result.jac_condition: float | None` and
`result.hessian_condition_estimate: float | None`.  Neither blocks any
downstream feature; purely diagnostic.

Module: extension to `pympcc/_diagnostics.py`.

### 6.3. Time limit with feasible incumbent — (S) · ✅ shipped

IPOPT's `max_cpu_secs` already stops the inner solve, but pympcc returns
whatever IPOPT had at that point without marking it as an "incumbent".
Wrap `MPCCSolver.solve()` with a `time_limit` parameter:

* Track the best feasible iterate seen across outer iterations
  (via the per-iteration callback).
* When `time_limit` is reached, stop the outer loop and return the best
  incumbent rather than the incomplete current iterate.
* `result.time_limit_hit: bool` flag for downstream detection.

Module: extension to `pympcc/solver.py`.

### 6.4. CUTEst / AMPL model library benchmark — (M) · *AMPL ✅ · CUTEst deferred*

Extend `pympcc.benchmarks` beyond the original 13-problem MacMPEC subset.

**AMPL `.nl` reader** ✅ — :mod:`pympcc.frontend.ampl` — a
self-contained text-format ``.nl`` parser, no external runtime deps.
Operator coverage targets the ~30 ops MacMPEC uses; expressions are
walked via reverse-mode AD over the parsed op-tree (no JAX dependency).

Staging plan:

1. ✅ Header parser (`parse_header`), op-tree reader (`_read_optree`),
   `_TokenStream` infrastructure.
2. ✅ Body-segment parsers (`C`, `O`, `b`, `r`, `k`, `J`, `G`, `S`,
   `x`, `d`); bound-type-5 in the ``b`` segment carries the
   complementarity pair `(var → constraint)` mapping used by AMPL/MacMPEC.
3. ✅ Forward op-tree evaluator (`eval_value`) + forward-mode AD
   (`eval_grad`); op-trees in `.nl` are typically narrow so forward-mode
   matches reverse-mode performance without the bookkeeping.
4. ✅ `from_nl(path) -> MPCCProblem` builder; complementarity rows
   route into `comp_var_pairs_bulk` (vectorised MCP form, §4.5).
   Verified end-to-end: hand-authored `.nl` → `pympcc.solve()` →
   converges to the unique global optimum.
5. ✅ Three hand-authored `.nl` fixtures (`simple`, `kth1`, `ralph1`)
   under `tests/fixtures/nl/`, parity-tested against
   :mod:`pympcc.benchmarks._problems`; loader at
   :func:`pympcc.benchmarks._nl_loader.load_nl_directory`; CLI flag
   `python -m pympcc.benchmarks.macmpec --from-nl <dir>` lands the
   `.nl`-driven benchmark path.
6. ✅ Full MacMPEC `.nl` fixture set (≈168 problems) generated offline
   via AMPL and committed under `tests/fixtures/nl/`.  The loader and
   CLI consume the directory unchanged.

Out of scope for the first ship: binary `.nl` format, defined functions
(``f<N>``), piecewise-linear terms (``o63``), AMPL extensions.

**CUTEst** (deferred) — `pycutest` requires the CUTEST/SIFDECODE/MASTSIF
runtime *and* per-problem MPCC pair-mapping (CUTEst itself doesn't tag
complementarity structure).  Re-evaluate once the `.nl` path lands and
the benchmark suite is at ~50 problems.

Output: Leyffer-style results table (problem, strategy, f_opt_gap,
comp_residual, CQ_class, stationarity, n_iter, time) for paper figures.

CLI (planned): `python -m pympcc.benchmarks.macmpec --from-nl path/`.

### 6.5. Stateful warm hot-start solver object — (M) · *priority 12*

Today `MPCCSolver.solve()` is one-shot.  For MPC rolling-horizon control,
parametric continuation, and outer hyperparameter loops, the dominant
cost is *reconstruction*: building the NLP, factoring KKT once, refining
multipliers.  Repeated solves on near-identical problems should reuse:

* Last `x*`, `λ*`, `μ*` (multipliers for `g`, `h`, `G`, `H`).
* Last barrier parameter `μ` and step `α` from IPOPT.
* Active-set summary from §2.6 TNLP refinement.
* Presolve `PresolveMap` (when problem dimensions are stable).
* Optionally, the IPOPT internal state via cyipopt's `solve()` reuse.

API:

```python
solver = pympcc.MPCCSolver(problem, ...)
result_0 = solver.solve()            # cold
problem_1 = update_parameters(...)   # change F(x; θ_1)
result_1 = solver.resolve(problem_1) # warm — reuses x*, λ*, μ
```

`solver.resolve()` validates that the *structure* (n, m, sparsity) is
unchanged; only numeric values may differ.  Falls back to cold start if
structure changed.  `result_k.warmstart_savings_iter` reports how many
inner iterations were saved vs the cold baseline.

This is the canonical KNITRO / PATH usage pattern for repeated solves
and unblocks pympcc as an MPC inner solver.

Module: extension to `pympcc/solver.py` (`MPCCSolver.resolve()`,
state-retention struct).

### 6.6. NLPEC modifier matrix — (M) · *priority 17*

GAMS-NLPEC's 33-row `equreform` is built from the cross-product of
~6 NCP functions × 4 slack types (`none` / `free` / `positive` / `one`)
× 2 constraint types (`eq` / `ineq`) × 3 aggregate levels × 4 NCPBounds
levels.  pympcc currently treats each row as a separate strategy class.

Refactor to expose the modifiers as orthogonal options:

```python
result = pympcc.solve(
    problem,
    strategy="ncp",
    ncp_function="fischer-burmeister",   # or "min", "ChenChenKanzow", …
    slack="positive",                     # how G, H rows are slack-lifted
    constraint="ineq",                    # phi(G,H) ≤ ε vs phi(G,H) = ε
    aggregate="partial",                  # collapse rows of φ
    ncp_bounds="all",                     # auto-bounds inferred for slacks
)
```

Yields ~12-20 *effective* reformulations from the existing NCP code with
no new theory.  Strategies become thin: pure orchestration over a single
"NCP-reformulation builder" that consumes the modifier dict.

Module: refactor of `pympcc/strategies/_base.py` and
`pympcc/strategies/ncp.py`; new `pympcc/_reformulation.py` builder.

### 6.7. Reformulated-NLP dump — (S)

NLPEC's `dotGams` writes the reformulated NLP to a file for inspection
(`m.lst` / `m.gms`).  pympcc users currently have no way to see the NLP
that IPOPT actually solves after a strategy reformulation has been
applied — useful for debugging and for cross-solver comparison.

Targets:

* `result.dump_reformulated_nlp(path, format="nl")` — writes the
  reformulated problem as an AMPL `.nl` file (reusing §6.4
  infrastructure in reverse).
* `format="pyomo"` — emits a `pyomo.ConcreteModel` for users who want
  to introspect or re-solve via a different backend.
* `MPCCSolver(..., dump_nlp_path=...)` — auto-dump on solve.

Module: new `pympcc/_dump.py`; `.nl` writer is the inverse of the
existing `pympcc.frontend.ampl` reader.

---

## Conventions

* Every presolve pass is opt-in via `presolve=True`; default behaviour
  is unchanged.
* Every presolve pass returns `(reduced_problem, PresolveMap)`;
  `is_identity` is `True` when nothing changed.
* `PresolveMap.expand_result` is the single point that lifts solver
  output back to the original variable / comp-pair indexing.
* Reduction happens once at solver construction time, never inside an
  outer iteration.
* Reduction is COO-sparsity-aware; passes that need structural
  information fall back to a no-op when the user supplied dense
  Jacobians without sparsity patterns.
* Diagnostics (§2) run after the strategy returns and never mutate
  `result.x` or `result.mult_g`.  They populate side fields and emit
  `UserWarning` only when they detect something genuinely wrong.
