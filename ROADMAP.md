# pympcc Roadmap

This roadmap tracks parity with commercial MPCC solvers (KNITRO MPEC,
GAMS-NLPEC, FilterMPEC, BARON-MPCC) across presolve, diagnostics,
solution methods, and modeling UX.

Items are tagged ✅ *shipped*, *in progress*, or *planned*.  Scope
estimates: **S** (≲200 lines + tests), **M** (~500–1500), **L** (multi-week).

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

### 2.3. Second-order condition (MPCC-SOSC) — (M)

Checks the reduced Lagrangian Hessian is positive definite on the
critical cone — verifies `x*` is a strict local minimiser, not a
saddle.  Requires the Lagrangian Hessian (already exposed via
`MPCCProblem.lagrangian_hessian` when supplied; else finite-difference
fallback).

Surface as `result.sosc` ∈ {`True`, `False`, `None`}.

### 2.4. IIS / minimal infeasible subsystem — (L)

Replaces roadmap C3.  When a strategy returns infeasible, deletion-
filter or Chinneck-style elimination identifies a minimal subset of
constraints whose joint infeasibility certifies the original.  Useful
for debugging large bilevel formulations where infeasibility is hard
to localise.

### 2.5. Solver telemetry — (S)

Per-iteration log of ε, complementarity residual, KKT residual,
biactive-set size, CQ rank deficit.  Most pieces already on
`IterationInfo`; needs a single `result.summary()` formatter.

---

## 3. Solution methods

### 3.1. Branch-and-bound on disjunctions — (L)

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

### 3.3. Multi-start wrapper — (S)

Run any strategy from `K` randomly perturbed starts; return the best
local optimum plus all encountered stationary points.  Cheap publishable
addition; directly addresses single-start bias in the bilevel paper.

Module: `pympcc/multistart.py`.  API: `pympcc.solve(problem, ...,
n_starts=16, perturb_scale=0.1)`.

### 3.4. Elastic-mode penalty — (M)

Anitescu / Leyffer ℓ₁-elastic penalty as a fallback when MFCQ collapses.
Replaces hard equality `G·H = 0` by penalised slacks; complementary
to Scholtes / smoothing when those stall on highly degenerate problems.

Module: `pympcc/strategies/elastic.py`.

---

## 4. Modeling & user experience

### 4.1. Pyomo / mpec.complementarity bridge — (M)

Today users hand-roll callables and COO sparsity.  A Pyomo backend
would let users write
`m.comp = Complementarity(expr=complements(m.x >= 0, m.y >= 0))` and
have pympcc compile it down to `MPCCProblem`.  Eliminates most of the
COO bookkeeping in `bilevel_mpcc_imaging/problem.py`.

Module: `pympcc/frontend/pyomo.py`.

### 4.2. Default JAX-AD path — (S)

`_jax.py` exists but isn't the default.  Lift it so users can pass
`objective` / `comp_G` / `comp_H` as JAX-traceable functions and have
gradients, Jacobians, and sparsity patterns derived automatically.

### 4.3. Auto pair-scaling — (S)

`comp_G_scale` / `comp_H_scale` fields exist but no detector
populates them.  Add a probe that evaluates `|G_i(x0)|, |H_i(x0)|`
across a small batch of perturbations and rescales pairs whose
typical magnitudes differ by > 1e3.

Module: `pympcc/_autoscale.py`.

### 4.4. Result repr / summary formatter — (S)

Single `result.summary(verbosity=...)` that prints obj, comp residual,
CQ class, stationarity class, B-stat verdict, SOSC verdict, biactive
set size.

---

## 5. Bilevel / parametric extensions

### 5.1. Parametric sensitivity (sIPOPT-style) — (M)

Compute `dx*/dp` for parameters `p` entering the MPCC.  Directly
applicable to hyperparameter learning (the bilevel TV use case): the
outer-loop gradient becomes a single linear solve at the inner-loop
optimum instead of unrolled differentiation.

Requires the KKT system at `x*` (already available from IPOPT) and
implicit-function differentiation through the active set.

Module: `pympcc/sensitivity.py`.  API: `pympcc.sensitivity(result,
dp, ...)`.

### 5.2. Multiplier warm-start — (S)

Today the strategy layer warm-starts only the primal `x`.  Carrying
`mult_g`, `mult_x_L`, `mult_x_U` between outer iterations can
dramatically speed up ε-continuation and parametric sweeps.

### 5.3. EPEC / VI extension — (L)

Equilibrium problems with equilibrium constraints; out of scope for
the SIAM imaging paper but a natural follow-up.

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
