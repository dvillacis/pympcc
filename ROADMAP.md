# pympcc Roadmap

This document describes what is already implemented, what is planned, and why each item matters for the MPCC research community.  Items are grouped by effort and impact rather than by strict version number, because priorities shift as user feedback arrives.

---

## Current state (v0.1)

| Feature | Status |
|---|---|
| `MPCCProblem` — numeric interface with dense Jacobians | ✅ |
| `StructuredMPCC` — explicit linear + nonlinear constraint layers | ✅ |
| Direct NLP strategy | ✅ |
| Scholtes relaxation (sequential, warm-started) | ✅ |
| Fischer-Burmeister smoothing (sequential, warm-started) | ✅ |
| Lin-Fukushima regularization (sequential, warm-started) | ✅ |
| Augmented Lagrangian strategy (PHR method, penalty in objective) | ✅ |
| MacMPEC benchmark suite (8 problems, 3 checks × 4 strategies) | ✅ |
| `MPCCResult` with per-iteration history | ✅ |
| Finite-difference Jacobian fallback (`"fd"` sentinel) | ✅ |
| Stationarity classification (`result.stationarity`, `classify_stationarity`) | ✅ ⚠️ |
| Dual warm-starting between outer iterations (`dual_warmstart=True`) | ✅ |
| Sparse Jacobian support (COO format via `*_jacobian_sparsity` fields) | ✅ |
| Slack (lifting) strategy — zero x-block in complementarity rows | ✅ |

⚠️ = implemented and correct, but see known limitation under item 2.

---

## Near-term

These items have the highest impact-to-effort ratio and directly lower the barrier to adoption.

### 1. Slack (lifting) strategy ✅

**Why it matters:** In every non-lifted strategy, the complementarity rows `∂(G_i·H_i)/∂x`
have `n` entries each.  For large-n problems (imaging, contact mechanics, traffic networks
with `n ≫ n_comp`) this makes the Jacobian dominated by the complementarity block.

The slack strategy introduces explicit slack variables `s_G = G(x)` and `s_H = H(x)` and
rewrites the complementarity condition as `s_G · s_H ≤ ε` on the slacks alone.  The
Jacobian rows for the complementarity block have **zero entries in x** and only
`2 × n_comp` nonzeros total, independent of `n`:

```
    x (n)          s_G (n_comp)    s_H (n_comp)
[  JG           |      −I         |       0      ]   ← G − s_G
[  JH           |       0         |      −I      ]   ← H − s_H
[   0           |  diag(s_H)      |  diag(s_G)   ]   ← s_G · s_H
```

**Implemented:**
- `SlackStrategy` in `pympcc/strategies/slack.py`
- Always uses sparse NLP adapter; sparsity structure built once before the outer loop
- Outer loop structure identical to Scholtes (`ε → 0`, warm-started between iterations)
- Exposed as `strategy="slack"`; full `dual_warmstart`, `epsilon_0`, `reduction`,
  `epsilon_min`, `max_iter` option support
- `result.stationarity`: `"S-stationary"` (no biactive pairs) or `"unknown"` (biactive
  pairs present — lifted multiplier signs are not constrained)

**Impact for imaging applications (n=10,000, n_comp=50):**

| Strategy | comp-row nnz | total Jacobian entries |
|----------|-------------|------------------------|
| Scholtes | 10,000 / row | 50 × 10,000 = 500,000 |
| Slack    | 2 / row      | 50 × 2 = 100 (+pinning)|

---

### 2. Finite-difference Jacobian fallback

**Why it matters:** Providing exact Jacobians by hand is the single largest friction point for new users.  A finite-difference (FD) fallback would let anyone prototype quickly and switch to exact derivatives only when performance demands it.

**Plan:**
- Forward-difference and central-difference modes, selectable per function (`gradient`, `eq_jacobian`, `comp_G_jacobian`, …)
- Accept `"fd"` as the Jacobian argument: `comp_G_jacobian="fd"`
- Scalar step size option with a sensible default (e.g. `h = sqrt(eps_machine)`)
- Warn when FD is active so users do not leave it on by accident in production

**Impact:** Dramatically reduces onboarding cost; makes `pympcc` accessible to practitioners who do not want to derive derivatives analytically.

---

### 2. Stationarity classification ✅ (implemented — known limitation)

**Why it matters:** NLP solvers find stationary points of the reformulated NLP, but at an MPCC solution several distinct stationarity concepts exist — from weakest (W-stationary) to strongest (S-stationary) — and they carry very different optimality guarantees.  Reporting only `comp_residual` leaves users with no information about the quality of the solution they obtained.

**Stationarity hierarchy (weakest → strongest):**

```
W-stationary  ⊂  C-stationary  ⊂  M-stationary  ⊂  S-stationary  ⊂  B-stationary
```

**Implemented:**
- `result.stationarity: str` populated post-solve by all three strategies
- `pympcc.classify_stationarity(result, problem, tol)` as a standalone utility
- Correct classification logic for all levels based on biactive set and multiplier signs
- Edge cases: `"not stationary"` (failed solve), `"unknown"` (no multipliers available)

**Known limitation — IPOPT sign convention:**
IPOPT's interior-point method uses the Lagrangian `L = f + λᵀc`, so at an active
lower bound `G_i = 0` the KKT multiplier satisfies `λ_G ≤ 0`, and the literature
convention `μ_G = −λ_G ≥ 0` is **structurally guaranteed** for any well-converged
solution.  As a result:

- Non-biactive solutions (G_i > tol or H_i > tol) → always vacuously S-stationary
- Biactive solutions → always genuinely S-stationary (IPOPT's KKT forces `μ ≥ 0`)

The W/C/M distinction is therefore not observable through IPOPT's `mult_g` for
converged solutions.  The classification correctly reports `"not stationary"` for
failed solves and `"unknown"` when multipliers are unavailable, which retains
diagnostic value.

**Meaningful use cases:**
- Confirming that a converged solution has the strongest possible stationarity quality
- Detecting solver failures via the `"not stationary"` / `"unknown"` branches
- Post-processing results from non-interior-point solvers (SNOPT, KNITRO, SQP
  methods) that can produce multipliers of any sign — these would show true W/C/M
  differentiation when pympcc gains alternative NLP backends (see item 13)

**Impact:** Infrastructure is in place and correct; full W/C/M/S differentiation
requires either a non-interior-point backend or a reformulation that bypasses the
sign constraint imposed by IPOPT's KKT conditions.

---

### 3. Lin-Fukushima regularization strategy ✅

**Why it matters:** Scholtes relaxation replaces `G·H ≤ ε` but does not improve the constraint qualification structure.  The Lin-Fukushima regularization (Lin & Fukushima, 2003) uses a tighter perturbation that guarantees MPCC-MFCQ holds at the regularized solution even when MPCC-LICQ fails, which translates to better practical convergence on degenerate problems.

**Reformulation:**

```
G_i(x) * H_i(x) ≤ ε,   G_i(x) + H_i(x) ≥ ε    (Lin-Fukushima)
vs.
G_i(x) * H_i(x) ≤ ε                              (Scholtes)
```

The extra lower bound on `G + H` prevents both variables from simultaneously approaching zero from below — the source of MFCQ failure in Scholtes.

**Implemented:**
- `LinFukushimaStrategy` in `pympcc/strategies/lin_fukushima.py`
- Constraint layout `[g, h, G, H, G·H, G+H]` — one extra row per complementarity pair vs. Scholtes
- Same outer loop structure (sequential warm-started solves, `epsilon_min` stopping criterion)
- Exposed as `strategy="lin_fukushima"`; included in MacMPEC benchmark suite

**Impact:** More robust on problems where Scholtes stalls; differentiates `pympcc` from packages that only implement the basic relaxation.

---

### 4. Dual warm-starting between outer iterations ✅

**Why it matters:** The iterative strategies (Scholtes, smoothing, Lin-Fukushima) previously warm-started only the primal variable `x`.  Providing the dual variables (multipliers) from the previous solve as the starting point for the next often cuts the inner IPOPT iteration count by 30–60% on smooth sequences.

**Implemented:**
- `dual_warmstart=True` (default) in all three iterative strategies
- First outer iteration always cold-starts; from iteration 2 onwards `mult_g`, `mult_x_L`, `mult_x_U` are passed to `nlp.solve()` with `warm_start_init_point = "yes"`
- Disable with `dual_warmstart=False` — produces identical solutions, useful for benchmarking

---

### 5. Expand the MacMPEC benchmark suite

**Why it matters:** The current 8-problem suite covers common structures but misses important categories: nonlinear KKT bilevel, Nash equilibrium (gnash), design-centering, and the larger ex9 family from Luo-Pang-Ralph (1996).

**Plan:**
- Add the following problems with verified optimal values:
  - `ex9.1.1` – `ex9.1.10` (LP/QP inner problems, closed-form optima)
  - `gnash1` (Nash equilibrium, n=18 variables)
  - `dempe` (nonlinear equality from bilevel theory)
  - `outrata32` – `outrata34` (parametric family)
  - `desilva` (design-centering application)
- Automate optimal-value verification by cross-checking scholtes vs. smoothing

---

## Medium-term

These items require more design work but are important for scaling to real research problems.

### 6. Sparse Jacobian support ✅

**Why it matters:** Dense Jacobians of shape `(m, n)` are stored and sent to IPOPT even when 90% of entries are zero.  For problems with `n > 200` or `m > 100` (e.g. contact mechanics, traffic networks), this becomes the bottleneck.

**Implemented:**
- Four optional `*_jacobian_sparsity` fields on `MPCCProblem` (COO format: `(row_indices, col_indices)`)
- When a sparsity field is set, the corresponding `*_jacobian` callable returns a 1-D values array of `nnz` entries instead of a dense `(n_rows, n)` matrix
- `MPCCProblem.is_sparse` property; `_SparseNLP` adapter in `_nlp.py` passes the structure to `jacobianstructure()`
- All five strategies pre-compute the assembled global NLP sparsity structure once before the solve loop and dispatch to `_SparseNLP` automatically
- Derived blocks (G·H, G+H, φ_ε) use the union of the G and H sparsity patterns
- Union index maps (`map1`, `map2`) are precomputed once per solve; derived block values are assembled in O(nnz) from flat user-supplied values — no dense `(m, n)` matrix is ever allocated on hot-path Jacobian callbacks
- Fully backward-compatible: dense path unchanged when no sparsity fields are set

---

### 7. Lagrangian Hessian support

**Why it matters:** IPOPT defaults to a limited-memory BFGS Hessian approximation (`hessian_approximation=limited-memory`), which can be slow near MPCC solutions where the Lagrangian is non-smooth.  Providing the exact Hessian of the Lagrangian

```
W(x, σ, λ) = σ ∇²f + Σ λᵢ ∇²cᵢ
```

typically reduces outer IPOPT iterations by 2–5× on medium-scale problems.

**Plan:**
- Optional `hessian` and `hessianstructure` callbacks on `MPCCProblem`
- Each strategy assembles the Hessian of its full constraint vector (including the complementarity/relaxation terms) and delegates to the user-provided Hessians of `f`, `g`, `h`, `G`, `H`
- Validate shapes at construction

---

### 8. Augmented Lagrangian strategy ✅

**Why it matters:** Sequential quadratic programming (SQP) and augmented Lagrangian (AL) methods handle MPCC degeneracy differently from pure interior-point methods.  An AL outer loop with IPOPT as the inner solver can escape saddle points that IPOPT-direct cannot.

**Reformulation (PHR method for inequality `G_i · H_i ≤ 0`):**

```
min  f(x) + (1/2ρ) Σ_i [max(0, μ_i + ρ G_i H_i)² − μ_i²]
s.t. G(x) ≥ 0,  H(x) ≥ 0,  g(x) ≤ 0,  h(x) = 0
```

The complementarity condition lives entirely in the objective.  No G·H constraint appears — MFCQ holds trivially at every inner-NLP feasible point.

**Implemented:**
- `AugmentedLagrangianStrategy` in `pympcc/strategies/augmented_lagrangian.py`
- Constraint layout `[g, h, G, H]` — only 2 complementarity rows vs 3–4 in Scholtes/Lin-Fukushima
- Outer loop: multiplier update `μ_i ← max(0, μ_i + ρ G_i H_i)`, penalty growth `ρ ← min(τρ, ρ_max)` when residual not decreasing by factor `η`
- `result.history[k].epsilon` stores the current penalty `ρ` (not a relaxation parameter)
- Early termination when `comp_residual < comp_tol`; all standard options (`dual_warmstart`, sparse Jacobians) supported
- Exposed as `strategy="augmented_lagrangian"`; key parameters: `rho_0`, `rho_max`, `tau`, `eta`, `comp_tol`

---

### 9. Multi-start for global search

**Why it matters:** MPCC has exponentially many B-stationary points (one per active-set pattern).  Solvers reliably find *a* local solution; finding the *global* solution requires systematic exploration of starting points.

**Plan:**
- `pympcc.multistart(problem, n_starts, strategy, seed)` utility
- Generates diverse starting points (Latin hypercube, random, grid) within `[xl, xu]`
- Runs solves in parallel via `concurrent.futures.ProcessPoolExecutor`
- Returns all converged results sorted by objective value
- Flags the globally best result

---

### 10. Automatic bilevel-to-MPCC reformulation

**Why it matters:** Bilevel optimization is the largest application domain for MPCC.  Currently users must manually derive and code the lower-level KKT conditions, as in the `bard1` example.  This is error-prone and discourages non-experts.

**Plan:**
- `pympcc.bilevel.BilevelProblem` class accepting:
  - Upper-level objective and constraints
  - Lower-level objective, constraints, and variable partition
- Auto-derives KKT conditions (stationarity, complementarity slackness) under convexity of the lower level
- Validates that the lower-level satisfies LICQ at `x0` (necessary for valid KKT reformulation)
- Converts to `StructuredMPCC` (linear stationarity → `A_eq`, nonlinear complementarity → `comp_G/H`)

---

### 11. Pyomo interoperability

**Why it matters:** A large share of the optimization community already models problems in Pyomo.  Pyomo has an `mpec` extension that can represent complementarity conditions but relies on external transformations and solvers (PATH, IPOPT with manual NLP transformation).  A bridge from `pyomo.mpec` models to `pympcc` would give those users access to the dedicated strategies here.

**Plan:**
- `pympcc.from_pyomo(model)` that reads a `ConcreteModel` with `Complementarity` components
- Extracts variables, objective, standard constraints, and complementarity pairs
- Uses Pyomo's NL writer + numeric evaluation to generate the callbacks
- Requires `pyomo` as an optional dependency

---

## Long-term

These items define what v1.0 looks like as a mature, production-ready research package.

### 12. Automatic differentiation backends

Eliminate the need for any handwritten derivatives by integrating with:

| Backend | Audience |
|---|---|
| **JAX** (`jax.grad`, `jax.jacfwd`) | ML researchers, optimal control |
| **CasADi** | Control engineering, robotics |
| **PyTorch** (`torch.autograd`) | Deep learning + optimization |

**Plan:** A `pympcc.ad` submodule that wraps any differentiable function into the `(fn, jac_fn)` pair expected by `MPCCProblem`.

---

### 13. Alternative NLP solver backends

IPOPT is the default and the most capable open-source option, but other solvers offer distinct advantages:

| Solver | Advantage |
|---|---|
| **KNITRO** (via `knitropy`) | Faster on medium-scale; native MPCC mode |
| **SNOPT** | Robust for highly nonlinear problems |
| **OSQP** | Extremely fast for LP/QP subproblems in iterative strategies |
| **Clarabel** | Conic solver for structured subproblems |

**Plan:** Abstract the NLP solve interface behind a `Backend` protocol so strategies do not depend on cyipopt directly.

---

### 14. Convergence diagnostics and plotting

Researchers need to visualise and compare algorithm behaviour:

- `result.plot_history()` — convergence of objective and `comp_residual` vs outer iteration
- `pympcc.benchmark(problems, strategies)` — table of solve times, iteration counts, final stationarity types, and failures across a problem set
- Export results to pandas `DataFrame` for downstream analysis

---

### 15. Full API documentation (Sphinx + Read the Docs)

- Sphinx `autodoc` with `numpydoc` style
- Mathematical background section (MPCC theory, strategy derivations)
- Application-domain tutorials as Jupyter notebooks:
  - Traffic equilibrium (Wardrop conditions)
  - Bilevel parameter estimation
  - Contact mechanics with Coulomb friction
  - Nash equilibrium computation

---

## What will not be added

To keep scope manageable, `pympcc` will not:

- Implement a full NLP solver internally (cyipopt / alternative backends are always used for the inner problems)
- Support mixed-integer complementarity (MICP) — a fundamentally different class
- Provide a modelling language — problem definition stays purely numerical/callable-based; symbolic layers belong in CasADi / Pyomo / JuMP

---

## Contributing

Contributions are welcome at any priority tier.  The near-term items are the most tractable starting points for new contributors:

1. Open an issue to discuss the approach before implementing
2. Add tests to `tests/` before or alongside the implementation (test-driven is preferred)
3. All new strategies must pass the MacMPEC benchmark suite at the same tolerances used by the existing strategies

---

## References

- Scholtes, S. (2001). *Convergence properties of a regularization scheme for MPCCs*. SIAM J. Optim.
- Lin, G.-H., & Fukushima, M. (2003). *New relaxation method for MPEC*. J. Optim. Theory Appl.
- Kadrani, A., Dussault, J.-P., & Benchakroun, A. (2009). *A new regularization scheme for MPECs*. SIAM J. Optim.
- Kanzow, C., & Schwartz, A. (2013). *A new regularization method for MPCCs*. SIAM J. Optim.
- Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). *Mathematical Programs with Equilibrium Constraints*. Cambridge University Press.
- Leyffer, S. (2006). *Solving MPCCs as NLPs*. Optim. Methods Softw.
- Flegel, M. L., & Kanzow, C. (2005). *Abadie-type constraint qualification for MPECs*. Math. Program.
