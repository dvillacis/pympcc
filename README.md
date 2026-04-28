# pympcc
[![PyPI](https://img.shields.io/pypi/v/pympcc)](https://pypi.org/project/pympcc/)
[![CI](https://github.com/dvillacis/pympcc/actions/workflows/tests.yml/badge.svg)](https://github.com/dvillacis/pympcc/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python package for solving **Mathematical Programs with Complementarity Constraints (MPCC)** using [IPOPT](https://github.com/coin-or/Ipopt) via [cyipopt](https://github.com/mechmotum/cyipopt), with an optional SciPy backend for small IPOPT-free runs.

## Problem form

```
min  f(x)
s.t. g(x) ≤ 0              (inequality constraints)
     h(x) = 0              (equality constraints)
     G(x) ≥ 0  }
     H(x) ≥ 0  }           (complementarity)
     G(x)ᵀ H(x) = 0  }
```

---

## Installation

**Requirements:** Python ≥ 3.11. The default IPOPT backend also requires a working IPOPT installation and `cyipopt`.

```bash
# macOS (Homebrew)
brew install ipopt

# Linux (Debian/Ubuntu)
sudo apt-get install coinor-libipopt-dev

# Install the package with the default IPOPT backend
pip install "pympcc[ipopt]"

# Minimal install for the SciPy backend only
pip install "pympcc[scipy]"

# With development dependencies (pytest, coverage, IPOPT backend)
pip install "pympcc[dev]"

# Optional: Numba JIT kernels for large sparse problems (~2–5× speedup on hot paths)
pip install "pympcc[numba]"
```

The custom IPOPT linear-solver bridge is optional and only needed when passing
`linear_solver_fn`. Build it in-place after installing IPOPT and the development
dependencies:

```bash
uv run python setup.py build_ext --inplace

# Optional overrides when IPOPT is not in a standard prefix:
IPOPT_INCLUDE_DIR=/path/to/include/coin-or IPOPT_LIB_DIR=/path/to/lib \
  uv run python setup.py build_ext --inplace
```

---

## Strategies

Six NLP-based reformulation strategies are available. Each transforms the MPCC into one or more standard NLPs solved with IPOPT.

| Strategy | How it works | Loop | Best for |
|---|---|---|---|
| `"direct"` | Replaces `G·H = 0` with `G·H ≤ 0`; single solve | No | Quick feasibility checks |
| `"scholtes"` | Relaxes to `G·H ≤ ε`, drives `ε → 0` (Scholtes 2001) | Yes | **Default — most robust** |
| `"smoothing"` | Fischer-Burmeister equation `φ_ε(G,H) = 0`, drives `ε → 0` | Yes | Tight complementarity residuals |
| `"lin_fukushima"` | `G·H ≤ ε` and `G+H ≥ ε`; guarantees MPCC-MFCQ (Lin & Fukushima 2003) | Yes | Degenerate problems |
| `"augmented_lagrangian"` | PHR penalty in the objective; complementarity never enters the NLP constraints | Yes | Problems where MFCQ fails |
| `"slack"` | Lifts `G`, `H` to slack variables; complementarity rows have zero x-entries | Yes | Large-n problems (`n ≫ n_comp`) |

> **Practical advice:** `"scholtes"` is the safest default. `"smoothing"` often achieves tighter complementarity residuals on smooth problems. `"lin_fukushima"` is more robust on degenerate problems where Scholtes stalls. `"slack"` is the best choice when `n` is large (hundreds to thousands) and `n_comp` is small — the Jacobian of the complementarity block is `O(n_comp)` rather than `O(n_comp × n)`.

---

## Quick start

```python
import numpy as np
import pympcc

# Solve:  min (x0-2)² + (x1-1)²
#         s.t. x0 ≥ 0, x1 ≥ 0, x0·x1 = 0
# Global optimum: x* = (2, 0), f* = 1

problem = pympcc.MPCCProblem(
    n=2,
    n_comp=1,
    x0=np.array([0.5, 0.5]),
    xl=np.zeros(2),
    objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
    gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

result = pympcc.solve(problem, strategy="scholtes")

print(result)
# MPCCResult(strategy='scholtes', success=True, obj=1.0, comp_residual=2.0e-08, status=0)

print(result.x)            # solution vector
print(result.stationarity) # "S-stationary"
```

---

## API reference

### `MPCCProblem`

The core numeric problem definition. All callables are validated at construction by evaluating them at `x0`.

```python
pympcc.MPCCProblem(
    n,                         # int — number of decision variables
    n_comp,                    # int — number of complementarity pairs
    x0,                        # (n,) — initial guess
    objective,                 # f(x)  → float
    gradient,                  # ∇f(x) → (n,)   or "fd" / "jax"

    # Complementarity — standard form:
    comp_G=None,               # G(x) → (n_comp,)   must be ≥ 0
    comp_G_jacobian=None,      # ∇G(x) → (n_comp, n)  or 1-D nnz values if sparse
    comp_H=None,               # H(x) → (n_comp,)   must be ≥ 0
    comp_H_jacobian=None,      # ∇H(x) → (n_comp, n)  or 1-D nnz values if sparse

    # Complementarity — MCP variable-paired form (alternative to comp_G / comp_H):
    comp_var_pairs=None,       # list of (var_idx, h_fn) or (var_idx, h_fn, h_jac_fn)
                               # declares x[var_idx] ≥ 0 ⊥ h_fn(x) ≥ 0 per entry

    xl=None,                   # (n,) lower bounds on x  (default: −∞)
    xu=None,                   # (n,) upper bounds on x  (default: +∞)

    n_ineq=0,                  # number of inequality constraints g(x) ≤ 0
    ineq_constraints=None,     # g(x)  → (n_ineq,)
    ineq_jacobian=None,        # ∇g(x) → (n_ineq, n)  or "fd" / "jax"

    n_eq=0,                    # number of equality constraints h(x) = 0
    eq_constraints=None,       # h(x)  → (n_eq,)
    eq_jacobian=None,          # ∇h(x) → (n_eq, n)  or "fd" / "jax"

    # Derivative shorthand — fills all unset derivative fields at once:
    derivatives=None,          # "fd" or "jax"

    # Sparse Jacobian support (COO format) — see "Sparse Jacobians" section
    comp_G_jacobian_sparsity=None,   # (row_indices, col_indices)
    comp_H_jacobian_sparsity=None,
    ineq_jacobian_sparsity=None,
    eq_jacobian_sparsity=None,

    # Analytical Lagrangian Hessian (optional) — see "Analytical Hessian" section
    lagrangian_hessian=None,          # (x, lagrange, obj_factor) → nnz values
    lagrangian_hessian_sparsity=None, # (row_indices, col_indices) — lower triangle, row ≥ col
    lagrangian_hessian_slack=None,    # same but z=[x,s_G,s_H]-space; slack strategy only
    lagrangian_hessian_slack_sparsity=None,

    # Finite-difference options:
    fd_h=1.49e-8,              # step size (default: sqrt(machine epsilon))
    fd_mode="forward",         # "forward" or "central"
)
```

#### Variable-paired complementarity (`comp_var_pairs`)

Instead of writing `comp_G = lambda x: x[[j, k]]` manually, declare variable-paired complementarity at the variable level:

```python
problem = pympcc.MPCCProblem(
    n=3, n_comp=2, x0=np.array([0.5, 0.5, 0.5]),
    objective=..., gradient=...,
    comp_var_pairs=[
        # (var_idx, h_fn)            → x[0] ≥ 0 ⊥ x[1] ≥ 0   (fd Jacobian for h)
        (0, lambda x: np.array([x[1]])),
        # (var_idx, h_fn, h_jac_fn)  → x[1] ≥ 0 ⊥ x[2] ≥ 0   (exact Jacobian)
        (1, lambda x: np.array([x[2]]), lambda x: np.array([0., 0., 1.])),
    ],
)
```

- `xl[var_idx]` is silently clamped to `max(xl[var_idx], 0.0)`.
- The G-side Jacobian (identity rows) is always exact; `h_jac_fn` may be omitted to use forward FD.
- **Mixed mode**: provide both `comp_G`/`comp_H` (for the first `n_comp − k` pairs) and `comp_var_pairs` (for the last `k` pairs). `n_comp` must equal the sum.

### `StructuredMPCC`

Higher-level interface that accepts linear constraints as matrices (`A_eq`, `b_eq`, `A_ineq`, `b_ineq`) alongside nonlinear callables. Converted to `MPCCProblem` automatically by `solve()`.

```python
model = pympcc.StructuredMPCC(
    n=5, n_comp=2,
    x0=np.ones(5),
    objective=..., gradient=...,
    comp_G=..., comp_G_jacobian=...,
    comp_H=..., comp_H_jacobian=...,

    # Linear equality:  A_eq @ x = b_eq
    A_eq=np.array([[1, 1, 0, 0, 0]]),
    b_eq=np.array([1.0]),

    # Nonlinear inequality:  g_nl(x) ≤ 0
    n_nl_ineq=1,
    ineq_nl=lambda x: np.array([x[0]**2 + x[1] - 2]),
    jac_ineq_nl=lambda x: np.array([[2*x[0], 1, 0, 0, 0]]),
)
result = pympcc.solve(model)
```

### `solve`

```python
result = pympcc.solve(
    problem,                   # MPCCProblem or StructuredMPCC
    strategy="scholtes",       # see strategy table above
    ipopt_options=None,        # dict of IPOPT options, e.g. {"max_iter": 500}

    # Iterative strategy options:
    epsilon_0=1.0,             # initial relaxation / smoothing / penalty parameter
    reduction=0.1,             # multiplicative reduction per outer iteration
    max_iter=20,               # maximum outer iterations
    epsilon_min=1e-8,          # stop when ε < epsilon_min
    dual_warmstart=True,       # warm-start dual variables between iterations

    # augmented_lagrangian-specific:
    rho_0=10.0,                # initial penalty
    rho_max=1e6,               # penalty cap
    tau=10.0,                  # penalty growth factor
    eta=0.25,                  # progress threshold
    comp_tol=1e-8,             # early stop on complementarity residual

    # Diagnostics (CQ, B-stationarity, SOSC):
    diagnostics=False,         # run §2.1 / §2.2 / §2.3 diagnostics after solve
    b_stat_max_biactive=10,    # max biactive pairs for B-stat branch enumeration

    # TNLP certified multiplier refinement:
    tnlp_refine=False,         # re-solve with fixed active set → certified multipliers
    tnlp_max_iter=500,         # IPOPT iteration limit for the TNLP solve

    # Presolve:
    presolve=False,            # run pinned-variable elimination + FBBT before solve
)
```

### `MPCCResult`

| Field | Type | Description |
|---|---|---|
| `x` | `(n,)` | Solution vector |
| `obj` | `float` | Objective value `f(x)` |
| `G`, `H` | `(n_comp,)` | Complementarity values at solution |
| `comp_residual` | `float` | `max_i |G_i · H_i|` |
| `success` | `bool` | `True` when IPOPT converged (status 0, 1, or 3) |
| `status` | `int` | Raw IPOPT exit code |
| `message` | `str` | Human-readable IPOPT status |
| `strategy` | `str` | Strategy name used |
| `stationarity` | `str` | `"S-stationary"`, `"W-stationary"`, or `"unknown"` |
| `kkt_residual` | `float\|None` | `‖∇f + Jgᵀλ_g + Jhᵀλ_h + JGᵀμ_G + JHᵀμ_H‖_∞` |
| `history` | `list[IterationInfo]` | Per-iteration diagnostics (iterative strategies) |
| `mult_g` | `(m,)\|None` | Constraint multipliers from the last inner IPOPT solve |
| `per_pair_status` | `list[str]` | Per-pair label: `"G_active"`, `"H_active"`, `"biactive"`, `"inactive"` |
| `cq` | `str\|None` | CQ at solution: `"MPCC-LICQ"`, `"MPCC-MFCQ"`, `"none"`, `"unknown"` (requires `diagnostics=True`) |
| `cq_rank_deficit` | `int\|None` | Rows minus rank of active-gradient matrix; 0 ⇔ LICQ |
| `b_stationary` | `str\|None` | `"B-stationary"`, `"not B-stationary"`, `"intractable"` (requires `diagnostics=True`) |
| `sosc` | `bool\|None` | SOSC: `True` = strict local min, `False` = not, `None` = skipped (requires `diagnostics=True`) |
| `sosc_min_eigenvalue` | `float\|None` | Minimum eigenvalue of the reduced Hessian W = Z^T ∇²L Z |
| `sosc_skipped_reason` | `str\|None` | Why SOSC was skipped: `"biactive_pairs"`, `"not_converged"`, or `None` |
| `mult_comp_G_mpcc` | `(n_comp,)\|None` | Certified MPCC multipliers μ_G (requires `tnlp_refine=True`) |
| `mult_comp_H_mpcc` | `(n_comp,)\|None` | Certified MPCC multipliers μ_H (requires `tnlp_refine=True`) |
| `tnlp_refined` | `TNLPResult\|None` | Full TNLP sub-result (requires `tnlp_refine=True`) |

`IterationInfo` fields: `epsilon`, `x`, `obj`, `status`, `message`, `comp_residual`, `comp_residual_mean`, `n_ipopt_iter`, `iter_time`.

---

## Examples

### With inequality and equality constraints

```python
import numpy as np
import pympcc

# bard1 (MacMPEC) — bilevel problem via KKT reformulation
# Variables: [x, y, λ1, λ2, λ3]
# Optimal:   x*=1, y*=0, f*=17

problem = pympcc.MPCCProblem(
    n=5, n_comp=3, n_eq=1,
    x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
    xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
    objective=lambda x: (x[0]-5)**2 + (2*x[1]+1)**2,
    gradient=lambda x: np.array([2*(x[0]-5), 4*(2*x[1]+1), 0, 0, 0]),
    eq_constraints=lambda x: np.array([
        2*(x[1]-1) - 1.5*x[0] + x[2] - 0.5*x[3] + x[4]
    ]),
    eq_jacobian=lambda x: np.array([[-1.5, 2.0, 1.0, -0.5, 1.0]]),
    comp_G=lambda x: np.array([3*x[0]-x[1]-3, -x[0]+0.5*x[1]+4, -x[0]-x[1]+7]),
    comp_G_jacobian=lambda x: np.array([
        [ 3, -1, 0, 0, 0],
        [-1, .5, 0, 0, 0],
        [-1, -1, 0, 0, 0],
    ], dtype=float),
    comp_H=lambda x: x[2:5],
    comp_H_jacobian=lambda x: np.eye(3, 5, k=2),
)

result = pympcc.solve(problem, strategy="scholtes")
print(f"x={result.x[:2]}, f*={result.obj:.4f}")   # x=[1. 0.], f*=17.0000
```

### Variable-paired complementarity

For simple `x[j] ≥ 0 ⊥ F(x) ≥ 0` pairs, use `comp_var_pairs` to skip writing `comp_G` manually:

```python
# min (x0-1)² + (x1-1)²  s.t.  x0 ≥ 0 ⊥ x1 ≥ 0
problem = pympcc.MPCCProblem(
    n=2, n_comp=1, x0=np.array([0.5, 0.5]),
    objective=lambda x: (x[0]-1)**2 + (x[1]-1)**2,
    gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
    comp_var_pairs=[
        (0, lambda x: np.array([x[1]]), lambda x: np.array([0., 1.])),
    ],
)
result = pympcc.solve(problem, strategy="scholtes")
```

### Finite-difference derivatives

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # suppress FD UserWarning
    problem = pympcc.MPCCProblem(
        n=2, n_comp=1, x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0]-1)**2 + (x[1]-1)**2,
        comp_var_pairs=[(0, lambda x: np.array([x[1]]))],
        derivatives="fd",                    # fills gradient + all Jacobians
    )
result = pympcc.solve(problem, strategy="scholtes")
```

### Inspecting iteration history

```python
result = pympcc.solve(problem, strategy="scholtes", max_iter=10)

for it in result.history:
    print(f"ε={it.epsilon:.2e}  obj={it.obj:.6f}  comp={it.comp_residual:.2e}")
```

### Per-pair status and structured export

`result.per_pair_status` is always populated; each entry is one of `"G_active"`, `"H_active"`, `"biactive"`, or `"inactive"`.

```python
result = pympcc.solve(problem, strategy="scholtes")
print(result.per_pair_status)   # ["H_active"]

# Serialize to JSON:
json_str = result.to_json()

# Per-pair DataFrame (requires pandas):
df = result.to_dataframe()
# Columns: pair, G, H, GH, status, mu_G, mu_H (last two when tnlp_refine=True)
```

### Diagnostics: CQ, B-stationarity, and SOSC

```python
result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)

print(result.cq)               # "MPCC-LICQ"
print(result.b_stationary)     # "B-stationary"
print(result.sosc)             # True  — strict local minimiser
print(result.sosc_min_eigenvalue)  # > 0
```

**SOSC** checks positive definiteness of the reduced Lagrangian Hessian on the critical cone:
- `True` — `x*` is a certified strict local minimiser.
- `False` — saddle point or only a local max in some direction.
- `None` — skipped (biactive pairs present, or solve did not converge); check `result.sosc_skipped_reason`.

### TNLP certified multipliers

```python
result = pympcc.solve(problem, strategy="scholtes",
                      diagnostics=True, tnlp_refine=True)

print(result.mult_comp_G_mpcc)  # certified μ_G (≥ 0 at S-stationary points)
print(result.mult_comp_H_mpcc)  # certified μ_H
print(result.stationarity)      # "S-stationary" or "W-stationary"
```

TNLP refinement re-solves a tightened NLP with the active set fixed as equalities. This extracts MPCC-clean multipliers and certifies S- or W-stationarity. The FD-based SOSC check uses TNLP multipliers when available (most accurate).

### Sparse Jacobians (large-n problems)

When `n_comp` Jacobian rows each have only a few non-zero columns, pass sparsity patterns in COO format. Jacobian callables then return 1-D arrays of `nnz` values instead of dense `(n_comp, n)` matrices.

```python
# Example: n=1000, comp_G only depends on variables 0 and 1
rows = np.array([0, 0])   # row indices
cols = np.array([0, 1])   # column indices

problem = pympcc.MPCCProblem(
    n=1000, n_comp=1, ...,
    comp_G_jacobian=lambda x: np.array([2*x[0], 3*x[1]]),  # returns nnz values
    comp_G_jacobian_sparsity=(rows, cols),
)
```

All six strategies dispatch to the sparse NLP adapter automatically when any sparsity field is set.

### Slack strategy for large-n problems

The `"slack"` strategy introduces explicit slack variables `s_G = G(x)`, `s_H = H(x)` so the complementarity Jacobian rows have **zero entries in x** — only `2 × n_comp` nonzeros regardless of `n`.

```python
result = pympcc.solve(problem, strategy="slack")
```

For an imaging application with `n=10,000` and `n_comp=50`:

| Strategy | comp-row nnz | total Jacobian entries |
|---|---|---|
| Scholtes / smoothing | 10,000 / row | 50 × 10,000 = 500,000 |
| Slack | 2 / row | 50 × 2 = 100 (+ pinning) |

### Analytical Lagrangian Hessian

By default IPOPT uses L-BFGS. Supplying the exact Hessian improves inner-NLP convergence:

```python
def my_hessian(x, lagrange, obj_factor):
    # Multiplier ordering (direct / scholtes / lin_fukushima):
    #   lagrange = [λ_g (n_ineq), λ_h (n_eq), λ_G (n_comp), λ_H (n_comp)]
    ...
    return nnz_values  # shape (nnz,), lower triangle

problem = pympcc.MPCCProblem(
    ...,
    lagrangian_hessian=my_hessian,
    lagrangian_hessian_sparsity=(hess_rows, hess_cols),  # (row_indices, col_indices), row ≥ col
)
result = pympcc.solve(problem, strategy="scholtes")
```

The `slack` strategy uses `lagrangian_hessian_slack` with `z = [x, s_G, s_H]`.

**Notes:**
- Compatible: `direct`, `scholtes`, `lin_fukushima` (via `lagrangian_hessian`); `slack` (via `lagrangian_hessian_slack`).
- Not compatible: `augmented_lagrangian`, `smoothing` — their objective perturbations add second-derivative terms not captured by a static callable.
- The SOSC check uses the user-supplied Hessian when present, eliminating FD cost.

### Benchmark runner

```python
from pympcc.benchmarks import run_benchmark, print_table, save_csv

results = run_benchmark(strategies=["scholtes", "smoothing"], verbose=True)
print_table(results)
save_csv(results, "results.csv")
```

CLI:

```bash
python -m pympcc.benchmarks.macmpec --strategy scholtes,smoothing --out results.csv
```

13 MacMPEC problems are bundled (`pympcc/benchmarks/_problems.py`). See `ROADMAP.md §6.4` for paths to the full 150-problem Leyffer collection.

---

## Testing

```bash
uv run pytest tests/ -v
```

The test suite covers all six strategies, all diagnostic modules, and the MacMPEC benchmark collection.

| File | Contents |
|---|---|
| `tests/test_macmpec.py` | Parametrized benchmark: convergence, objective accuracy, complementarity, KKT |
| `tests/test_strategies.py` | Unit tests for all strategies, options, result fields, and problem validation |
| `tests/test_diagnostics.py` | CQ classification and active-set tests |
| `tests/test_sosc.py` | SOSC direct unit tests and solver integration (16 cases) |
| `tests/test_tnlp.py` | TNLP refinement, stationarity classification, biactivity pre-screen |
| `tests/test_mcp_var_pairs.py` | Variable-paired complementarity: construction, mixed mode, solve equivalence (21 cases) |
| `tests/test_per_pair_status.py` | Per-pair status, `to_json`, `to_dataframe` |
| `tests/test_kernels.py` | Hot-path numerical kernel correctness |
| `tests/test_stationarity.py` | Stationarity classification and KKT residual |

**MacMPEC problems included:**

| Problem | n | n_comp | f* |
|---|---|---|---|
| kth1 | 2 | 1 | 0 |
| kth2 | 4 | 2 | 0 |
| ralph1 | 2 | 1 | 0 |
| simple | 2 | 1 | 1 |
| simple_ineq | 2 | 1 | 1 |
| gauvin | 3 | 2 | 20 |
| bard1 | 5 | 3 | 17 |
| scholtes1 | 3 | 1 | 2 |
| scholtes2 | 3 | 1 | 15 |
| chain2 | 3 | 2 | 4 |
| bilevel1 | 4 | 2 | 0 |
| outrata31 | 6 | 2 | 0 |
| outrata32 | 9 | 3 | 0 |

> The `direct` strategy tests are marked `xfail(strict=False)` because LICQ generically fails at MPCC feasible points.

---

## References

- Scholtes, S. (2001). Convergence properties of a regularization scheme for mathematical programs with complementarity constraints. *SIAM Journal on Optimization*, 11(4), 918–936.
- Lin, G.-H., & Fukushima, M. (2003). New relaxation method for mathematical programs with complementarity constraints. *Journal of Optimization Theory and Applications*, 118(1), 81–116.
- Huang, X.-X., Yang, X.-Q., & Zhu, D.-L. (2006). A sequential smooth penalization approach to mathematical programming with equilibrium constraints. *Numerical Functional Analysis and Optimization*, 27(1).
- Fischer, A. (1992). A special Newton-type optimization method. *Optimization*, 24(3–4), 269–284.
- Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). *Mathematical Programs with Equilibrium Constraints*. Cambridge University Press.
- Leyffer, S. (2000). MacMPEC — AMPL collection of MPECs. Argonne National Laboratory.
