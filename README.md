# pympcc

A Python package for solving **Mathematical Programs with Complementarity Constraints (MPCC)** using [IPOPT](https://github.com/coin-or/Ipopt) via [cyipopt](https://github.com/mechmotum/cyipopt).

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

**Requirements:** Python ≥ 3.11, a working IPOPT installation, and `cyipopt`.

```bash
# macOS (Homebrew)
brew install ipopt

# Linux (Debian/Ubuntu)
sudo apt-get install coinor-libipopt-dev

# Install the package
pip install pympcc

# With development dependencies (pytest, coverage)
pip install "pympcc[dev]"

# Optional: Numba JIT kernels for large sparse problems (~2–5× speedup on hot paths)
pip install "pympcc[numba]"
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
    gradient,                  # ∇f(x) → (n,)

    comp_G,                    # G(x) → (n_comp,)   must be ≥ 0
    comp_G_jacobian,           # ∇G(x) → (n_comp, n)  or 1-D nnz values if sparse
    comp_H,                    # H(x) → (n_comp,)   must be ≥ 0
    comp_H_jacobian,           # ∇H(x) → (n_comp, n)  or 1-D nnz values if sparse

    xl=None,                   # (n,) lower bounds on x  (default: −∞)
    xu=None,                   # (n,) upper bounds on x  (default: +∞)

    n_ineq=0,                  # number of inequality constraints g(x) ≤ 0
    ineq_constraints=None,     # g(x)  → (n_ineq,)
    ineq_jacobian=None,        # ∇g(x) → (n_ineq, n)

    n_eq=0,                    # number of equality constraints h(x) = 0
    eq_constraints=None,       # h(x)  → (n_eq,)
    eq_jacobian=None,          # ∇h(x) → (n_eq, n)

    # Sparse Jacobian support (COO format) — see "Sparse Jacobians" section
    comp_G_jacobian_sparsity=None,   # (row_indices, col_indices)
    comp_H_jacobian_sparsity=None,
    ineq_jacobian_sparsity=None,
    eq_jacobian_sparsity=None,

    # Analytical Lagrangian Hessian (optional) — see "Analytical Hessian" section
    lagrangian_hessian=None,          # (x, lagrange, obj_factor) → nnz values; x-space (direct/scholtes/lin_fukushima)
    lagrangian_hessian_sparsity=None, # (row_indices, col_indices) — lower triangle, row ≥ col
    lagrangian_hessian_slack=None,    # same signature but z=[x,s_G,s_H]-space; slack strategy only
    lagrangian_hessian_slack_sparsity=None,
)
```

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

    # Finite-difference gradient (prototyping only):
    # gradient="fd",
)
result = pympcc.solve(model)
```

`StructuredMPCC` accepts `"fd"` as the value for `gradient`, `comp_G_jacobian`, `comp_H_jacobian`, `jac_eq_nl`, or `jac_ineq_nl`. `MPCCProblem` supports the same `"fd"` sentinel for `gradient`, `comp_G_jacobian`, `comp_H_jacobian`, `ineq_jacobian`, and `eq_jacobian`. Both warn at construction when FD is active. Use `fd_mode="central"` for higher accuracy.

### `solve`

```python
result = pympcc.solve(
    problem,                   # MPCCProblem or StructuredMPCC
    strategy="scholtes",       # see strategy table above
    ipopt_options=None,        # dict of IPOPT options, e.g. {"max_iter": 500}
    # Common strategy options (apply to all iterative strategies):
    epsilon_0=1.0,             # initial relaxation/smoothing/penalty parameter
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
| `stationarity` | `str` | Stationarity type: `"S-stationary"`, `"unknown"`, or `"not stationary"` |
| `kkt_residual` | `float` or `None` | MPCC stationarity residual `‖∇f + Jgᵀλ_g + Jhᵀλ_h + JGᵀμ_G + JHᵀμ_H − z_L + z_U‖_∞`; `None` when multipliers unavailable |
| `history` | `list[IterationInfo]` | Per-iteration diagnostics (iterative strategies only) |
| `mult_g` | `(m,)` or `None` | Constraint multipliers from last inner IPOPT solve |

`IterationInfo` fields: `epsilon`, `x`, `obj`, `status`, `message`, `comp_residual`.

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

### Inspecting iteration history

```python
result = pympcc.solve(problem, strategy="scholtes", max_iter=10)

for it in result.history:
    print(f"ε={it.epsilon:.2e}  obj={it.obj:.6f}  comp={it.comp_residual:.2e}")
```

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

All six strategies dispatch to the sparse NLP adapter automatically when any sparsity field is set. Derived blocks (`G·H`, `G+H`, `φ_ε`) use the union of the G and H sparsity patterns — no dense `(m, n)` matrix is ever allocated on hot-path callbacks.

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

By default IPOPT uses a limited-memory BFGS (L-BFGS) Hessian approximation. For problems where you can provide the exact Lagrangian Hessian you can supply it directly to get better convergence in the inner NLP solves.

```python
# H_L(x, λ, σ) = σ·∇²f + Σ_i λ_i ∇²c_i
# Return: 1-D array of nnz values (lower triangle, row ≥ col)

def my_hessian(x, lagrange, obj_factor):
    # For a problem with n_eq equality constraints and n_comp complementarity pairs.
    # Multiplier ordering (direct / scholtes / lin_fukushima):
    #   lagrange = [λ_g (n_ineq), λ_h (n_eq), λ_G (n_comp), λ_H (n_comp), λ_GH (n_comp)]
    # lin_fukushima appends one extra λ_GpH (n_comp) block (G+H is linear → zero Hessian).
    ...
    return nnz_values  # shape (nnz,)

hess_rows = np.array([...])  # row indices, row ≥ col
hess_cols = np.array([...])  # col indices

problem = pympcc.MPCCProblem(
    ...,
    lagrangian_hessian=my_hessian,
    lagrangian_hessian_sparsity=(hess_rows, hess_cols),
)
result = pympcc.solve(problem, strategy="scholtes")
```

The `slack` strategy operates in the lifted space `z = [x, s_G, s_H]` (length `n + 2·n_comp`). Its Hessian callable receives a `z` vector and multipliers in the lifted-constraint ordering `[λ_h, λ_{G−s_G}, λ_{H−s_H}, λ_{s_G s_H}]`:

```python
problem = pympcc.MPCCProblem(
    ...,
    lagrangian_hessian_slack=my_hessian_slack,
    lagrangian_hessian_slack_sparsity=(slack_rows, slack_cols),
)
result = pympcc.solve(problem, strategy="slack")
```

**Notes:**
- Supports `direct`, `scholtes`, and `lin_fukushima` via `lagrangian_hessian`; supports `slack` via `lagrangian_hessian_slack`.
- `augmented_lagrangian` and `smoothing` are not supported — the PHR penalty and Fischer-Burmeister smoothing introduce second-derivative terms that cannot be expressed as a static callable.
- If only one of the two Hessian fields is set, the supported strategies use the provided Hessian while the remaining strategies fall back to L-BFGS.
- If JAX is installed (`pip install "pympcc[jax]"`), autodiff Hessians are computed automatically when no manual Hessian is provided.
- The exact Hessian can be indefinite near MPCC solutions (LICQ typically fails there). If IPOPT enters a restoration phase with the exact Hessian, switching to L-BFGS (i.e., omitting the Hessian fields) is often more robust.

### Verbose IPOPT output

```python
result = pympcc.solve(problem, ipopt_options={"print_level": 5})
```

### Stationarity classification and KKT residual

```python
result = pympcc.solve(problem, strategy="scholtes")
print(result.stationarity)     # "S-stationary"
print(result.kkt_residual)     # ~1e-10 for converged solves

# Or call directly with a tolerance:
from pympcc import classify_stationarity
stat = classify_stationarity(result, problem, tol=1e-6)
```

> **Note:** Because IPOPT uses a primal-dual interior-point method, the barrier forces
> all active lower-bound constraint multipliers to be non-negative at convergence, so
> `result.stationarity` is almost always `"S-stationary"` for fully-converged IPOPT
> solutions. Use `result.kkt_residual` as the primary quality metric — values below
> `1e-6` indicate a well-converged KKT point.

---

## Testing

```bash
pytest tests/ -v
```

The test suite covers all six strategies against the [MacMPEC benchmark collection](https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC) (Leyffer, 2000).

| File | Contents |
|---|---|
| `tests/macmpec_problems.py` | 13 MacMPEC problems with exact Jacobians and known optima |
| `tests/test_macmpec.py` | Parametrized benchmark tests: convergence, objective accuracy, complementarity feasibility, KKT residual |
| `tests/test_strategies.py` | Unit tests for all strategies, options, result fields, and problem validation |
| `tests/test_kernels.py` | Correctness tests for the hot-path numerical kernels |
| `tests/test_stationarity.py` | Stationarity classification logic |

**MacMPEC problems included:**

| Problem | n | n_comp | f* | Notes |
|---|---|---|---|---|
| kth1 | 2 | 1 | 0 | Trivial LPEC |
| kth2 | 4 | 2 | 0 | Two-comp extension of kth1 |
| ralph1 | 2 | 1 | 0 | B-stationary only |
| simple | 2 | 1 | 1 | Quadratic |
| simple_ineq | 2 | 1 | 1 | Simple with active inequality constraint |
| gauvin | 3 | 2 | 20 | Gauvin-Savard |
| bard1 | 5 | 3 | 17 | KKT bilevel (Bard 1991) |
| scholtes1 | 3 | 1 | 2 | Nonlinear (Scholtes 1997) |
| scholtes2 | 3 | 1 | 15 | Nonlinear (Scholtes 1997) |
| chain2 | 3 | 2 | 4 | Chain network, two comp pairs |
| bilevel1 | 4 | 2 | 0 | Bilevel with G=H=0 at optimum |
| outrata31 | 6 | 2 | 0 | KKT bilevel with 2 equalities |
| outrata32 | 9 | 3 | 0 | KKT bilevel with 3 equalities (Outrata 1994) |

> The `direct` strategy tests are marked `xfail(strict=False)` because LICQ generically fails at MPCC feasible points.

---

## References

- Scholtes, S. (2001). Convergence properties of a regularization scheme for mathematical programs with complementarity constraints. *SIAM Journal on Optimization*, 11(4), 918–936.
- Lin, G.-H., & Fukushima, M. (2003). New relaxation method for mathematical programs with complementarity constraints. *Journal of Optimization Theory and Applications*, 118(1), 81–116.
- Huang, X.-X., Yang, X.-Q., & Zhu, D.-L. (2006). A sequential smooth penalization approach to mathematical programming with equilibrium constraints. *Numerical Functional Analysis and Optimization*, 27(1).
- Fischer, A. (1992). A special Newton-type optimization method. *Optimization*, 24(3–4), 269–284.
- Luo, Z.-Q., Pang, J.-S., & Ralph, D. (1996). *Mathematical Programs with Equilibrium Constraints*. Cambridge University Press.
- Leyffer, S. (2000). MacMPEC — AMPL collection of MPECs. Argonne National Laboratory.
