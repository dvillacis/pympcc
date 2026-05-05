---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Sparse Jacobians

For larger MPCCs, supplying Jacobians in **COO sparse format** avoids allocating a dense `(m, n)` matrix on every IPOPT callback. The four `*_jacobian_sparsity` fields on `MPCCProblem` accept `(row_indices, col_indices)` tuples; the corresponding `*_jacobian` callable then returns a 1-D values array (one entry per nonzero) instead of a dense matrix.

## Dense vs. sparse — same problem

```{code-cell} ipython3
import warnings
import numpy as np
import pympcc

# min (x0 - 2)^2 + (x1 - 1)^2  s.t.  x0 ⊥ x1
def build_dense():
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),    # 1×2 dense (2 entries)
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),    # 1×2 dense
    )

def build_sparse():
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([1.0]),                          # 1 nnz
        comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),            # row 0, col 0
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([1.0]),                          # 1 nnz
        comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),            # row 0, col 1
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r_d = pympcc.solve(build_dense(),  strategy="scholtes")
    r_s = pympcc.solve(build_sparse(), strategy="scholtes")

print(f"dense  x* = {r_d.x},   f* = {r_d.obj:.6f}")
print(f"sparse x* = {r_s.x},   f* = {r_s.obj:.6f}")
print(f"agree: {np.allclose(r_d.x, r_s.x, atol=1e-6)}")
```

## A larger problem where sparsity actually matters

Suppose we have $n = 50$ variables and $n_\text{comp} = 25$ pairs, where each pair uses only **two** of the variables. Dense storage costs `25 × 50 = 1250` entries per Jacobian; sparse costs `25 × 1 = 25`.

```{code-cell} ipython3
n = 50
n_comp = 25
rng = np.random.default_rng(0)

# Pair k: x[2k] >= 0  ⊥  x[2k+1] >= 0   (each pair touches exactly 2 variables)
G_rows = np.arange(n_comp)
G_cols = 2 * np.arange(n_comp)         # x[0], x[2], x[4], ...
H_rows = np.arange(n_comp)
H_cols = 2 * np.arange(n_comp) + 1     # x[1], x[3], x[5], ...

target = rng.uniform(0.5, 1.5, size=n)

problem = pympcc.MPCCProblem(
    n=n, n_comp=n_comp,
    x0=np.full(n, 0.3),
    xl=np.zeros(n),
    objective=lambda x: float(np.sum((x - target) ** 2)),
    gradient=lambda x: 2.0 * (x - target),
    comp_G=lambda x: x[G_cols],
    comp_G_jacobian=lambda x: np.ones(n_comp),          # all ones, 1 per pair
    comp_G_jacobian_sparsity=(G_rows, G_cols),
    comp_H=lambda x: x[H_cols],
    comp_H_jacobian=lambda x: np.ones(n_comp),
    comp_H_jacobian_sparsity=(H_rows, H_cols),
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    r = pympcc.solve(problem, strategy="scholtes")

print(f"converged: {r.success}")
print(f"obj      : {r.obj:.4f}")
print(f"comp_res : {r.comp_residual:.2e}")
print(f"nnz / dense Jacobian entries: {n_comp} / {n_comp * n} "
      f"({100 * n_comp / (n_comp * n):.1f}%)")
```

The 25-nnz sparse Jacobian sends 50× less data to IPOPT per callback than the dense `25 × 50` matrix.

## Tips

- **Build sparsity arrays once at construction.** Don't rebuild `(rows, cols)` per call.
- **Order doesn't matter** — IPOPT internalises the sparsity pattern; pass any consistent ordering, then return values in the same order from the callable.
- **Auto-sparsity from JAX.** With `derivatives="jax"`, the package detects sparsity automatically and skips the dense path. See the [JAX backend example](solve_jax_demo.md).
- **Diagnostic.** `result.kkt_residual` and `result.comp_residual` should match the dense path within solver tolerance; if they diverge, double-check that your nnz values come out in the same order as your `(rows, cols)`.
