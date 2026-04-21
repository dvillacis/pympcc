"""
Example 8 — Sparse Jacobians for a parallel complementarity problem
====================================================================

Demonstrates the memory saving from sparse Jacobians on a problem with
k decoupled complementarity pairs.

Problem (k pairs, n = 2k variables):

    min  Σ_{i=0}^{k-1} [(x_{2i} - 2)² + (x_{2i+1} - 1)²]
    s.t. x_{2i} >= 0,  x_{2i+1} >= 0,  x_{2i} · x_{2i+1} = 0,
         i = 0, …, k−1

Global solution: x_{2i} = 2, x_{2i+1} = 0 for all i   →   f* = k.

Jacobian structure (i-th row of JG / JH):

    JG[i, j] = 1  if j = 2i,  else 0   →  k nonzeros in a k×2k matrix
    JH[i, j] = 1  if j = 2i+1, else 0  →  k nonzeros in a k×2k matrix

Dense entries per callback:  2k²  (both JG and JH are k×2k).
Sparse entries per callback: 2k   (one nonzero per row, diagonal pattern).

    k =  8 : 128 → 16  entries  (8×  reduction)
    k = 20 : 800 → 40  entries  (20× reduction)
    k = 50 : 5000 → 100 entries  (50× reduction)
"""
from __future__ import annotations

import numpy as np
import pympcc


def build_dense(k: int) -> pympcc.MPCCProblem:
    """Dense formulation of the parallel complementarity problem."""
    n = 2 * k
    x0 = np.full(n, 0.5)

    def objective(x):
        return float(np.sum((x[0::2] - 2.0)**2 + (x[1::2] - 1.0)**2))

    def gradient(x):
        g = np.empty(n)
        g[0::2] = 2.0 * (x[0::2] - 2.0)
        g[1::2] = 2.0 * (x[1::2] - 1.0)
        return g

    def comp_G(x):
        return x[0::2].copy()

    def comp_G_jacobian(x):
        J = np.zeros((k, n))
        J[np.arange(k), np.arange(0, n, 2)] = 1.0
        return J

    def comp_H(x):
        return x[1::2].copy()

    def comp_H_jacobian(x):
        J = np.zeros((k, n))
        J[np.arange(k), np.arange(1, n, 2)] = 1.0
        return J

    return pympcc.MPCCProblem(
        n=n, n_comp=k, x0=x0,
        objective=objective, gradient=gradient,
        comp_G=comp_G, comp_G_jacobian=comp_G_jacobian,
        comp_H=comp_H, comp_H_jacobian=comp_H_jacobian,
    )


def build_sparse(k: int) -> pympcc.MPCCProblem:
    """
    Sparse COO formulation of the same problem.

    JG has nonzeros at (row=i, col=2i) for i=0…k−1.
    JH has nonzeros at (row=i, col=2i+1) for i=0…k−1.
    Both value arrays are all-ones (constant Jacobians).
    """
    n = 2 * k
    x0 = np.full(n, 0.5)

    G_rows = np.arange(k)
    G_cols = np.arange(0, n, 2)   # 0, 2, 4, …, 2k-2

    H_rows = np.arange(k)
    H_cols = np.arange(1, n, 2)   # 1, 3, 5, …, 2k-1

    def objective(x):
        return float(np.sum((x[0::2] - 2.0)**2 + (x[1::2] - 1.0)**2))

    def gradient(x):
        g = np.empty(n)
        g[0::2] = 2.0 * (x[0::2] - 2.0)
        g[1::2] = 2.0 * (x[1::2] - 1.0)
        return g

    def comp_G(x):
        return x[0::2].copy()

    def comp_G_jacobian(x):
        return np.ones(k)   # k flat nnz values (all 1s)

    def comp_H(x):
        return x[1::2].copy()

    def comp_H_jacobian(x):
        return np.ones(k)   # k flat nnz values (all 1s)

    return pympcc.MPCCProblem(
        n=n, n_comp=k, x0=x0,
        objective=objective, gradient=gradient,
        comp_G=comp_G,
        comp_G_jacobian=comp_G_jacobian,
        comp_G_jacobian_sparsity=(G_rows, G_cols),
        comp_H=comp_H,
        comp_H_jacobian=comp_H_jacobian,
        comp_H_jacobian_sparsity=(H_rows, H_cols),
    )


def main():
    k = 8
    n = 2 * k
    f_star = float(k)

    dense  = build_dense(k)
    sparse = build_sparse(k)

    dense_nnz  = 2 * k * n   # two full (k, n) matrices
    sparse_nnz = 2 * k       # k nonzeros each in JG and JH

    print(f"Parallel complementarity  (k={k} pairs, n={n} variables)")
    print("=" * 60)
    print(f"  Dense  Jacobian entries per callback: {dense_nnz:>5d}  ({k}×{n} × 2)")
    print(f"  Sparse Jacobian entries per callback: {sparse_nnz:>5d}  (diagonal, {k} nnz each)")
    print(f"  Memory reduction: {dense_nnz // sparse_nnz}×")
    print(f"  Known optimal value: f* = {f_star:.1f}")
    print()

    strategies = ("scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian")
    print(f"  {'strategy':<22}  {'dense f*':>8}  {'sparse f*':>9}  {'comp_res':>10}  ok?")
    print(f"  {'─'*22}  {'─'*8}  {'─'*9}  {'─'*10}  {'─'*4}")
    for s in strategies:
        r_d = pympcc.solve(dense,  strategy=s)
        r_s = pympcc.solve(sparse, strategy=s)
        agree = abs(r_d.obj - r_s.obj) < 1e-4 and r_s.success
        print(
            f"  {s:<22}  {r_d.obj:>8.4f}  {r_s.obj:>9.4f}  "
            f"{r_s.comp_residual:>10.2e}  {'✓' if agree else '✗'}"
        )

    print()
    print("Sparsity pattern:")
    print(f"  JG nonzero cols: {sparse.comp_G_jacobian_sparsity[1].tolist()}")
    print(f"  JH nonzero cols: {sparse.comp_H_jacobian_sparsity[1].tolist()}")


if __name__ == "__main__":
    main()
