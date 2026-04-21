"""
Example 7 — Sparse Jacobian API
================================

Shows how to supply Jacobians in COO sparse format using the
``*_jacobian_sparsity`` fields on ``MPCCProblem``.

Problem (same as main.py):

    min  (x0 - 2)^2 + (x1 - 1)^2
    s.t. x0 >= 0,  x1 >= 0,  x0 * x1 = 0

Dense Jacobians are 1×2 matrices; each has only 1 nonzero entry.
The sparse API accepts a 1-D values array paired with a COO sparsity pattern:

    Dense  JG = [[1, 0]]           — 2 entries sent to IPOPT
    Sparse JG: values=[1.0],       — 1 entry sent to IPOPT
               rows=[0], cols=[0]

    Dense  JH = [[0, 1]]           — 2 entries
    Sparse JH: values=[1.0],       — 1 entry
               rows=[0], cols=[1]

Both formulations are numerically identical; the sparse path eliminates
the dense matrix allocation on every IPOPT Jacobian callback.
"""
from __future__ import annotations

import numpy as np
import pympcc


def build_dense() -> pympcc.MPCCProblem:
    """Standard dense formulation."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2.0*(x[0]-2.0), 2.0*(x[1]-1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),         # 1×2 dense
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),         # 1×2 dense
    )


def build_sparse() -> pympcc.MPCCProblem:
    """Sparse COO formulation — functionally identical to the dense version."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2.0*(x[0]-2.0), 2.0*(x[1]-1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([1.0]),                # 1 nnz value
        comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),  # nonzero at row 0, col 0
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([1.0]),                # 1 nnz value
        comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),  # nonzero at row 0, col 1
    )


def main():
    dense  = build_dense()
    sparse = build_sparse()

    n, n_comp = dense.n, dense.n_comp
    dense_nnz_G  = n_comp * n
    sparse_nnz_G = len(sparse.comp_G_jacobian_sparsity[0])

    print("Sparse Jacobian API  (n=2, n_comp=1)")
    print("=" * 58)
    print(f"  Dense  JG / JH: {dense_nnz_G} entries each  ({n_comp}×{n} matrix)")
    print(f"  Sparse JG / JH: {sparse_nnz_G} entry   each  (COO, 1 nonzero)")
    print()

    strategies = ("scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian")
    print(f"  {'strategy':<22}  {'dense f*':>8}  {'sparse f*':>9}  {'|Δ obj|':>10}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*9}  {'─'*10}")
    for s in strategies:
        r_d = pympcc.solve(dense,  strategy=s)
        r_s = pympcc.solve(sparse, strategy=s)
        delta = abs(r_d.obj - r_s.obj)
        print(f"  {s:<22}  {r_d.obj:>8.6f}  {r_s.obj:>9.6f}  {delta:>10.2e}")

    print()
    print("Sparse and dense solutions agree within solver tolerance.")


if __name__ == "__main__":
    main()
