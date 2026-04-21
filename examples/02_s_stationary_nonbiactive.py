"""
Example 2 — S-stationary, non-biactive solution
=================================================

    min  (x₀ − 2)² + (x₁ − 1)²
    s.t. x₀ ≥ 0,  x₁ ≥ 0,  x₀ · x₁ = 0   (complementarity)

The feasible set splits into two branches:
  • Branch A: x₀ = 0  →  minimiser at (0, 1),  f* = 4
  • Branch B: x₁ = 0  →  minimiser at (2, 0),  f* = 1   ← global

At x* = (2, 0):
  G = x₀ = 2   (strictly positive)
  H = x₁ = 0   (zero, lower bound active)

Because G = 2 >> 0 the complementarity pair is **not biactive** — only H is
at zero. The biactive set I₀₀ is empty, so x* is **vacuously S-stationary**
regardless of what the multipliers look like.

Moral: when all complementarity pairs have at least one component clearly
positive (non-degenerate solution), the solver trivially achieves the
strongest stationarity level.
"""
from __future__ import annotations

import numpy as np

import pympcc


def main():
    problem = pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )

    print("Simple quadratic MPCC — non-biactive solution")
    print("─" * 55)

    for strategy in ("direct", "scholtes", "smoothing"):
        result = pympcc.solve(problem, strategy=strategy)
        p = problem
        off = p.n_ineq + p.n_eq
        mu_G = -result.mult_g[off : off + p.n_comp]
        mu_H = -result.mult_g[off + p.n_comp : off + 2 * p.n_comp]

        print(f"\n[{strategy}]")
        print(f"  x*          = {result.x}")
        print(f"  f*          = {result.obj:.6f}   (expected 1.0)")
        print(f"  G = x₀      = {result.G[0]:.4f}   (clearly > 0 → not biactive)")
        print(f"  H = x₁      = {result.H[0]:.2e}")
        print(f"  μ_G         = {mu_G[0]:.4f}   (irrelevant: pair not biactive)")
        print(f"  μ_H         = {mu_H[0]:.4f}")
        print(f"  stationarity = {result.stationarity!r}  ← vacuous (I₀₀ = ∅)")


if __name__ == "__main__":
    main()
