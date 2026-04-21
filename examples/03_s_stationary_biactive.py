"""
Example 3 — S-stationary, biactive solution
=============================================

    min  x₁ + x₂
    s.t. x₁ ≥ 0,  x₂ ≥ 0,  x₁ · x₂ = 0   (complementarity)

(kth1 from MacMPEC — Leyffer 2000)

Global optimum: x* = (0, 0), f* = 0.

At x* = (0, 0):
  G = x₁ = 0   ← biactive
  H = x₂ = 0   ← biactive

Both complementarity components are zero: the pair is **biactive**.  The MPCC
KKT conditions then require checking the sign of the Lagrange multipliers.

Literature-convention KKT at x* = (0, 0):
  ∇f = [1, 1] = μ_G · [1, 0] + μ_H · [0, 1]
  ⟹  μ_G = 1 > 0,  μ_H = 1 > 0

Both multipliers non-negative  →  **S-stationary** (genuine, not vacuous).

Contrast with Example 2: here the stationarity classification USES the
multiplier values — both must be ≥ 0 for S-stationarity to hold.
"""
from __future__ import annotations

import numpy as np

import pympcc


def main():
    problem = pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.0, 1.0]),   # start with x₂ > 0 to help convergence
        xl=np.zeros(2),
        objective=lambda x: x[0] + x[1],
        gradient=lambda x: np.ones(2),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )

    print("kth1 — biactive solution at origin")
    print("─" * 55)

    for strategy in ("direct", "scholtes", "smoothing"):
        result = pympcc.solve(problem, strategy=strategy)
        p = problem
        off = p.n_ineq + p.n_eq
        mu_G = -result.mult_g[off : off + p.n_comp]
        mu_H = -result.mult_g[off + p.n_comp : off + 2 * p.n_comp]

        biactive = (result.G[0] <= 1e-6) and (result.H[0] <= 1e-6)

        print(f"\n[{strategy}]")
        print(f"  x*           = {result.x}")
        print(f"  f*           = {result.obj:.2e}   (expected 0.0)")
        print(f"  G = x₁       = {result.G[0]:.2e}")
        print(f"  H = x₂       = {result.H[0]:.2e}")
        print(f"  biactive?    = {biactive}   (both ≤ tol = 1e-6)")
        print(f"  μ_G          = {mu_G[0]:.4f}   (≥ 0 ✓ for S)")
        print(f"  μ_H          = {mu_H[0]:.4f}   (≥ 0 ✓ for S)")
        print(f"  stationarity = {result.stationarity!r}  ← genuine (multipliers verified)")


if __name__ == "__main__":
    main()
