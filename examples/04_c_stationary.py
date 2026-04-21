"""
Example 4 — C-stationary point
================================

C-stationary points arise when the MPCC Lagrange multipliers at a biactive
pair are BOTH NEGATIVE.  These are degenerate critical points that satisfy
weaker optimality conditions than S or M stationarity.

    Theoretical MPCC (ralph1 variant):
        min  −x₁ − x₂
        s.t. x₁ ≥ 0,  x₂ ≥ 0,  x₁ · x₂ = 0

    Biactive point: x* = (0, 0)

    Literature KKT at (0, 0):
        ∇f = [−1, −1] = μ_G · [1, 0] + μ_H · [0, 1]
        ⟹  μ_G = −1 < 0,  μ_H = −1 < 0

    Product: μ_G · μ_H = (−1)(−1) = +1 ≥ 0  →  C-stationary (not M, not S)

Why C-stationary?
-----------------
At (0, 0), the objective gradient [−1, −1] points into the complementarity
cone interior (both x₁ and x₂ "want" to increase).  Each branch offers a
better solution:
  • Branch x₁ = 0: minimiser at x₂ → ∞ (unbounded if unconstrained)
  • Branch x₂ = 0: minimiser at x₁ → ∞ (unbounded if unconstrained)

So (0, 0) is NOT a local MPCC minimum — it is a C-stationary saddle point.
This illustrates that C-stationary ≠ locally optimal.

In practice, IPOPT's interior-point method avoids such saddle points and
converges to S-stationary solutions when they exist.  The example below
demonstrates C-stationarity by directly calling classify_stationarity() with
the theoretical multipliers, then comparing against what the solver actually
finds.

Reference: Scholtes (2001) — limit points of regularisation sequences are
C-stationary (Theorem 4.1), but do not have to be local minima.
"""
from __future__ import annotations

import numpy as np

import pympcc


def main():
    n_comp = 1

    # ── Theoretical C-stationary point ──────────────────────────────
    print("Theoretical C-stationary point at x* = (0, 0)")
    print("  (multipliers computed from MPCC KKT conditions)")
    print("─" * 55)

    problem = pympcc.MPCCProblem(
        n=2, n_comp=n_comp,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        xu=np.ones(2),       # x₁, x₂ ∈ [0, 1] to bound the problem
        objective=lambda x: -(x[0] + x[1]),
        gradient=lambda x: np.array([-1.0, -1.0]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )

    mu_G_lit = -1.0   # from ∇f = μ_G ∇G + μ_H ∇H → μ_G = −1
    mu_H_lit = -1.0   # μ_H = −1

    result_theory = pympcc.MPCCResult(
        x=np.zeros(2), obj=0.0, status=0, message="",
        G=np.array([0.0]), H=np.array([0.0]),   # biactive
        comp_residual=0.0, comp_residual_mean=0.0, success=True,
        strategy="theoretical",
        mult_g=np.array([-mu_G_lit, -mu_H_lit, 0.0]),  # IPOPT sign
    )
    level = pympcc.classify_stationarity(result_theory, problem)

    print(f"  G = 0,  H = 0  (biactive)")
    print(f"  μ_G = {mu_G_lit:+.1f}  (< 0)")
    print(f"  μ_H = {mu_H_lit:+.1f}  (< 0)")
    print(f"  μ_G · μ_H = {mu_G_lit * mu_H_lit:.1f}  (> 0 → C holds; < 0 fails → not W only)")
    print(f"  stationarity = {level!r}")
    print()
    print("  Check: S fails (μ_G < 0), M fails (both < 0, product ≠ 0),")
    print("         C holds (product ≥ 0), W trivially holds.")
    print()

    # ── What the solver actually finds ──────────────────────────────
    print("What the solver finds (interior-point avoids saddle points):")
    print("─" * 55)
    for strategy in ("scholtes", "smoothing"):
        result = pympcc.solve(problem, strategy=strategy)
        off = problem.n_ineq + problem.n_eq
        mu_G = -result.mult_g[off : off + n_comp]
        mu_H = -result.mult_g[off + n_comp : off + 2 * n_comp]
        print(f"\n[{strategy}]")
        print(f"  x*            = {result.x}")
        print(f"  f*            = {result.obj:.4f}")
        print(f"  comp_residual = {result.comp_residual:.2e}")
        print(f"  G             = {result.G[0]:.2e}")
        print(f"  H             = {result.H[0]:.2e}")
        print(f"  μ_G           = {mu_G[0]:.4f}")
        print(f"  μ_H           = {mu_H[0]:.4f}")
        print(f"  stationarity  = {result.stationarity!r}")

    print()
    print("Note: the solver avoids the C-stationary saddle at (0, 0).")
    print("The global MPCC optima are (1, 0) and (0, 1) with f* = −1.")
    print("Both are non-biactive (only one complementarity component is zero)")
    print("→ vacuously S-stationary.  The interior-point method naturally")
    print("reaches these better solutions rather than the degenerate (0, 0).")


if __name__ == "__main__":
    main()
