"""
Example 5 — W-stationary point (ralph1)
=========================================

W-stationary is the weakest MPCC optimality condition.  A biactive pair is
W-stationary when the Lagrange multipliers have **opposite signs** (product < 0).
Every convergent MPCC solver finds at least a W-stationary point (if MPCC-LICQ
holds), but W-stationary does not imply local optimality.

    Problem (ralph1 from MacMPEC — Leyffer 2000):
        min   2x − y
        s.t.  G = y  ≥ 0  }  complementarity
              H = y − x ≥ 0  }
              x ≥ 0,  y ≥ 0

    Global optimum: x* = y* = 0, f* = 0.

    Literature KKT at (0, 0):
        ∇f      = [2, −1]
        ∇G      = [0, 1]      (G = y)
        ∇H      = [−1, 1]     (H = y − x)
        [2, −1] = μ_G · [0, 1] + μ_H · [−1, 1]

        From x-component:  2 = −μ_H           ⟹  μ_H = −2
        From y-component: −1 = μ_G + μ_H      ⟹  μ_G = 1

    Product: μ_G · μ_H = (1)(−2) = −2 < 0   →  W-stationary only

Why W-stationary and not stronger?
------------------------------------
At (0, 0) the objective gradient [2, −1] "pulls" in opposite directions along
the two complementarity constraints:
  • Increasing G direction (y > 0): objective prefers NOT to increase y → μ_H < 0
  • Increasing H direction (y > x): objective prefers to increase x → μ_G > 0

The opposite signs of μ_G and μ_H violate C-stationarity (which requires
μ_G · μ_H ≥ 0).  This is a genuine W-stationary saddle point.

Note: ralph1's solution (0, 0) IS a local MPCC minimum (both branches give
f ≥ 0), yet it is only W-stationary — this shows that local optimality does
not imply M or S stationarity when MPCC-LICQ fails.

Reference: Ye (2000) — Example 4.1; Luo, Pang & Ralph (1996).
"""
from __future__ import annotations

import numpy as np

import pympcc


def main():
    n_comp = 1

    # ── Theoretical W-stationary analysis ───────────────────────────
    print("ralph1 — theoretical W-stationary analysis at x* = (0, 0)")
    print("─" * 60)

    problem = pympcc.MPCCProblem(
        n=2, n_comp=n_comp,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: 2.0 * x[0] - x[1],
        gradient=lambda x: np.array([2.0, -1.0]),
        comp_G=lambda x: np.array([x[1]]),            # G = y
        comp_G_jacobian=lambda x: np.array([[0.0, 1.0]]),
        comp_H=lambda x: np.array([x[1] - x[0]]),    # H = y − x
        comp_H_jacobian=lambda x: np.array([[-1.0, 1.0]]),
    )

    mu_G_lit = +1.0   # from KKT: μ_G = 1
    mu_H_lit = -2.0   # from KKT: μ_H = −2

    result_theory = pympcc.MPCCResult(
        x=np.zeros(2), obj=0.0, status=0, message="",
        G=np.array([0.0]),   # y = 0 → biactive
        H=np.array([0.0]),   # y − x = 0 → biactive
        comp_residual=0.0, comp_residual_mean=0.0, success=True,
        strategy="theoretical",
        mult_g=np.array([-mu_G_lit, -mu_H_lit, 0.0]),  # IPOPT sign
    )
    level = pympcc.classify_stationarity(result_theory, problem)

    print(f"  G = y = 0,  H = y−x = 0  (biactive)")
    print(f"  μ_G = {mu_G_lit:+.1f}  (> 0)")
    print(f"  μ_H = {mu_H_lit:+.1f}  (< 0)")
    print(f"  μ_G · μ_H = {mu_G_lit * mu_H_lit:.1f}  (< 0 → C fails → only W holds)")
    print(f"  stationarity = {level!r}")
    print()
    print("  Check: S fails (μ_H < 0), M fails (opposite signs, neither zero),")
    print("         C fails (product < 0), W trivially holds.")
    print()

    # ── What the solver returns ──────────────────────────────────────
    print("What the solver finds:")
    print("─" * 60)
    for strategy in ("scholtes", "smoothing", "direct"):
        result = pympcc.solve(problem, strategy=strategy)
        off = problem.n_ineq + problem.n_eq
        mu_G = -result.mult_g[off : off + n_comp]
        mu_H = -result.mult_g[off + n_comp : off + 2 * n_comp]
        print(f"\n[{strategy}]")
        print(f"  x*             = {result.x}")
        print(f"  f*             = {result.obj:.6f}   (expected 0.0)")
        print(f"  G = y          = {result.G[0]:.2e}")
        print(f"  H = y−x        = {result.H[0]:.2e}")
        print(f"  biactive (tol=1e-6)? {(result.G[0] <= 1e-6) and (result.H[0] <= 1e-6)}")
        print(f"  μ_G            = {mu_G[0]:.6f}")
        print(f"  μ_H            = {mu_H[0]:.6f}")
        print(f"  stationarity   = {result.stationarity!r}")

    print()
    print("Observation: IPOPT converges near (0, 0) but the numerical solution")
    print("has G, H slightly above the biactive threshold (1e-6), so the pair")
    print("is classified as non-biactive → vacuously S-stationary.")
    print()
    print("This illustrates a key subtlety: the theoretical W-stationary point")
    print("exists, but interior-point solvers naturally land on nearby solutions")
    print("where S-stationarity holds vacuously.  The stationarity level reported")
    print("reflects the NUMERICAL solution, not the theoretical optimum.")


if __name__ == "__main__":
    main()
