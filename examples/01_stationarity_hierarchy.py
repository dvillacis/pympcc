"""
Example 1 — MPCC stationarity hierarchy
========================================

Demonstrates the four stationarity levels (S, M, C, W) using
classify_stationarity() with hand-crafted Lagrange multipliers.

Background
----------
At the optimal solution x* of an MPCC, a complementarity pair (G_i, H_i) is
called **biactive** when G_i ≈ 0 AND H_i ≈ 0.  For biactive pairs, the sign
of the MPCC Lagrange multipliers (μ_G, μ_H) determines the stationarity level:

    Level        Condition for every biactive pair
    ─────────    ──────────────────────────────────────────
    S-stationary  μ_G ≥ 0  AND  μ_H ≥ 0          (strongest)
    M-stationary  μ_G · μ_H = 0  OR  both > 0
    C-stationary  μ_G · μ_H ≥ 0
    W-stationary  (always, if solver converged)    (weakest)

The hierarchy is S ⊆ M ⊆ C ⊆ W:  every S-stationary point is also M, C, W.
When no biactive pair exists the solution is vacuously S-stationary.

IPOPT / cyipopt sign convention
--------------------------------
IPOPT returns mult_g with the opposite sign from the literature convention
(mult_g[i] < 0 when the i-th constraint is at its active lower bound).
pympcc.classify_stationarity() negates mult_g internally, so you always work
with the literature-convention multipliers μ = −mult_g.
"""
from __future__ import annotations

import numpy as np

import pympcc


# ------------------------------------------------------------------ #
# Helper: build a minimal (MPCCResult, MPCCProblem) pair with         #
# prescribed literature-convention multipliers for a biactive pair.   #
# ------------------------------------------------------------------ #

def _make_example(mu_G_lit: float, mu_H_lit: float, label: str):
    """
    Single complementarity pair, both G = H = 0 (biactive).
    Multipliers are given in literature convention (μ ≥ 0 at S-stat).
    Internally converted to IPOPT convention: mult_g = −μ_lit.
    """
    n_comp = 1
    mult_g = np.array([-mu_G_lit, -mu_H_lit, 0.0])  # [G, H, product]

    problem = pympcc.MPCCProblem(
        n=2, n_comp=n_comp,
        x0=np.zeros(2),
        objective=lambda x: 0.0,
        gradient=lambda x: np.zeros(2),
        comp_G=lambda x: np.zeros(n_comp),
        comp_G_jacobian=lambda x: np.zeros((n_comp, 2)),
        comp_H=lambda x: np.zeros(n_comp),
        comp_H_jacobian=lambda x: np.zeros((n_comp, 2)),
    )
    result = pympcc.MPCCResult(
        x=np.zeros(2), obj=0.0, status=0, message="",
        G=np.zeros(n_comp), H=np.zeros(n_comp),
        comp_residual=0.0, comp_residual_mean=0.0, success=True,
        strategy="manual", mult_g=mult_g,
    )
    level = pympcc.classify_stationarity(result, problem)
    print(f"  {label:<30s}  μ_G = {mu_G_lit:+.1f},  μ_H = {mu_H_lit:+.1f}  →  {level}")
    return level


def main():
    print("=" * 65)
    print("MPCC stationarity hierarchy — single biactive pair (G=H=0)")
    print("=" * 65)
    print()

    # S-stationary cases
    print("── S-stationary (both multipliers non-negative) ──────────────")
    _make_example(+1.0, +1.0, "both positive")
    _make_example( 0.0, +1.0, "zero and positive")
    _make_example( 0.0,  0.0, "both zero")
    print()

    # M-stationary but not S
    print("── M-stationary (product = 0, at least one negative) ─────────")
    _make_example( 0.0, -1.0, "zero and negative")
    _make_example(-1.0,  0.0, "negative and zero")
    print()

    # C-stationary but not M
    print("── C-stationary (product > 0, both negative) ─────────────────")
    _make_example(-1.0, -1.0, "both negative")
    _make_example(-2.0, -0.5, "both negative, different mag.")
    print()

    # W-stationary only
    print("── W-stationary only (product < 0, opposite signs) ───────────")
    _make_example(+1.0, -1.0, "positive and negative")
    _make_example(-1.0, +2.0, "negative and positive")
    print()

    # No biactive pairs → vacuously S
    print("── Vacuously S-stationary (no biactive pairs) ─────────────────")
    n_comp = 1
    problem = pympcc.MPCCProblem(
        n=2, n_comp=n_comp,
        x0=np.zeros(2),
        objective=lambda x: 0.0,
        gradient=lambda x: np.zeros(2),
        comp_G=lambda x: np.zeros(n_comp),
        comp_G_jacobian=lambda x: np.zeros((n_comp, 2)),
        comp_H=lambda x: np.zeros(n_comp),
        comp_H_jacobian=lambda x: np.zeros((n_comp, 2)),
    )
    result = pympcc.MPCCResult(
        x=np.zeros(2), obj=0.0, status=0, message="",
        G=np.array([1.0]),   # G = 1.0 >> tol → not biactive
        H=np.array([0.0]),
        comp_residual=0.0, comp_residual_mean=0.0, success=True,
        strategy="manual",
        mult_g=np.array([-999.0, -999.0, 0.0]),  # multipliers irrelevant
    )
    level = pympcc.classify_stationarity(result, problem)
    print(f"  {'G=1 (not biactive)':<30s}  μ_G = ???,  μ_H = ???  →  {level}")
    print("  (multipliers at non-biactive pairs are irrelevant)")
    print()

    print("Key: S ⊆ M ⊆ C ⊆ W  (S is strongest, W is weakest)")


if __name__ == "__main__":
    main()
