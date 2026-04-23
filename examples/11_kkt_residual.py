"""
Example 11 — KKT stationarity residual
=======================================

``result.kkt_residual`` is the infinity-norm of the MPCC stationarity residual:

    ‖∇f(x) + Jg(x)ᵀλ_g + Jh(x)ᵀλ_h + JG(x)ᵀμ_G + JH(x)ᵀμ_H − z_L + z_U‖_∞

where μ_G and μ_H are strategy-specific effective MPCC multipliers derived
from the raw IPOPT constraint multipliers, and z_L / z_U are the variable
bound multipliers from IPOPT.

This residual measures how close the iterate is to satisfying the MPCC
first-order stationarity conditions, independent of the complementarity gap.
A value below 1e-6 is a strong indicator of a high-quality KKT point.

Why not just use ``result.stationarity``?
-----------------------------------------
IPOPT's interior-point barrier forces all active lower-bound multipliers to
be non-negative at convergence, so ``result.stationarity`` is almost always
``"S-stationary"`` regardless of the true stationarity type.
``result.kkt_residual`` is a continuous measure and is more informative.

This example solves the same simple problem with all five iterative strategies
and prints the KKT residual alongside the complementarity residual.
"""

import numpy as np
import pympcc

# Simple MPCC: min (x0-2)² + (x1-1)²  s.t. x0 ⊥ x1, x0,x1 ≥ 0
# Optimal solution: x0*=2, x1*=0, f*=1.
problem = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    objective=lambda x: (x[0] - 2) ** 2 + (x[1] - 1) ** 2,
    gradient=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
)

strategies = ["scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian", "slack"]

print(f"{'strategy':<25}  {'obj':>8}  {'comp':>10}  {'kkt':>10}  stat")
print("-" * 72)
for strategy in strategies:
    result = pympcc.solve(problem, strategy=strategy)
    kkt_str = f"{result.kkt_residual:.2e}" if result.kkt_residual is not None else "N/A"
    print(
        f"{strategy:<25}  {result.obj:>8.4f}  "
        f"{result.comp_residual:>10.2e}  {kkt_str:>10}  {result.stationarity}"
    )
