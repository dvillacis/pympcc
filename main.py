"""
Simple MPCC example:

    min  (x0 - 2)^2 + (x1 - 1)^2
    s.t. x0 >= 0,  x1 >= 0,  x0 * x1 = 0   (complementarity)

The complementarity splits the feasible set into two branches:
  - Branch A: x0 = 0  →  minimiser at (0, 1),  f* = 4
  - Branch B: x1 = 0  →  minimiser at (2, 0),  f* = 1   ← global

Expected output: x* ≈ (2, 0), f* ≈ 1.
"""

import numpy as np
import pympcc


def main():
    problem = pympcc.MPCCProblem(
        n=2,
        n_comp=1,
        x0=np.array([0.5, 0.5]),
        # Objective: f(x) = (x0-2)^2 + (x1-1)^2
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        # Complementarity: x0 >= 0, x1 >= 0, x0 * x1 = 0
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )

    for strategy in ("direct", "scholtes", "smoothing", "lin_fukushima",
                     "augmented_lagrangian"):
        result = pympcc.solve(problem, strategy=strategy)
        print(
            f"[{strategy:10s}]  x={result.x}  "
            f"obj={result.obj:.6f}  "
            f"comp_res={result.comp_residual:.2e}  "
            f"success={result.success}  "
            f"stationarity={result.stationarity!r}"
        )


if __name__ == "__main__":
    main()
