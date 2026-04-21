"""
Example 6 — Stationarity across strategies and problems
=========================================================

Runs the three built-in strategies (direct, scholtes, smoothing) on several
benchmark problems and reports the stationarity level alongside the standard
convergence diagnostics.

In practice, all three strategies converge to S-stationary solutions for these
well-conditioned problems.  This is expected:

  • Scholtes (2001) proves that limit points of the regularisation sequence are
    at least C-stationary (Theorem 4.1).
  • In practice, the interior-point solver naturally reaches S-stationary points
    because they satisfy the NLP KKT conditions with the correct multiplier signs.
  • Getting C or W stationarity computationally requires degenerate starting
    conditions or specific problem structure (see examples 04 and 05).

The 'stationarity' field is most informative when the solution is biactive
(both G_i ≈ 0 and H_i ≈ 0 for some i).  The kth1 problem has a biactive
solution at (0, 0) — the report below will show 'S-stationary' with multipliers
that are explicitly positive (genuine, not vacuous).
"""
from __future__ import annotations

import math

import numpy as np

import pympcc

# ── Problem definitions ─────────────────────────────────────────────────── #

PROBLEMS: dict[str, pympcc.MPCCProblem] = {
    "simple (non-biactive)": pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    ),
    "kth1 (biactive at origin)": pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.0, 1.0]),
        xl=np.zeros(2),
        objective=lambda x: x[0] + x[1],
        gradient=lambda x: np.ones(2),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    ),
    "bard1 (bilevel, 3 compl.)": pympcc.MPCCProblem(
        n=5, n_comp=3, n_eq=1,
        x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
        xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
        objective=lambda x: (x[0]-5)**2 + (2*x[1]+1)**2,
        gradient=lambda x: np.array([2*(x[0]-5), 4*(2*x[1]+1), 0, 0, 0]),
        eq_constraints=lambda x: np.array([
            2*(x[1]-1) - 1.5*x[0] + x[2] - 0.5*x[3] + x[4]
        ]),
        eq_jacobian=lambda x: np.array([[-1.5, 2.0, 1.0, -0.5, 1.0]]),
        comp_G=lambda x: np.array([3*x[0]-x[1]-3, -x[0]+0.5*x[1]+4, -x[0]-x[1]+7]),
        comp_G_jacobian=lambda x: np.array([
            [3, -1, 0, 0, 0], [-1, 0.5, 0, 0, 0], [-1, -1, 0, 0, 0]
        ]),
        comp_H=lambda x: np.array([x[2], x[3], x[4]]),
        comp_H_jacobian=lambda x: np.array([
            [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]
        ]),
    ),
    "scholtes1 (nonlinear G)": pympcc.MPCCProblem(
        n=3, n_comp=1,
        x0=np.array([1.0, 1.0, 1.0]),
        xl=np.array([0.0, -np.inf, 0.0]),
        objective=lambda x: (x[0]+1)**2 + (x[1]-2.5)**2 + (x[2]+1)**2,
        gradient=lambda x: np.array([2*(x[0]+1), 2*(x[1]-2.5), 2*(x[2]+1)]),
        comp_G=lambda x: np.array([-math.exp(x[0]) + x[1] - math.exp(x[2])]),
        comp_G_jacobian=lambda x: np.array([[-math.exp(x[0]), 1.0, -math.exp(x[2])]]),
        comp_H=lambda x: np.array([x[0]]),
        comp_H_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
    ),
}


def _biactive_info(result: pympcc.MPCCResult, problem: pympcc.MPCCProblem, tol: float = 1e-6) -> str:
    """Return a short description of biactive pairs."""
    mask = (result.G <= tol) & (result.H <= tol)
    n_ba = int(np.sum(mask))
    total = problem.n_comp
    if n_ba == 0:
        return f"no biactive pairs (vacuous)"
    return f"{n_ba}/{total} biactive pair(s)"


def main():
    strategies = ["direct", "scholtes", "smoothing"]

    for prob_name, problem in PROBLEMS.items():
        print("=" * 65)
        print(f"Problem: {prob_name}")
        print("=" * 65)
        print(f"  {'strategy':<10s}  {'f*':>10s}  {'comp_res':>10s}  {'biactive':>22s}  stationarity")
        print(f"  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*22}  {'─'*14}")

        for strategy in strategies:
            try:
                result = pympcc.solve(problem, strategy=strategy)
                bi = _biactive_info(result, problem)
                print(
                    f"  {strategy:<10s}  {result.obj:>10.4f}  "
                    f"{result.comp_residual:>10.2e}  {bi:>22s}  "
                    f"{result.stationarity}"
                )
            except Exception as exc:
                print(f"  {strategy:<10s}  ERROR: {exc}")

        print()


if __name__ == "__main__":
    main()
