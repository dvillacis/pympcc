"""
MPCC stationarity classification using IPOPT Lagrange multipliers.

Stationarity hierarchy (weakest → strongest, Ye 2000):

    W-stationary ⊂ C-stationary ⊂ M-stationary ⊂ S-stationary

For biactive pairs I_00 = {i : G_i ≤ tol AND H_i ≤ tol}, let μ_G_i and μ_H_i
be the MPCC multipliers for the G_i ≥ 0 and H_i ≥ 0 constraints respectively:

* **W-stationary**: KKT of the relaxed NLP is satisfied.  Always true when
  IPOPT converged successfully.
* **C-stationary**: W + μ_G_i · μ_H_i ≥ 0 for each biactive pair.
  Captures "same-sign" multipliers; "both negative" satisfies C but not M.
* **M-stationary**: C + NOT(both strictly negative) for each biactive pair.
  Formally: (μ_G_i > 0 AND μ_H_i > 0) OR (μ_G_i · μ_H_i = 0).
* **S-stationary**: μ_G_i ≥ 0 AND μ_H_i ≥ 0 for every biactive pair.
  The strongest KKT-based condition; implies MPCC-LICQ holds at the solution.

Sign convention note
--------------------
IPOPT solves min f(x) s.t. cl ≤ c(x) ≤ cu with KKT ∇f + J(x)ᵀ λ = 0.
For a constraint at its *lower* bound (e.g. G_i ≥ 0 with G_i = 0), IPOPT
returns λ_i ≤ 0.  The literature convention is μ_G_i = −λ_i ≥ 0.
All functions in this module negate the raw ``mult_g`` slices before testing.
"""
from __future__ import annotations

import numpy as np

from .problem import MPCCProblem
from .result import MPCCResult


def classify_stationarity(
    result: MPCCResult,
    problem: MPCCProblem,
    tol: float = 1e-6,
) -> str:
    """
    Classify the MPCC stationarity type of a solved result.

    Returns the strongest level satisfied from the hierarchy
    ``W-stationary ⊂ C-stationary ⊂ M-stationary ⊂ S-stationary``
    (W is weakest, S is strongest).

    Parameters
    ----------
    result : MPCCResult
        A solved MPCC result (from :func:`pympcc.solve`).
    problem : MPCCProblem
        The problem instance used to produce the result.
    tol : float, optional
        Numerical tolerance (default ``1e-6``).  Controls two things:

        * **Biactive detection**: pair *i* is biactive when
          ``G_i ≤ tol AND H_i ≤ tol``.
        * **Sign comparison**: multipliers within ``[-tol, tol]`` are
          treated as zero.

    Returns
    -------
    str
        One of: ``"S-stationary"``, ``"M-stationary"``, ``"C-stationary"``,
        ``"W-stationary"``, ``"unknown"``, or ``"not stationary"``.

        * ``"unknown"`` — multipliers were not stored (``result.mult_g`` is
          ``None``).
        * ``"not stationary"`` — the solve did not converge
          (``result.success`` is ``False``).

    Notes
    -----
    The classification uses the Lagrange multipliers for the G(x) ≥ 0 and
    H(x) ≥ 0 constraints, located at indices
    ``[n_ineq + n_eq : n_ineq + n_eq + 2*n_comp]`` of ``result.mult_g``.
    The product/smoothing constraint multipliers (the last ``n_comp`` entries)
    are intentionally excluded — they are an artifact of the NLP reformulation.

    Practical interpretation:
        * **C-stationary**: sufficient for many engineering applications.
        * **S-stationary**: implies MPCC-LICQ holds; equivalent to
          B-stationary under MPCC-LICQ (Scheel & Scholtes 2000).
        * **W/M-stationary**: solution may be a saddle point; consider
          multi-start or a different strategy.

    References
    ----------
    Ye, J.J. (2000). Necessary optimality conditions for MPECs with
    equilibrium constraints. *Mathematics of Operations Research*, 25(4).

    Scheel, H. and Scholtes, S. (2000). Mathematical programs with
    complementarity constraints: stationarity, optimality, and sensitivity.
    *Mathematics of Operations Research*, 25(1), 1–22.
    """
    if not result.success:
        return "not stationary"
    if result.mult_g is None:
        return "unknown"

    offset = problem.n_ineq + problem.n_eq
    # Convert IPOPT sign convention to literature convention (negate).
    # In IPOPT: KKT ∇f + Jᵀλ = 0 → λ ≤ 0 at active lower bound.
    # In literature: μ = −λ ≥ 0 at S-stationary points.
    mu_G = -result.mult_g[offset           : offset + problem.n_comp]
    mu_H = -result.mult_g[offset + problem.n_comp : offset + 2 * problem.n_comp]

    # Biactive set I_00 = {i : G_i ≤ tol AND H_i ≤ tol}
    mask = (result.G <= tol) & (result.H <= tol)

    # Vacuously S-stationary when there are no degenerate pairs.
    if not np.any(mask):
        return "S-stationary"

    mu_G_ba = mu_G[mask]
    mu_H_ba = mu_H[mask]

    # S-stationary: both multipliers ≥ 0 for every biactive pair.
    # Allow small negative values (within tol) as numerical noise.
    if np.all(mu_G_ba >= -tol) and np.all(mu_H_ba >= -tol):
        return "S-stationary"

    # M-stationary: for each biactive pair, either
    #   (a) both strictly positive, or
    #   (b) at least one is ≈ 0 (product ≈ 0)
    m_cond = (
        ((mu_G_ba > tol) & (mu_H_ba > tol))
        | ((np.abs(mu_G_ba) <= tol) | (np.abs(mu_H_ba) <= tol))
    )
    if np.all(m_cond):
        return "M-stationary"

    # C-stationary: product ≥ 0 for every biactive pair.
    # Handles the "both negative" case (product > 0) which fails M.
    if np.all(mu_G_ba * mu_H_ba >= -(tol ** 2)):
        return "C-stationary"

    # W-stationary: KKT satisfied (always true when IPOPT converged).
    return "W-stationary"
