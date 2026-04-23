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

Practical limitation: interior-point bias toward S-stationarity
---------------------------------------------------------------
IPOPT uses a primal-dual interior-point method where the barrier parameter
drives *all* multipliers for active lower-bound constraints (G ≥ 0, H ≥ 0)
toward non-negative values at convergence.  Concretely, for a biactive pair
(G_i ≈ 0, H_i ≈ 0), both μ_G_i = −λ_G_i ≥ 0 and μ_H_i = −λ_H_i ≥ 0 hold
by NLP complementarity, regardless of the true MPCC stationarity type.
As a result, ``classify_stationarity`` will almost always report
``"S-stationary"`` for fully converged IPOPT solutions.

The classification is most informative when:

* Comparing partially converged iterates (e.g., intermediate ``ε`` values in
  Scholtes / smoothing), where the barrier has not fully enforced positivity.
* Checking the sign of multipliers at a *prescribed* candidate solution (not
  one produced by IPOPT).
* Using a large IPOPT tolerance (``tol ≥ 1e-4``) that leaves multipliers
  further from their limit, making sign variations observable.

For a definitive stationarity check, use ``compute_kkt_residual`` to verify
that the optimality residual is small, and inspect ``result.comp_residual``
to confirm complementarity feasibility.  The stationarity label supplements
these measures; it does not replace them.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .problem import MPCCProblem
from .result import MPCCResult

__all__ = ["classify_stationarity", "compute_kkt_residual"]


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


def _dense_jac(
    values: np.ndarray,
    sparsity: Optional[tuple],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """Return a dense ``(n_rows, n_cols)`` Jacobian from sparse or dense input."""
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 2:
        return vals
    if sparsity is not None:
        dense = np.zeros((n_rows, n_cols))
        dense[np.asarray(sparsity[0]), np.asarray(sparsity[1])] = vals
        return dense
    return vals.reshape(n_rows, n_cols)


def _jac_T_vec(
    values: np.ndarray,
    sparsity: Optional[tuple],
    n_rows: int,
    n_cols: int,
    vec: np.ndarray,
) -> np.ndarray:
    """Compute ``J.T @ vec`` without materialising the dense J.

    For sparse COO Jacobians ``J.T @ vec`` via ``_dense_jac`` requires an
    ``(n_rows × n_cols)`` dense allocation.  At N=128 with n_rows=81920 and
    n_cols=99328 that is ~65 GB.  Instead, scatter-add directly::

        out[cols[k]] += vals[k] * vec[rows[k]]   for each nonzero k

    which is O(nnz) in time and O(n_cols) in memory.
    """
    vals = np.asarray(values, dtype=float)
    vec  = np.asarray(vec, dtype=float)
    if vals.ndim == 2:
        return vals.T @ vec
    if sparsity is None:
        return vals.reshape(n_rows, n_cols).T @ vec
    rows = np.asarray(sparsity[0])
    cols = np.asarray(sparsity[1])
    return np.bincount(cols, weights=vals * vec[rows], minlength=n_cols)


def compute_kkt_residual(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    mpcc_mult_G: Optional[np.ndarray] = None,
    mpcc_mult_H: Optional[np.ndarray] = None,
    mult_x_L: Optional[np.ndarray] = None,
    mult_x_U: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    MPCC stationarity residual:
    ``||∇f + Jg^T λ_g + Jh^T λ_h + JG^T μ_G + JH^T μ_H - z_L + z_U||_∞``.

    Parameters
    ----------
    result : MPCCResult
        A solved MPCC result.
    problem : MPCCProblem
        The problem instance used to produce the result.
    mpcc_mult_G : ndarray, shape (n_comp,)
        MPCC multipliers for the ``G(x) ≥ 0`` constraints.  For Scholtes /
        direct these are ``λ_G + H ⊙ λ_GH``; for lin-fukushima additionally
        ``+ λ_GPH``; for smoothing ``+ (1 − G/r) ⊙ λ_φ``; for augmented
        Lagrangian ``λ_G + μ_AL ⊙ H``; for slack ``λ_{G−s_G}`` directly.
    mpcc_mult_H : ndarray, shape (n_comp,)
        MPCC multipliers for the ``H(x) ≥ 0`` constraints (analogous).
    mult_x_L : ndarray, shape ≥ n, optional
        Variable lower-bound multipliers ``z_L ≥ 0`` from IPOPT.
        Only the first ``n`` entries are used.  ``None`` is treated as zero.
    mult_x_U : ndarray, shape ≥ n, optional
        Variable upper-bound multipliers ``z_U ≥ 0`` from IPOPT.
        Only the first ``n`` entries are used.  ``None`` is treated as zero.

    Returns
    -------
    float or None
        Infinity-norm of the stationarity residual, or ``None`` if
        ``mpcc_mult_G`` / ``mpcc_mult_H`` were not provided.
    """
    if result.mult_g is None or mpcc_mult_G is None or mpcc_mult_H is None:
        return None

    p = problem
    x = result.x
    n_g, n_h = p.n_ineq, p.n_eq

    r = np.asarray(p.gradient(x), dtype=float).copy()
    if mult_x_L is not None:
        r -= mult_x_L[:p.n]
    if mult_x_U is not None:
        r += mult_x_U[:p.n]

    if n_g:
        r += _jac_T_vec(p.ineq_jacobian(x), p.ineq_jacobian_sparsity, n_g, p.n, result.mult_g[:n_g])
    if n_h:
        r += _jac_T_vec(p.eq_jacobian(x), p.eq_jacobian_sparsity, n_h, p.n, result.mult_g[n_g : n_g + n_h])

    r += _jac_T_vec(p.comp_G_jacobian(x), p.comp_G_jacobian_sparsity, p.n_comp, p.n, mpcc_mult_G)
    r += _jac_T_vec(p.comp_H_jacobian(x), p.comp_H_jacobian_sparsity, p.n_comp, p.n, mpcc_mult_H)

    return float(np.max(np.abs(r)))
