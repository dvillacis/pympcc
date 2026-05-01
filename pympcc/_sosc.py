"""MPCC second-order sufficient conditions (SOSC) check.

At a non-biactive MPCC stationary point x* with multipliers (μ_G, μ_H),
MPCC-SOSC holds when the reduced Lagrangian Hessian is positive definite
on the critical cone — i.e. every d ≠ 0 that is first-order feasible
satisfies d^T ∇²L(x*) d > 0.

For implementation we use the reduced-Hessian approach:

1. Build the active constraint gradient matrix A (rows from ∇h, ∇G_{I_G},
   ∇H_{I_H}, ∇g_{I_g}, and unit bound rows).
2. Compute a null-space basis Z of A via SVD.
3. Form the reduced Hessian W = Z^T H Z where H = ∇²_xx L(x*, λ).
4. SOSC holds iff min eigenvalue of W > 0.

Biactive pairs (I_00 ≠ ∅) make the critical cone non-convex; the check
returns ``None`` in that case.  When ``result.success`` is ``False`` the
check is also skipped.

The Lagrangian Hessian H is obtained from:
  * ``problem.lagrangian_hessian`` (user-supplied), or
  * central finite differences of ∇_x L using the best available
    multipliers (TNLP-refined if present, else zero — conservative).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._diagnostics import _stack_active_gradient_matrix, active_sets
from ._stationarity import _dense_jac
from .problem import MPCCProblem
from .result import MPCCResult

_log = logging.getLogger(__name__)

__all__ = ["sosc_check"]

_SKIPPED_NOT_CONVERGED = "not_converged"
_SKIPPED_BIACTIVE = "biactive_pairs"
_SKIPPED_NO_HESSIAN = "no_hessian_callable_and_fd_failed"


def _null_space(A: np.ndarray, *, rank_tol: Optional[float] = None) -> np.ndarray:
    """Return an orthonormal basis for the null space of A (shape m×n).

    Returns a matrix of shape (n, null_dim).  When A is empty or has rank
    zero, the full identity is returned (whole R^n is the null space).
    """
    n = A.shape[1]
    if A.shape[0] == 0:
        return np.eye(n)
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    if rank_tol is None:
        rank_tol = max(A.shape) * np.finfo(float).eps * float(s[0]) if s.size else 0.0
    rank = int(np.sum(s > rank_tol))
    return Vt[rank:].T   # shape (n, n - rank)


def _fd_lagrangian_hessian(
    x: np.ndarray,
    problem: MPCCProblem,
    lam_g: np.ndarray,
    lam_h: np.ndarray,
    lam_G: np.ndarray,
    lam_H: np.ndarray,
    fd_h: float,
) -> np.ndarray:
    """Finite-difference approximation of ∇²_xx L using central differences.

    The Lagrangian gradient at a point y is:
        ∇_y L = ∇f(y) + J_g(y)^T λ_g + J_h(y)^T λ_h + J_G(y)^T μ_G + J_H(y)^T μ_H

    Each column j of the Hessian is approximated by:
        H[:, j] ≈ (∇L(y + h e_j) − ∇L(y − h e_j)) / (2h)

    Cost: 2n gradient/Jacobian evaluations.
    """
    p = problem
    n = p.n

    def grad_L(y: np.ndarray) -> np.ndarray:
        g = np.asarray(p.gradient(y), dtype=float).copy()  # type: ignore[misc, operator]
        if p.n_ineq > 0 and p.ineq_jacobian is not None and lam_g.size:
            Jg = _dense_jac(p.ineq_jacobian(y), p.ineq_jacobian_sparsity, p.n_ineq, n)  # type: ignore[operator]
            g += Jg.T @ lam_g
        if p.n_eq > 0 and p.eq_jacobian is not None and lam_h.size:
            Jh = _dense_jac(p.eq_jacobian(y), p.eq_jacobian_sparsity, p.n_eq, n)  # type: ignore[operator]
            g += Jh.T @ lam_h
        JG = _dense_jac(p.comp_G_jacobian(y), p.comp_G_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
        JH = _dense_jac(p.comp_H_jacobian(y), p.comp_H_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
        g += JG.T @ lam_G + JH.T @ lam_H
        return g

    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = fd_h
        H[:, j] = (grad_L(x + e) - grad_L(x - e)) / (2.0 * fd_h)

    return 0.5 * (H + H.T)


def _build_hessian(
    x: np.ndarray,
    problem: MPCCProblem,
    lam_g: np.ndarray,
    lam_h: np.ndarray,
    lam_G: np.ndarray,
    lam_H: np.ndarray,
    fd_h: float,
) -> np.ndarray:
    """Construct the full n×n Lagrangian Hessian matrix."""
    p = problem
    n = p.n

    if p.lagrangian_hessian is not None:
        # User-supplied Hessian in IPOPT convention (lower-triangle COO or dense).
        lagrange = np.concatenate([lam_g, lam_h, lam_G, lam_H])
        raw = np.asarray(p.lagrangian_hessian(x, lagrange, 1.0), dtype=float)

        if p.lagrangian_hessian_sparsity is not None:
            rows = np.asarray(p.lagrangian_hessian_sparsity[0], dtype=np.intp)
            cols = np.asarray(p.lagrangian_hessian_sparsity[1], dtype=np.intp)
            H = np.zeros((n, n))
            H[rows, cols] = raw
            # Lower-triangle COO → symmetrise
            upper = rows != cols
            H[cols[upper], rows[upper]] = raw[upper]
        else:
            H = raw.reshape(n, n)
            H = 0.5 * (H + H.T)
        return H

    return _fd_lagrangian_hessian(x, problem, lam_g, lam_h, lam_G, lam_H, fd_h)


def sosc_check(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 0.0,
    fd_h: Optional[float] = None,
    active_tol: float = 1e-6,
) -> dict:
    """Check MPCC second-order sufficient conditions at ``result.x``.

    Parameters
    ----------
    result : MPCCResult
        Converged solve result.
    problem : MPCCProblem
        Source problem (original space, not presolve-reduced).
    tol : float
        Eigenvalue threshold: SOSC holds when min_eigenvalue > tol
        (default ``0.0`` — strict positive definiteness).
    fd_h : float or None
        Step size for FD Hessian approximation when
        ``problem.lagrangian_hessian`` is not provided.
        Defaults to ``problem.fd_h``.
    active_tol : float
        Constraint tolerance for active-set detection (default ``1e-6``).

    Returns
    -------
    dict
        ``sosc`` : bool or None
            True = SOSC holds, False = SOSC violated, None = check skipped.
        ``min_eigenvalue`` : float or None
            Minimum eigenvalue of the reduced Hessian W = Z^T H Z.
            ``None`` when the check was skipped.
        ``null_space_dim`` : int or None
            Dimension of the critical cone (columns in Z).
        ``n_active`` : int or None
            Number of active constraint rows in A.
        ``skipped_reason`` : str or None
            Explanation when ``sosc is None``.
    """
    _empty = {"sosc": None, "min_eigenvalue": None, "cond_W": None,
               "null_space_dim": None, "n_active": None, "skipped_reason": None}

    if not result.success:
        return {**_empty, "skipped_reason": _SKIPPED_NOT_CONVERGED}

    p = problem
    x = np.asarray(result.x, dtype=float)
    _fd_h = fd_h if fd_h is not None else p.fd_h

    sets = active_sets(result, p, tol=active_tol)

    if sets["I_00"].size > 0:
        return {**_empty, "skipped_reason": _SKIPPED_BIACTIVE}

    # ------------------------------------------------------------------ #
    # Active constraint gradient matrix and null space.                   #
    # ------------------------------------------------------------------ #
    A, _ = _stack_active_gradient_matrix(p, x, sets)
    n_active = int(A.shape[0])
    Z = _null_space(A)
    null_dim = int(Z.shape[1])

    # ------------------------------------------------------------------ #
    # Multipliers: TNLP-refined (best) → zeros (conservative).            #
    # ------------------------------------------------------------------ #
    n_g = p.n_ineq
    n_h = p.n_eq
    n_c = p.n_comp

    lam_g = np.zeros(n_g)
    lam_h = np.zeros(n_h)
    lam_G = np.zeros(n_c)
    lam_H = np.zeros(n_c)

    tnlp = getattr(result, "tnlp_refined", None)
    if tnlp is not None and getattr(tnlp, "success", False):
        lam_G = np.asarray(tnlp.mult_comp_G, dtype=float)
        lam_H = np.asarray(tnlp.mult_comp_H, dtype=float)
        if tnlp.mult_ineq is not None:
            lam_g = np.asarray(tnlp.mult_ineq, dtype=float)
        if tnlp.mult_eq is not None:
            lam_h = np.asarray(tnlp.mult_eq, dtype=float)

    # ------------------------------------------------------------------ #
    # Lagrangian Hessian.                                                  #
    # ------------------------------------------------------------------ #
    try:
        H_mat = _build_hessian(x, p, lam_g, lam_h, lam_G, lam_H, _fd_h)
    except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
        _log.debug("sosc: skipping (Hessian build failed), %s: %s",
                   type(exc).__name__, exc)
        return {**_empty, "skipped_reason": _SKIPPED_NO_HESSIAN}

    # ------------------------------------------------------------------ #
    # Reduced Hessian and SOSC test.                                       #
    # ------------------------------------------------------------------ #
    cond_W: Optional[float]
    if null_dim == 0:
        # Critical cone = {0}: SOSC trivially satisfied.
        min_eig = float("inf")
        cond_W = None
        sosc = True
    else:
        W = Z.T @ H_mat @ Z
        eigvals = np.linalg.eigvalsh(W)
        min_eig = float(eigvals.min())
        max_abs = float(np.abs(eigvals).max())
        # Condition of the reduced Hessian.  Defined only when min_eig > 0
        # (W is PD); for non-PD W we leave it None — the user already sees
        # SOSC=False, and a "negative" condition number would be misleading.
        cond_W = (max_abs / min_eig) if min_eig > 0.0 else float("inf")
        sosc = bool(min_eig > tol)

    return {
        "sosc": sosc,
        "min_eigenvalue": min_eig,
        "cond_W": cond_W,
        "null_space_dim": null_dim,
        "n_active": n_active,
        "skipped_reason": None,
    }
