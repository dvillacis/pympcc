"""MPCC constraint-qualification diagnostics.

Operates on a solved :class:`MPCCResult` plus its source
:class:`MPCCProblem`.  Two tests are exposed:

* :func:`classify_cq` — strongest CQ that holds at ``result.x`` from
  the chain ``MPCC-LICQ ⇒ MPCC-MFCQ``.
* :func:`active_sets` — partition of the constraint indices into the
  active subsets used by both CQ and B-stationarity routines.

The implementation mirrors :mod:`pympcc._stationarity`: dense
Jacobians via :func:`_stationarity._dense_jac`, dense LP via
:func:`scipy.optimize.linprog`.  Intended as a diagnostic for
small-to-medium MPCCs; the CQ rank test is ``O((n_active)^2 · n)``
through SVD.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._stationarity import _dense_jac
from .problem import MPCCProblem
from .result import MPCCResult

__all__ = ["classify_cq", "active_sets"]


def active_sets(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 1e-6,
) -> dict:
    """Partition constraint indices into active subsets at ``result.x``.

    Returns a dict with the integer index arrays:

    * ``I_g``    — active inequality rows (``g_j(x*) ≥ -tol``)
    * ``I_G``    — comp pairs with ``G_i(x*) ≤ tol``
    * ``I_H``    — comp pairs with ``H_i(x*) ≤ tol``
    * ``I_00``   — biactive pairs ``I_G ∩ I_H``
    * ``I_xL``   — variables at active lower bound (``x_j ≈ xl_j``)
    * ``I_xU``   — variables at active upper bound (``x_j ≈ xu_j``)
    """
    p = problem
    x = np.asarray(result.x, dtype=float)
    G = np.asarray(result.G, dtype=float)
    H = np.asarray(result.H, dtype=float)

    if p.n_ineq and p.ineq_constraints is not None:
        gvals = np.asarray(p.ineq_constraints(x), dtype=float)
        I_g = np.where(gvals >= -tol)[0].astype(np.intp)
    else:
        I_g = np.empty(0, dtype=np.intp)

    I_G = np.where(G <= tol)[0].astype(np.intp)
    I_H = np.where(H <= tol)[0].astype(np.intp)
    I_00 = np.intersect1d(I_G, I_H, assume_unique=True).astype(np.intp)

    xl = np.asarray(p.xl, dtype=float) if p.xl is not None else np.full(p.n, -np.inf)
    xu = np.asarray(p.xu, dtype=float) if p.xu is not None else np.full(p.n,  np.inf)
    I_xL = np.where(np.isfinite(xl) & (np.abs(x - xl) <= tol))[0].astype(np.intp)
    I_xU = np.where(np.isfinite(xu) & (np.abs(xu - x) <= tol))[0].astype(np.intp)

    return {
        "I_g":  I_g,
        "I_G":  I_G,
        "I_H":  I_H,
        "I_00": I_00,
        "I_xL": I_xL,
        "I_xU": I_xU,
    }


def _stack_active_gradient_matrix(
    problem: MPCCProblem,
    x: np.ndarray,
    sets: dict,
) -> tuple[np.ndarray, dict]:
    """Build the full active-gradient matrix M used for the LICQ rank test.

    Rows in order:
        h (all)  |  G_{I_G}  |  H_{I_H}  |  g_{I_g}  |  e_{I_xL ∪ I_xU}

    Returns ``(M, row_offsets)`` where ``row_offsets`` is a dict mapping
    block name → ``(start, stop)`` for downstream slicing.
    """
    p = problem
    n = p.n
    blocks: list[np.ndarray] = []
    offsets: dict = {}
    cursor = 0

    def _add(name, mat):
        nonlocal cursor
        if mat.shape[0] == 0:
            offsets[name] = (cursor, cursor)
            return
        blocks.append(mat)
        offsets[name] = (cursor, cursor + mat.shape[0])
        cursor += mat.shape[0]

    if p.n_eq and p.eq_jacobian is not None:
        Jh = _dense_jac(p.eq_jacobian(x), p.eq_jacobian_sparsity, p.n_eq, n)  # type: ignore[operator]
        _add("h", Jh)
    else:
        _add("h", np.zeros((0, n)))

    JG = _dense_jac(p.comp_G_jacobian(x), p.comp_G_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
    JH = _dense_jac(p.comp_H_jacobian(x), p.comp_H_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
    _add("G", JG[sets["I_G"]])
    _add("H", JH[sets["I_H"]])

    if p.n_ineq and p.ineq_jacobian is not None and sets["I_g"].size:
        Jg = _dense_jac(p.ineq_jacobian(x), p.ineq_jacobian_sparsity, p.n_ineq, n)  # type: ignore[operator]
        _add("g", Jg[sets["I_g"]])
    else:
        _add("g", np.zeros((0, n)))

    bound_idx = np.union1d(sets["I_xL"], sets["I_xU"]).astype(np.intp)
    if bound_idx.size:
        E = np.zeros((bound_idx.size, n))
        E[np.arange(bound_idx.size), bound_idx] = 1.0
        _add("bnd", E)
    else:
        _add("bnd", np.zeros((0, n)))

    if blocks:
        M = np.vstack(blocks)
    else:
        M = np.zeros((0, n))
    return M, offsets


def _matrix_rank(M: np.ndarray, *, tol: Optional[float] = None) -> int:
    """Numerically robust rank via SVD with a relative tolerance."""
    if M.size == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0:
        return 0
    if tol is None:
        tol = max(M.shape) * np.finfo(float).eps * float(s[0])
    return int(np.sum(s > tol))


def _mfcq_lp_feasible(
    problem: MPCCProblem,
    x: np.ndarray,
    sets: dict,
    *,
    lp_tol: float = 1e-8,
) -> bool:
    """Return True iff a strict MPCC-MFCQ direction exists.

    Solves::

        max  t
        s.t. ∇h·d              = 0
             ∇G_i·d            = 0   (i ∈ I_G)
             ∇H_i·d            = 0   (i ∈ I_H)
             ∇g_j·d + t        ≤ 0   (j ∈ I_g)
             −d_j  + t         ≤ 0   (j ∈ I_xL)
              d_j  + t         ≤ 0   (j ∈ I_xU)
             t ≥ 0,    |d| ≤ 1.

    MPCC-MFCQ holds iff the optimum ``t*`` exceeds ``lp_tol``.
    Pinned variables (in ``I_xL ∩ I_xU``) force ``t = 0`` so MFCQ
    always fails when one is present — but the presolve pass
    eliminates them upstream.
    """
    from scipy.optimize import linprog

    p = problem
    n = p.n
    n_var = n + 1   # d (n) + t (1)
    c = np.zeros(n_var)
    c[-1] = -1.0   # maximise t  ⇔  minimise −t

    # Equality block.
    A_eq_blocks: list[np.ndarray] = []
    if p.n_eq and p.eq_jacobian is not None:
        Jh = _dense_jac(p.eq_jacobian(x), p.eq_jacobian_sparsity, p.n_eq, n)  # type: ignore[operator]
        A_eq_blocks.append(np.hstack([Jh, np.zeros((Jh.shape[0], 1))]))
    if sets["I_G"].size:
        JG = _dense_jac(p.comp_G_jacobian(x), p.comp_G_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
        A_eq_blocks.append(np.hstack([JG[sets["I_G"]],
                                      np.zeros((sets["I_G"].size, 1))]))
    if sets["I_H"].size:
        JH = _dense_jac(p.comp_H_jacobian(x), p.comp_H_jacobian_sparsity, p.n_comp, n)  # type: ignore[misc, operator]
        A_eq_blocks.append(np.hstack([JH[sets["I_H"]],
                                      np.zeros((sets["I_H"].size, 1))]))
    A_eq = np.vstack(A_eq_blocks) if A_eq_blocks else None
    b_eq = np.zeros(A_eq.shape[0]) if A_eq is not None else None

    # Inequality block.
    A_ub_blocks: list[np.ndarray] = []
    if sets["I_g"].size and p.n_ineq and p.ineq_jacobian is not None:
        Jg = _dense_jac(p.ineq_jacobian(x), p.ineq_jacobian_sparsity, p.n_ineq, n)  # type: ignore[operator]
        Jg_act = Jg[sets["I_g"]]
        A_ub_blocks.append(np.hstack([Jg_act,
                                      np.ones((Jg_act.shape[0], 1))]))
    if sets["I_xL"].size:
        rows = np.zeros((sets["I_xL"].size, n_var))
        rows[np.arange(sets["I_xL"].size), sets["I_xL"]] = -1.0
        rows[:, -1] = 1.0
        A_ub_blocks.append(rows)
    if sets["I_xU"].size:
        rows = np.zeros((sets["I_xU"].size, n_var))
        rows[np.arange(sets["I_xU"].size), sets["I_xU"]] = 1.0
        rows[:, -1] = 1.0
        A_ub_blocks.append(rows)
    A_ub = np.vstack(A_ub_blocks) if A_ub_blocks else None
    b_ub = np.zeros(A_ub.shape[0]) if A_ub is not None else None

    bounds = [(-1.0, 1.0)] * n + [(0.0, None)]
    lp = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method="highs")
    if not lp.success:
        return False
    t_star = float(-lp.fun)
    return t_star > lp_tol


def classify_cq(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 1e-6,
    lp_tol: float = 1e-8,
) -> dict:
    """Classify the strongest MPCC constraint qualification at ``result.x``.

    The chain tested (strongest first):

    * **MPCC-LICQ** — gradients of all active equality, comp G/H, active
      inequality, and active bound rows are linearly independent.
    * **MPCC-MFCQ** — equality-style block (h, G_{I_G}, H_{I_H}) has
      full row rank, and a strict descent direction exists for every
      active inequality and active bound (see :func:`_mfcq_lp_feasible`).

    Parameters
    ----------
    result, problem : MPCCResult, MPCCProblem
        Solved result and its source problem.
    tol : float
        Active-set tolerance (default ``1e-6``).
    lp_tol : float
        Strictness threshold for the MPCC-MFCQ LP (default ``1e-8``).

    Returns
    -------
    dict
        ``cq`` ∈ {``"MPCC-LICQ"``, ``"MPCC-MFCQ"``, ``"none"``,
        ``"unknown"``};
        ``active_set_sizes`` — counts per active subset
        (``{"g","h","G","H","biactive","xL","xU"}``);
        ``rank_deficit`` — ``n_active_rows − rank(M)`` (0 ⇒ LICQ);
        ``n_active_rows`` — total rows in the LICQ matrix.

    ``cq == "unknown"`` when ``result.success`` is ``False`` (no point
    classifying a non-converged iterate).
    """
    if not result.success:
        return {
            "cq": "unknown",
            "active_set_sizes": None,
            "rank_deficit": None,
            "n_active_rows": None,
        }

    sets = active_sets(result, problem, tol=tol)
    sizes = {
        "g":        int(sets["I_g"].size),
        "h":        int(problem.n_eq),
        "G":        int(sets["I_G"].size),
        "H":        int(sets["I_H"].size),
        "biactive": int(sets["I_00"].size),
        "xL":       int(sets["I_xL"].size),
        "xU":       int(sets["I_xU"].size),
    }

    x = np.asarray(result.x, dtype=float)
    M, offsets = _stack_active_gradient_matrix(problem, x, sets)
    n_rows = M.shape[0]
    rk = _matrix_rank(M)
    rank_deficit = n_rows - rk

    if n_rows == 0 or rank_deficit == 0:
        return {
            "cq": "MPCC-LICQ",
            "active_set_sizes": sizes,
            "rank_deficit": int(rank_deficit),
            "n_active_rows": int(n_rows),
        }

    # MPCC-MFCQ requires the equality-style block (h, G_{I_G}, H_{I_H})
    # to be linearly independent on its own.
    h_lo, h_hi = offsets["h"]
    G_lo, G_hi = offsets["G"]
    H_lo, H_hi = offsets["H"]
    eq_rows = list(range(h_lo, h_hi)) + list(range(G_lo, G_hi)) + list(range(H_lo, H_hi))
    M_eq = M[eq_rows] if eq_rows else np.zeros((0, problem.n))
    eq_rank = _matrix_rank(M_eq) if M_eq.shape[0] else 0
    if eq_rank < M_eq.shape[0]:
        return {
            "cq": "none",
            "active_set_sizes": sizes,
            "rank_deficit": int(rank_deficit),
            "n_active_rows": int(n_rows),
        }

    if _mfcq_lp_feasible(problem, x, sets, lp_tol=lp_tol):
        return {
            "cq": "MPCC-MFCQ",
            "active_set_sizes": sizes,
            "rank_deficit": int(rank_deficit),
            "n_active_rows": int(n_rows),
        }
    return {
        "cq": "none",
        "active_set_sizes": sizes,
        "rank_deficit": int(rank_deficit),
        "n_active_rows": int(n_rows),
    }
