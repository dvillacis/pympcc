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

__all__ = [
    "classify_cq",
    "active_sets",
    "jac_condition_number",
    "merit_cross_check",
    "jac_norms",
    "initial_point_statistics",
    "degeneracy_report",
]


_ZERO_NORM_TOL = 1e-10


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


def jac_condition_number(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 1e-6,
) -> Optional[float]:
    """Condition number κ₂(M_active) of the active-constraint Jacobian.

    Builds the same active-gradient stack used by :func:`classify_cq`
    (rows: equality, comp G/H on the active sides, active inequality,
    active variable bounds) and returns ``np.linalg.cond`` of that
    matrix.  Large values indicate near-rank-deficiency, which usually
    coincides with MPCC-LICQ failure or numerically ill-conditioned
    multipliers.

    Returns ``None`` when ``result.success`` is False or when the active
    matrix is empty.  An infinite condition number (rank deficient) is
    returned as ``float('inf')``.
    """
    if not result.success:
        return None
    sets = active_sets(result, problem, tol=tol)
    x = np.asarray(result.x, dtype=float)
    M, _ = _stack_active_gradient_matrix(problem, x, sets)
    if M.size == 0:
        return None
    return float(np.linalg.cond(M))


# --------------------------------------------------------------------------- #
# §2.7 — PATH-style multi-merit & degeneracy diagnostics                       #
# --------------------------------------------------------------------------- #

def _fischer_burmeister(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Pointwise Fischer-Burmeister residual ``a + b − sqrt(a² + b²)``."""
    return G + H - np.sqrt(G * G + H * H)


def merit_cross_check(
    result: MPCCResult,
    problem: MPCCProblem,
) -> dict:
    """Cross-check three independent MPCC merit functions at ``result.x``.

    Disagreement between merits localises numerical trouble: one merit
    can mask issues another exposes (PATH ships this cross-check at
    termination time).  All three should be comparable in magnitude on a
    healthy converged point.

    Returns a dict with:

    * ``fb_max`` / ``fb_mean`` — `|G + H − √(G² + H²)|` (Fischer-Burmeister).
    * ``min_map_max`` / ``min_map_mean`` — `|min(G, H)|` (min-map).
    * ``inner_product_max`` / ``inner_product_mean`` — `|G · H|` (matches
      ``result.comp_residual`` / ``result.comp_residual_mean``).
    * ``disagreement_ratio`` — `max / min` of the three max-norms after
      flooring the denominator at ``1e-16`` (ratio close to 1 ⇒ merits
      agree; large ratio ⇒ scaling mismatch or ill-conditioning).
    """
    G = np.asarray(result.G, dtype=float)
    H = np.asarray(result.H, dtype=float)
    if G.size == 0:
        return {
            "fb_max": 0.0, "fb_mean": 0.0,
            "min_map_max": 0.0, "min_map_mean": 0.0,
            "inner_product_max": 0.0, "inner_product_mean": 0.0,
            "disagreement_ratio": 1.0,
        }
    fb = np.abs(_fischer_burmeister(G, H))
    mm = np.abs(np.minimum(G, H))
    ip = np.abs(G * H)

    fb_max = float(np.max(fb))
    mm_max = float(np.max(mm))
    ip_max = float(np.max(ip))
    norms = np.array([fb_max, mm_max, ip_max])
    denom = max(float(np.min(norms)), 1e-16)
    return {
        "fb_max":              fb_max,
        "fb_mean":             float(np.mean(fb)),
        "min_map_max":         mm_max,
        "min_map_mean":        float(np.mean(mm)),
        "inner_product_max":   ip_max,
        "inner_product_mean":  float(np.mean(ip)),
        "disagreement_ratio":  float(np.max(norms) / denom),
    }


def jac_norms(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 1e-6,
    zero_tol: float = _ZERO_NORM_TOL,
) -> dict:
    """Row- and column-norm summary of the active-constraint Jacobian.

    Builds the same active-gradient stack used by :func:`classify_cq`
    (rows: equality, comp G/H on the active sides, active inequality,
    active variable bounds) and returns row/column norm extrema plus
    near-zero counts.

    Near-zero rows or columns are usually degeneracy signals (a row at
    norm ≈ 0 contributes no constraint information at the iterate; a
    column at norm ≈ 0 means a variable has no influence on any active
    constraint).

    Returns
    -------
    dict
        ``{
            "row": {"max": …, "min": …, "n_zero": …, "n_rows": …},
            "col": {"max": …, "min": …, "n_zero": …, "n_cols": …},
        }``

    Returns empty dicts under each key when the active matrix is empty
    or the result did not converge.
    """
    empty = {
        "row": {"max": None, "min": None, "n_zero": 0, "n_rows": 0},
        "col": {"max": None, "min": None, "n_zero": 0, "n_cols": 0},
    }
    if not result.success:
        return empty
    sets = active_sets(result, problem, tol=tol)
    x = np.asarray(result.x, dtype=float)
    M, _ = _stack_active_gradient_matrix(problem, x, sets)
    if M.size == 0:
        return empty
    row_n = np.linalg.norm(M, axis=1)
    col_n = np.linalg.norm(M, axis=0)
    return {
        "row": {
            "max":    float(np.max(row_n)),
            "min":    float(np.min(row_n)),
            "n_zero": int(np.sum(row_n <= zero_tol)),
            "n_rows": int(M.shape[0]),
        },
        "col": {
            "max":    float(np.max(col_n)),
            "min":    float(np.min(col_n)),
            "n_zero": int(np.sum(col_n <= zero_tol)),
            "n_cols": int(M.shape[1]),
        },
    }


def initial_point_statistics(problem: MPCCProblem) -> dict:
    """Replicate PATH's ``output_initial_point_statistics`` at ``x0``.

    Reports residuals at the user's starting point so the user sees the
    starting state before any work has been done (a complementarity-
    feasible ``x0`` often dramatically changes solver behaviour).

    Returns a dict with:

    * ``comp_residual`` — `max_i |G_i(x0) · H_i(x0)|`.
    * ``min_map_residual`` — `max_i |min(G_i(x0), H_i(x0))|`.
    * ``max_bound_violation`` — largest amount any ``x0[j]`` lies outside
      ``[xl[j], xu[j]]`` (0 if feasible).
    * ``ineq_residual`` — `max_j max(g_j(x0), 0)` (0 if feasible).
    * ``eq_residual`` — `max_k |h_k(x0)|`.
    """
    p = problem
    x0 = np.asarray(p.x0, dtype=float)
    out: dict = {}
    if p.comp_G is not None and p.comp_H is not None:
        G0 = np.asarray(p.comp_G(x0), dtype=float)
        H0 = np.asarray(p.comp_H(x0), dtype=float)
        out["comp_residual"]    = float(np.max(np.abs(G0 * H0))) if G0.size else 0.0
        out["min_map_residual"] = float(np.max(np.abs(np.minimum(G0, H0)))) if G0.size else 0.0
    else:
        out["comp_residual"] = 0.0
        out["min_map_residual"] = 0.0
    xl = np.asarray(p.xl, dtype=float) if p.xl is not None else np.full(p.n, -np.inf)
    xu = np.asarray(p.xu, dtype=float) if p.xu is not None else np.full(p.n,  np.inf)
    lo_v = np.where(np.isfinite(xl), np.maximum(xl - x0, 0.0), 0.0)
    hi_v = np.where(np.isfinite(xu), np.maximum(x0 - xu, 0.0), 0.0)
    out["max_bound_violation"] = float(np.max(np.concatenate([lo_v, hi_v])))
    if p.n_ineq and p.ineq_constraints is not None:
        g0 = np.asarray(p.ineq_constraints(x0), dtype=float)
        out["ineq_residual"] = float(np.max(np.maximum(g0, 0.0))) if g0.size else 0.0
    else:
        out["ineq_residual"] = 0.0
    if p.n_eq and p.eq_constraints is not None:
        h0 = np.asarray(p.eq_constraints(x0), dtype=float)
        out["eq_residual"] = float(np.max(np.abs(h0))) if h0.size else 0.0
    else:
        out["eq_residual"] = 0.0
    return out


def degeneracy_report(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    tol: float = 1e-6,
    zero_tol: float = _ZERO_NORM_TOL,
) -> dict:
    """Combined degeneracy summary at ``result.x``.

    Aggregates signals already produced by :func:`active_sets`,
    :func:`jac_norms`, and :func:`merit_cross_check` into a single
    structured dict for one-glance health assessment.

    Returns
    -------
    dict
        ``n_biactive`` — biactive-pair count |I_00|.
        ``n_zero_rows`` — rows of the active Jacobian at norm ≤ zero_tol.
        ``n_zero_cols`` — columns at norm ≤ zero_tol.
        ``min_singular_value`` — smallest singular value of the active
        Jacobian (None when active matrix is empty / not converged).
        ``merit_disagreement_ratio`` — see :func:`merit_cross_check`.
    """
    if not result.success:
        return {
            "n_biactive": None,
            "n_zero_rows": None,
            "n_zero_cols": None,
            "min_singular_value": None,
            "merit_disagreement_ratio": None,
        }
    sets = active_sets(result, problem, tol=tol)
    x = np.asarray(result.x, dtype=float)
    M, _ = _stack_active_gradient_matrix(problem, x, sets)

    if M.size:
        s = np.linalg.svd(M, compute_uv=False)
        min_sv = float(s[-1]) if s.size else None
        row_n = np.linalg.norm(M, axis=1)
        col_n = np.linalg.norm(M, axis=0)
        n_zero_rows = int(np.sum(row_n <= zero_tol))
        n_zero_cols = int(np.sum(col_n <= zero_tol))
    else:
        min_sv = None
        n_zero_rows = 0
        n_zero_cols = 0

    mc = merit_cross_check(result, problem)
    return {
        "n_biactive":               int(sets["I_00"].size),
        "n_zero_rows":              n_zero_rows,
        "n_zero_cols":              n_zero_cols,
        "min_singular_value":       min_sv,
        "merit_disagreement_ratio": mc["disagreement_ratio"],
    }
