"""Feasibility-Based Bound Tightening (FBBT) for the presolve pipeline.

Sweeps every linear ineq / eq row, tightening ``[xl, xu]`` from the
implied bounds until no bound moves more than :data:`_FBBT_TOL` or the
budget :data:`_FBBT_BUDGET` is exhausted.  Returns a fresh
``(xl, xu)`` pair plus an ``infeasible`` flag.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ._constants import FBBT_TOL as _FBBT_TOL
from ._presolve_detect import _identify_linear_rows
from .problem import MPCCProblem

#: Hard cap on outer FBBT sweeps.
_FBBT_BUDGET = 50


def _fbbt(
    p: MPCCProblem, *, rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Tighten ``xl`` / ``xu`` from linear ineq/eq rows.

    Returns ``(xl_new, xu_new, infeasible)``.  When no linear rows are
    detected (or sparsity is unavailable), returns the original bounds
    untouched.

    The algorithm sweeps every linear row; for each variable in the
    row, it computes the tightest implied bound from the other
    variables' current bounds and updates if strictly tighter (by
    ``_FBBT_TOL``) and consistent with ``x0``.  Sweeps repeat until no
    bound moves more than ``_FBBT_TOL`` or the budget is exhausted.
    """
    if rng is None:
        rng = np.random.default_rng(0xFB87)
    xl = np.asarray(p.xl, dtype=float).copy()
    xu = np.asarray(p.xu, dtype=float).copy()
    x0 = np.asarray(p.x0, dtype=float)

    lin_g, A_g, c_g = _identify_linear_rows(
        p.ineq_constraints, p.ineq_jacobian, p.ineq_jacobian_sparsity,
        p.n_ineq, p.n, x0, xl, xu, rng=rng,
    )
    lin_h, A_h, c_h = _identify_linear_rows(
        p.eq_constraints, p.eq_jacobian, p.eq_jacobian_sparsity,
        p.n_eq, p.n, x0, xl, xu, rng=rng,
    )

    if not lin_g.any() and not lin_h.any():
        return xl, xu, False

    # Pre-collect the rows we'll iterate over (kind, A_dict, c_value).
    # kind: 0 = ineq (Σ aⱼ xⱼ + c ≤ 0); 1 = eq (Σ aⱼ xⱼ + c = 0).
    rows_iter: list[tuple[int, dict[int, float], float]] = []
    for i in np.where(lin_g)[0]:
        if A_g[i]:
            rows_iter.append((0, A_g[i], c_g[i]))
    for i in np.where(lin_h)[0]:
        if A_h[i]:
            rows_iter.append((1, A_h[i], c_h[i]))

    if not rows_iter:
        return xl, xu, False

    def _row_min_max(A: dict[int, float], skip: int) -> tuple[float, float]:
        """Compute (min, max) of Σ_{k≠skip} aₖ xₖ over current bounds."""
        rmin = 0.0
        rmax = 0.0
        for k, ak in A.items():
            if k == skip:
                continue
            if ak > 0:
                rmin = (-np.inf if not np.isfinite(xl[k]) else rmin + ak * xl[k]) \
                    if np.isfinite(rmin) else rmin
                rmax = (+np.inf if not np.isfinite(xu[k]) else rmax + ak * xu[k]) \
                    if np.isfinite(rmax) else rmax
            else:
                rmin = (-np.inf if not np.isfinite(xu[k]) else rmin + ak * xu[k]) \
                    if np.isfinite(rmin) else rmin
                rmax = (+np.inf if not np.isfinite(xl[k]) else rmax + ak * xl[k]) \
                    if np.isfinite(rmax) else rmax
        return rmin, rmax

    for _ in range(_FBBT_BUDGET):
        moved = False
        for kind, A, c in rows_iter:
            # Whole-row min/max for the infeasibility check.
            row_min, row_max = _row_min_max(A, skip=-1)
            row_min += c
            row_max += c
            if kind == 0 and np.isfinite(row_min) and row_min > _FBBT_TOL:
                return xl, xu, True
            if kind == 1:
                if (np.isfinite(row_min) and row_min > _FBBT_TOL) or \
                   (np.isfinite(row_max) and row_max < -_FBBT_TOL):
                    return xl, xu, True

            for j, aj in A.items():
                # Skip degenerate coefficients (e.g. duplicates that
                # cancelled to zero during _identify_linear_rows).
                if abs(aj) < 1e-15:
                    continue
                rest_min_no_c, rest_max_no_c = _row_min_max(A, skip=j)
                rest_min = rest_min_no_c + c
                rest_max = rest_max_no_c + c

                # Inequality: aⱼ xⱼ + rest_min ≤ 0  ⇒  aⱼ xⱼ ≤ −rest_min
                # Equality:   aⱼ xⱼ ≥ −rest_max  AND  aⱼ xⱼ ≤ −rest_min
                if kind == 0:
                    if np.isfinite(rest_min):
                        ub_aj = -rest_min
                        if aj > 0:
                            new_xu = ub_aj / aj
                            if new_xu < xu[j] - _FBBT_TOL and new_xu >= x0[j] - _FBBT_TOL:
                                xu[j] = new_xu
                                moved = True
                        else:
                            new_xl = ub_aj / aj
                            if new_xl > xl[j] + _FBBT_TOL and new_xl <= x0[j] + _FBBT_TOL:
                                xl[j] = new_xl
                                moved = True
                else:  # equality
                    if np.isfinite(rest_min):
                        ub_aj = -rest_min
                        if aj > 0:
                            new_xu = ub_aj / aj
                            if new_xu < xu[j] - _FBBT_TOL and new_xu >= x0[j] - _FBBT_TOL:
                                xu[j] = new_xu
                                moved = True
                        else:
                            new_xl = ub_aj / aj
                            if new_xl > xl[j] + _FBBT_TOL and new_xl <= x0[j] + _FBBT_TOL:
                                xl[j] = new_xl
                                moved = True
                    if np.isfinite(rest_max):
                        lb_aj = -rest_max
                        if aj > 0:
                            new_xl = lb_aj / aj
                            if new_xl > xl[j] + _FBBT_TOL and new_xl <= x0[j] + _FBBT_TOL:
                                xl[j] = new_xl
                                moved = True
                        else:
                            new_xu = lb_aj / aj
                            if new_xu < xu[j] - _FBBT_TOL and new_xu >= x0[j] - _FBBT_TOL:
                                xu[j] = new_xu
                                moved = True

                # Bound crossing → infeasibility (keep ε so we don't trip on roundoff).
                if xl[j] > xu[j] + _FBBT_TOL:
                    return xl, xu, True

        if not moved:
            break

    return xl, xu, False
