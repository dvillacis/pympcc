"""Detection passes for the presolve pipeline.

Each function inspects the original :class:`MPCCProblem` and returns
indices / masks describing what can be eliminated.  No mutation; the
reduction itself happens in :mod:`pympcc._presolve_reduce`.

Public entry points re-exported from :mod:`pympcc._presolve` for
backward compatibility with existing tests:

* :func:`_detect_pinned`, :func:`_detect_pinned_from_bounds`
* :func:`_detect_empty_rows`, :func:`_detect_empty_cols`
* :func:`_detect_dead`, :func:`_detect_forced`
* :func:`_detect_prefix_eq`
* :func:`_identify_linear_rows`, :func:`_row_interval`
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ._constants import (
    DEAD_VAL_TOL as _DEAD_VAL_TOL,
)
from ._constants import (
    EMPTY_ROW_TOL as _EMPTY_ROW_TOL,
)
from ._constants import (
    FREE_VAR_GRAD_TOL as _FREE_GRAD_TOL,
)
from ._constants import (
    LINEARITY_TOL as _LINEARITY_TOL,
)
from .problem import MPCCProblem

_log = logging.getLogger(__name__)


def _detect_pinned(p: MPCCProblem) -> tuple[np.ndarray, np.ndarray]:
    return _detect_pinned_from_bounds(
        np.asarray(p.xl, dtype=float),
        np.asarray(p.xu, dtype=float),
    )


def _detect_pinned_from_bounds(
    xl: np.ndarray, xu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_eq = np.isfinite(xl) & np.isfinite(xu) & (xl == xu)
    pinned = np.where(finite_eq)[0]
    return pinned, xl[pinned].copy()


def _detect_empty_rows(p: MPCCProblem) -> tuple[np.ndarray, np.ndarray, bool]:
    """Find ineq/eq rows whose Jacobian sparsity is structurally empty.

    For each such row, evaluate the constraint at ``x0`` once: drop if
    feasible (``≤ tol`` for ineq, ``|·| ≤ tol`` for eq), flag infeasible
    otherwise.  Returns ``(drop_ineq_mask, drop_eq_mask, infeasible)``.

    Falls back to all-False masks when sparsity is missing (we deliberately
    do not flag rows constant just by sampling — fragile for non-trivial
    nonlinear callbacks that happen to vanish at ``x0``).
    """
    drop_ineq = np.zeros(p.n_ineq, dtype=bool) if p.n_ineq else np.empty(0, dtype=bool)
    drop_eq   = np.zeros(p.n_eq,   dtype=bool) if p.n_eq   else np.empty(0, dtype=bool)
    infeasible = False

    if (p.n_ineq and p.ineq_jacobian_sparsity is not None
            and p.ineq_constraints is not None):
        rows = np.asarray(p.ineq_jacobian_sparsity[0], dtype=np.intp)
        rows_with_entry = np.unique(rows) if rows.size else np.empty(0, dtype=np.intp)
        empty = np.setdiff1d(np.arange(p.n_ineq, dtype=np.intp),
                             rows_with_entry, assume_unique=False)
        if empty.size:
            g0 = np.asarray(p.ineq_constraints(p.x0), dtype=float)
            for i in empty:
                if g0[i] <= _EMPTY_ROW_TOL:
                    drop_ineq[i] = True
                else:
                    infeasible = True

    if (p.n_eq and p.eq_jacobian_sparsity is not None
            and p.eq_constraints is not None):
        rows = np.asarray(p.eq_jacobian_sparsity[0], dtype=np.intp)
        rows_with_entry = np.unique(rows) if rows.size else np.empty(0, dtype=np.intp)
        empty = np.setdiff1d(np.arange(p.n_eq, dtype=np.intp),
                             rows_with_entry, assume_unique=False)
        if empty.size:
            h0 = np.asarray(p.eq_constraints(p.x0), dtype=float)
            for i in empty:
                if abs(h0[i]) <= _EMPTY_ROW_TOL:
                    drop_eq[i] = True
                else:
                    infeasible = True

    return drop_ineq, drop_eq, infeasible


def _detect_empty_cols(
    p: MPCCProblem, xl: np.ndarray, xu: np.ndarray,
    *, rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Find columns absent from every Jacobian sparsity *and* zero in the
    objective gradient at two random probe points.

    Returns the array of free-variable indices.  Conservative: if any
    constraint exposes no sparsity, returns empty (we can't certify
    structural absence without a sparsity pattern).
    """
    n = p.n
    # Refuse to act when any present Jacobian is dense (no sparsity to inspect).
    if ((p.comp_G_jacobian is not None and p.comp_G_jacobian_sparsity is None) or
        (p.comp_H_jacobian is not None and p.comp_H_jacobian_sparsity is None) or
        (p.n_ineq and p.ineq_jacobian is not None and p.ineq_jacobian_sparsity is None) or
        (p.n_eq   and p.eq_jacobian   is not None and p.eq_jacobian_sparsity   is None)):
        return np.empty(0, dtype=np.intp)

    seen = set()
    for sp in (p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity,
               p.ineq_jacobian_sparsity, p.eq_jacobian_sparsity):
        if sp is None:
            continue
        cols = np.asarray(sp[1], dtype=np.intp)
        if cols.size:
            seen.update(cols.tolist())
    if len(seen) == n:
        return np.empty(0, dtype=np.intp)

    candidates = np.setdiff1d(
        np.arange(n, dtype=np.intp),
        np.array(sorted(seen), dtype=np.intp) if seen else np.empty(0, dtype=np.intp),
        assume_unique=False,
    )
    if candidates.size == 0:
        return candidates

    if rng is None:
        rng = np.random.default_rng(0xE6C0)
    x0 = np.asarray(p.x0, dtype=float)
    span = np.where(np.isfinite(xu - xl), xu - xl, 1.0)
    span = np.where(span > 0, span, 1.0)
    delta = 0.1 * span * (2.0 * rng.random(n) - 1.0)
    x1 = np.clip(x0 + delta, xl, xu)

    try:
        g0 = np.asarray(p.gradient(x0), dtype=float)  # type: ignore[misc, operator]
        g1 = np.asarray(p.gradient(x1), dtype=float)  # type: ignore[misc, operator]
    except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
        _log.debug("presolve: free-var probe failed, %s: %s",
                   type(exc).__name__, exc)
        return np.empty(0, dtype=np.intp)

    free_mask = ((np.abs(g0[candidates]) <= _FREE_GRAD_TOL)
                 & (np.abs(g1[candidates]) <= _FREE_GRAD_TOL))
    return candidates[free_mask].astype(np.intp)


def _detect_dead(p: MPCCProblem) -> np.ndarray:
    n_comp = p.n_comp
    dead = np.zeros(n_comp, dtype=bool)

    sG_sp = p.comp_G_jacobian_sparsity
    sH_sp = p.comp_H_jacobian_sparsity
    if sG_sp is None and sH_sp is None:
        return dead  # no sparsity → no structural-zero detection

    G_x0 = np.asarray(p.comp_G(p.x0), dtype=float)  # type: ignore[misc]
    H_x0 = np.asarray(p.comp_H(p.x0), dtype=float)  # type: ignore[misc]

    if sG_sp is not None:
        rows_with_entry = np.unique(np.asarray(sG_sp[0]))
        empty = np.setdiff1d(np.arange(n_comp), rows_with_entry,
                             assume_unique=False)
        for i in empty:
            if G_x0[i] > _DEAD_VAL_TOL:
                dead[i] = True

    if sH_sp is not None:
        rows_with_entry = np.unique(np.asarray(sH_sp[0]))
        empty = np.setdiff1d(np.arange(n_comp), rows_with_entry,
                             assume_unique=False)
        for i in empty:
            if H_x0[i] > _DEAD_VAL_TOL:
                dead[i] = True

    return dead


def _detect_forced(
    p: MPCCProblem, dead: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect comp pairs whose H_i (resp. G_i) is structurally identically zero.

    Returns ``(promote_G, promote_H, extra_dead)``:

    * ``promote_G[k]`` — pair where ``H_i`` has empty Jacobian sparsity row
      and ``|H_i(x0)| ≤ tol``.  ``G_i ≥ 0`` is promoted to a regular ineq.
    * ``promote_H[k]`` — pair where ``G_i`` has empty Jacobian sparsity row
      and ``|G_i(x0)| ≤ tol``.  ``H_i ≥ 0`` is promoted to a regular ineq.
    * ``extra_dead`` — pairs where *both* G_i and H_i are structurally zero
      at ``x0``; the entire pair is dead.

    Pairs already in ``dead`` (from B1) are skipped.  Restricted to the
    COO-sparsity case; falls back to empty arrays when sparsity is
    missing on either side.
    """
    empty_arr = np.empty(0, dtype=np.intp)
    sG_sp = p.comp_G_jacobian_sparsity
    sH_sp = p.comp_H_jacobian_sparsity
    if sG_sp is None or sH_sp is None:
        return empty_arr, empty_arr, empty_arr
    # Need a sparse ineq Jacobian to safely append promoted rows.
    if p.n_ineq and p.ineq_jacobian_sparsity is None:
        return empty_arr, empty_arr, empty_arr

    rows_g = np.asarray(sG_sp[0], dtype=np.intp)
    rows_h = np.asarray(sH_sp[0], dtype=np.intp)
    with_entry_g = np.unique(rows_g) if rows_g.size else empty_arr
    with_entry_h = np.unique(rows_h) if rows_h.size else empty_arr
    empty_G_set = set(int(i) for i in np.setdiff1d(np.arange(p.n_comp), with_entry_g))
    empty_H_set = set(int(i) for i in np.setdiff1d(np.arange(p.n_comp), with_entry_h))

    G_x0 = np.asarray(p.comp_G(p.x0), dtype=float)  # type: ignore[misc]
    H_x0 = np.asarray(p.comp_H(p.x0), dtype=float)  # type: ignore[misc]

    promote_G: list[int] = []
    promote_H: list[int] = []
    extra_dead: list[int] = []
    for i in range(p.n_comp):
        if dead[i]:
            continue
        gi_const = i in empty_G_set
        hi_const = i in empty_H_set
        gi_zero = abs(G_x0[i]) <= _DEAD_VAL_TOL
        hi_zero = abs(H_x0[i]) <= _DEAD_VAL_TOL
        if gi_const and hi_const and gi_zero and hi_zero:
            extra_dead.append(i)
        elif hi_const and hi_zero:
            promote_G.append(i)
        elif gi_const and gi_zero:
            promote_H.append(i)

    return (np.asarray(promote_G, dtype=np.intp),
            np.asarray(promote_H, dtype=np.intp),
            np.asarray(extra_dead, dtype=np.intp))


def _row_interval(
    A: dict[int, float], c: float,
    xl: np.ndarray, xu: np.ndarray,
) -> tuple[float, float]:
    """Compute (min, max) of ``c + Σ aⱼ xⱼ`` over ``[xl, xu]`` with
    inf-safe arithmetic.
    """
    rmin, rmax = c, c
    for k, ak in A.items():
        if ak == 0.0:
            continue
        if ak > 0:
            lo, hi = xl[k], xu[k]
        else:
            lo, hi = xu[k], xl[k]
        rmin = -np.inf if (np.isinf(rmin) and rmin < 0) or not np.isfinite(lo) \
            else rmin + ak * lo
        rmax = +np.inf if (np.isinf(rmax) and rmax > 0) or not np.isfinite(hi) \
            else rmax + ak * hi
    return rmin, rmax


def _detect_prefix_eq(
    p: MPCCProblem, xl: np.ndarray, xu: np.ndarray, skip: np.ndarray,
    *, rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """B3 — pre-fix complementarity pairs via linear sign analysis.

    For each pair ``i`` not yet handled by B1/B2:

    * If ``G_i`` is linear (per :func:`_identify_linear_rows`) and its
      interval over ``[xl, xu]`` lies strictly above zero, complementarity
      forces ``H_i = 0``.  Pair index goes to ``prefix_H_eq``.
    * Symmetric for ``H_i``.

    If the interval check fires on *both* sides of the same pair, the
    feasible set is empty (G·H = 0 cannot hold with both positive); the
    second return slot is ``infeasible = True``.
    """
    n_comp = p.n_comp
    empty = np.empty(0, dtype=np.intp)
    if n_comp == 0:
        return empty, empty, False
    sG_sp = p.comp_G_jacobian_sparsity
    sH_sp = p.comp_H_jacobian_sparsity
    if sG_sp is None and sH_sp is None:
        return empty, empty, False

    if rng is None:
        rng = np.random.default_rng(0xB30E)
    x0 = np.asarray(p.x0, dtype=float)

    lin_G, A_G, c_G = _identify_linear_rows(
        p.comp_G, p.comp_G_jacobian, sG_sp, n_comp, p.n,
        x0, xl, xu, rng=rng,
    )
    lin_H, A_H, c_H = _identify_linear_rows(
        p.comp_H, p.comp_H_jacobian, sH_sp, n_comp, p.n,
        x0, xl, xu, rng=rng,
    )

    prefix_H: list[int] = []
    prefix_G: list[int] = []
    for i in range(n_comp):
        if skip[i]:
            continue
        gpos = False
        hpos = False
        if lin_G[i]:
            gmin, _ = _row_interval(A_G[i], c_G[i], xl, xu)
            gpos = np.isfinite(gmin) and gmin > _DEAD_VAL_TOL
        if lin_H[i]:
            hmin, _ = _row_interval(A_H[i], c_H[i], xl, xu)
            hpos = np.isfinite(hmin) and hmin > _DEAD_VAL_TOL
        if gpos and hpos:
            return empty, empty, True
        if gpos:
            prefix_H.append(i)
        elif hpos:
            prefix_G.append(i)

    return (np.asarray(prefix_H, dtype=np.intp),
            np.asarray(prefix_G, dtype=np.intp), False)


# --------------------------------------------------------------------------- #
# Linear-row identification                                                    #
# --------------------------------------------------------------------------- #

def _identify_linear_rows(
    g_callable, jac_callable, sparsity, n_con: int, n: int,
    x0: np.ndarray, xl: np.ndarray, xu: np.ndarray,
    *, rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[int, float]], np.ndarray]:
    """Probe ``g`` for per-row linearity and return ``(linear_mask, A_rows, c)``.

    A row ``i`` is flagged linear when, for a single bounded random
    perturbation ``δ``,

        |g(x0 + δ)[i] − g(x0)[i] − (J(x0)·δ)[i]|  <  tol · (1 + |·|).

    Restricted to the COO-sparsity case; returns all-False when
    ``sparsity is None`` (we deliberately do not call dense Jacobians
    just for FBBT — the cost is rarely worth it).

    Parameters mirror ``MPCCProblem`` constraint accessors.  ``A_rows[i]``
    is ``{col: coefficient}`` (only populated when row ``i`` is linear).
    ``c[i] = g(x0)[i] − Σ_j A[i,j] · x0[j]``.
    """
    linear = np.zeros(n_con, dtype=bool)
    A_rows: list[dict[int, float]] = [dict() for _ in range(n_con)]
    c = np.zeros(n_con, dtype=float)
    if g_callable is None or jac_callable is None or sparsity is None:
        return linear, A_rows, c
    if n_con == 0:
        return linear, A_rows, c

    rows = np.asarray(sparsity[0], dtype=np.intp)
    cols = np.asarray(sparsity[1], dtype=np.intp)

    try:
        vals0 = np.asarray(jac_callable(x0), dtype=float)
        g0    = np.asarray(g_callable(x0),   dtype=float)
    except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
        _log.debug("presolve: linearity probe at x0 failed, %s: %s",
                   type(exc).__name__, exc)
        return linear, A_rows, c

    # Bounded perturbation: ±10% of finite range, else ±1e-2.
    span = np.where(np.isfinite(xu - xl), xu - xl, 1.0)
    span = np.where(span > 0, span, 1.0)
    delta = 0.1 * span * (2.0 * rng.random(n) - 1.0)
    x1 = np.clip(x0 + delta, xl, xu)
    delta_eff = x1 - x0  # actual step after clipping

    try:
        g1 = np.asarray(g_callable(x1), dtype=float)
    except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
        _log.debug("presolve: linearity probe at x1 failed, %s: %s",
                   type(exc).__name__, exc)
        return linear, A_rows, c

    # Predicted change per row using J(x0): Σ_j A[i,j] · δ_j.
    pred = np.zeros(n_con, dtype=float)
    np.add.at(pred, rows, vals0 * delta_eff[cols])
    actual = g1 - g0
    scale = 1.0 + np.maximum(np.abs(g0), np.abs(g1))
    err = np.abs(actual - pred)
    linear = err < (_LINEARITY_TOL * scale)

    # For linear rows, build A and c from x0-evaluation.
    for i in np.where(linear)[0]:
        mask_i = rows == i
        for col, val in zip(cols[mask_i], vals0[mask_i]):
            A_rows[i][int(col)] = A_rows[i].get(int(col), 0.0) + float(val)
        affine = sum(coef * x0[j] for j, coef in A_rows[i].items())
        c[i] = g0[i] - affine

    return linear, A_rows, c
