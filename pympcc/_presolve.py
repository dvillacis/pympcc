"""Presolve passes for :class:`~pympcc.MPCCProblem`.

Two passes are applied (both opt-in via ``MPCCSolver(..., presolve=True)``):

1. **Pinned-variable elimination** — variables with ``xl[j] == xu[j]``
   (finite, equal) are substituted out of the optimisation. The reduced
   problem has fewer variables; callbacks are wrapped to reinject the
   fixed values before evaluation.
2. **Dead-pair pruning** — complementarity pairs whose Jacobian row is
   structurally empty (no x-dependence) and whose value at ``x0`` is
   strictly positive are dropped. The constraint ``G_i ≥ 0`` (resp.
   ``H_i ≥ 0``) is trivially satisfied by a positive constant.
   Restricted to the COO-sparsity case; falls back to a no-op when
   sparsity is not provided.

The mapping ``PresolveMap`` is used by the solver to expand the result
back to the original variable / comp-pair space before returning.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .problem import MPCCProblem
from .result import IterationInfo, MPCCResult

__all__ = ["PresolveMap", "presolve"]


_DEAD_VAL_TOL = 1e-12
_EMPTY_ROW_TOL = 1e-9    # constant rows must satisfy |g(x0)| ≤ this
_FREE_GRAD_TOL = 1e-10   # |∂f/∂x_j| at probe points must be ≤ this to prune


@dataclass
class PresolveMap:
    """Mapping between an original :class:`MPCCProblem` and its reduced form.

    Use :meth:`expand_result` to lift a result on the reduced problem
    back to the original variable / comp-pair indexing.
    """

    keep_vars:   np.ndarray             # original variable indices kept
    fixed_vars:  np.ndarray             # original variable indices pinned
    fixed_vals:  np.ndarray             # value each pinned variable was fixed to
    keep_comp:   np.ndarray             # original comp-pair indices kept
    n_orig:      int                    # original number of variables
    n_comp_orig: int                    # original number of comp pairs
    n_ineq:      int = 0                # original number of inequality constraints
    n_eq:        int = 0                # original number of equality constraints
    keep_ineq:   Optional[np.ndarray] = None  # original ineq rows kept (None ⇒ all)
    keep_eq:     Optional[np.ndarray] = None  # original eq rows kept   (None ⇒ all)
    # B2 — forced-pair pruning: pairs whose H_i ≡ 0 (resp. G_i ≡ 0)
    # structurally were dropped from the comp block; their G_i (resp. H_i)
    # was promoted to a regular inequality.  Multipliers on those promoted
    # rows are dropped on expand_result (info loss; documented).
    promote_G:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    promote_H:   np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    # B3 — pre-fixing on linear sign analysis: pairs where one side is
    # linear and provably > 0 over [xl, xu] are dropped; the other side
    # is enforced as an equality (H_i = 0 if G_i > 0; G_i = 0 if H_i > 0).
    # Multipliers on those promoted eq rows are dropped on expand_result.
    prefix_H_eq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    prefix_G_eq: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    @property
    def is_identity(self) -> bool:
        n_ineq_kept = self.n_ineq if self.keep_ineq is None else self.keep_ineq.size
        n_eq_kept   = self.n_eq   if self.keep_eq   is None else self.keep_eq.size
        return (self.fixed_vars.size == 0
                and self.keep_comp.size == self.n_comp_orig
                and n_ineq_kept == self.n_ineq
                and n_eq_kept == self.n_eq
                and self.promote_G.size == 0
                and self.promote_H.size == 0
                and self.prefix_H_eq.size == 0
                and self.prefix_G_eq.size == 0)

    def expand_x(self, x_red: np.ndarray) -> np.ndarray:
        """Scatter a reduced-space ``x`` back to the original ``n_orig`` slots."""
        x_red = np.asarray(x_red, dtype=float)
        if self.fixed_vars.size == 0:
            return x_red.copy()
        x_full = np.empty(self.n_orig, dtype=float)
        x_full[self.keep_vars]  = x_red
        x_full[self.fixed_vars] = self.fixed_vals
        return x_full

    def expand_comp(self, vec_red: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
        """Scatter a length-``len(keep_comp)`` vector back to ``n_comp_orig``."""
        vec_red = np.asarray(vec_red, dtype=float)
        if self.keep_comp.size == self.n_comp_orig:
            return vec_red.copy()
        out = np.full(self.n_comp_orig, fill, dtype=float)
        out[self.keep_comp] = vec_red
        return out

    def expand_result(
        self,
        result: MPCCResult,
        problem_orig: MPCCProblem,
    ) -> MPCCResult:
        """Expand ``result`` (built on the reduced problem) into original space.

        Mutates ``result`` in place and also returns it.
        """
        if self.is_identity:
            return result

        # Primal solution.
        x_full = self.expand_x(result.x)

        # G, H: re-evaluate on the original problem so values reflect the
        # original constraint, including pruned (constant-positive) pairs.
        G_full = np.asarray(problem_orig.comp_G(x_full), dtype=float)
        H_full = np.asarray(problem_orig.comp_H(x_full), dtype=float)

        # mult_g: layout is [ineq | eq | <comp blocks>], comp blocks each
        # of length ``n_comp_red``.  Rebuild with zeros at pruned ineq/eq
        # rows and dropped comp pairs.  The reduced ineq block is
        # ``[orig_kept | promoted_G | promoted_H]``; only the orig-kept
        # slice maps back — promoted multipliers are dropped (info loss).
        mult_g = result.mult_g
        if mult_g is not None:
            n_ineq_kept = self.n_ineq if self.keep_ineq is None else self.keep_ineq.size
            n_promote   = self.promote_G.size + self.promote_H.size
            n_ineq_red  = n_ineq_kept + n_promote
            n_eq_kept   = self.n_eq   if self.keep_eq   is None else self.keep_eq.size
            n_prefix_eq = self.prefix_H_eq.size + self.prefix_G_eq.size
            n_eq_red    = n_eq_kept + n_prefix_eq
            n_comp_red  = self.keep_comp.size
            n_pre_red   = n_ineq_red + n_eq_red

            ineq_red = np.asarray(mult_g[:n_ineq_kept])  # promoted slice ignored
            eq_red   = np.asarray(mult_g[n_ineq_red:n_ineq_red + n_eq_kept])  # prefix slice ignored
            tail     = np.asarray(mult_g[n_pre_red:])

            ineq_full = np.zeros(self.n_ineq)
            if self.keep_ineq is None:
                ineq_full = ineq_red
            else:
                ineq_full[self.keep_ineq] = ineq_red

            eq_full = np.zeros(self.n_eq)
            if self.keep_eq is None:
                eq_full = eq_red
            else:
                eq_full[self.keep_eq] = eq_red

            tail_full = tail
            if (n_comp_red > 0 and tail.size % n_comp_red == 0
                    and self.keep_comp.size != self.n_comp_orig):
                K = tail.size // n_comp_red
                blocks_red = tail.reshape(K, n_comp_red)
                blocks_full = np.zeros((K, self.n_comp_orig))
                blocks_full[:, self.keep_comp] = blocks_red
                tail_full = blocks_full.reshape(-1)

            if (n_ineq_red != self.n_ineq
                    or n_eq_red != self.n_eq
                    or self.keep_comp.size != self.n_comp_orig):
                mult_g = np.concatenate([ineq_full, eq_full, tail_full])

        # comp_*_scale: pad pruned slots with 1.0 (no rescaling on those pairs).
        cG_scale = result.comp_G_scale
        cH_scale = result.comp_H_scale
        if cG_scale is not None and len(cG_scale) != self.n_comp_orig:
            cG_scale = self.expand_comp(cG_scale, fill=1.0)
        if cH_scale is not None and len(cH_scale) != self.n_comp_orig:
            cH_scale = self.expand_comp(cH_scale, fill=1.0)

        # history: scatter each iter's primal back to n_orig.
        new_history: list[IterationInfo] = []
        for info in result.history:
            new_history.append(IterationInfo(
                epsilon=info.epsilon,
                x=self.expand_x(info.x),
                obj=info.obj,
                status=info.status,
                message=info.message,
                comp_residual=info.comp_residual,
                comp_residual_mean=info.comp_residual_mean,
                n_ipopt_iter=info.n_ipopt_iter,
                iter_time=info.iter_time,
                kkt_residual=info.kkt_residual,
                restoration_iter_count=info.restoration_iter_count,
                entered_restoration=info.entered_restoration,
            ))

        result.x = x_full
        result.G = G_full
        result.H = H_full
        result.mult_g = mult_g
        result.comp_G_scale = cG_scale
        result.comp_H_scale = cH_scale
        result.history = new_history
        return result


# --------------------------------------------------------------------------- #
# Detection                                                                    #
# --------------------------------------------------------------------------- #

_FBBT_TOL  = 1e-9       # bounds must improve by more than this to count
_FBBT_BUDGET = 50       # hard cap on outer FBBT sweeps


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
        g0 = np.asarray(p.gradient(x0), dtype=float)
        g1 = np.asarray(p.gradient(x1), dtype=float)
    except Exception:
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

    G_x0 = np.asarray(p.comp_G(p.x0), dtype=float)
    H_x0 = np.asarray(p.comp_H(p.x0), dtype=float)

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

    G_x0 = np.asarray(p.comp_G(p.x0), dtype=float)
    H_x0 = np.asarray(p.comp_H(p.x0), dtype=float)

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
    except Exception:
        return linear, A_rows, c

    # Bounded perturbation: ±10% of finite range, else ±1e-2.
    span = np.where(np.isfinite(xu - xl), xu - xl, 1.0)
    span = np.where(span > 0, span, 1.0)
    delta = 0.1 * span * (2.0 * rng.random(n) - 1.0)
    x1 = np.clip(x0 + delta, xl, xu)
    delta_eff = x1 - x0  # actual step after clipping

    try:
        g1 = np.asarray(g_callable(x1), dtype=float)
    except Exception:
        return linear, A_rows, c

    # Predicted change per row using J(x0): Σ_j A[i,j] · δ_j.
    pred = np.zeros(n_con, dtype=float)
    np.add.at(pred, rows, vals0 * delta_eff[cols])
    actual = g1 - g0
    scale = 1.0 + np.maximum(np.abs(g0), np.abs(g1))
    err = np.abs(actual - pred)
    linear = err < (1e-9 * scale)

    # For linear rows, build A and c from x0-evaluation.
    for i in np.where(linear)[0]:
        mask_i = rows == i
        for col, val in zip(cols[mask_i], vals0[mask_i]):
            A_rows[i][int(col)] = A_rows[i].get(int(col), 0.0) + float(val)
        affine = sum(coef * x0[j] for j, coef in A_rows[i].items())
        c[i] = g0[i] - affine

    return linear, A_rows, c


# --------------------------------------------------------------------------- #
# FBBT (Feasibility-Based Bound Tightening)                                    #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Reduction                                                                    #
# --------------------------------------------------------------------------- #

def _build_reduced(
    p: MPCCProblem,
    pmap: PresolveMap,
    *,
    xl_full: Optional[np.ndarray] = None,
    xu_full: Optional[np.ndarray] = None,
    drop_ineq: Optional[np.ndarray] = None,
    drop_eq:   Optional[np.ndarray] = None,
) -> MPCCProblem:
    keep   = pmap.keep_vars
    fixed  = pmap.fixed_vars
    fvals  = pmap.fixed_vals
    kcomp  = pmap.keep_comp
    n_red  = keep.size
    n_orig = p.n
    n_comp_orig = p.n_comp
    xl_eff = np.asarray(p.xl if xl_full is None else xl_full, dtype=float)
    xu_eff = np.asarray(p.xu if xu_full is None else xu_full, dtype=float)

    col_remap = -np.ones(n_orig, dtype=np.intp)
    col_remap[keep] = np.arange(n_red)
    comp_remap = -np.ones(n_comp_orig, dtype=np.intp)
    comp_remap[kcomp] = np.arange(kcomp.size)

    def lift_x(x_red: np.ndarray) -> np.ndarray:
        x_full = np.empty(n_orig, dtype=float)
        x_full[keep] = x_red
        x_full[fixed] = fvals
        return x_full

    x0_red = np.asarray(p.x0, dtype=float)[keep].copy()
    xl_red = xl_eff[keep].copy()
    xu_red = xu_eff[keep].copy()
    # Defensive: clip x0 to (possibly tightened) bounds. FBBT preserves
    # x0-feasibility analytically, but roundoff can push x0 a hair past
    # the new bound; clipping keeps `_validate` from rejecting it.
    x0_red = np.clip(x0_red, xl_red, xu_red)

    objective = lambda x: float(p.objective(lift_x(x)))                  # noqa: E731
    gradient  = lambda x: np.asarray(p.gradient(lift_x(x)))[keep]        # noqa: E731
    comp_G    = lambda x: np.asarray(p.comp_G(lift_x(x)))[kcomp]         # noqa: E731
    comp_H    = lambda x: np.asarray(p.comp_H(lift_x(x)))[kcomp]         # noqa: E731

    def reduce_jac(orig_jac, sparsity, row_remap, *, drop_empty_rows=False):
        """Reduce a Jacobian by dropping pruned rows and pinned cols.

        ``row_remap[orig_row] = new_row`` (or ``-1`` if pruned).
        Returns ``(wrapped_callable, new_sparsity, surviving_orig_rows)``.

        When ``drop_empty_rows=True`` (used for ineq/eq blocks), rows whose
        every column was pinned out are also dropped — they're trivially
        ``c = 0`` and pollute multiplier bookkeeping otherwise.
        ``surviving_orig_rows`` is the original-row indices kept (or None
        when no row drops happened).
        """
        if orig_jac is None:
            return None, None, None
        if sparsity is None:
            kept_rows = np.where(row_remap >= 0)[0]
            def jac_dense(x_red):
                Jfull = np.asarray(orig_jac(lift_x(x_red)))
                return Jfull[np.ix_(kept_rows, keep)]
            surviving_orig_rows_dense = (
                kept_rows.astype(np.intp)
                if (drop_empty_rows and kept_rows.size != row_remap.size)
                else None
            )
            return jac_dense, None, surviving_orig_rows_dense
        rows = np.asarray(sparsity[0])
        cols = np.asarray(sparsity[1])
        nr = row_remap[rows]
        nc = col_remap[cols]
        mask = (nr >= 0) & (nc >= 0)
        idx_kept = np.where(mask)[0]

        surviving_orig_rows = None
        new_rows_local = nr[idx_kept]
        if drop_empty_rows:
            # Find which "new" rows have at least one surviving entry.
            n_new_rows_max = (row_remap.max() + 1) if row_remap.size else 0
            row_has_entry = np.zeros(int(n_new_rows_max), dtype=bool)
            if new_rows_local.size:
                row_has_entry[new_rows_local] = True
            kept_new_rows = np.where(row_has_entry)[0]
            initially_full = bool(np.all(row_remap >= 0)) if row_remap.size else True
            if kept_new_rows.size != n_new_rows_max or not initially_full:
                # Row map: old new_row → compact new_row (or -1 if dropped).
                if kept_new_rows.size != n_new_rows_max:
                    compact = -np.ones(int(n_new_rows_max), dtype=np.intp)
                    compact[kept_new_rows] = np.arange(kept_new_rows.size)
                    new_rows_local = compact[new_rows_local]
                # Surviving original rows = original rows whose row_remap
                # value lands in kept_new_rows.
                kept_orig = np.where(row_remap >= 0)[0]
                in_kept = np.isin(row_remap[kept_orig], kept_new_rows)
                surviving_orig_rows = kept_orig[in_kept].astype(np.intp)

        new_sp = (new_rows_local.astype(np.intp), nc[idx_kept].astype(np.intp))
        def jac_sparse(x_red):
            return np.asarray(orig_jac(lift_x(x_red)))[idx_kept]
        return jac_sparse, new_sp, surviving_orig_rows

    cG_jac, sG_sp, _ = reduce_jac(
        p.comp_G_jacobian, p.comp_G_jacobian_sparsity, comp_remap)
    cH_jac, sH_sp, _ = reduce_jac(
        p.comp_H_jacobian, p.comp_H_jacobian_sparsity, comp_remap)

    surviving_ineq: Optional[np.ndarray] = None
    surviving_eq:   Optional[np.ndarray] = None

    if p.n_ineq:
        ineq_id = np.full(p.n_ineq, -1, dtype=np.intp)
        keep_mask_i = ~np.asarray(drop_ineq, dtype=bool) if drop_ineq is not None \
            else np.ones(p.n_ineq, dtype=bool)
        ineq_id[keep_mask_i] = np.arange(int(keep_mask_i.sum()))
        iJ_jac, iJ_sp, surviving_ineq = reduce_jac(
            p.ineq_jacobian, p.ineq_jacobian_sparsity, ineq_id,
            drop_empty_rows=True,
        )
        if surviving_ineq is None:
            ineq_fn = lambda x: np.asarray(p.ineq_constraints(lift_x(x)))    # noqa: E731
            n_ineq_red = p.n_ineq
        else:
            sel = surviving_ineq
            n_ineq_red = int(sel.size)
            if n_ineq_red == 0:
                ineq_fn, iJ_jac, iJ_sp = None, None, None
            else:
                ineq_fn = lambda x: np.asarray(p.ineq_constraints(lift_x(x)))[sel]  # noqa: E731
    else:
        ineq_fn, iJ_jac, iJ_sp, n_ineq_red = None, None, None, 0

    # ----------------------------------------------------------------- #
    # B2 — augment ineq with promoted G_i ≥ 0 / H_i ≥ 0 rows.
    # ----------------------------------------------------------------- #
    n_promote_G = pmap.promote_G.size
    n_promote_H = pmap.promote_H.size
    if n_promote_G or n_promote_H:
        sG_full_rows = np.asarray(p.comp_G_jacobian_sparsity[0], dtype=np.intp)
        sG_full_cols = np.asarray(p.comp_G_jacobian_sparsity[1], dtype=np.intp)
        sH_full_rows = np.asarray(p.comp_H_jacobian_sparsity[0], dtype=np.intp)
        sH_full_cols = np.asarray(p.comp_H_jacobian_sparsity[1], dtype=np.intp)

        pos_G = -np.ones(n_comp_orig, dtype=np.intp)
        pos_G[pmap.promote_G] = np.arange(n_promote_G)
        pos_H = -np.ones(n_comp_orig, dtype=np.intp)
        pos_H[pmap.promote_H] = np.arange(n_promote_H)

        new_rows_G = pos_G[sG_full_rows]
        nc_G       = col_remap[sG_full_cols]
        keep_G_mask = (new_rows_G >= 0) & (nc_G >= 0)
        src_G_idx  = np.where(keep_G_mask)[0].astype(np.intp)
        aug_rows_G = (n_ineq_red + new_rows_G[src_G_idx]).astype(np.intp)
        aug_cols_G = nc_G[src_G_idx].astype(np.intp)

        new_rows_H = pos_H[sH_full_rows]
        nc_H       = col_remap[sH_full_cols]
        keep_H_mask = (new_rows_H >= 0) & (nc_H >= 0)
        src_H_idx  = np.where(keep_H_mask)[0].astype(np.intp)
        aug_rows_H = (n_ineq_red + n_promote_G + new_rows_H[src_H_idx]).astype(np.intp)
        aug_cols_H = nc_H[src_H_idx].astype(np.intp)

        if iJ_sp is not None:
            aug_sp_rows = np.concatenate([iJ_sp[0], aug_rows_G, aug_rows_H]).astype(np.intp)
            aug_sp_cols = np.concatenate([iJ_sp[1], aug_cols_G, aug_cols_H]).astype(np.intp)
        else:
            aug_sp_rows = np.concatenate([aug_rows_G, aug_rows_H]).astype(np.intp)
            aug_sp_cols = np.concatenate([aug_cols_G, aug_cols_H]).astype(np.intp)

        promote_G_idx = pmap.promote_G
        promote_H_idx = pmap.promote_H
        orig_ineq_fn = ineq_fn        # already wrapped with lift_x + sel_ineq
        orig_iJ_jac  = iJ_jac          # already wrapped
        cG_full     = p.comp_G
        cH_full     = p.comp_H
        cG_jac_full = p.comp_G_jacobian
        cH_jac_full = p.comp_H_jacobian

        empty = np.empty(0, dtype=float)

        def aug_ineq_fn(
            x_red,
            _orig=orig_ineq_fn, _G=cG_full, _H=cH_full,
            _gi=promote_G_idx, _hi=promote_H_idx,
            _ng=n_promote_G, _nh=n_promote_H,
        ):
            xf = lift_x(x_red)
            parts = []
            if _orig is not None:
                parts.append(np.asarray(_orig(x_red), dtype=float))
            if _ng:
                parts.append(-np.asarray(_G(xf), dtype=float)[_gi])
            if _nh:
                parts.append(-np.asarray(_H(xf), dtype=float)[_hi])
            return np.concatenate(parts) if parts else empty

        def aug_iJ_jac(
            x_red,
            _orig=orig_iJ_jac, _Gj=cG_jac_full, _Hj=cH_jac_full,
            _sg=src_G_idx, _sh=src_H_idx,
            _ng=n_promote_G, _nh=n_promote_H,
        ):
            xf = lift_x(x_red)
            parts = []
            if _orig is not None:
                parts.append(np.asarray(_orig(x_red), dtype=float))
            if _ng:
                parts.append(-np.asarray(_Gj(xf), dtype=float)[_sg])
            if _nh:
                parts.append(-np.asarray(_Hj(xf), dtype=float)[_sh])
            return np.concatenate(parts) if parts else empty

        ineq_fn = aug_ineq_fn
        iJ_jac  = aug_iJ_jac
        iJ_sp   = (aug_sp_rows, aug_sp_cols)
        n_ineq_red = n_ineq_red + n_promote_G + n_promote_H

    if p.n_eq:
        eq_id = np.full(p.n_eq, -1, dtype=np.intp)
        keep_mask_e = ~np.asarray(drop_eq, dtype=bool) if drop_eq is not None \
            else np.ones(p.n_eq, dtype=bool)
        eq_id[keep_mask_e] = np.arange(int(keep_mask_e.sum()))
        eJ_jac, eJ_sp, surviving_eq = reduce_jac(
            p.eq_jacobian, p.eq_jacobian_sparsity, eq_id,
            drop_empty_rows=True,
        )
        if surviving_eq is None:
            eq_fn = lambda x: np.asarray(p.eq_constraints(lift_x(x)))        # noqa: E731
            n_eq_red = p.n_eq
        else:
            sel_eq = surviving_eq
            n_eq_red = int(sel_eq.size)
            if n_eq_red == 0:
                eq_fn, eJ_jac, eJ_sp = None, None, None
            else:
                eq_fn = lambda x: np.asarray(p.eq_constraints(lift_x(x)))[sel_eq]  # noqa: E731
    else:
        eq_fn, eJ_jac, eJ_sp, n_eq_red = None, None, None, 0

    # ----------------------------------------------------------------- #
    # B3 — augment eq with prefix rows H_i = 0 / G_i = 0.
    # ----------------------------------------------------------------- #
    n_prefix_H = pmap.prefix_H_eq.size  # G_i > 0 → H_i = 0
    n_prefix_G = pmap.prefix_G_eq.size  # H_i > 0 → G_i = 0
    if n_prefix_H or n_prefix_G:
        sG_full_rows = np.asarray(p.comp_G_jacobian_sparsity[0], dtype=np.intp)
        sG_full_cols = np.asarray(p.comp_G_jacobian_sparsity[1], dtype=np.intp)
        sH_full_rows = np.asarray(p.comp_H_jacobian_sparsity[0], dtype=np.intp)
        sH_full_cols = np.asarray(p.comp_H_jacobian_sparsity[1], dtype=np.intp)

        pos_pH = -np.ones(n_comp_orig, dtype=np.intp)
        pos_pH[pmap.prefix_H_eq] = np.arange(n_prefix_H)  # H_i row offsets
        pos_pG = -np.ones(n_comp_orig, dtype=np.intp)
        pos_pG[pmap.prefix_G_eq] = np.arange(n_prefix_G)  # G_i row offsets

        new_rows_pH = pos_pH[sH_full_rows]
        nc_pH       = col_remap[sH_full_cols]
        keep_pH_mask = (new_rows_pH >= 0) & (nc_pH >= 0)
        src_pH_idx  = np.where(keep_pH_mask)[0].astype(np.intp)
        aug_rows_pH = (n_eq_red + new_rows_pH[src_pH_idx]).astype(np.intp)
        aug_cols_pH = nc_pH[src_pH_idx].astype(np.intp)

        new_rows_pG = pos_pG[sG_full_rows]
        nc_pG       = col_remap[sG_full_cols]
        keep_pG_mask = (new_rows_pG >= 0) & (nc_pG >= 0)
        src_pG_idx  = np.where(keep_pG_mask)[0].astype(np.intp)
        aug_rows_pG = (n_eq_red + n_prefix_H + new_rows_pG[src_pG_idx]).astype(np.intp)
        aug_cols_pG = nc_pG[src_pG_idx].astype(np.intp)

        if eJ_sp is not None:
            aug_eq_rows = np.concatenate([eJ_sp[0], aug_rows_pH, aug_rows_pG]).astype(np.intp)
            aug_eq_cols = np.concatenate([eJ_sp[1], aug_cols_pH, aug_cols_pG]).astype(np.intp)
        else:
            aug_eq_rows = np.concatenate([aug_rows_pH, aug_rows_pG]).astype(np.intp)
            aug_eq_cols = np.concatenate([aug_cols_pH, aug_cols_pG]).astype(np.intp)

        prefix_H_idx = pmap.prefix_H_eq
        prefix_G_idx = pmap.prefix_G_eq
        orig_eq_fn  = eq_fn
        orig_eJ_jac = eJ_jac
        cG_full     = p.comp_G
        cH_full     = p.comp_H
        cG_jac_full = p.comp_G_jacobian
        cH_jac_full = p.comp_H_jacobian
        empty = np.empty(0, dtype=float)

        def aug_eq_fn(
            x_red,
            _orig=orig_eq_fn, _G=cG_full, _H=cH_full,
            _hi=prefix_H_idx, _gi=prefix_G_idx,
            _nph=n_prefix_H, _npg=n_prefix_G,
        ):
            xf = lift_x(x_red)
            parts = []
            if _orig is not None:
                parts.append(np.asarray(_orig(x_red), dtype=float))
            if _nph:
                parts.append(np.asarray(_H(xf), dtype=float)[_hi])
            if _npg:
                parts.append(np.asarray(_G(xf), dtype=float)[_gi])
            return np.concatenate(parts) if parts else empty

        def aug_eJ_jac(
            x_red,
            _orig=orig_eJ_jac, _Gj=cG_jac_full, _Hj=cH_jac_full,
            _sh=src_pH_idx, _sg=src_pG_idx,
            _nph=n_prefix_H, _npg=n_prefix_G,
        ):
            xf = lift_x(x_red)
            parts = []
            if _orig is not None:
                parts.append(np.asarray(_orig(x_red), dtype=float))
            if _nph:
                parts.append(np.asarray(_Hj(xf), dtype=float)[_sh])
            if _npg:
                parts.append(np.asarray(_Gj(xf), dtype=float)[_sg])
            return np.concatenate(parts) if parts else empty

        eq_fn  = aug_eq_fn
        eJ_jac = aug_eJ_jac
        eJ_sp  = (aug_eq_rows, aug_eq_cols)
        n_eq_red = n_eq_red + n_prefix_H + n_prefix_G

    # Record dropped rows on the map so expand_result can pad zeros.
    pmap.keep_ineq = surviving_ineq
    pmap.keep_eq   = surviving_eq

    cG_scale = (np.asarray(p.comp_G_scale)[kcomp].copy()
                if p.comp_G_scale is not None else None)
    cH_scale = (np.asarray(p.comp_H_scale)[kcomp].copy()
                if p.comp_H_scale is not None else None)

    return MPCCProblem(
        n=n_red, n_comp=int(kcomp.size),
        x0=x0_red, xl=xl_red, xu=xu_red,
        objective=objective, gradient=gradient,
        comp_G=comp_G, comp_G_jacobian=cG_jac,
        comp_G_jacobian_sparsity=sG_sp,
        comp_H=comp_H, comp_H_jacobian=cH_jac,
        comp_H_jacobian_sparsity=sH_sp,
        n_ineq=n_ineq_red, ineq_constraints=ineq_fn, ineq_jacobian=iJ_jac,
        ineq_jacobian_sparsity=iJ_sp,
        n_eq=n_eq_red, eq_constraints=eq_fn, eq_jacobian=eJ_jac,
        eq_jacobian_sparsity=eJ_sp,
        comp_G_scale=cG_scale, comp_H_scale=cH_scale,
    )


# --------------------------------------------------------------------------- #
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

def _identity_map(p: MPCCProblem) -> PresolveMap:
    return PresolveMap(
        keep_vars=np.arange(p.n, dtype=np.intp),
        fixed_vars=np.empty(0, dtype=np.intp),
        fixed_vals=np.empty(0, dtype=float),
        keep_comp=np.arange(p.n_comp, dtype=np.intp),
        n_orig=p.n,
        n_comp_orig=p.n_comp,
        n_ineq=p.n_ineq,
        n_eq=p.n_eq,
    )


def presolve(problem: MPCCProblem) -> tuple[MPCCProblem, PresolveMap]:
    """Reduce *problem* by FBBT, pinned-var elimination and dead-pair pruning.

    Returns ``(reduced_problem, map)``.  When nothing can be eliminated,
    or when reduction would yield an invalid problem (e.g. an empty
    Jacobian sparsity pattern), returns ``(problem, identity_map)``.

    Detected linear infeasibility (an FBBT bound crossing) emits a
    ``UserWarning`` and falls back to identity so the original problem
    can surface the issue through the strategy's normal infeasibility
    path.
    """
    # 1. Empty-row detection on the original sparsity.
    drop_ineq, drop_eq, infeasible = _detect_empty_rows(problem)
    if infeasible:
        warnings.warn(
            "pympcc.presolve: structurally-empty constraint row violates its "
            "RHS at x0; falling back to identity (no presolve).",
            UserWarning,
            stacklevel=2,
        )
        return problem, _identity_map(problem)

    # 2. FBBT: tighten xl/xu from linear ineq/eq rows.
    xl_t, xu_t, infeasible = _fbbt(problem)
    if infeasible:
        warnings.warn(
            "pympcc.presolve: FBBT detected an infeasible linear constraint; "
            "falling back to identity (no presolve).",
            UserWarning,
            stacklevel=2,
        )
        return problem, _identity_map(problem)

    # 3. Empty-column detection — pin free variables via xl=xu=clip(0,...).
    free_cols = _detect_empty_cols(problem, xl_t, xu_t)
    if free_cols.size:
        for j in free_cols:
            lo, hi = xl_t[j], xu_t[j]
            fix = 0.0
            if np.isfinite(lo) and np.isfinite(hi):
                fix = float(np.clip(0.0, lo, hi))
            elif np.isfinite(lo):
                fix = float(max(0.0, lo))
            elif np.isfinite(hi):
                fix = float(min(0.0, hi))
            xl_t[j] = fix
            xu_t[j] = fix

    # 4. Pinned-var detection on the (possibly tightened) bounds.
    pinned_idx, pinned_vals = _detect_pinned_from_bounds(xl_t, xu_t)
    dead = _detect_dead(problem)
    promote_G, promote_H, extra_dead = _detect_forced(problem, dead)
    if extra_dead.size:
        dead = dead.copy()
        dead[extra_dead] = True
    dropped_pairs = dead.copy()
    if promote_G.size:
        dropped_pairs[promote_G] = True
    if promote_H.size:
        dropped_pairs[promote_H] = True

    # B3 — linear sign analysis using FBBT-tightened bounds.
    prefix_H_eq, prefix_G_eq, b3_infeasible = _detect_prefix_eq(
        problem, xl_t, xu_t, dropped_pairs,
    )
    if b3_infeasible:
        warnings.warn(
            "pympcc.presolve: linear sign analysis detected infeasible "
            "complementarity (G_i > 0 and H_i > 0 simultaneously); "
            "falling back to identity (no presolve).",
            UserWarning,
            stacklevel=2,
        )
        return problem, _identity_map(problem)
    if prefix_H_eq.size:
        dropped_pairs[prefix_H_eq] = True
    if prefix_G_eq.size:
        dropped_pairs[prefix_G_eq] = True
    keep_comp = np.where(~dropped_pairs)[0].astype(np.intp)

    bounds_changed = (
        not np.array_equal(xl_t, np.asarray(problem.xl, dtype=float))
        or not np.array_equal(xu_t, np.asarray(problem.xu, dtype=float))
    )
    rows_dropped = bool(drop_ineq.any() or drop_eq.any())
    if (pinned_idx.size == 0
            and keep_comp.size == problem.n_comp
            and not bounds_changed
            and not rows_dropped
            and promote_G.size == 0
            and promote_H.size == 0
            and prefix_H_eq.size == 0
            and prefix_G_eq.size == 0):
        return problem, _identity_map(problem)

    # MPCC requires n_comp >= 1; bail out of comp pruning rather than
    # destroy that invariant.
    if keep_comp.size == 0:
        keep_comp = np.arange(problem.n_comp, dtype=np.intp)
        promote_G = np.empty(0, dtype=np.intp)
        promote_H = np.empty(0, dtype=np.intp)
        prefix_H_eq = np.empty(0, dtype=np.intp)
        prefix_G_eq = np.empty(0, dtype=np.intp)
        if pinned_idx.size == 0 and not bounds_changed and not rows_dropped:
            return problem, _identity_map(problem)

    keep_vars = np.setdiff1d(
        np.arange(problem.n, dtype=np.intp),
        pinned_idx,
        assume_unique=True,
    )
    pmap = PresolveMap(
        keep_vars=keep_vars,
        fixed_vars=pinned_idx.astype(np.intp),
        fixed_vals=np.asarray(pinned_vals, dtype=float),
        keep_comp=keep_comp,
        n_orig=problem.n,
        n_comp_orig=problem.n_comp,
        n_ineq=problem.n_ineq,
        n_eq=problem.n_eq,
        promote_G=promote_G,
        promote_H=promote_H,
        prefix_H_eq=prefix_H_eq,
        prefix_G_eq=prefix_G_eq,
    )

    try:
        reduced = _build_reduced(
            problem, pmap, xl_full=xl_t, xu_full=xu_t,
            drop_ineq=drop_ineq, drop_eq=drop_eq,
        )
    except ValueError as e:
        warnings.warn(
            f"pympcc.presolve: reduction would invalidate the problem ({e}); "
            "falling back to identity (no presolve).",
            UserWarning,
            stacklevel=3,
        )
        return problem, _identity_map(problem)

    return reduced, pmap
