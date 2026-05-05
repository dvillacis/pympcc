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

Module layout
-------------
The detection passes live in :mod:`pympcc._presolve_detect`, FBBT in
:mod:`pympcc._presolve_fbbt`, and the reduction step in
:mod:`pympcc._presolve_reduce`.  This module owns :class:`PresolveMap`
and the public :func:`presolve` entry point, and re-exports the
private helpers so existing call sites that import them via
``pympcc._presolve`` keep working.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Re-export the split-out helpers so external callers (and the test
# suite) can keep importing them via ``pympcc._presolve``.  Each of the
# three submodules is single-purpose and self-contained.
from ._presolve_detect import (
    _detect_dead,
    _detect_empty_cols,
    _detect_empty_rows,
    _detect_forced,
    _detect_pinned_from_bounds,
    _detect_prefix_eq,
)
from ._presolve_fbbt import _fbbt
from ._presolve_reduce import _build_reduced, _identity_map
from .problem import MPCCProblem
from .result import IterationInfo, MPCCResult

_log = logging.getLogger(__name__)

__all__ = ["PresolveMap", "presolve"]


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
        G_full = np.asarray(problem_orig.comp_G(x_full), dtype=float)  # type: ignore[misc]
        H_full = np.asarray(problem_orig.comp_H(x_full), dtype=float)  # type: ignore[misc]

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
# Public entry point                                                           #
# --------------------------------------------------------------------------- #

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
    # cyipopt requires n >= 1.  When *every* variable is pinned, fall
    # back to identity so the strategy / NLP layer evaluates the
    # original problem at its single feasible point (where IPOPT
    # terminates after one iteration); avoids constructing an
    # n_red == 0 NLP that cyipopt rejects.
    if keep_vars.size == 0:
        return problem, _identity_map(problem)

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
