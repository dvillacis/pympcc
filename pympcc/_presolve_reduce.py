"""Reduction step of the presolve pipeline.

Builds the reduced :class:`MPCCProblem` from an original problem and a
:class:`PresolveMap` populated by the detection passes.  Wraps every
user callable with the lift / drop / promotion logic that keeps the
reduced and original problems mathematically equivalent.

Public entry points re-exported from :mod:`pympcc._presolve` for
backward compatibility:

* :func:`_build_reduced`
* :func:`_identity_map`
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .problem import MPCCProblem

if TYPE_CHECKING:
    from ._presolve import PresolveMap


def _build_reduced(
    p: MPCCProblem,
    pmap: "PresolveMap",
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
    gradient  = lambda x: np.asarray(p.gradient(lift_x(x)))[keep]  # type: ignore[misc, operator]  # noqa: E731
    comp_G    = lambda x: np.asarray(p.comp_G(lift_x(x)))[kcomp]  # type: ignore[misc]  # noqa: E731
    comp_H    = lambda x: np.asarray(p.comp_H(lift_x(x)))[kcomp]  # type: ignore[misc]  # noqa: E731

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
            ineq_fn = lambda x: np.asarray(p.ineq_constraints(lift_x(x)))  # type: ignore[misc]  # noqa: E731
            n_ineq_red = p.n_ineq
        else:
            sel = surviving_ineq
            n_ineq_red = int(sel.size)
            if n_ineq_red == 0:
                ineq_fn, iJ_jac, iJ_sp = None, None, None
            else:
                ineq_fn = lambda x: np.asarray(p.ineq_constraints(lift_x(x)))[sel]  # type: ignore[misc]  # noqa: E731
    else:
        ineq_fn, iJ_jac, iJ_sp, n_ineq_red = None, None, None, 0

    # ----------------------------------------------------------------- #
    # B2 — augment ineq with promoted G_i ≥ 0 / H_i ≥ 0 rows.
    # ----------------------------------------------------------------- #
    n_promote_G = pmap.promote_G.size
    n_promote_H = pmap.promote_H.size
    if n_promote_G or n_promote_H:
        sG_full_rows = np.asarray(p.comp_G_jacobian_sparsity[0], dtype=np.intp)  # type: ignore[index]
        sG_full_cols = np.asarray(p.comp_G_jacobian_sparsity[1], dtype=np.intp)  # type: ignore[index]
        sH_full_rows = np.asarray(p.comp_H_jacobian_sparsity[0], dtype=np.intp)  # type: ignore[index]
        sH_full_cols = np.asarray(p.comp_H_jacobian_sparsity[1], dtype=np.intp)  # type: ignore[index]

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
            eq_fn = lambda x: np.asarray(p.eq_constraints(lift_x(x)))  # type: ignore[misc]  # noqa: E731
            n_eq_red = p.n_eq
        else:
            sel_eq = surviving_eq
            n_eq_red = int(sel_eq.size)
            if n_eq_red == 0:
                eq_fn, eJ_jac, eJ_sp = None, None, None
            else:
                eq_fn = lambda x: np.asarray(p.eq_constraints(lift_x(x)))[sel_eq]  # type: ignore[misc]  # noqa: E731
    else:
        eq_fn, eJ_jac, eJ_sp, n_eq_red = None, None, None, 0

    # ----------------------------------------------------------------- #
    # B3 — augment eq with prefix rows H_i = 0 / G_i = 0.
    # ----------------------------------------------------------------- #
    n_prefix_H = pmap.prefix_H_eq.size  # G_i > 0 → H_i = 0
    n_prefix_G = pmap.prefix_G_eq.size  # H_i > 0 → G_i = 0
    if n_prefix_H or n_prefix_G:
        sG_full_rows = np.asarray(p.comp_G_jacobian_sparsity[0], dtype=np.intp)  # type: ignore[index]
        sG_full_cols = np.asarray(p.comp_G_jacobian_sparsity[1], dtype=np.intp)  # type: ignore[index]
        sH_full_rows = np.asarray(p.comp_H_jacobian_sparsity[0], dtype=np.intp)  # type: ignore[index]
        sH_full_cols = np.asarray(p.comp_H_jacobian_sparsity[1], dtype=np.intp)  # type: ignore[index]

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


def _identity_map(p: MPCCProblem) -> "PresolveMap":
    # Late import to avoid a cycle: PresolveMap lives in `_presolve`
    # which itself imports the helpers in this module.
    from ._presolve import PresolveMap

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
