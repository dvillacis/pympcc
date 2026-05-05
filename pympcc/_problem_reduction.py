"""Var-pair / box-pair / doubly-bounded normalisation for
:class:`~pympcc.MPCCProblem`.

These helpers translate the high-level entry forms

* ``comp_var_pairs`` / ``comp_var_pairs_bulk``  — variable-paired MCPs,
* ``comp_box_pairs``                            — box-MCPs

into the canonical ``comp_G`` / ``comp_H`` block plus their Jacobians /
sparsity patterns, augmenting the eq / ineq blocks where needed.  All
functions mutate the supplied ``MPCCProblem`` in place.

Order of orchestration in :meth:`MPCCProblem.__post_init__`::

    normalize_var_pairs(p)   # comp_var_pairs / comp_var_pairs_bulk
    normalize_box_pairs(p)   # comp_box_pairs (lower / upper / free / doubly-bounded)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .problem import MPCCProblem


def normalize_var_pairs(p: "MPCCProblem") -> None:
    """Merge ``comp_var_pairs`` declarations into ``comp_G`` / ``comp_H`` and their Jacobians.

    After this function runs:
    * ``comp_G`` and ``comp_H`` are non-None callables.
    * ``comp_G_jacobian`` is an exact callable (identity rows for var-pair block).
    * ``comp_H_jacobian`` is a callable (user-provided or fd per var-pair row).

    Three entry forms are accepted:

    * ``(var_idx, h_fn)`` — fd dense row (current default).
    * ``(var_idx, h_fn, h_jac_fn)`` — dense row; ``h_jac_fn(x) -> ndarray (n,)``.
    * ``(var_idx, h_fn, h_jac_fn, h_cols)`` — sparse row; ``h_jac_fn(x)``
      returns only the values at the columns ``h_cols``.

    When **all** entries use the 4-tuple sparse form, ``comp_G_jacobian`` and
    ``comp_H_jacobian`` are emitted as values-only callbacks and the
    corresponding ``*_jacobian_sparsity`` patterns are auto-set (unless
    already supplied). 2-/3-tuple and 4-tuple entries cannot be mixed in
    the same list.
    """
    if p.comp_var_pairs_bulk is not None:
        if p.comp_var_pairs:
            raise ValueError(
                "comp_var_pairs and comp_var_pairs_bulk are mutually exclusive"
            )
        apply_var_pairs_bulk(p)
        return

    if not p.comp_var_pairs:
        if p.comp_G is None or p.comp_H is None:
            if p.comp_box_pairs:
                return  # comp_box_pairs will populate comp_G / comp_H
            raise ValueError(
                "comp_G and comp_H are required when neither comp_var_pairs "
                "nor comp_box_pairs is used"
            )
        return

    from ._fd import fd_jacobian as _fd_jac

    pairs = list(p.comp_var_pairs)
    k = len(pairs)
    n = p.n
    h_fd = p.fd_h
    h_mode = p.fd_mode

    # Parse tuples: 2-, 3-, or 4-tuple forms.
    var_idxs: list[int] = []
    h_fns: list = []
    h_jac_fns: list = []   # None → build fd inline
    h_cols_list: list = []  # None for dense rows; ndarray of intp for sparse rows
    n_sparse = 0
    for entry in pairs:
        if len(entry) == 2:
            vi, hf = entry
            hj = None
            hc = None
        elif len(entry) == 3:
            vi, hf, hj = entry
            hc = None
        elif len(entry) == 4:
            vi, hf, hj, hc = entry
            if hj is None:
                raise ValueError(
                    "comp_var_pairs 4-tuple form requires h_jac_fn (3rd element); "
                    "use 2-tuple form for fd"
                )
            hc = np.asarray(hc, dtype=np.intp).ravel()
            if hc.size > 0 and (hc.min() < 0 or hc.max() >= n):
                raise ValueError(
                    f"comp_var_pairs: h_cols entries must be in [0, {n})"
                )
            n_sparse += 1
        else:
            raise ValueError(
                "comp_var_pairs entries must be (var_idx, h_fn), "
                "(var_idx, h_fn, h_jac_fn), or (var_idx, h_fn, h_jac_fn, h_cols)"
            )
        if not (0 <= int(vi) < n):
            raise ValueError(
                f"comp_var_pairs: var_idx {vi} is out of range [0, {n})"
            )
        var_idxs.append(int(vi))
        h_fns.append(hf)
        h_jac_fns.append(hj)
        h_cols_list.append(hc)

    if 0 < n_sparse < k:
        raise ValueError(
            "comp_var_pairs entries must be all-sparse (4-tuple) or all-dense "
            f"(2-/3-tuple); got {n_sparse} sparse and {k - n_sparse} dense"
        )
    sparse_mode = (n_sparse == k)

    var_idx_arr = np.array(var_idxs, dtype=np.intp)

    # Enforce lower bound >= 0 for each paired variable
    assert p.xl is not None
    for vi in var_idxs:
        p.xl[vi] = max(p.xl[vi], 0.0)

    # Guard the fd fallback at scale: fd evaluates h_fn ~n+1 times per row
    # per Jacobian call.  At k=1e4, n=1e3 that's 1e7 user calls per Jacobian
    # — typically 1000x slower than supplying h_jac_fn explicitly.
    n_fd_rows = sum(1 for hj in h_jac_fns if hj is None)
    if n_fd_rows > 0 and (n_fd_rows * (n + 1) > 10_000_000):
        raise ValueError(
            f"comp_var_pairs has {n_fd_rows} entries with finite-difference "
            f"H Jacobians and n={n}; this would evaluate h_fn ~"
            f"{n_fd_rows * (n + 1):_} times per Jacobian call. "
            "Pass h_jac_fn explicitly (3-tuple form), use sparse 4-tuple "
            "form, or use comp_var_pairs_bulk for vectorized MCPs."
        )

    # Build per-row H Jacobians (resolve fd now for any missing — fd is only
    # reachable from 2-tuple form, which is dense by definition)
    resolved_h_jac: list = []
    for hf, hj in zip(h_fns, h_jac_fns):
        if hj is not None:
            resolved_h_jac.append(hj)
        else:
            def _scalar_fd(x, _f=hf, _h=h_fd, _m=h_mode):
                return _fd_jac(_f, 1, n, _h, _m)(x)[0]
            resolved_h_jac.append(_scalar_fd)

    # One-shot construction-time validation for sparse-mode rows: each
    # h_jac_fn(x0) must return exactly len(h_cols) values.  Caught here
    # so the per-callback hot path is lean (no per-row size check).
    if sparse_mode:
        for i, (jf, hc) in enumerate(zip(resolved_h_jac, h_cols_list)):
            vals = np.asarray(jf(p.x0)).ravel()
            if vals.size != hc.size:
                raise ValueError(
                    f"comp_var_pairs row {i}: h_jac_fn returned "
                    f"{vals.size} values but h_cols has {hc.size}"
                )

    # ------------------------------------------------------------------
    # Common comp_G / comp_H value callables (identical for sparse/dense).
    # ------------------------------------------------------------------

    if p.comp_G is None:
        # All-var-pairs mode: every comp pair is declared via comp_var_pairs
        if k != p.n_comp:
            raise ValueError(
                f"comp_var_pairs has {k} entries but n_comp={p.n_comp}; "
                "when comp_G is None, len(comp_var_pairs) must equal n_comp"
            )

        def _G(x, _idx=var_idx_arr):
            return np.asarray(x, dtype=float)[_idx]

        def _H(x, _hfs=h_fns, _k=k):
            out = np.empty(_k)
            for i, hf in enumerate(_hfs):
                out[i] = np.asarray(hf(x)).ravel()[0]
            return out

        p.comp_G = _G
        p.comp_H = _H

        if sparse_mode:
            G_rows_arr = np.arange(k, dtype=np.intp)
            G_cols_arr = var_idx_arr.copy()
            H_row_blocks = [np.full(hc.size, i, dtype=np.intp)
                            for i, hc in enumerate(h_cols_list)]
            H_rows_arr = (np.concatenate(H_row_blocks)
                          if H_row_blocks else np.empty(0, dtype=np.intp))
            H_cols_arr = (np.concatenate(h_cols_list)
                          if h_cols_list else np.empty(0, dtype=np.intp))
            row_sizes = np.array([hc.size for hc in h_cols_list], dtype=np.intp)
            row_offsets = np.concatenate(
                ([0], np.cumsum(row_sizes))
            ).astype(np.intp)
            H_nnz = int(row_sizes.sum())
            G_vals_const = np.ones(k)

            def _G_jac_sparse(x, _v=G_vals_const):
                return _v

            def _H_jac_sparse(x, _jfs=resolved_h_jac, _off=row_offsets,
                              _nnz=H_nnz):
                out = np.empty(_nnz)
                for i, jf in enumerate(_jfs):
                    out[_off[i]:_off[i + 1]] = jf(x)
                return out

            p.comp_G_jacobian = _G_jac_sparse
            p.comp_H_jacobian = _H_jac_sparse
            if p.comp_G_jacobian_sparsity is None:
                p.comp_G_jacobian_sparsity = (G_rows_arr, G_cols_arr)
            if p.comp_H_jacobian_sparsity is None:
                p.comp_H_jacobian_sparsity = (H_rows_arr, H_cols_arr)
        else:
            def _G_jac(x, _idx=var_idx_arr, _k=k, _n=n):
                J = np.zeros((_k, _n))
                J[np.arange(_k), _idx] = 1.0
                return J

            def _H_jac(x, _jfs=resolved_h_jac):
                return np.stack([jf(x) for jf in _jfs], axis=0)

            p.comp_G_jacobian = _G_jac
            p.comp_H_jacobian = _H_jac

    else:
        # Mixed mode: base comp_G/comp_H + k var-pair rows appended at the end
        n_base = p.n_comp - k
        if n_base <= 0:
            raise ValueError(
                f"comp_var_pairs has {k} entries but n_comp={p.n_comp}; "
                "mixed mode requires n_comp > len(comp_var_pairs) (at least one base pair)"
            )

        base_G = p.comp_G
        base_H = p.comp_H
        base_G_jac = p.comp_G_jacobian
        base_H_jac = p.comp_H_jacobian
        base_G_sp = p.comp_G_jacobian_sparsity
        base_H_sp = p.comp_H_jacobian_sparsity

        if not callable(base_G_jac):
            raise ValueError(
                "Mixed comp_var_pairs requires comp_G_jacobian to be a resolved callable. "
                "Pass derivatives='fd', derivatives='jax', or supply comp_G_jacobian explicitly."
            )
        if not callable(base_H_jac):
            raise ValueError(
                "Mixed comp_var_pairs requires comp_H_jacobian to be a resolved callable. "
                "Pass derivatives='fd', derivatives='jax', or supply comp_H_jacobian explicitly."
            )

        def _G(x, _bG=base_G, _idx=var_idx_arr):  # type: ignore[misc]
            return np.concatenate([
                np.asarray(_bG(x), dtype=float),
                np.asarray(x, dtype=float)[_idx],
            ])

        def _H(x, _bH=base_H, _hfs=h_fns, _k=k):  # type: ignore[misc]
            base_vals = np.asarray(_bH(x), dtype=float)
            out = np.empty(base_vals.size + _k)
            out[:base_vals.size] = base_vals
            offset = base_vals.size
            for i, hf in enumerate(_hfs):
                out[offset + i] = np.asarray(hf(x)).ravel()[0]
            return out

        p.comp_G = _G
        p.comp_H = _H

        if sparse_mode:
            if base_G_sp is None or base_H_sp is None:
                raise ValueError(
                    "comp_var_pairs sparse (4-tuple) form in mixed mode requires "
                    "both comp_G_jacobian_sparsity and comp_H_jacobian_sparsity "
                    "to be set on the base problem"
                )

            base_G_rows = np.asarray(base_G_sp[0], dtype=np.intp)
            base_G_cols = np.asarray(base_G_sp[1], dtype=np.intp)
            base_H_rows = np.asarray(base_H_sp[0], dtype=np.intp)
            base_H_cols = np.asarray(base_H_sp[1], dtype=np.intp)

            tail_G_rows = (n_base + np.arange(k, dtype=np.intp))
            tail_G_cols = var_idx_arr.copy()
            G_rows_arr = np.concatenate([base_G_rows, tail_G_rows])
            G_cols_arr = np.concatenate([base_G_cols, tail_G_cols])

            H_row_blocks = [np.full(hc.size, n_base + i, dtype=np.intp)
                            for i, hc in enumerate(h_cols_list)]
            tail_H_rows = (np.concatenate(H_row_blocks)
                           if H_row_blocks else np.empty(0, dtype=np.intp))
            tail_H_cols = (np.concatenate(h_cols_list)
                           if h_cols_list else np.empty(0, dtype=np.intp))
            H_rows_arr = np.concatenate([base_H_rows, tail_H_rows])
            H_cols_arr = np.concatenate([base_H_cols, tail_H_cols])

            n_base_G = int(base_G_rows.size)
            n_base_H = int(base_H_rows.size)
            row_sizes = np.array([hc.size for hc in h_cols_list], dtype=np.intp)
            row_offsets = np.concatenate(
                ([0], np.cumsum(row_sizes))
            ).astype(np.intp)
            tail_H_nnz = int(row_sizes.sum())
            G_vals_tail_const = np.ones(k)

            def _G_jac_sparse_mixed(x, _bJac=base_G_jac, _tail=G_vals_tail_const,
                                    _nb=n_base_G):
                out = np.empty(_nb + _tail.size)
                out[:_nb] = np.asarray(_bJac(x), dtype=float).ravel()
                out[_nb:] = _tail
                return out

            def _H_jac_sparse_mixed(x, _bJac=base_H_jac, _jfs=resolved_h_jac,
                                    _off=row_offsets, _nb=n_base_H,
                                    _tail_nnz=tail_H_nnz):
                out = np.empty(_nb + _tail_nnz)
                out[:_nb] = np.asarray(_bJac(x), dtype=float).ravel()
                for i, jf in enumerate(_jfs):
                    out[_nb + _off[i]:_nb + _off[i + 1]] = jf(x)
                return out

            p.comp_G_jacobian = _G_jac_sparse_mixed
            p.comp_H_jacobian = _H_jac_sparse_mixed
            p.comp_G_jacobian_sparsity = (G_rows_arr, G_cols_arr)
            p.comp_H_jacobian_sparsity = (H_rows_arr, H_cols_arr)
        else:
            def _G_jac(x, _bJac=base_G_jac, _idx=var_idx_arr, _k=k, _n=n):  # type: ignore[misc]
                base_rows = np.asarray(_bJac(x), dtype=float)
                id_rows = np.zeros((_k, _n))
                id_rows[np.arange(_k), _idx] = 1.0
                return np.vstack([base_rows, id_rows])

            def _H_jac(x, _bJac=base_H_jac, _jfs=resolved_h_jac):  # type: ignore[misc]
                base_rows = np.asarray(_bJac(x), dtype=float)
                extra_rows = np.stack([jf(x) for jf in _jfs], axis=0)
                return np.vstack([base_rows, extra_rows])

            p.comp_G_jacobian = _G_jac
            p.comp_H_jacobian = _H_jac


def apply_var_pairs_bulk(p: "MPCCProblem") -> None:
    """Wire ``comp_var_pairs_bulk`` into ``comp_G`` / ``comp_H`` and Jacobians.

    Expects ``p.comp_var_pairs_bulk`` to be a 4-tuple
    ``(var_idxs, h_bulk_fn, h_bulk_jac_fn, h_bulk_jac_sparsity)``.
    Sets ``comp_G_jacobian_sparsity`` to identity-on-var_idxs and forwards
    the user's H sparsity unchanged.  Mutually exclusive with the per-row
    ``comp_var_pairs`` form.
    """
    bulk = p.comp_var_pairs_bulk
    if not isinstance(bulk, tuple) or len(bulk) != 4:
        raise ValueError(
            "comp_var_pairs_bulk must be a 4-tuple "
            "(var_idxs, h_bulk_fn, h_bulk_jac_fn, h_bulk_jac_sparsity)"
        )
    var_idxs_in, h_bulk_fn, h_bulk_jac_fn, h_bulk_jac_sp = bulk

    n = p.n
    var_idxs = np.asarray(var_idxs_in, dtype=np.intp).ravel()
    k = var_idxs.size
    if k != p.n_comp:
        raise ValueError(
            f"comp_var_pairs_bulk: var_idxs has {k} entries but "
            f"n_comp={p.n_comp}; lengths must match"
        )
    if k > 0 and (var_idxs.min() < 0 or var_idxs.max() >= n):
        raise ValueError(
            f"comp_var_pairs_bulk: var_idxs entries must be in [0, {n})"
        )
    if not callable(h_bulk_fn):
        raise ValueError("comp_var_pairs_bulk: h_bulk_fn must be callable")
    if not callable(h_bulk_jac_fn):
        raise ValueError("comp_var_pairs_bulk: h_bulk_jac_fn must be callable")
    if (not isinstance(h_bulk_jac_sp, tuple)) or len(h_bulk_jac_sp) != 2:
        raise ValueError(
            "comp_var_pairs_bulk: h_bulk_jac_sparsity must be a (rows, cols) tuple"
        )
    H_rows = np.asarray(h_bulk_jac_sp[0], dtype=np.intp).ravel()
    H_cols = np.asarray(h_bulk_jac_sp[1], dtype=np.intp).ravel()
    if H_rows.size != H_cols.size:
        raise ValueError(
            f"comp_var_pairs_bulk: H sparsity rows/cols size mismatch "
            f"({H_rows.size} vs {H_cols.size})"
        )
    if H_rows.size > 0 and (H_rows.min() < 0 or H_rows.max() >= k):
        raise ValueError(
            f"comp_var_pairs_bulk: H sparsity rows must be in [0, {k})"
        )
    if H_cols.size > 0 and (H_cols.min() < 0 or H_cols.max() >= n):
        raise ValueError(
            f"comp_var_pairs_bulk: H sparsity cols must be in [0, {n})"
        )

    assert p.xl is not None
    p.xl[var_idxs] = np.maximum(p.xl[var_idxs], 0.0)

    G_vals_const = np.ones(k)

    def _G(x, _idx=var_idxs):
        return np.asarray(x, dtype=float)[_idx]

    def _G_jac(x, _v=G_vals_const):
        return _v

    p.comp_G = _G
    p.comp_H = h_bulk_fn
    p.comp_G_jacobian = _G_jac
    p.comp_H_jacobian = h_bulk_jac_fn
    if p.comp_G_jacobian_sparsity is None:
        p.comp_G_jacobian_sparsity = (
            np.arange(k, dtype=np.intp), var_idxs.copy()
        )
    if p.comp_H_jacobian_sparsity is None:
        p.comp_H_jacobian_sparsity = (H_rows, H_cols)


def normalize_box_pairs(p: "MPCCProblem") -> None:
    """Dispatch ``comp_box_pairs`` entries by bound finiteness.

    Each entry ``(var_idx, F_fn[, F_jac_fn])`` is routed to one of:

    * **Free** (``xl[var_idx] = -inf, xu[var_idx] = +inf``) → append
      ``F(x) = 0`` to the equality block.
    * **Lower-only** (``xl`` finite, ``xu = +inf``) → append comp pair
      ``(x[var_idx] - xl) >= 0  ⊥  F(x) >= 0``.
    * **Upper-only** (``xl = -inf``, ``xu`` finite) → append comp pair
      ``(xu - x[var_idx]) >= 0  ⊥  -F(x) >= 0``.
    * **Doubly-bounded** (both finite) → Billups slack split into
      two extra slack variables + one equality + two comp pairs.

    ``n_comp`` and ``n_eq`` are auto-bumped to include the synthesized
    rows; the user supplies them for the *base* problem only.

    Sparse base ``comp_G/H`` or sparse ``eq_jacobian`` are rejected in
    this first ship (deferred).  Mutually exclusive with
    ``comp_var_pairs`` / ``comp_var_pairs_bulk``.
    """
    if not p.comp_box_pairs:
        return

    if p.comp_var_pairs or p.comp_var_pairs_bulk:
        raise ValueError(
            "comp_box_pairs cannot be combined with comp_var_pairs or "
            "comp_var_pairs_bulk in this release"
        )

    from ._fd import fd_jacobian as _fd_jac

    n = p.n
    h_fd = p.fd_h
    h_mode = p.fd_mode
    assert p.xl is not None and p.xu is not None

    # ---- Parse entries ------------------------------------------------
    parsed: list = []
    for entry in p.comp_box_pairs:
        if len(entry) == 2:
            vi, F_fn = entry
            F_jac = None
        elif len(entry) == 3:
            vi, F_fn, F_jac = entry
        else:
            raise ValueError(
                "comp_box_pairs entries must be (var_idx, F_fn) or "
                "(var_idx, F_fn, F_jac_fn)"
            )
        if not (0 <= int(vi) < n):
            raise ValueError(
                f"comp_box_pairs: var_idx {vi} is out of range [0, {n})"
            )
        if not callable(F_fn):
            raise ValueError(
                f"comp_box_pairs: F_fn for var_idx={vi} must be callable"
            )
        if F_jac is not None and not callable(F_jac):
            raise ValueError(
                f"comp_box_pairs: F_jac_fn for var_idx={vi} must be callable or None"
            )
        ell = float(p.xl[int(vi)])
        u = float(p.xu[int(vi)])
        parsed.append((int(vi), F_fn, F_jac, ell, u))

    # Reject duplicate var_idx entries (ambiguous semantics).
    seen: set[int] = set()
    for vi, *_ in parsed:
        if vi in seen:
            raise ValueError(
                f"comp_box_pairs: var_idx={vi} appears in more than one entry"
            )
        seen.add(vi)

    # ---- Categorize by bound finiteness -------------------------------
    free_entries: list = []
    lower_entries: list = []
    upper_entries: list = []
    doubly_bounded_entries: list = []
    for vi, F_fn, F_jac, ell, u in parsed:
        ell_finite = bool(np.isfinite(ell))
        u_finite = bool(np.isfinite(u))
        if ell_finite and u_finite:
            if ell >= u:
                raise ValueError(
                    f"comp_box_pairs: var_idx={vi} has degenerate box "
                    f"[{ell}, {u}] (lower >= upper)."
                )
            doubly_bounded_entries.append((vi, F_fn, F_jac, ell, u))
            continue
        if ell_finite:
            lower_entries.append((vi, F_fn, F_jac, ell))
        elif u_finite:
            upper_entries.append((vi, F_fn, F_jac, u))
        else:
            free_entries.append((vi, F_fn, F_jac))

    # ---- Resolve any None F_jac_fn to a per-row forward fd callable ----
    def _resolve_F_jac(F_fn, F_jac):
        if F_jac is not None:
            return F_jac
        def _fd_row(x, _f=F_fn, _h=h_fd, _m=h_mode):
            return _fd_jac(_f, 1, n, _h, _m)(x)[0]
        return _fd_row

    # ---- Lower / upper-only → extend comp_G / comp_H block -------------
    if lower_entries or upper_entries:
        _extend_comp_with_box(p, lower_entries, upper_entries, _resolve_F_jac)

    # ---- Free → extend eq block ---------------------------------------
    if free_entries:
        _extend_eq_with_free(p, free_entries, _resolve_F_jac)

    # ---- Doubly-bounded → lift via Billups slack split ----------------
    # MUST run last: the lift extends ``n`` and wraps every user
    # callable (objective, gradient, comp_G/H, ineq, eq) onto the
    # original n-block, so any rebinding done by the helpers above
    # has to already be in place.
    if doubly_bounded_entries:
        _extend_with_doubly_bounded(p, doubly_bounded_entries, _resolve_F_jac)

    if p.comp_G is None:
        raise ValueError(
            "comp_box_pairs: every entry is free (both bounds infinite) "
            "and no base comp_G / comp_H is provided. The result is a "
            "pure square nonlinear system (CNS), not an MPCC. pympcc "
            "requires at least one complementarity pair; use a CNS / "
            "rootfinding solver if that is what you need."
        )


def _extend_comp_with_box(p: "MPCCProblem", lower_entries, upper_entries,
                          resolve_F_jac) -> None:
    """Append lower- and upper-only box-MCP rows to the comp_G/H block."""
    if (p.comp_G is not None
            and (p.comp_G_jacobian_sparsity is not None
                 or p.comp_H_jacobian_sparsity is not None)):
        raise NotImplementedError(
            "comp_box_pairs with sparse base comp_G/comp_H is not yet supported. "
            "Pass dense base Jacobians or wait for §4.9 Phase 2."
        )

    n = p.n
    n_lower = len(lower_entries)
    n_upper = len(upper_entries)
    n_box_comp = n_lower + n_upper

    lower_idx = np.array([vi for vi, *_ in lower_entries], dtype=np.intp)
    lower_ell = np.array([ell for *_, ell in lower_entries], dtype=float)
    upper_idx = np.array([vi for vi, *_ in upper_entries], dtype=np.intp)
    upper_u = np.array([u for *_, u in upper_entries], dtype=float)

    lower_F_fns = [F_fn for _, F_fn, *_ in lower_entries]
    upper_F_fns = [F_fn for _, F_fn, *_ in upper_entries]
    lower_F_jacs = [resolve_F_jac(F_fn, F_jac)
                    for _, F_fn, F_jac, _ in lower_entries]
    upper_F_jacs = [resolve_F_jac(F_fn, F_jac)
                    for _, F_fn, F_jac, _ in upper_entries]

    base_G = p.comp_G
    base_H = p.comp_H
    base_G_jac = p.comp_G_jacobian if callable(p.comp_G_jacobian) else None
    base_H_jac = p.comp_H_jacobian if callable(p.comp_H_jacobian) else None
    n_base = p.n_comp  # base count BEFORE we extend

    if base_G is not None:
        if base_G_jac is None or base_H_jac is None:
            raise ValueError(
                "comp_box_pairs requires comp_G_jacobian and comp_H_jacobian "
                "to be resolved callables when base comp_G / comp_H are set. "
                "Pass derivatives='fd' / 'jax' or supply Jacobians explicitly."
            )

        def _G(x, _bG=base_G, _li=lower_idx, _le=lower_ell,
               _ui=upper_idx, _uu=upper_u):
            base_vals = np.asarray(_bG(x), dtype=float).ravel()
            xv = np.asarray(x, dtype=float)
            box_lower = (xv[_li] - _le) if _li.size else np.empty(0)
            box_upper = (_uu - xv[_ui]) if _ui.size else np.empty(0)
            return np.concatenate([base_vals, box_lower, box_upper])

        def _H(x, _bH=base_H, _lf=lower_F_fns, _uf=upper_F_fns):
            base_vals = np.asarray(_bH(x), dtype=float).ravel()
            lower_vals = (np.array([np.asarray(f(x)).ravel()[0] for f in _lf])
                          if _lf else np.empty(0))
            upper_vals = (-np.array([np.asarray(f(x)).ravel()[0] for f in _uf])
                          if _uf else np.empty(0))
            return np.concatenate([base_vals, lower_vals, upper_vals])

        def _G_jac(x, _bJ=base_G_jac, _li=lower_idx, _ui=upper_idx,
                   _nb=n_base, _nbox=n_box_comp, _n=n):
            J = np.zeros((_nb + _nbox, _n))
            J[:_nb, :] = np.asarray(_bJ(x), dtype=float)
            for i in range(_li.size):
                J[_nb + i, _li[i]] = 1.0
            for i in range(_ui.size):
                J[_nb + _li.size + i, _ui[i]] = -1.0
            return J

        def _H_jac(x, _bJ=base_H_jac, _lj=lower_F_jacs, _uj=upper_F_jacs,
                   _nb=n_base, _nbox=n_box_comp, _n=n):
            J = np.zeros((_nb + _nbox, _n))
            J[:_nb, :] = np.asarray(_bJ(x), dtype=float)
            for i, jf in enumerate(_lj):
                J[_nb + i, :] = np.asarray(jf(x), dtype=float).ravel()
            for i, jf in enumerate(_uj):
                J[_nb + len(_lj) + i, :] = -np.asarray(jf(x), dtype=float).ravel()
            return J

        p.comp_G = _G
        p.comp_H = _H
        p.comp_G_jacobian = _G_jac
        p.comp_H_jacobian = _H_jac
    else:
        if n_base != 0:
            raise ValueError(
                f"comp_box_pairs: comp_G is None but n_comp={n_base}; "
                "with comp_box_pairs only (no base comp_G/H), set n_comp=0 "
                "(box-pair contributions are auto-counted)."
            )

        def _G_box(x, _li=lower_idx, _le=lower_ell, _ui=upper_idx, _uu=upper_u):
            xv = np.asarray(x, dtype=float)
            box_lower = (xv[_li] - _le) if _li.size else np.empty(0)
            box_upper = (_uu - xv[_ui]) if _ui.size else np.empty(0)
            return np.concatenate([box_lower, box_upper])

        def _H_box(x, _lf=lower_F_fns, _uf=upper_F_fns):
            lower_vals = (np.array([np.asarray(f(x)).ravel()[0] for f in _lf])
                          if _lf else np.empty(0))
            upper_vals = (-np.array([np.asarray(f(x)).ravel()[0] for f in _uf])
                          if _uf else np.empty(0))
            return np.concatenate([lower_vals, upper_vals])

        def _G_jac_box(x, _li=lower_idx, _ui=upper_idx, _nbox=n_box_comp, _n=n):
            J = np.zeros((_nbox, _n))
            for i in range(_li.size):
                J[i, _li[i]] = 1.0
            for i in range(_ui.size):
                J[_li.size + i, _ui[i]] = -1.0
            return J

        def _H_jac_box(x, _lj=lower_F_jacs, _uj=upper_F_jacs, _nbox=n_box_comp, _n=n):
            J = np.zeros((_nbox, _n))
            for i, jf in enumerate(_lj):
                J[i, :] = np.asarray(jf(x), dtype=float).ravel()
            for i, jf in enumerate(_uj):
                J[len(_lj) + i, :] = -np.asarray(jf(x), dtype=float).ravel()
            return J

        p.comp_G = _G_box
        p.comp_H = _H_box
        p.comp_G_jacobian = _G_jac_box
        p.comp_H_jacobian = _H_jac_box

    p.n_comp = n_base + n_box_comp


def _extend_eq_with_free(p: "MPCCProblem", free_entries, resolve_F_jac) -> None:
    """Append free-variable box-pair entries as ``F(x) = 0`` rows."""
    if p.eq_jacobian_sparsity is not None:
        raise NotImplementedError(
            "comp_box_pairs free entries with sparse base eq_jacobian "
            "are not yet supported."
        )

    n = p.n
    n_free = len(free_entries)
    free_F_fns = [F_fn for _, F_fn, *_ in free_entries]
    free_F_jacs = [resolve_F_jac(F_fn, F_jac)
                   for _, F_fn, F_jac in free_entries]

    n_eq_base = p.n_eq
    base_eq = p.eq_constraints
    base_eq_jac = p.eq_jacobian if callable(p.eq_jacobian) else None

    if base_eq is not None and base_eq_jac is None:
        raise ValueError(
            "comp_box_pairs free entries require eq_jacobian to be a "
            "resolved callable when base eq_constraints is set. "
            "Pass derivatives='fd' / 'jax' or supply eq_jacobian explicitly."
        )

    if base_eq is not None:
        def _eq(x, _be=base_eq, _ffs=free_F_fns):
            base_vals = np.asarray(_be(x), dtype=float).ravel()
            free_vals = np.array([np.asarray(f(x)).ravel()[0] for f in _ffs])
            return np.concatenate([base_vals, free_vals])

        def _eq_jac(x, _bJ=base_eq_jac, _jfs=free_F_jacs,
                    _nb=n_eq_base, _nf=n_free, _n=n):
            J = np.zeros((_nb + _nf, _n))
            J[:_nb, :] = np.asarray(_bJ(x), dtype=float)
            for i, jf in enumerate(_jfs):
                J[_nb + i, :] = np.asarray(jf(x), dtype=float).ravel()
            return J

        p.eq_constraints = _eq
        p.eq_jacobian = _eq_jac
    else:
        if n_eq_base != 0:
            raise ValueError(
                f"comp_box_pairs: eq_constraints is None but n_eq={n_eq_base}; "
                "with free box-pair entries only, set n_eq=0 (free "
                "contributions are auto-counted)."
            )

        def _eq_free(x, _ffs=free_F_fns):
            return np.array([np.asarray(f(x)).ravel()[0] for f in _ffs])

        def _eq_jac_free(x, _jfs=free_F_jacs, _nf=n_free, _n=n):
            J = np.zeros((_nf, _n))
            for i, jf in enumerate(_jfs):
                J[i, :] = np.asarray(jf(x), dtype=float).ravel()
            return J

        p.eq_constraints = _eq_free
        p.eq_jacobian = _eq_jac_free

    p.n_eq = n_eq_base + n_free


def _extend_with_doubly_bounded(p: "MPCCProblem", db_entries: list,
                                resolve_F_jac) -> None:
    """Lift doubly-bounded ``ℓ ≤ x[j] ≤ u  ⊥  F(x)`` entries via the
    Billups / Mangasarian slack split.

    For each entry ``(j, F_fn, F_jac_fn)`` with finite ``xl[j] = ℓ``
    and ``xu[j] = u``, the lift introduces two new variables
    ``s₋, s₊ ≥ 0`` and rewrites the box-MCP as

    * one equality row  ``F(x) − s₋ + s₊ = 0``;
    * two complementarity pairs:
          ``(x[j] − ℓ) ≥ 0  ⊥  s₋ ≥ 0``
          ``(u − x[j]) ≥ 0  ⊥  s₊ ≥ 0``.

    Why this works: at ``x[j] = ℓ`` the second pair forces ``s₊ = 0``
    and the first leaves ``s₋`` free, giving ``F = s₋ ≥ 0``.
    At ``x[j] = u``, by symmetry, ``F = −s₊ ≤ 0``.
    Strictly inside the box both pairs force ``s₋ = s₊ = 0`` so
    ``F = 0``.  These are exactly the PATH semantics for the
    doubly-bounded MCP.

    The lift is universal: every existing strategy sees a standard
    :class:`MPCCProblem` and does not need to know about it.
    Originally-named decision variables stay at indices ``0..n_orig``
    in the lifted ``x`` vector, so users can reach them via
    ``result.x[:problem.n_orig_doubly_bounded]`` (the slack tail
    lives at ``result.x[n_orig:]``).
    """
    if p.is_sparse:
        raise NotImplementedError(
            "comp_box_pairs doubly-bounded entries do not support sparse "
            "base Jacobians in this release. Pass dense Jacobians."
        )

    n_orig = p.n
    k = len(db_entries)
    n_new = n_orig + 2 * k

    j_idxs = np.array([entry[0] for entry in db_entries], dtype=np.intp)
    ells = np.array([entry[3] for entry in db_entries], dtype=float)
    us = np.array([entry[4] for entry in db_entries], dtype=float)
    F_fns = [entry[1] for entry in db_entries]
    F_jacs = [resolve_F_jac(entry[1], entry[2]) for entry in db_entries]

    # Slack columns interleave per entry: [s₋_0, s₊_0, s₋_1, s₊_1, ...].
    s_minus_cols = np.arange(n_orig, n_new, 2, dtype=np.intp)
    s_plus_cols = np.arange(n_orig + 1, n_new, 2, dtype=np.intp)

    # ---- Variable bounds + x0 extension --------------------------------
    assert p.xl is not None and p.xu is not None
    p.xl = np.concatenate([p.xl, np.zeros(2 * k)])
    p.xu = np.concatenate([p.xu, np.full(2 * k, np.inf)])
    p.x0 = np.concatenate([p.x0, np.zeros(2 * k)])
    p.n = n_new
    p.n_orig_doubly_bounded = n_orig
    p.n_doubly_bounded_pairs = k

    # ---- Wrap objective + gradient -------------------------------------
    base_obj = p.objective
    base_grad = p.gradient
    if not callable(base_grad):
        raise ValueError(
            "comp_box_pairs doubly-bounded entries require a resolved "
            "gradient callable. Pass derivatives='fd' or 'jax', or "
            "supply gradient explicitly."
        )

    def _obj_lifted(x, _bo=base_obj, _no=n_orig):
        return _bo(np.asarray(x)[:_no])

    def _grad_lifted(x, _bg=base_grad, _no=n_orig, _nn=n_new):
        out = np.zeros(_nn)
        out[:_no] = np.asarray(_bg(np.asarray(x)[:_no]), dtype=float).ravel()
        return out

    p.objective = _obj_lifted
    p.gradient = _grad_lifted

    # ---- Wrap comp_G / comp_H + Jacobians; append 2k new pairs --------
    base_G = p.comp_G  # may be None when only doubly-bounded entries are present
    base_H = p.comp_H
    base_G_jac = p.comp_G_jacobian if callable(p.comp_G_jacobian) else None
    base_H_jac = p.comp_H_jacobian if callable(p.comp_H_jacobian) else None
    n_comp_base = p.n_comp

    if base_G is not None and (base_G_jac is None or base_H_jac is None):
        raise ValueError(
            "comp_box_pairs doubly-bounded entries require comp_G_jacobian "
            "and comp_H_jacobian to be resolved callables when base "
            "comp_G / comp_H are set. Pass derivatives='fd' or 'jax', or "
            "supply Jacobians explicitly."
        )
    if base_G is None and n_comp_base != 0:
        raise ValueError(
            f"comp_box_pairs: comp_G is None but n_comp={n_comp_base}; "
            "with doubly-bounded entries only (no base comp_G/H), set "
            "n_comp=0 (box-pair contributions are auto-counted)."
        )

    def _comp_G_lifted(x, _bG=base_G, _ji=j_idxs, _ell=ells, _us=us,
                       _no=n_orig, _k=k):
        x = np.asarray(x)
        x_orig = x[:_no]
        base_vals = (np.asarray(_bG(x_orig), dtype=float).ravel()
                     if _bG is not None else np.empty(0))
        new_vals = np.empty(2 * _k)
        new_vals[0::2] = x_orig[_ji] - _ell
        new_vals[1::2] = _us - x_orig[_ji]
        return np.concatenate([base_vals, new_vals])

    def _comp_H_lifted(x, _bH=base_H, _smi=s_minus_cols, _spi=s_plus_cols,
                       _no=n_orig, _k=k):
        x = np.asarray(x)
        x_orig = x[:_no]
        base_vals = (np.asarray(_bH(x_orig), dtype=float).ravel()
                     if _bH is not None else np.empty(0))
        new_vals = np.empty(2 * _k)
        new_vals[0::2] = x[_smi]   # s₋
        new_vals[1::2] = x[_spi]   # s₊
        return np.concatenate([base_vals, new_vals])

    def _comp_G_jac_lifted(x, _bJ=base_G_jac, _ji=j_idxs, _no=n_orig,
                           _nn=n_new, _ncb=n_comp_base, _k=k):
        J = np.zeros((_ncb + 2 * _k, _nn))
        if _bJ is not None and _ncb > 0:
            J[:_ncb, :_no] = np.asarray(_bJ(np.asarray(x)[:_no]), dtype=float)
        for i in range(_k):
            J[_ncb + 2 * i, _ji[i]] = 1.0       # ∂(x[j]-ℓ)/∂x[j]
            J[_ncb + 2 * i + 1, _ji[i]] = -1.0  # ∂(u-x[j])/∂x[j]
        return J

    def _comp_H_jac_lifted(x, _bJ=base_H_jac, _smi=s_minus_cols,
                           _spi=s_plus_cols, _no=n_orig, _nn=n_new,
                           _ncb=n_comp_base, _k=k):
        J = np.zeros((_ncb + 2 * _k, _nn))
        if _bJ is not None and _ncb > 0:
            J[:_ncb, :_no] = np.asarray(_bJ(np.asarray(x)[:_no]), dtype=float)
        for i in range(_k):
            J[_ncb + 2 * i, _smi[i]] = 1.0       # ∂s₋/∂s₋
            J[_ncb + 2 * i + 1, _spi[i]] = 1.0   # ∂s₊/∂s₊
        return J

    p.comp_G = _comp_G_lifted
    p.comp_H = _comp_H_lifted
    p.comp_G_jacobian = _comp_G_jac_lifted
    p.comp_H_jacobian = _comp_H_jac_lifted
    p.n_comp = n_comp_base + 2 * k

    # ---- Wrap eq_constraints + Jacobian; append k new rows ------------
    base_eq = p.eq_constraints
    base_eq_jac = p.eq_jacobian if callable(p.eq_jacobian) else None
    n_eq_base = p.n_eq

    if base_eq is not None and base_eq_jac is None:
        raise ValueError(
            "comp_box_pairs doubly-bounded entries require eq_jacobian "
            "to be a resolved callable when base eq_constraints is set. "
            "Pass derivatives='fd' or 'jax', or supply eq_jacobian "
            "explicitly."
        )

    def _eq_lifted(x, _be=base_eq, _ffns=F_fns, _smi=s_minus_cols,
                   _spi=s_plus_cols, _no=n_orig, _k=k):
        x = np.asarray(x)
        x_orig = x[:_no]
        base_vals = (np.asarray(_be(x_orig), dtype=float).ravel()
                     if _be is not None else np.empty(0))
        new_vals = np.empty(_k)
        for i, F in enumerate(_ffns):
            new_vals[i] = (
                float(np.asarray(F(x_orig)).ravel()[0])
                - x[_smi[i]] + x[_spi[i]]
            )
        return np.concatenate([base_vals, new_vals])

    def _eq_jac_lifted(x, _bJ=base_eq_jac, _Fjacs=F_jacs,
                       _smi=s_minus_cols, _spi=s_plus_cols, _no=n_orig,
                       _nn=n_new, _neb=n_eq_base, _k=k):
        x_orig = np.asarray(x)[:_no]
        J = np.zeros((_neb + _k, _nn))
        if _bJ is not None and _neb > 0:
            J[:_neb, :_no] = np.asarray(_bJ(x_orig), dtype=float)
        for i, jf in enumerate(_Fjacs):
            J[_neb + i, :_no] = np.asarray(jf(x_orig), dtype=float).ravel()
            J[_neb + i, _smi[i]] = -1.0
            J[_neb + i, _spi[i]] = 1.0
        return J

    p.eq_constraints = _eq_lifted
    p.eq_jacobian = _eq_jac_lifted
    p.n_eq = n_eq_base + k

    # ---- Wrap ineq_constraints + Jacobian (zero-pad slack columns) ----
    if p.n_ineq > 0 and p.ineq_constraints is not None:
        base_ineq = p.ineq_constraints
        base_ineq_jac = (p.ineq_jacobian
                         if callable(p.ineq_jacobian) else None)
        if base_ineq_jac is None:
            raise ValueError(
                "comp_box_pairs doubly-bounded entries require "
                "ineq_jacobian to be a resolved callable when "
                "ineq_constraints is set."
            )
        n_ineq = p.n_ineq

        def _ineq_lifted(x, _bi=base_ineq, _no=n_orig):
            return _bi(np.asarray(x)[:_no])

        def _ineq_jac_lifted(x, _bJ=base_ineq_jac, _no=n_orig,
                             _nn=n_new, _ng=n_ineq):
            J = np.zeros((_ng, _nn))
            J[:, :_no] = np.asarray(_bJ(np.asarray(x)[:_no]), dtype=float)
            return J

        p.ineq_constraints = _ineq_lifted
        p.ineq_jacobian = _ineq_jac_lifted


__all__ = [
    "normalize_var_pairs",
    "apply_var_pairs_bulk",
    "normalize_box_pairs",
]
