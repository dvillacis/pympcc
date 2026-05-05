"""Abstract base class for MPCC reformulation strategies.

Owns ``BaseStrategy``, the dataclass-flavoured base used by every
ε-continuation strategy.  The ε-continuation harness, the active-set
cleanup polish, and the safeguard option-parsing live in three sibling
mixins (`_continuation_mixin`, `_cleanup_mixin`, `_safeguards_mixin`)
and are folded in via the ``BaseStrategy(SafeguardsMixin,
CleanupMixin, ContinuationMixin, ABC)`` MRO.  ``BaseStrategy``
otherwise carries the constraint / Jacobian-evaluation helpers, the
NLP build, the cyipopt timed-solve adapter, and KKT diagnostics.
"""
from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from .._constants import (
    INNER_TOL_FLOOR as _INNER_TOL_FLOOR,
)
from .._constants import (
    KKT_TERMINATION_TOL as _KKT_TERMINATION_TOL,
)
from .._kernels import coo_to_dense as _coo_kernel
from .._kernels import eval_weighted_union as _wu_kernel
from .._kernels import weighted_row_sum as _wrs_kernel
from .._stationarity import compute_kkt_residual as _compute_kkt_residual
from .._typing import BackendName
from ..problem import MPCCProblem
from ..result import IPOPTStatus, MPCCResult
from ._cleanup_mixin import CleanupMixin
from ._continuation_mixin import ContinuationMixin
from ._safeguards_mixin import SafeguardsMixin

# Safeguard option defaults.  Merged into each ε-continuation strategy's
# ``_DEFAULTS`` so that callers can pass these as ``strategy_options``.
# Defaults preserve the previous (un-safeguarded) outer-loop behaviour.
# Active-set cleanup option defaults.  Merged into each ε-continuation
# strategy's ``_DEFAULTS`` so callers can pass them as ``strategy_options``.
# Defaults preserve the previous (no-cleanup) behaviour.
CLEANUP_DEFAULTS: dict = dict(
    cleanup=False,                  # False | True | "auto"
    cleanup_biactive_tol=None,      # None ⇒ derive from comp_residual
    cleanup_active_set=None,        # None | (I_G_active, I_H_active)
    cleanup_obj_worsen_tol=1e-3,    # accept iff Δobj/(|obj|+1) ≤ this; 1e-3
                                    # tolerates the tiny objective slip a polished
                                    # active-set NLP often produces vs. the
                                    # ε-relaxed iterate, while rejecting genuine
                                    # cleanup-induced regressions on MacMPEC.
    cleanup_max_iter=300,           # IPOPT max_iter for the cleanup NLP
    cleanup_tol=None,               # None ⇒ max(ipopt_options['tol'], 1e-6);
                                    # set explicitly to override.  L-BFGS on
                                    # large problems rarely converges below
                                    # 1e-6 in 300 iters, so the default
                                    # avoids spurious MAX_ITER rejections.
)


SAFEGUARD_DEFAULTS: dict = dict(
    safeguards=None,                # "all" turns on all five safeguards at once
    safeguard_rollback=False,       # snapshot/rollback when inner solve misbehaves
    safeguard_adaptive_eps=False,   # cautious ε reduction when last solve was loose
    safeguard_kkt_termination=False,  # break early on small MPCC-KKT residual
    safeguard_plateau=False,        # break early when (obj, comp) stop improving
    kkt_tol=_KKT_TERMINATION_TOL,   # threshold for safeguard_kkt_termination
    inner_tol_mode="linear",        # "linear" | "quadratic" (Leyffer) | "matched"
    inner_tol_factor=0.1,           # ``matched`` mode: tol = factor · ε
    inner_tol_floor=_INNER_TOL_FLOOR,  # ``matched`` mode: hard floor on tol
    comp_eps_ratio_theta=10.0,      # "tracked ε" means comp ≤ θ·ε.  θ=10 lets
                                    # IPOPT lag one order of magnitude behind ε
                                    # before adaptive_eps treats the iterate as
                                    # "loose" — tighter values cause spurious
                                    # holds early in continuation; looser values
                                    # let comp residual drift on hard MacMPEC.
    rollback_lambda_jump=1e3,       # reject if ‖mult_g‖∞ jumps more than this
    rollback_max_count=3,           # abort after this many consecutive rollbacks
    eps_hold_factor=3.0,            # ε multiplier on rollback (>1 backs off
                                    # toward last good ε; capped at
                                    # last_good_eps).  3.0 ≈ one Scholtes step
                                    # of reduction=0.1: backing off by exactly
                                    # the same factor that reduction=1/3 would
                                    # have advanced, which empirically resolves
                                    # transient restoration without ping-pong.
    restoration_iter_threshold=5,   # rollback if IPOPT spent ≥ this many iters in restoration
    pre_post_blowup_factor=10.0,    # rollback if post-NLP comp residual > factor × pre-NLP comp residual
    plateau_tol_obj=1e-4,           # relative |Δobj| below this counts as plateau
    plateau_tol_comp=1e-3,          # relative |Δcomp_residual| below this counts as plateau
    plateau_window=2,               # consecutive plateau iters required to terminate
    plateau_comp_target=1e-4,       # plateau breaks only fire when comp_residual ≤ this
)


class BaseStrategy(SafeguardsMixin, CleanupMixin, ContinuationMixin, ABC):
    """
    Abstract base for MPCC reformulation strategies.

    Each concrete strategy transforms the MPCC into one or more standard NLPs
    and delegates the actual solving to cyipopt via :class:`_DenseNLP`.
    """

    name: str = "base"

    # Declared here for type checking; set by iterative-continuation subclasses.
    epsilon_0: float
    epsilon_min: float
    max_iter: int
    dual_warmstart: bool
    comp_tol: float | None
    reduction: float

    def __init__(self, problem: MPCCProblem, ipopt_options: dict, *,
                 backend: BackendName = "ipopt",
                 solver_options: dict | None = None,
                 linear_solver_fn=None, **kwargs) -> None:
        self.problem = problem
        self.ipopt_options = ipopt_options
        self.backend = backend
        self.solver_options = solver_options or {}
        self._linear_solver_fn = linear_solver_fn
        self.callback = kwargs.pop("callback", None)
        self.inner_callback = kwargs.pop("inner_callback", None)
        self.time_limit: float | None = kwargs.pop("time_limit", None)
        self._time_limit_hit = False
        # Outer-solve start time, set by ``_run_epsilon_continuation`` and
        # consulted by ``_maybe_run_cleanup`` so cleanup respects the same
        # wall-clock budget.  ``None`` for single-shot strategies.
        self._wall_t0 = None
        # Hot-start machinery (§6.5).  ``_initial_warm_dual`` is a one-shot
        # seed consumed by the strategy's first ``nlp.solve`` call when
        # :meth:`MPCCSolver.resolve` injects state from a previous solve.
        # ``_last_solve_state`` is refreshed at the end of every
        # ``_timed_solve`` so :meth:`MPCCSolver.resolve` can fish out the
        # final-iterate multipliers regardless of which strategy ran.
        self._initial_warm_dual = None
        self._last_solve_state: dict | None = None
        # Strategies that accept no extra kwargs (e.g. DirectStrategy) inherit
        # this base __init__; unknown kwargs are silently ignored so that
        # callers can always pass e.g. epsilon_0/max_iter without branching.

        # Pre-gather complementarity-pair scales by sparsity rows so that
        # _eval_comp_jac_raw can apply per-row scaling with one elementwise
        # multiply (sparse path) instead of an indexed gather every call.
        self._sG_jac_flat: np.ndarray | None = None
        self._sH_jac_flat: np.ndarray | None = None
        if problem.has_comp_scale:
            if (problem.comp_G_scale is not None
                    and problem.comp_G_jacobian_sparsity is not None):
                self._sG_jac_flat = np.asarray(
                    problem.comp_G_scale)[problem.comp_G_jacobian_sparsity[0]]
            if (problem.comp_H_scale is not None
                    and problem.comp_H_jacobian_sparsity is not None):
                self._sH_jac_flat = np.asarray(
                    problem.comp_H_scale)[problem.comp_H_jacobian_sparsity[0]]

    @abstractmethod
    def solve(self) -> MPCCResult:
        """Solve the MPCC and return the result."""

    # ------------------------------------------------------------------ #
    # Helpers shared by all strategies                                     #
    # ------------------------------------------------------------------ #

    def _init_continuation_options(
        self,
        problem: MPCCProblem,
        ipopt_options: dict,
        defaults: dict,
        kwargs: dict,
    ) -> None:
        """Shared init boilerplate for ε-continuation strategies.

        Single source of truth for the option-merge ordering used by
        Scholtes / Smoothing / Lin-Fukushima / Slack / NCP-base.  Pops the
        five standard ``BaseStrategy`` keyword arguments (``backend``,
        ``solver_options``, ``callback``, ``inner_callback``,
        ``time_limit``) into :meth:`BaseStrategy.__init__`, merges the
        remaining ``kwargs`` over ``defaults``, validates the canonical
        ε-continuation options, and runs the safeguard / cleanup setup
        hooks.

        The caller's ``kwargs`` dict is mutated in-place: the five
        ``BaseStrategy`` keys are popped so ``defaults | kwargs`` does
        not pull them back in.

        Subclasses with additional strategy-specific kwargs (CCK ``lam``,
        Billups ``gamma``, etc.) must pop those *before* calling this
        helper so the merge picks up the correct continuation defaults
        only.
        """
        BaseStrategy.__init__(
            self, problem, ipopt_options,
            backend=kwargs.pop("backend", "ipopt"),
            solver_options=kwargs.pop("solver_options", None),
            callback=kwargs.pop("callback", None),
            inner_callback=kwargs.pop("inner_callback", None),
            time_limit=kwargs.pop("time_limit", None),
        )
        opts = {**defaults, **kwargs}
        opts = self._maybe_resolve_auto_epsilon_0(opts)
        self._validate_continuation_options(
            epsilon_0=opts["epsilon_0"],
            reduction=opts["reduction"],
            max_iter=opts["max_iter"],
            epsilon_min=opts["epsilon_min"],
            comp_tol=opts["comp_tol"],
        )
        self.epsilon_0    = float(opts["epsilon_0"])
        self.reduction    = float(opts["reduction"])
        self.max_iter     = int(opts["max_iter"])
        self.epsilon_min  = float(opts["epsilon_min"])
        self.dual_warmstart = bool(opts["dual_warmstart"])
        self.comp_tol     = opts["comp_tol"]
        self._init_safeguards(opts)
        self._init_cleanup(opts, user_kwargs=kwargs)

    def _build_nlp(
        self,
        cl: np.ndarray,
        cu: np.ndarray,
        con_fn,
        jac_fn,
        jac_structure=None,
        obj_fn=None,
        grad_fn=None,
        hess_fn=None,
        hess_sparsity=None,
        linear_solver_fn=None,
    ):
        """
        Construct and configure an NLP for this problem.

        The active backend (``self.backend``) determines which adapter is
        built:

        * ``"ipopt"`` (default) — builds :class:`_DenseNLP` or
          :class:`_SparseNLP` (cyipopt adapter, current behaviour).
          If *jac_structure* ``(rows, cols)`` is provided, the sparse
          variant is used; otherwise the dense one.  When *hess_fn* and
          *hess_sparsity* are given the NLP class is dynamically extended
          with :class:`_HessianMixin`.

        * ``"filterSQP"`` — builds a :class:`~pyfiltersqp._FilterSQPAdapter`
          that presents the same ``solve()`` / ``add_option()`` interface.
          When *jac_structure* is provided (sparse-native path), the Jacobian
          is transparently densified before being handed to the adapter.

        *obj_fn* and *grad_fn* override ``problem.objective`` and
        ``problem.gradient`` respectively (useful for augmented-Lagrangian
        strategies that augment the objective each iteration).
        """
        p = self.problem
        _obj = obj_fn if obj_fn is not None else p.objective
        _grad = grad_fn if grad_fn is not None else p.gradient

        # ------------------------------------------------------------------ #
        # filterSQP backend                                                    #
        # ------------------------------------------------------------------ #
        if self.backend == "filterSQP":
            try:
                from .._filtersqp_adapter import _FilterSQPAdapter
            except ImportError as exc:
                raise ImportError(
                    "backend='filterSQP' requires the pyfiltersqp package. "
                    "Install it via `pip install pyfiltersqp` (or the "
                    "`pympcc[filtersqp]` extra) or switch to backend='ipopt'."
                ) from exc
            _jac_fn = jac_fn
            if jac_structure is not None:
                import scipy.sparse as _sp
                _m, _n = len(cl), p.n
                _rows = np.asarray(jac_structure[0])
                _cols = np.asarray(jac_structure[1])
                _sparse_jac_fn = jac_fn
                def _jac_fn(x, _r=_rows, _c=_cols, _m=_m, _n=_n,
                            _fn=_sparse_jac_fn):
                    vals = np.asarray(_fn(x), dtype=float)
                    if vals.ndim == 1:
                        return _sp.csr_matrix((vals, (_r, _c)), shape=(_m, _n))
                    return vals
            adapter = _FilterSQPAdapter(
                n=p.n, m=len(cl),
                xl=p.xl, xu=p.xu, cl=cl, cu=cu,  # type: ignore[arg-type]
                obj_fn=_obj, grad_fn=_grad,  # type: ignore[arg-type]
                con_fn=con_fn, jac_fn=_jac_fn,
                solver_options=self.solver_options,
            )
            for key, val in self.ipopt_options.items():
                adapter.add_option(key, val)
            return adapter

        # ------------------------------------------------------------------ #
        # scipy backend                                                        #
        # ------------------------------------------------------------------ #
        if self.backend == "scipy":  # pragma: no cover
            from .._scipy_adapter import _ScipyAdapter
            return _ScipyAdapter(
                n=p.n, m=len(cl),
                xl=p.xl, xu=p.xu, cl=cl, cu=cu,  # type: ignore[arg-type]
                obj_fn=_obj, grad_fn=_grad,
                con_fn=con_fn, jac_fn=jac_fn,
                jac_rows=jac_structure[0] if jac_structure is not None else None,
                jac_cols=jac_structure[1] if jac_structure is not None else None,
                solver_options=self.solver_options,
            )

        # ------------------------------------------------------------------ #
        # IPOPT backend (default)                                              #
        # ------------------------------------------------------------------ #
        from .._nlp import _DenseNLP, _HessianMixin, _SparseNLP

        kwargs = dict(
            n=p.n, m=len(cl), xl=p.xl, xu=p.xu, cl=cl, cu=cu,
            obj_fn=_obj, grad_fn=_grad,
            con_fn=con_fn, jac_fn=jac_fn,
            hess_fn=hess_fn, hess_sparsity=hess_sparsity,
            inner_callback=self.inner_callback,
        )
        base: type[_DenseNLP] | type[_SparseNLP]
        # Resolve linear_solver_fn: explicit arg > instance attribute
        _lsf = linear_solver_fn if linear_solver_fn is not None else self._linear_solver_fn
        if jac_structure is not None:
            base = _SparseNLP
            extra = dict(jac_rows=jac_structure[0], jac_cols=jac_structure[1])
            if _lsf is not None:
                extra["linear_solver_fn"] = _lsf
        else:
            base = _DenseNLP
            extra = {}

        if hess_fn is not None:
            cls = type("_NLPWithHess", (_HessianMixin, base), {})
        else:
            cls = base

        nlp = cls(**kwargs, **extra)
        for key, val in self.ipopt_options.items():
            nlp.add_option(key, val)
        return nlp

    def _has_jax_hessian(self) -> bool:
        """True if the problem requests exact JAX Lagrangian Hessians."""
        return getattr(self.problem, "use_jax_hessian", False)

    def _has_manual_hessian(self) -> bool:
        """True if the problem supplies an exact Lagrangian Hessian callback."""
        return getattr(self.problem, "lagrangian_hessian", None) is not None

    @staticmethod
    def _decode_msg(msg) -> str:
        """Decode cyipopt status_msg (bytes or str)."""
        return msg.decode() if isinstance(msg, bytes) else str(msg)

    @staticmethod
    def _to_dense_block(
        values_or_dense,
        sparsity,
        n_rows: int,
        n: int,
    ) -> np.ndarray:
        """
        Return a dense ``(n_rows, n)`` array from either a dense matrix or
        a flat 1-D values array paired with a COO sparsity structure.
        """
        if sparsity is None:
            return np.asarray(values_or_dense, dtype=float)
        rows, cols = sparsity
        dense = np.zeros((n_rows, n))
        _coo_kernel(
            np.asarray(rows, dtype=np.intp),
            np.asarray(cols, dtype=np.intp),
            np.asarray(values_or_dense, dtype=float),
            dense,
        )
        return dense

    @staticmethod
    def _new_callback_cache() -> dict:
        """Create a small per-NLP callback cache keyed by the current x value."""
        return {"x": None}

    @staticmethod
    def _ensure_cache_current(cache: dict | None, x: np.ndarray) -> bool:
        """
        Return True when *cache* already belongs to *x*; otherwise reset it.

        IPOPT may pass mutable arrays to callbacks, so the cache stores a copy
        of the last x instead of relying on object identity.
        """
        if cache is None:
            return False
        x_ref = cache.get("x")
        if x_ref is not None and np.array_equal(x_ref, x):
            return True
        cache.clear()
        cache["x"] = np.asarray(x, dtype=float).copy()
        return False

    def _eval_comp_values(
        self, x: np.ndarray, cache: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate or retrieve cached complementarity values ``(G, H)``.

        When ``problem.comp_G_scale`` / ``comp_H_scale`` are set, the
        returned values are scaled element-wise.  The user's underlying
        ``comp_G(x)`` / ``comp_H(x)`` buffers are never mutated; a fresh
        scaled array is allocated.
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "comp_values" in cache:
            return cache["comp_values"]
        G = np.asarray(p.comp_G(x))  # type: ignore[misc]
        H = np.asarray(p.comp_H(x))  # type: ignore[misc]
        if p.comp_G_scale is not None:
            G = G * p.comp_G_scale
        if p.comp_H_scale is not None:
            H = H * p.comp_H_scale
        values = (G, H)
        if cache is not None:
            cache["comp_values"] = values
        return values

    def _eval_comp_jac_raw(
        self, x: np.ndarray, cache: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate or retrieve cached raw complementarity Jacobian outputs.

        Per-row scaling (``s_G_i`` / ``s_H_i``) is applied here so every
        downstream consumer (sparse assembly, dense densification,
        weighted-row kernels) sees a consistently scaled Jacobian.
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "comp_jac_raw" in cache:
            return cache["comp_jac_raw"]
        vG = np.asarray(p.comp_G_jacobian(x), dtype=float)  # type: ignore[misc, operator]
        vH = np.asarray(p.comp_H_jacobian(x), dtype=float)  # type: ignore[misc, operator]
        if p.comp_G_scale is not None:
            if vG.ndim == 2:
                vG = vG * p.comp_G_scale[:, None]
            else:
                vG = vG * self._sG_jac_flat
        if p.comp_H_scale is not None:
            if vH.ndim == 2:
                vH = vH * p.comp_H_scale[:, None]
            else:
                vH = vH * self._sH_jac_flat
        values = (vG, vH)
        if cache is not None:
            cache["comp_jac_raw"] = values
        return values

    @staticmethod
    def _union_sparsity(s1, s2):
        """
        Union of two COO patterns, sorted in row-major order.
        Returns ``None`` (dense) if either input is ``None``.
        """
        if s1 is None or s2 is None:
            return None
        r = np.concatenate([np.asarray(s1[0]), np.asarray(s2[0])])
        c = np.concatenate([np.asarray(s1[1]), np.asarray(s2[1])])
        order = np.lexsort((c, r))
        r, c = r[order], c[order]
        mask = np.concatenate([[True], (r[1:] != r[:-1]) | (c[1:] != c[:-1])])
        return r[mask], c[mask]

    def _build_comp_jacobians(
        self, x: np.ndarray, cache: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate complementarity functions and Jacobians.

        Returns ``(G, H, JG, JH)`` where ``JG`` and ``JH`` are always dense
        ``(n_comp, n)`` arrays regardless of whether the problem uses sparse
        structures.
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "comp_jac_dense" in cache:
            return cache["comp_jac_dense"]
        G, H = self._eval_comp_values(x, cache)
        vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
        JG = self._to_dense_block(
            vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n
        )
        JH = self._to_dense_block(
            vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n
        )
        values = (G, H, JG, JH)
        if cache is not None:
            cache["comp_jac_dense"] = values
        return values

    def _make_jac_structure(
        self,
        blocks: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assemble global NLP Jacobian ``(rows, cols)`` from a block list.

        Each element of *blocks* is ``(n_block_rows, sparsity_or_None)``.
        Dense blocks (``None``) contribute all ``n`` columns; sparse blocks
        use the supplied COO indices.
        """
        p = self.problem
        all_rows: list = []
        all_cols: list = []
        row_offset = 0
        for n_block_rows, sparsity in blocks:
            if n_block_rows > 0:
                if sparsity is None:
                    rows = np.repeat(
                        np.arange(row_offset, row_offset + n_block_rows), p.n
                    )
                    cols = np.tile(np.arange(p.n), n_block_rows)
                else:
                    rows = np.asarray(sparsity[0]) + row_offset
                    cols = np.asarray(sparsity[1])
                all_rows.append(rows)
                all_cols.append(cols)
            row_offset += n_block_rows
        return np.concatenate(all_rows), np.concatenate(all_cols)

    def _make_union_maps(self, s1, s2):
        """
        Compute union COO pattern and index maps for two sparsity patterns.

        Returns ``(union_sp, map1, map2)`` or ``None`` if either input is ``None``.

        - ``union_sp = (rows, cols)`` — sorted row-major
        - ``map1[k]`` = flat index in s1 values array for union entry k (-1 if absent)
        - ``map2[k]`` = same for s2
        """
        if s1 is None or s2 is None:
            return None
        n = self.problem.n
        r1 = np.asarray(s1[0], dtype=np.intp)
        c1 = np.asarray(s1[1], dtype=np.intp)
        r2 = np.asarray(s2[0], dtype=np.intp)
        c2 = np.asarray(s2[1], dtype=np.intp)
        r_all = np.concatenate([r1, r2])
        c_all = np.concatenate([c1, c2])
        order = np.lexsort((c_all, r_all))
        r_sort, c_sort = r_all[order], c_all[order]
        unique_mask = np.concatenate(
            [[True], (r_sort[1:] != r_sort[:-1]) | (c_sort[1:] != c_sort[:-1])]
        )
        r_u, c_u = r_sort[unique_mask], c_sort[unique_mask]
        nnz_u = len(r_u)
        key_u = r_u * n + c_u   # sorted; n is a safe stride (all col indices < n)
        map1 = np.full(nnz_u, -1, dtype=np.intp)
        map2 = np.full(nnz_u, -1, dtype=np.intp)
        if len(r1):
            map1[np.searchsorted(key_u, r1 * n + c1)] = np.arange(len(r1))
        if len(r2):
            map2[np.searchsorted(key_u, r2 * n + c2)] = np.arange(len(r2))
        return (r_u, c_u), map1, map2

    @staticmethod
    def _eval_weighted_union(
        v_G: np.ndarray,
        v_H: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        r_u: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute flat values of ``(alpha_i * JG + beta_i * JH)`` at union
        positions without allocating a dense matrix.

        Derived-block callers:

        - G·H block : ``alpha=H``, ``beta=G``
        - G+H block : ``alpha=beta=ones``
        - φ_ε block : ``alpha=1-G/r``, ``beta=1-H/r``

        Parameters
        ----------
        out : ndarray of shape (nnz_union,), optional
            Pre-allocated output buffer.  When supplied the kernel writes
            directly into it (zero allocation on the hot path).  When
            omitted a fresh array is allocated.

        Returns
        -------
        out : ndarray, shape (nnz_union,)
        """
        if out is None:
            out = np.empty(len(r_u))
        _wu_kernel(v_G, v_H, alpha, beta, r_u, map1, map2, out)
        return out

    def _eval_standard_con_values(
        self, x: np.ndarray, cache: dict | None = None
    ) -> list[np.ndarray]:
        """
        Evaluate standard constraint *values* only — no Jacobians.

        Use this in ``constraints(x)`` callbacks where the Jacobian is
        not needed.  Calling ``_build_standard_constraints`` from a
        constraints callback unnecessarily invokes the user's Jacobian
        callables and allocates dense ``(n_rows, n)`` blocks that are
        immediately discarded.
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "std_values" in cache:
            return list(cache["std_values"])
        parts: list[np.ndarray] = []
        if p.n_ineq > 0:
            parts.append(np.asarray(p.ineq_constraints(x)))  # type: ignore[misc]
        if p.n_eq > 0:
            parts.append(np.asarray(p.eq_constraints(x)))  # type: ignore[misc]
        if cache is not None:
            cache["std_values"] = tuple(parts)
        return parts

    @staticmethod
    def _weighted_row_sum(
        alpha: np.ndarray,
        A: np.ndarray,
        beta: np.ndarray,
        B: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Fill ``out[i,j] = alpha[i]*A[i,j] + beta[i]*B[i,j]`` in-place.

        Dispatches to the Numba kernel when available, otherwise uses a
        one-temporary NumPy fallback.  *out* must be pre-allocated with
        shape ``(len(alpha), A.shape[1])``.
        """
        _wrs_kernel(alpha, A, beta, B, out)

    def _build_std_jac_flat(
        self, x: np.ndarray, cache: dict | None = None
    ) -> np.ndarray:
        """
        Return flat 1-D Jacobian values for the standard ``[g, h]`` blocks.

        Sparse blocks contribute their nnz values as-is; dense blocks are
        ravelled in row-major order (matching the ``jac_structure`` layout).
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "std_jac_flat" in cache:
            return cache["std_jac_flat"]
        parts: list = []
        if p.n_ineq > 0:
            J = np.asarray(p.ineq_jacobian(x), dtype=float)  # type: ignore[misc, operator]
            parts.append(J if J.ndim == 1 else J.ravel())
        if p.n_eq > 0:
            J = np.asarray(p.eq_jacobian(x), dtype=float)  # type: ignore[misc, operator]
            parts.append(J if J.ndim == 1 else J.ravel())
        values = np.concatenate(parts) if parts else np.empty(0, dtype=float)
        if cache is not None:
            cache["std_jac_flat"] = values
        return values

    def _timed_solve(
        self, nlp, x: np.ndarray, warm_dual: dict
    ) -> tuple[np.ndarray, dict, float]:
        """Call ``nlp.solve()`` with optional warm-start and return ``(x, info, elapsed)``.

        *elapsed* is the wall-clock seconds spent inside ``nlp.solve()``.
        A non-empty *warm_dual* is forwarded as ``lagrange``/``zl``/``zu``
        kwargs regardless of ``self.dual_warmstart``: the flag controls
        whether outer iterations *populate* a warm dual between solves; once
        a warm dual is in hand, it always gets forwarded.  This lets
        :meth:`MPCCSolver.resolve` inject a warm seed into single-shot
        strategies (e.g. ``direct``) that don't carry their own dual
        warm-start machinery.

        On exit, refreshes ``self._last_solve_state`` with the final
        multipliers so ``MPCCSolver.resolve`` can fish them back out.
        """
        t0 = time.perf_counter()
        if warm_dual:
            x, info = nlp.solve(x, **warm_dual)
        else:
            x, info = nlp.solve(x)
        elapsed = time.perf_counter() - t0
        if info["status"] not in (0, 1, 3):
            try:
                status_name = IPOPTStatus(info["status"]).name
            except ValueError:
                status_name = str(info["status"])
            warnings.warn(
                f"pympcc ({self.name!r}): IPOPT returned {status_name!r} "
                f"(status {info['status']}). "
                "Outer loop continues with current iterate as warm-start.",
                UserWarning,
                stacklevel=3,
            )
        self._last_solve_state = {
            "lagrange": info.get("mult_g"),
            "zl":       info.get("mult_x_L"),
            "zu":       info.get("mult_x_U"),
        }
        # Strategies without a per-iteration ``history`` (e.g. ``direct``)
        # rely on this side channel to surface IPOPT iter counts to
        # :class:`MPCCSolver._populate_warmstart_fields`.
        self._last_n_ipopt_iter = int(getattr(nlp, "n_ipopt_iter", 0) or 0)
        return x, info, elapsed


    def _comp_residual(self, x: np.ndarray) -> float:
        """Complementarity infeasibility: max_i |G_i * H_i|."""
        p = self.problem
        G = np.asarray(p.comp_G(x))  # type: ignore[misc]
        H = np.asarray(p.comp_H(x))  # type: ignore[misc]
        return float(np.max(np.abs(G * H)))

    def _compute_kkt_iter(
        self,
        x: np.ndarray,
        mult_g: np.ndarray,
        mpcc_mult_G: np.ndarray,
        mpcc_mult_H: np.ndarray,
        mult_x_L,
        mult_x_U,
    ) -> float | None:
        """
        KKT stationarity residual (∞-norm) for an intermediate iterate.

        Builds a lightweight proxy with only the fields that
        ``compute_kkt_residual`` actually reads (``x`` and ``mult_g``),
        avoiding the cost of constructing a full :class:`MPCCResult`.
        """
        from types import SimpleNamespace
        proxy = cast(MPCCResult, SimpleNamespace(x=x, mult_g=mult_g))
        return _compute_kkt_residual(
            proxy, self.problem,
            mpcc_mult_G=mpcc_mult_G,
            mpcc_mult_H=mpcc_mult_H,
            mult_x_L=mult_x_L,
            mult_x_U=mult_x_U,
        )

    def _build_standard_constraints(self, x: np.ndarray, cache: dict | None = None):
        """
        Evaluate the standard (non-complementarity) part of the constraint
        vector and its Jacobian rows.

        Returns
        -------
        parts : list of ndarray
            Constraint values for g and h (may be empty).
        jac_rows : list of ndarray
            Jacobian rows corresponding to *parts*.
        """
        p = self.problem
        self._ensure_cache_current(cache, x)
        if cache is not None and "std_constraints" in cache:
            parts, jac_rows = cache["std_constraints"]
            return list(parts), list(jac_rows)
        parts, jac_rows = [], []

        if p.n_ineq > 0:
            parts.append(np.asarray(p.ineq_constraints(x)))  # type: ignore[misc]
            jac_rows.append(self._to_dense_block(
                p.ineq_jacobian(x), p.ineq_jacobian_sparsity, p.n_ineq, p.n  # type: ignore[misc, operator]
            ))

        if p.n_eq > 0:
            parts.append(np.asarray(p.eq_constraints(x)))  # type: ignore[misc]
            jac_rows.append(self._to_dense_block(
                p.eq_jacobian(x), p.eq_jacobian_sparsity, p.n_eq, p.n  # type: ignore[misc, operator]
            ))

        if cache is not None:
            cache["std_constraints"] = (tuple(parts), tuple(jac_rows))
        return parts, jac_rows
