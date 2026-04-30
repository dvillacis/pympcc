"""Abstract base class for MPCC reformulation strategies."""
from __future__ import annotations

import math
import time
import warnings
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from .._kernels import coo_to_dense as _coo_kernel
from .._kernels import eval_weighted_union as _wu_kernel
from .._kernels import weighted_row_sum as _wrs_kernel
from .._stationarity import compute_kkt_residual as _compute_kkt_residual
from ..problem import MPCCProblem
from ..result import IPOPTStatus, IterationInfo, MPCCResult

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
    cleanup_obj_worsen_tol=1e-3,    # accept iff Δobj/(|obj|+1) ≤ this
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
    kkt_tol=1e-6,                   # threshold for safeguard_kkt_termination
    inner_tol_mode="linear",        # "linear" | "quadratic" (Leyffer) | "matched"
    inner_tol_factor=0.1,           # ``matched`` mode: tol = factor · ε
    inner_tol_floor=1e-10,          # ``matched`` mode: hard floor on tol
    comp_eps_ratio_theta=10.0,      # "tracked ε" means comp ≤ θ·ε
    rollback_lambda_jump=1e3,       # reject if ‖mult_g‖∞ jumps more than this
    rollback_max_count=3,           # abort after this many consecutive rollbacks
    eps_hold_factor=3.0,            # ε multiplier on rollback (>1 backs off toward last good ε; capped at last_good_eps)
    restoration_iter_threshold=5,   # rollback if IPOPT spent ≥ this many iters in restoration
    pre_post_blowup_factor=10.0,    # rollback if post-NLP comp residual > factor × pre-NLP comp residual
    plateau_tol_obj=1e-4,           # relative |Δobj| below this counts as plateau
    plateau_tol_comp=1e-3,          # relative |Δcomp_residual| below this counts as plateau
    plateau_window=2,               # consecutive plateau iters required to terminate
    plateau_comp_target=1e-4,       # plateau breaks only fire when comp_residual ≤ this
)


class BaseStrategy(ABC):
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
                 backend: str = "ipopt", solver_options: dict | None = None,
                 linear_solver_fn=None, **kwargs) -> None:
        self.problem = problem
        self.ipopt_options = ipopt_options
        self.backend = backend
        self.solver_options = solver_options or {}
        self._linear_solver_fn = linear_solver_fn
        self.callback = kwargs.pop("callback", None)
        self.inner_callback = kwargs.pop("inner_callback", None)
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
        if self.backend == "filterSQP":  # pragma: no cover
            try:
                from pyfiltersqp import _FilterSQPAdapter
            except ImportError as exc:
                raise ImportError(
                    "backend='filterSQP' requires the pyfiltersqp package. "
                    "Install it or switch to backend='ipopt'."
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
                xl=p.xl, xu=p.xu, cl=cl, cu=cu,
                obj_fn=_obj, grad_fn=_grad,
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
    def _resolve_auto_epsilon_0(
        problem,
        *,
        theta: float = 1.0,
        lo: float = 1e-3,
        hi: float = 1.0,
    ) -> tuple[float, float]:
        """Pick ``ε₀ = clip(theta * max|G(x₀)·H(x₀)|, lo, hi)``.

        Returns ``(epsilon_0, raw_residual)`` so callers can log both.
        Uses the raw user-provided ``comp_G`` / ``comp_H`` (pre-scaling);
        the resolver runs once at strategy ``__init__``.
        """
        G = np.asarray(problem.comp_G(problem.x0))
        H = np.asarray(problem.comp_H(problem.x0))
        raw = float(np.max(np.abs(G * H))) if G.size else 0.0
        eps0 = float(np.clip(theta * raw, lo, hi))
        return eps0, raw

    def _maybe_resolve_auto_epsilon_0(self, opts: dict) -> dict:
        """Replace ``opts['epsilon_0'] == 'auto'`` with a concrete float.

        Stashes ``self._auto_eps0_origin = (raw_residual, resolved_eps0)``
        so :meth:`_log_auto_eps0` can print one diagnostic line; left as
        ``None`` when the user passed a numeric ``epsilon_0``.
        """
        self._auto_eps0_origin: tuple[float, float] | None = None
        if opts.get("epsilon_0") == "auto":
            eps0, raw = self._resolve_auto_epsilon_0(self.problem)
            self._auto_eps0_origin = (raw, eps0)
            opts = {**opts, "epsilon_0": eps0}
        return opts

    def _log_auto_eps0(self) -> None:
        """One-line report when ε₀ was resolved from the ``"auto"`` sentinel.

        Quietly does nothing for numeric ``epsilon_0``, or when the user
        hasn't requested output (no per-iteration callback installed —
        which is the same gate ``verbose=True`` uses to wire up
        :func:`_default_verbose_callback`).
        """
        origin = getattr(self, "_auto_eps0_origin", None)
        if origin is None or self.callback is None:
            return
        raw, eps0 = origin
        clipped = " (clipped)" if eps0 != raw else ""
        print(f"auto-ε₀: max|G·H|(x0) = {raw:.2e} → ε₀ = {eps0:.2e}{clipped}")

    @staticmethod
    def _validate_continuation_options(
        *,
        epsilon_0: float,
        reduction: float,
        max_iter: int,
        epsilon_min: float,
        comp_tol: float | None,
    ) -> None:
        """Validate common continuation-strategy scalar options."""
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if epsilon_0 <= 0.0:
            raise ValueError("epsilon_0 must be > 0")
        if not (0.0 < reduction < 1.0):
            raise ValueError("reduction must satisfy 0 < reduction < 1")
        if epsilon_min < 0.0:
            raise ValueError("epsilon_min must be >= 0")
        if epsilon_min >= epsilon_0:
            raise ValueError("epsilon_min must be smaller than epsilon_0")
        if comp_tol is not None and comp_tol <= 0.0:
            raise ValueError("comp_tol must be > 0 when provided")

    def _init_cleanup(self, opts: dict, user_kwargs: dict | None = None) -> None:
        """Install active-set cleanup attributes on ``self``.

        Strategies built on :meth:`_run_epsilon_continuation` call this once
        after ``_init_safeguards``.  The meta key ``opts["safeguards"] ==
        "all"`` also flips ``cleanup`` to ``"auto"`` when the user has *not*
        explicitly supplied a ``cleanup`` value, so that ``--safeguards``
        becomes the single switch to opt into both the continuation guards
        and the polish phase.  Passing ``cleanup=False`` explicitly always
        suppresses cleanup regardless of ``safeguards``.

        *user_kwargs* is the raw ``**kwargs`` dict before merging with
        defaults; it is used to detect whether the caller explicitly set
        ``cleanup``.  When omitted (legacy callers) the old behaviour is
        preserved.
        """
        cleanup_raw = opts.get("cleanup", False)
        user_set_cleanup = user_kwargs is not None and "cleanup" in user_kwargs
        if opts.get("safeguards") == "all" and not user_set_cleanup and cleanup_raw is False:
            cleanup_raw = "auto"
        if cleanup_raw not in (False, True, "auto"):
            raise ValueError(
                f"cleanup must be False, True, or 'auto', got {cleanup_raw!r}"
            )

        bi_tol = opts.get("cleanup_biactive_tol", None)
        if bi_tol is not None and float(bi_tol) <= 0.0:
            raise ValueError("cleanup_biactive_tol must be > 0 when provided")

        manual = opts.get("cleanup_active_set", None)
        if manual is not None:
            if not (isinstance(manual, tuple) and len(manual) == 2):
                raise ValueError(
                    "cleanup_active_set must be a 2-tuple "
                    "(I_G_active, I_H_active) of integer arrays"
                )

        worsen = float(opts.get("cleanup_obj_worsen_tol", 1e-3))
        max_it = int(opts.get("cleanup_max_iter", 300))
        cleanup_tol_raw = opts.get("cleanup_tol", None)
        if worsen < 0.0:
            raise ValueError("cleanup_obj_worsen_tol must be >= 0")
        if max_it < 1:
            raise ValueError("cleanup_max_iter must be >= 1")
        if cleanup_tol_raw is not None and float(cleanup_tol_raw) <= 0.0:
            raise ValueError("cleanup_tol must be > 0 when provided")

        self.cleanup                = cleanup_raw
        self.cleanup_biactive_tol   = float(bi_tol) if bi_tol is not None else None
        self.cleanup_active_set     = manual
        self.cleanup_obj_worsen_tol = worsen
        self.cleanup_max_iter       = max_it
        self.cleanup_tol            = (float(cleanup_tol_raw)
                                       if cleanup_tol_raw is not None else None)

    def _init_safeguards(self, opts: dict) -> None:
        """Install safeguard attributes on ``self`` from a merged options dict.

        Strategies built on :meth:`_run_epsilon_continuation` call this once
        after ``_validate_continuation_options``.  When ``opts["safeguards"] ==
        "all"``, the three boolean safeguards are forced on and the inner-tol
        mode is set to ``"quadratic"`` (one-line opt-in for callers).
        """
        all_on = (opts.get("safeguards") == "all")

        rb_on  = bool(opts.get("safeguard_rollback", False))         or all_on
        ad_on  = bool(opts.get("safeguard_adaptive_eps", False))      or all_on
        kk_on  = bool(opts.get("safeguard_kkt_termination", False))   or all_on
        pl_on  = bool(opts.get("safeguard_plateau", False))           or all_on
        mode   = opts.get("inner_tol_mode", "linear")
        if mode not in ("linear", "quadratic", "matched"):
            raise ValueError(
                "inner_tol_mode must be 'linear', 'quadratic', or 'matched',"
                f" got {mode!r}"
            )

        theta    = float(opts.get("comp_eps_ratio_theta", 10.0))
        lam_jump = float(opts.get("rollback_lambda_jump", 1e3))
        rb_cap   = int(opts.get("rollback_max_count", 3))
        eps_hold = float(opts.get("eps_hold_factor", 1.0))
        kkt_tol  = float(opts.get("kkt_tol", 1e-6))
        rest_th  = int(opts.get("restoration_iter_threshold", 5))
        blowup   = float(opts.get("pre_post_blowup_factor", 10.0))
        tol_fac  = float(opts.get("inner_tol_factor", 0.1))
        tol_floor = float(opts.get("inner_tol_floor", 1e-10))
        pl_tobj  = float(opts.get("plateau_tol_obj", 1e-4))
        pl_tcomp = float(opts.get("plateau_tol_comp", 1e-3))
        pl_win   = int(opts.get("plateau_window", 2))
        pl_targ  = float(opts.get("plateau_comp_target", 1e-4))

        if theta <= 0.0:
            raise ValueError("comp_eps_ratio_theta must be > 0")
        if lam_jump <= 1.0:
            raise ValueError("rollback_lambda_jump must be > 1")
        if rb_cap < 1:
            raise ValueError("rollback_max_count must be >= 1")
        if eps_hold <= 0.0:
            raise ValueError("eps_hold_factor must be > 0")
        if kkt_tol <= 0.0:
            raise ValueError("kkt_tol must be > 0")
        if rest_th < 1:
            raise ValueError("restoration_iter_threshold must be >= 1")
        if blowup <= 1.0:
            raise ValueError("pre_post_blowup_factor must be > 1")
        if tol_fac <= 0.0:
            raise ValueError("inner_tol_factor must be > 0")
        if tol_floor <= 0.0:
            raise ValueError("inner_tol_floor must be > 0")
        if pl_tobj <= 0.0:
            raise ValueError("plateau_tol_obj must be > 0")
        if pl_tcomp <= 0.0:
            raise ValueError("plateau_tol_comp must be > 0")
        if pl_win < 1:
            raise ValueError("plateau_window must be >= 1")
        if pl_targ <= 0.0:
            raise ValueError("plateau_comp_target must be > 0")

        self.safeguard_rollback         = rb_on
        self.safeguard_adaptive_eps     = ad_on
        self.safeguard_kkt_termination  = kk_on
        self.safeguard_plateau          = pl_on
        self.kkt_tol                    = kkt_tol
        self.inner_tol_mode             = mode
        self.comp_eps_ratio_theta       = theta
        self.rollback_lambda_jump       = lam_jump
        self.rollback_max_count         = rb_cap
        self.eps_hold_factor            = eps_hold
        self.restoration_iter_threshold = rest_th
        self.pre_post_blowup_factor     = blowup
        self.inner_tol_factor           = tol_fac
        self.inner_tol_floor            = tol_floor
        self.plateau_tol_obj            = pl_tobj
        self.plateau_tol_comp           = pl_tcomp
        self.plateau_window             = pl_win
        self.plateau_comp_target        = pl_targ

    @staticmethod
    def _validate_augmented_lagrangian_options(
        *,
        rho_0: float,
        rho_max: float,
        tau: float,
        eta: float,
        max_iter: int,
        comp_tol: float,
        stagnation_iters: int,
    ) -> None:
        """Validate augmented-Lagrangian scalar options."""
        if int(max_iter) != max_iter or max_iter < 1:
            raise ValueError("max_iter must be an integer >= 1")
        if rho_0 <= 0.0:
            raise ValueError("rho_0 must be > 0")
        if rho_max < rho_0:
            raise ValueError("rho_max must be >= rho_0")
        if tau <= 1.0:
            raise ValueError("tau must be > 1")
        if eta < 0.0:
            raise ValueError("eta must be >= 0")
        if comp_tol <= 0.0:
            raise ValueError("comp_tol must be > 0")
        if int(stagnation_iters) != stagnation_iters or stagnation_iters < 1:
            raise ValueError("stagnation_iters must be an integer >= 1")

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
        Warm-start multipliers are passed only when ``self.dual_warmstart`` is
        ``True`` (strategies without that attribute always skip warm-start).
        Emits a ``UserWarning`` when IPOPT returns a non-success status so that
        outer-loop failures are never silent.
        """
        t0 = time.perf_counter()
        if getattr(self, "dual_warmstart", False) and warm_dual:
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
        return x, info, elapsed

    # ------------------------------------------------------------------ #
    # Active-set cleanup phase                                              #
    # ------------------------------------------------------------------ #

    def _infer_active_set(
        self,
        G: np.ndarray,
        H: np.ndarray,
        mpcc_mult_G: np.ndarray | None,
        mpcc_mult_H: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partition ``range(n_comp)`` into ``(I_G_active, I_H_active)``.

        Every index is pinned on exactly one side: ``G_i = 0`` or ``H_i = 0``.
        Default rule is the magnitude rule (pin the smaller side); ties
        and biactive pairs (both within ``cleanup_biactive_tol``) are
        broken by the multiplier rule when MPCC multipliers are
        provided.  When ``self.cleanup_active_set`` is set, that
        partition is returned verbatim (no inference).
        """
        if self.cleanup_active_set is not None:
            I_G, I_H = self.cleanup_active_set
            return (
                np.asarray(I_G, dtype=np.intp).ravel(),
                np.asarray(I_H, dtype=np.intp).ravel(),
            )

        n_c = len(G)
        if n_c == 0:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty

        if self.cleanup_biactive_tol is not None:
            bi_tol = self.cleanup_biactive_tol
        else:
            comp = float(np.max(np.abs(G * H)))
            bi_tol = max(np.sqrt(comp), 1e-4)

        # Magnitude rule: pin the smaller of G_i, H_i.
        pin_G = G < H

        # Multiplier rule for truly biactive pairs (both small).
        if mpcc_mult_G is not None and mpcc_mult_H is not None:
            biactive = (G <= bi_tol) & (H <= bi_tol)
            if np.any(biactive):
                mu_G_bi = np.abs(mpcc_mult_G[biactive])
                mu_H_bi = np.abs(mpcc_mult_H[biactive])
                # Pin the side with the smaller multiplier (the side the
                # solver considers less binding).
                pin_G[biactive] = mu_G_bi <= mu_H_bi

        I_G = np.where(pin_G)[0].astype(np.intp)
        I_H = np.where(~pin_G)[0].astype(np.intp)
        return I_G, I_H

    def _build_cleanup_nlp(
        self,
        I_G_active: np.ndarray,
        I_H_active: np.ndarray,
    ):
        """Build the smooth active-set-fixed cleanup NLP in original ``x``.

        Layout: ``[g, h, G, H]`` where ``G_i`` (resp. ``H_i``) row has
        ``cl=cu=0`` for pinned indices and ``cl=0, cu=+∞`` otherwise.
        Mirrors :class:`DirectStrategy` minus the ``G·H`` rows.

        Returns ``(nlp, cache)`` ready for a single ``_timed_solve`` call.
        L-BFGS Hessian is forced (cleanup uses a different Lagrangian
        than continuation, so any JAX-built Hessian from the strategy
        does not apply here).
        """
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        _INF = 2e19

        cl = np.concatenate([
            np.full(n_g, -_INF), np.zeros(n_h),
            np.zeros(n_c), np.zeros(n_c),
        ])
        cu = np.concatenate([
            np.zeros(n_g), np.zeros(n_h),
            np.full(n_c, _INF), np.full(n_c, _INF),
        ])
        if len(I_G_active):
            cu[n_g + n_h + I_G_active] = 0.0
        if len(I_H_active):
            cu[n_g + n_h + n_c + I_H_active] = 0.0

        cache = self._new_callback_cache()

        def constraints(x):
            std_parts = self._eval_standard_con_values(x, cache)
            G, H = self._eval_comp_values(x, cache)
            return np.concatenate([*std_parts, G, H])

        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
        ]) if p.is_sparse else None

        if p.is_sparse:
            def jacobian(x):
                std_flat = self._build_std_jac_flat(x, cache)
                vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                return np.concatenate([std_flat, v_G, v_H])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                _, _, JG, JH = self._build_comp_jacobians(x, cache)
                jac_rows.extend([JG, JH])
                return np.vstack(jac_rows)

        hess_fn, hess_sp = self._build_cleanup_hessian()
        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure,
                              hess_fn=hess_fn, hess_sparsity=hess_sp)
        if hess_fn is None:
            # No exact Hessian available for this strategy — fall back to L-BFGS
            # and floor the inherited tol at 1e-6 to avoid IPOPT spinning on
            # quasi-Newton dual residuals it can't drive below the analytical
            # tolerance.
            nlp.add_option("hessian_approximation", "limited-memory")
            cu_tol_floor = 1e-6
        else:
            cu_tol_floor = 0.0
        nlp.add_option("max_iter", int(self.cleanup_max_iter))
        if self.cleanup_tol is not None:
            cu_tol = float(self.cleanup_tol)
        else:
            cu_tol = max(float(self.ipopt_options.get("tol", 1e-8)),
                         cu_tol_floor)
        nlp.add_option("tol", cu_tol)
        return nlp, cache

    def _build_cleanup_hessian(self):
        """Return ``(hess_fn, hess_sparsity)`` for the cleanup NLP, or
        ``(None, None)`` to use L-BFGS.

        The cleanup constraint layout is ``[g, h, G, H]`` (size
        ``n_g + n_h + 2·n_c``), so ``hess_fn(x, lam_cu, obj_factor)`` must
        accept multipliers in that layout.  The cleanup Lagrangian is
        the same across every strategy::

            L_cu = obj_factor·f + λ_g·g + λ_h·h + λ_G·G + λ_H·H

        so the universal default builds it via JAX autodiff whenever
        ``problem.use_jax_hessian`` is set; otherwise returns
        ``(None, None)``.  Strategies whose full Lagrangian is a strict
        superset of ``L_cu`` (Scholtes, lin_fukushima) override to wrap
        their existing Hessian with zero-padded multipliers — cheaper
        than a second JAX compile when a manual Hessian is supplied.
        """
        if not self._has_jax_hessian():
            return None, None
        return self._build_jax_cleanup_hessian()

    def _build_jax_cleanup_hessian(self):  # pragma: no cover
        """Build the cleanup Lagrangian Hessian via JAX (universal).

        Used by every strategy whose full Lagrangian is *not* a strict
        superset of ``L_cu`` (smoothing, slack, augmented_lagrangian),
        and as the JAX fallback for Scholtes/lin_fukushima when no
        manual Hessian is supplied.
        """
        import jax.numpy as jnp

        from .._jax import jax_hessian_lagrangian
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        m_cu = n_g + n_h + 2 * n_c  # cleanup layout: [g, h, G, H]

        def lagrangian_cu(x, lam, obj_factor):
            val = obj_factor * p.objective(x)
            if n_g:
                val = val + jnp.dot(lam[:n_g], p.ineq_constraints(x))
            if n_h:
                val = val + jnp.dot(lam[n_g:n_g + n_h], p.eq_constraints(x))
            off = n_g + n_h
            val = val + jnp.dot(lam[off:off + n_c], p.comp_G(x))
            val = val + jnp.dot(lam[off + n_c:off + 2 * n_c], p.comp_H(x))
            return val

        return jax_hessian_lagrangian(
            lagrangian_cu, p.n, p.x0, m_cu, p.jax_sparsity_tol,
        )

    def _maybe_run_cleanup(
        self,
        result: MPCCResult,
        last_info: dict,
        x_orig: np.ndarray,
        mpcc_mult_G: np.ndarray | None,
        mpcc_mult_H: np.ndarray | None,
    ) -> MPCCResult:
        """Run the active-set cleanup pass when enabled and patch ``result``.

        Cleanup is skipped silently when:

        * ``self.cleanup`` is ``False``;
        * ``self.cleanup == "auto"`` and the continuation result is not
          usable (``status not in {0,1,3}``) or ``comp_residual > 1e-2``.

        When cleanup runs, the diagnostic fields ``cleanup_status``,
        ``cleanup_n_iter``, ``cleanup_obj``, ``cleanup_active_set`` and
        ``cleanup_accepted`` are populated regardless of whether the
        polished iterate is accepted.  The top-level ``x``, ``obj``,
        ``G``, ``H``, ``comp_residual``, ``mult_g`` and ``stationarity``
        fields are replaced *only* when cleanup converges and does not
        worsen the objective by more than ``cleanup_obj_worsen_tol``.
        """
        from .._stationarity import (  # local import: avoid cycle
            classify_stationarity,
            compute_kkt_residual,
        )

        mode = self.cleanup
        if mode is False:
            return result
        if mode == "auto":
            if not result.success or result.comp_residual > 1e-2:
                return result

        p = self.problem
        try:
            # Evaluate G,H in the same (possibly scaled) space the multipliers
            # live in; otherwise the magnitude/multiplier rule mixes spaces
            # when comp_G_scale/comp_H_scale are active.
            G_init, H_init = self._eval_comp_values(x_orig, self._new_callback_cache())
            I_G, I_H = self._infer_active_set(G_init, H_init,
                                              mpcc_mult_G, mpcc_mult_H)
            nlp, _cache = self._build_cleanup_nlp(I_G, I_H)
            x_cu, info_cu, _t = self._timed_solve(nlp, x_orig, {})
        except Exception as exc:                          # pragma: no cover
            warnings.warn(
                f"pympcc cleanup raised {type(exc).__name__}: {exc}. "
                "Returning continuation result.",
                UserWarning, stacklevel=2,
            )
            result.cleanup_status   = -999
            result.cleanup_accepted = False
            return result

        cleanup_status = int(info_cu["status"])
        cleanup_obj    = float(info_cu["obj_val"])
        result.cleanup_status      = cleanup_status
        result.cleanup_n_iter      = int(getattr(nlp, "n_ipopt_iter", 0))
        result.cleanup_obj         = cleanup_obj
        result.cleanup_active_set  = (I_G, I_H)

        cleanup_success = cleanup_status in (0, 1, 3)
        worsen = (cleanup_obj - result.obj) / (abs(result.obj) + 1.0)
        accept = cleanup_success and (worsen <= self.cleanup_obj_worsen_tol)
        result.cleanup_accepted = accept

        if not accept:
            return result

        # Promote cleanup iterate.  Use _eval_comp_values so result.G/H stay
        # in the same (scaled) space that the rest of the solver and the
        # stored multipliers use — without this, accepting cleanup silently
        # flips result.G/H/comp_residual to unscaled space.
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
        G_cu, H_cu = self._eval_comp_values(x_cu, self._new_callback_cache())
        result.x                  = x_cu
        result.obj                = cleanup_obj
        result.G                  = G_cu
        result.H                  = H_cu
        result.comp_residual      = float(np.max(np.abs(G_cu * H_cu)))
        result.comp_residual_mean = float(np.mean(np.abs(G_cu * H_cu)))
        result.status             = cleanup_status
        result.message            = self._decode_msg(info_cu["status_msg"])
        result.success            = cleanup_success
        result.mult_g             = info_cu.get("mult_g")

        # Cleanup layout: [g, h, G, H].  The G and H multiplier blocks
        # carry both the original inequality multipliers and (where the
        # row was pinned to equality) the equality multipliers — sign
        # is unambiguous for stationarity classification because every
        # pair has exactly one pinned side.
        mult_g = info_cu.get("mult_g")
        if mult_g is not None:
            lam_G = mult_g[n_g + n_h           : n_g + n_h + n_c]
            lam_H = mult_g[n_g + n_h + n_c     : n_g + n_h + 2 * n_c]
            # Cleanup uses scaled constraints (s_G·G, s_H·H) via
            # _eval_comp_values.  Scale multipliers to original problem space
            # so compute_kkt_residual (which uses unscaled Jacobians) is correct.
            mu_G_orig = lam_G * np.asarray(p.comp_G_scale) if p.comp_G_scale is not None else lam_G
            mu_H_orig = lam_H * np.asarray(p.comp_H_scale) if p.comp_H_scale is not None else lam_H
            result.kkt_residual = compute_kkt_residual(
                result, p,
                mpcc_mult_G=mu_G_orig,
                mpcc_mult_H=mu_H_orig,
                mult_x_L=info_cu.get("mult_x_L"),
                mult_x_U=info_cu.get("mult_x_U"),
            )
            # Every pair is pinned on exactly one side, so the biactive
            # set in the cleaned MPCC is empty by construction.  In that
            # regime ``classify_stationarity`` returns ``"S-stationary"``
            # vacuously when the solve converged.
            result.stationarity = classify_stationarity(result, p)
        return result

    def _run_epsilon_continuation(
        self,
        nlp,
        x0: np.ndarray,
        eps_ref: list,
        make_iteration,
    ) -> tuple[np.ndarray, dict, float, list[IterationInfo]]:
        """
        Shared outer loop for epsilon-continuation strategies.

        Handles the common mechanics: epsilon updates, warm-starting, adaptive
        IPOPT tolerance, timing, history storage, callback invocation, and
        early stopping on ``comp_tol``.

        Optional safeguards (off by default; enabled via
        :data:`SAFEGUARD_DEFAULTS` keys forwarded as ``strategy_options``):

        * **rollback** — snapshot ``(x, warm_dual, eps)`` before each inner
          solve.  Reject and restore the snapshot when the inner solver
          fails, fails to track ε (``comp_residual > θ·ε``), or makes the
          dual multipliers explode.  After a reject ε is held (``×
          eps_hold_factor``) instead of being reduced.  After
          ``rollback_max_count`` consecutive rejects the loop terminates.
        * **adaptive ε reduction** — only apply the user's ``reduction``
          when the last solve was clean; otherwise back off to
          ``sqrt(reduction)``.
        * **inner-tol coupling** — switch the inner-solve tol from the
          linear ``ε·1e-2`` to the quadratic ``ε**1.5`` Leyffer schedule.
        * **MPCC-KKT termination** — break early once
          ``info.kkt_residual ≤ kkt_tol``.
        """
        self._log_auto_eps0()
        history: list[IterationInfo] = []
        x = np.asarray(x0, dtype=float).copy()
        eps = self.epsilon_0
        last_info: dict = {}
        warm_dual: dict = {}
        total_time: float = 0.0

        rollback_on    = getattr(self, "safeguard_rollback", False)
        adaptive_on    = getattr(self, "safeguard_adaptive_eps", False)
        kkt_term_on    = getattr(self, "safeguard_kkt_termination", False)
        plateau_on     = getattr(self, "safeguard_plateau", False)
        kkt_tol        = getattr(self, "kkt_tol", 1e-6)
        tol_mode       = getattr(self, "inner_tol_mode", "linear")
        theta          = getattr(self, "comp_eps_ratio_theta", 10.0)
        lam_jump_max   = getattr(self, "rollback_lambda_jump", 1e3)
        rollback_cap   = getattr(self, "rollback_max_count", 3)
        eps_hold       = getattr(self, "eps_hold_factor", 1.0)
        rest_threshold = getattr(self, "restoration_iter_threshold", 5)
        blowup_factor  = getattr(self, "pre_post_blowup_factor", 10.0)
        tol_factor     = getattr(self, "inner_tol_factor", 0.1)
        tol_floor      = getattr(self, "inner_tol_floor", 1e-10)
        plateau_tol_obj  = getattr(self, "plateau_tol_obj", 1e-4)
        plateau_tol_comp = getattr(self, "plateau_tol_comp", 1e-3)
        plateau_window   = getattr(self, "plateau_window", 2)
        plateau_target   = getattr(self, "plateau_comp_target", 1e-4)

        user_tol      = self.ipopt_options.get("tol", 1e-8)
        inner_max_iter = int(self.ipopt_options.get("max_iter", 3000))

        rollback_run   = 0
        prev_mult_inf  = None
        last_good_eps: float | None = None  # most recent ε of an accepted iterate
        warm_init_armed_at: int | None = None  # outer index where we toggled warm_start_init_point
        plateau_streak = 0
        prev_obj_acc: float | None = None
        prev_comp_acc: float | None = None

        k = 0
        while k < self.max_iter:
            eps_ref[0] = eps
            # Zero per-solve diagnostics (n_ipopt_iter + restoration counters).
            if hasattr(nlp, "reset_iter_counters"):
                nlp.reset_iter_counters()
            else:
                nlp.n_ipopt_iter = 0
            if (warm_init_armed_at is None and k >= 1 and self.dual_warmstart):
                nlp.add_option("warm_start_init_point", "yes")
                warm_init_armed_at = k

            # Inner solver tolerance.  ``linear`` and ``quadratic`` floor
            # at ``user_tol`` (typically 1e-8) — IPOPT keeps chasing
            # precision even when ε can no longer support it, which
            # spins MAX_ITER on degenerate Scholtes NLPs.  ``matched``
            # tracks ε directly with no user_tol floor: the inner tol
            # loosens as ε shrinks, so IPOPT terminates at a precision
            # the relaxation can actually deliver.
            if tol_mode == "quadratic":
                inner_tol = max(user_tol, eps ** 1.5)
            elif tol_mode == "matched":
                inner_tol = max(tol_floor, tol_factor * eps)
            else:
                inner_tol = max(user_tol, eps * 1e-2)
            nlp.add_option("tol", inner_tol)

            # Snapshot for rollback (before inner solve overwrites x/warm_dual).
            if rollback_on:
                snap_x        = x.copy()
                snap_warm     = dict(warm_dual)
                snap_eps      = eps
                snap_prev_inf = prev_mult_inf
                # Pre-NLP comp residual: lets us detect post-NLP iterates
                # that are dramatically worse than the warm-start (e.g.
                # IPOPT MAX_ITER returns a degraded iterate).
                snap_pre_comp = self._comp_residual(snap_x)

            x, last_info, iter_time = self._timed_solve(nlp, x, warm_dual)
            total_time += iter_time
            if self.dual_warmstart:
                warm_dual = {
                    "lagrange": last_info["mult_g"],
                    "zl":       last_info["mult_x_L"],
                    "zu":       last_info["mult_x_U"],
                }

            info = make_iteration(eps, x, last_info, nlp.n_ipopt_iter, iter_time)
            # Restoration-phase diagnostics — populated regardless of rollback
            # safeguards so callers can always inspect history[k].
            info.restoration_iter_count = int(getattr(nlp, "restoration_iter_count", 0))
            info.entered_restoration    = bool(getattr(nlp, "entered_restoration", False))
            history.append(info)
            if self.callback is not None:
                self.callback(len(history) - 1, history[-1])

            status_ok = last_info["status"] in (0, 1, 3)

            # ---------------- rollback decision ----------------
            rejected = False
            if rollback_on:
                tracked_eps = info.comp_residual <= theta * eps
                mult_g = last_info.get("mult_g")
                cur_inf = (float(np.max(np.abs(mult_g)))
                           if mult_g is not None and len(mult_g) else 0.0)
                mult_jump_ok = (
                    prev_mult_inf is None
                    or prev_mult_inf <= 0.0
                    or cur_inf <= lam_jump_max * prev_mult_inf
                )
                # Persistent restoration is a strong "ε too tight" signal —
                # force a rollback even when the other heuristics would
                # have accepted the iterate.
                restoration_excess = (
                    info.entered_restoration
                    and info.restoration_iter_count >= rest_threshold
                )
                # Post-NLP iterate dramatically worse than pre-NLP warm
                # start.  Catches MAX_ITER cases where the inner solver
                # returns a degraded iterate that tracked_eps alone may
                # accept (e.g. when ε is very small but pre_comp was
                # already smaller).
                post_blew_up = (
                    snap_pre_comp > 0.0
                    and info.comp_residual > blowup_factor * snap_pre_comp
                    and info.comp_residual > eps
                )
                rejected = (
                    not (status_ok and tracked_eps and mult_jump_ok)
                    or restoration_excess
                    or post_blew_up
                )

            if rejected:
                # Restore snapshot and back off ε.  ``eps_hold_factor > 1``
                # genuinely expands ε so the next attempt is less aggressive;
                # cap at ``last_good_eps`` so we never exceed the most recent
                # accepted value (otherwise a bad rollback could undo earlier
                # progress).  When no iterate has been accepted yet, fall
                # back to ``self.epsilon_0`` as the ceiling.
                x          = snap_x
                warm_dual  = snap_warm
                ceiling    = last_good_eps if last_good_eps is not None else self.epsilon_0
                eps        = min(snap_eps * eps_hold, ceiling)
                prev_mult_inf = snap_prev_inf
                plateau_streak = 0
                rollback_run += 1
                if rollback_run >= rollback_cap:
                    break
                if eps < self.epsilon_min:
                    break
                # If backoff didn't actually move ε (eps_hold==1 with no
                # ceiling change), we'd repeat the same NLP — break early.
                if eps <= snap_eps:
                    break
                k += 1
                continue

            rollback_run = 0
            last_good_eps = eps
            mult_g = last_info.get("mult_g")
            if mult_g is not None and len(mult_g):
                prev_mult_inf = float(np.max(np.abs(mult_g)))

            # ---------------- termination tests ----------------
            if (kkt_term_on
                    and info.kkt_residual is not None
                    and info.kkt_residual <= kkt_tol
                    and status_ok):
                break

            if (self.comp_tol is not None
                    and info.comp_residual < self.comp_tol
                    and status_ok):
                break

            # Plateau termination: stop once both obj and comp_residual
            # have stalled across consecutive accepted iterates and comp
            # is already below the target.  Catches the "obj converged
            # at outer iter k but loop keeps shrinking ε for nothing"
            # pattern that wastes most of the wall time on poorly-scaled
            # large problems.
            if plateau_on and prev_obj_acc is not None and prev_comp_acc is not None:
                d_obj = (abs(info.obj - prev_obj_acc)
                         / max(abs(info.obj), 1e-12))
                d_comp = (abs(info.comp_residual - prev_comp_acc)
                          / max(prev_comp_acc, 1e-12))
                comp_below_target = info.comp_residual <= plateau_target
                if (d_obj < plateau_tol_obj
                        and d_comp < plateau_tol_comp
                        and comp_below_target):
                    plateau_streak += 1
                    if plateau_streak >= plateau_window:
                        break
                else:
                    plateau_streak = 0
            prev_obj_acc = info.obj
            prev_comp_acc = info.comp_residual

            # ---------------- ε update ----------------
            if adaptive_on:
                clean = (
                    status_ok
                    and info.comp_residual <= theta * eps
                    and info.n_ipopt_iter < max(1, inner_max_iter // 2)
                )
                eps *= self.reduction if clean else math.sqrt(self.reduction)
            else:
                eps *= self.reduction

            if eps < self.epsilon_min:
                break
            k += 1

        return x, last_info, total_time, history

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
