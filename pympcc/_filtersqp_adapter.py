"""pyfiltersqp ↔ pympcc backend adapter.

pympcc's strategies build a single standard NLP::

    min  f(x)  s.t.  cl ≤ c(x) ≤ cu,  xl ≤ x ≤ xu

and hand it to a backend object that exposes the cyipopt-compatible
subset (``add_option`` / ``solve``).  pyfiltersqp uses a different
form — split equality / one-sided inequality with ``c_ineq(x) ≤ 0`` —
so this module is the translation layer that lets ``backend='filterSQP'``
plug into pympcc without changing any strategy code.

What the adapter has to translate
---------------------------------

1. **Constraint layout.**  Each row of ``c(x)`` is classified once at
   construction time:

   * ``cl[i] == cu[i]`` (finite)  →  pyfiltersqp eq row ``c[i] − cl[i] = 0``
   * ``cl[i] = −∞, cu[i]`` finite →  pyfiltersqp ineq ``c[i] − cu[i] ≤ 0``
   * ``cl[i]`` finite, ``cu[i] = +∞``  →  pyfiltersqp ineq ``cl[i] − c[i] ≤ 0``

   Range rows (both finite, ``cl < cu``) are not produced by any
   shipped pympcc strategy and are rejected with ``NotImplementedError``.

2. **Multiplier signs (cyipopt convention on the wire).**

   * eq row    →  ``mult_g[i] = lam_eq[k]``     (free sign)
   * upper row →  ``mult_g[i] = +lam_ineq[k]``  (≥ 0 at active upper)
   * lower row →  ``mult_g[i] = −lam_ineq[k]``  (≤ 0 at active lower,
     because pyfiltersqp returns ``λ ≥ 0`` for the rewritten
     ``cl − c ≤ 0`` form)

3. **Status codes.**  pyfiltersqp ``status`` ∈ {0, 1, 2, 3} maps to a
   distinct cyipopt-compatible negative int (so ``MPCCResult.status``
   keeps disambiguating).  ``success == True`` ⇒ status 0.

4. **Options.**  ``tol``, ``max_iter``, ``print_level`` translate to
   pyfiltersqp's ``tol_opt``, ``max_iter``, ``verbose``.  IPOPT-only
   options (``max_cpu_time``, ``warm_start_init_point``, ``sb``,
   ``hessian_approximation``) are silently ignored — there is no
   pyfiltersqp analogue.  Unknown keys are forwarded as raw kwargs.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Optional

import numpy as np

# Module-level import so ``ImportError`` surfaces cleanly to the
# wrapper in ``BaseStrategy._build_nlp`` (``backend='filterSQP'`` path).
# Anyone importing ``pympcc._filtersqp_adapter`` without pyfiltersqp
# installed gets the same error pympcc's strategy layer rewraps.
from pyfiltersqp import NLPProblem, SQPSolver  # noqa: E402, I001

_log = logging.getLogger(__name__)

#: Allowlist of kwargs ``pyfiltersqp.SQPSolver.__init__`` actually accepts,
#: introspected once at import time.  Any "unknown" cyipopt option that
#: is *not* in this set is dropped with a ``logger.debug`` message
#: instead of being forwarded as ``SQPSolver(**opts)`` and crashing with
#: ``TypeError: unexpected keyword argument``.
_SQP_KWARGS: frozenset[str] = frozenset(
    name for name in inspect.signature(SQPSolver.__init__).parameters
    if name != "self"
)

# IPOPT (and therefore pympcc strategies) emits ``±2e19`` as the
# "infinity" sentinel for bounds and constraint sides; values past
# this are functionally unbounded.  pyfiltersqp expects literal
# ``±np.inf``, so we sanitise on the way in.
_IPOPT_INF: float = 1e19


def _sanitise_inf(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with ``|x| > _IPOPT_INF`` replaced by ``±np.inf``."""
    out = arr.astype(float, copy=True)
    out[out >  _IPOPT_INF] =  np.inf
    out[out < -_IPOPT_INF] = -np.inf
    return out

#: pyfiltersqp ``status`` int → ``(cyipopt_status, status_msg)``.
#: pympcc strategies treat status in {0, 1, 3} as success
#: (see ``DirectStrategy.solve``); only success maps to 0 here, every
#: failure maps to a distinct negative int so ``_decode_msg`` can
#: surface the actual cause.
_STATUS_MAP: dict[int, tuple[int, str]] = {
    0: (0,   "Solve_Succeeded"),
    1: (-1,  "Maximum_Iterations_Exceeded"),
    2: (-3,  "Restoration_Failed"),
    3: (-10, "QP_Subproblem_Failed"),
}

#: cyipopt option names that have no pyfiltersqp analogue and are
#: silently dropped (debug-logged).
_IGNORED_OPTIONS: frozenset[str] = frozenset({
    "max_cpu_time",
    "max_wall_time",
    "warm_start_init_point",
    "sb",
    "hessian_approximation",
    "linear_solver_fn",
    "linear_solver",
})


class _FilterSQPAdapter:
    """pyfiltersqp backend adapter.

    Mirrors the cyipopt.Problem subset that
    :class:`pympcc.strategies._base.BaseStrategy` calls into:

    * ``add_option(key, val)`` once per IPOPT option.
    * ``solve(x0, lagrange=…, zl=…, zu=…)`` returning ``(x, info)``
      where ``info`` is a dict with the same keys
      ``status, status_msg, obj_val, mult_g, mult_x_L, mult_x_U``
      that ``cyipopt.Problem.solve`` returns.

    The constructor signature matches the ``adapter = _FilterSQPAdapter(...)``
    call in ``BaseStrategy._build_nlp``.
    """

    def __init__(
        self,
        *,
        n: int,
        m: int,
        xl: np.ndarray,
        xu: np.ndarray,
        cl: np.ndarray,
        cu: np.ndarray,
        obj_fn: Callable,
        grad_fn: Callable,
        con_fn: Optional[Callable],
        jac_fn: Optional[Callable],
        solver_options: Optional[dict] = None,
    ) -> None:
        self._n = int(n)
        self._m = int(m)
        # Sanitise IPOPT-style ``±2e19`` sentinels to ``±np.inf`` so
        # the row-classifier and pyfiltersqp's bound handling agree.
        self._xl = _sanitise_inf(np.asarray(xl, dtype=float))
        self._xu = _sanitise_inf(np.asarray(xu, dtype=float))
        cl_arr = _sanitise_inf(np.asarray(cl, dtype=float))
        cu_arr = _sanitise_inf(np.asarray(cu, dtype=float))
        self._con_fn = con_fn
        self._jac_fn = jac_fn
        # User-supplied solver_options is forwarded directly to SQPSolver,
        # so reject keys it doesn't accept here rather than at solve time.
        # (cyipopt-style options arrive via add_option below and follow a
        # different — silently-translating — path.)
        raw_opts = dict(solver_options or {})
        bad = [k for k in raw_opts if k not in _SQP_KWARGS]
        if bad:
            raise TypeError(
                f"_FilterSQPAdapter: solver_options contains keys not "
                f"accepted by SQPSolver: {bad}.  Valid keys are "
                f"{sorted(_SQP_KWARGS)}."
            )
        self._solver_options: dict[str, Any] = raw_opts
        self._option_overrides: dict[str, Any] = {}

        # --- Classify constraint rows --------------------------------- #
        if self._m > 0:
            eq_mask    = (cl_arr == cu_arr) & np.isfinite(cl_arr)
            upper_mask = (~eq_mask) & (~np.isfinite(cl_arr)) & np.isfinite(cu_arr)
            lower_mask = (~eq_mask) & np.isfinite(cl_arr) & (~np.isfinite(cu_arr))
            range_mask = (~eq_mask) & (~upper_mask) & (~lower_mask)
            if range_mask.any():
                offending = np.where(range_mask)[0].tolist()
                raise NotImplementedError(
                    "_FilterSQPAdapter: range constraints (both cl and cu "
                    "finite with cl < cu) are not supported.  Affected "
                    f"rows: {offending[:5]}"
                    f"{'…' if len(offending) > 5 else ''}"
                )
        else:
            eq_mask = upper_mask = lower_mask = np.zeros(0, dtype=bool)

        self._eq_rows    = np.where(eq_mask)[0].astype(np.intp)
        self._upper_rows = np.where(upper_mask)[0].astype(np.intp)
        self._lower_rows = np.where(lower_mask)[0].astype(np.intp)
        self._eq_rhs    = cl_arr[self._eq_rows]
        self._upper_rhs = cu_arr[self._upper_rows]
        self._lower_rhs = cl_arr[self._lower_rows]
        self._m_eq   = int(self._eq_rows.size)
        self._m_ineq = int(self._upper_rows.size + self._lower_rows.size)

        # --- Build pyfiltersqp NLPProblem ----------------------------- #
        self._problem = NLPProblem(
            n=self._n,
            m_eq=self._m_eq,
            m_ineq=self._m_ineq,
            xl=self._xl,
            xu=self._xu,
            objective=obj_fn,
            gradient=grad_fn,
            eq_constraints=self._build_eq_callable() if self._m_eq else None,
            eq_jacobian=self._build_eq_jacobian()    if self._m_eq else None,
            ineq_constraints=self._build_ineq_callable() if self._m_ineq else None,
            ineq_jacobian=self._build_ineq_jacobian()    if self._m_ineq else None,
        )

        # --- Diagnostic stubs the strategies read --------------------- #
        # BaseStrategy reads these via ``getattr``-style access in
        # `_run_epsilon_continuation`, so they must exist as plain ints.
        self.n_ipopt_iter: int = 0
        self.entered_restoration: bool = False
        self.restoration_iter_count: int = 0
        self.last_alg_mod: int = 0

    # ------------------------------------------------------------------ #
    # cyipopt.Problem-compatible surface                                 #
    # ------------------------------------------------------------------ #

    def reset_iter_counters(self) -> None:
        """Zero per-solve diagnostics so an outer-loop iteration sees a
        fresh count."""
        self.n_ipopt_iter = 0
        self.entered_restoration = False
        self.restoration_iter_count = 0
        self.last_alg_mod = 0

    def add_option(self, key: str, val: Any) -> None:
        """Translate a cyipopt option to its pyfiltersqp equivalent.

        Common options (``tol``, ``max_iter``, ``print_level``) are
        renamed.  Any other key is forwarded only when it appears in
        :data:`_SQP_KWARGS` (the allowlist of kwargs that
        :class:`pyfiltersqp.SQPSolver` actually accepts); everything
        else — IPOPT-only options like ``hsllib``, ``nlp_scaling_method``,
        and the explicit names in :data:`_IGNORED_OPTIONS` — is dropped
        with a ``logger.debug`` line.  This makes
        ``backend='ipopt' → 'filterSQP'`` a frictionless swap even for
        IPOPT options without an SQP analogue.
        """
        if key == "tol":
            self._option_overrides["tol_opt"] = float(val)
        elif key == "max_iter":
            self._option_overrides["max_iter"] = int(val)
        elif key == "print_level":
            self._option_overrides["verbose"] = bool(int(val))
        elif key in _IGNORED_OPTIONS:
            _log.debug("filterSQP: ignoring cyipopt-only option %r=%r",
                       key, val)
        elif key in _SQP_KWARGS:
            _log.debug("filterSQP: forwarding %r=%r to SQPSolver", key, val)
            self._option_overrides[key] = val
        else:
            _log.debug("filterSQP: dropping unknown / IPOPT-only option "
                       "%r=%r (not in SQPSolver kwargs)", key, val)

    def solve(
        self,
        x0: np.ndarray,
        lagrange: Optional[np.ndarray] = None,
        zl: Optional[np.ndarray] = None,
        zu: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run a single SQP solve and return ``(x, info)``."""
        opts = {**self._solver_options, **self._option_overrides}
        solver = SQPSolver(**opts)

        lam_eq0, lam_ineq0 = self._split_warm_lagrange(lagrange)

        result = solver.solve(
            self._problem,
            np.asarray(x0, dtype=float),
            lam_eq0=lam_eq0,
            lam_ineq0=lam_ineq0,
        )

        cyi_status, cyi_msg = _STATUS_MAP.get(
            result.status, (-99, f"pyfiltersqp_status_{result.status}")
        )
        info: dict[str, Any] = {
            "status":     cyi_status,
            "status_msg": cyi_msg,
            "obj_val":    float(result.obj),
            "mult_g":     self._merge_multipliers(result.lam_eq, result.lam_ineq),
            "mult_x_L":   np.asarray(result.lam_lb, dtype=float),
            "mult_x_U":   np.asarray(result.lam_ub, dtype=float),
        }
        self.n_ipopt_iter = int(result.iterations)
        return np.asarray(result.x, dtype=float), info

    # ------------------------------------------------------------------ #
    # Internal: callable wrappers                                        #
    # ------------------------------------------------------------------ #

    def _build_eq_callable(self) -> Callable:
        rows = self._eq_rows
        rhs  = self._eq_rhs
        con_fn = self._con_fn
        assert con_fn is not None  # m_eq > 0 ⇒ caller supplied con_fn

        def eq_c(x: np.ndarray) -> np.ndarray:
            return np.asarray(con_fn(x), dtype=float)[rows] - rhs
        return eq_c

    def _build_eq_jacobian(self) -> Callable:
        rows = self._eq_rows
        jac_fn = self._jac_fn
        assert jac_fn is not None

        def eq_J(x: np.ndarray):
            return _index_jacobian_rows(jac_fn(x), rows)
        return eq_J

    def _build_ineq_callable(self) -> Callable:
        u_rows, u_rhs = self._upper_rows, self._upper_rhs
        l_rows, l_rhs = self._lower_rows, self._lower_rhs
        con_fn = self._con_fn
        assert con_fn is not None

        def ineq_c(x: np.ndarray) -> np.ndarray:
            c_full = np.asarray(con_fn(x), dtype=float)
            return np.concatenate([
                c_full[u_rows] - u_rhs,    # c ≤ cu  →  c − cu ≤ 0
                l_rhs - c_full[l_rows],    # c ≥ cl  →  cl − c ≤ 0
            ])
        return ineq_c

    def _build_ineq_jacobian(self) -> Callable:
        u_rows, l_rows = self._upper_rows, self._lower_rows
        jac_fn = self._jac_fn
        assert jac_fn is not None

        def ineq_J(x: np.ndarray):
            J = jac_fn(x)
            J_u = _index_jacobian_rows(J, u_rows)
            J_l = _index_jacobian_rows(J, l_rows)
            # ``cl − c ≤ 0`` ⇒ Jacobian entry is ``−J``
            if u_rows.size == 0:
                return -J_l if l_rows.size else _empty_jacobian(self._n, J)
            if l_rows.size == 0:
                return J_u
            return _vstack_jac(J_u, _negate_jac(J_l))
        return ineq_J

    # ------------------------------------------------------------------ #
    # Internal: multiplier translation                                   #
    # ------------------------------------------------------------------ #

    def _merge_multipliers(
        self, lam_eq: np.ndarray, lam_ineq: np.ndarray,
    ) -> np.ndarray:
        """Reassemble pyfiltersqp's split multipliers into the
        per-row ``mult_g`` array of length ``self._m`` that pympcc
        strategies expect."""
        mult_g = np.zeros(self._m, dtype=float)
        if self._m_eq:
            mult_g[self._eq_rows] = np.asarray(lam_eq, dtype=float)
        if self._m_ineq:
            n_u = int(self._upper_rows.size)
            lam_arr = np.asarray(lam_ineq, dtype=float)
            if n_u:
                mult_g[self._upper_rows] = +lam_arr[:n_u]
            if self._lower_rows.size:
                mult_g[self._lower_rows] = -lam_arr[n_u:]
        return mult_g

    def _split_warm_lagrange(
        self, lagrange: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Inverse of :meth:`_merge_multipliers` for a warm-start
        ``lagrange`` vector.  Returns ``(None, None)`` when no
        warm-start is provided so the SQP solver uses its zero seed."""
        if lagrange is None:
            return None, None
        lag = np.asarray(lagrange, dtype=float)
        if lag.size == 0:
            return None, None
        lam_eq0 = lag[self._eq_rows] if self._m_eq else None
        lam_ineq0: Optional[np.ndarray] = None
        if self._m_ineq:
            lam_u = (+lag[self._upper_rows]
                     if self._upper_rows.size else np.empty(0))
            lam_l = (-lag[self._lower_rows]
                     if self._lower_rows.size else np.empty(0))
            lam_ineq0 = np.concatenate([lam_u, lam_l])
        return lam_eq0, lam_ineq0


# --------------------------------------------------------------------------- #
# Sparse / dense Jacobian helpers                                              #
# --------------------------------------------------------------------------- #

def _is_sparse(J: Any) -> bool:
    # Duck-type check so we don't import scipy.sparse unconditionally.
    return hasattr(J, "toarray") and hasattr(J, "shape")


def _index_jacobian_rows(J: Any, rows: np.ndarray):
    """Return ``J[rows]`` for either a dense (m, n) ndarray or a
    scipy sparse matrix, preserving the matrix kind."""
    if _is_sparse(J):
        return J[rows]
    return np.asarray(J, dtype=float)[rows]


def _negate_jac(J: Any):
    if _is_sparse(J):
        return -J
    return -np.asarray(J, dtype=float)


def _vstack_jac(A: Any, B: Any):
    """Stack two row blocks vertically, preserving sparse format
    when both inputs are sparse."""
    if _is_sparse(A) and _is_sparse(B):
        from scipy.sparse import vstack as _sp_vstack  # noqa: PLC0415
        return _sp_vstack([A, B])
    A_d = np.asarray(A, dtype=float) if not _is_sparse(A) else A.toarray()
    B_d = np.asarray(B, dtype=float) if not _is_sparse(B) else B.toarray()
    return np.concatenate([A_d, B_d], axis=0)


def _empty_jacobian(n_cols: int, like: Any):
    """Return an ``(0, n_cols)`` Jacobian in the same kind as *like*."""
    if _is_sparse(like):
        from scipy.sparse import csr_matrix  # noqa: PLC0415
        return csr_matrix((0, n_cols))
    return np.empty((0, n_cols), dtype=float)
