"""Abstract base class for MPCC reformulation strategies."""
from __future__ import annotations

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
from ..result import IPOPTStatus, MPCCResult


class BaseStrategy(ABC):
    """
    Abstract base for MPCC reformulation strategies.

    Each concrete strategy transforms the MPCC into one or more standard NLPs
    and delegates the actual solving to cyipopt via :class:`_DenseNLP`.
    """

    name: str = "base"

    def __init__(self, problem: MPCCProblem, ipopt_options: dict, *,
                 backend: str = "ipopt", solver_options: dict | None = None,
                 **kwargs) -> None:
        self.problem = problem
        self.ipopt_options = ipopt_options
        self.backend = backend
        self.solver_options = solver_options or {}
        self.callback = kwargs.pop("callback", None)
        # Strategies that accept no extra kwargs (e.g. DirectStrategy) inherit
        # this base __init__; unknown kwargs are silently ignored so that
        # callers can always pass e.g. epsilon_0/max_iter without branching.

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
        )
        base: type[_DenseNLP] | type[_SparseNLP]
        if jac_structure is not None:
            base = _SparseNLP
            extra = dict(jac_rows=jac_structure[0], jac_cols=jac_structure[1])
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
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate complementarity functions and Jacobians.

        Returns ``(G, H, JG, JH)`` where ``JG`` and ``JH`` are always dense
        ``(n_comp, n)`` arrays regardless of whether the problem uses sparse
        structures.
        """
        p = self.problem
        G  = np.asarray(p.comp_G(x))
        H  = np.asarray(p.comp_H(x))
        JG = self._to_dense_block(
            p.comp_G_jacobian(x), p.comp_G_jacobian_sparsity, p.n_comp, p.n  # type: ignore[operator]
        )
        JH = self._to_dense_block(
            p.comp_H_jacobian(x), p.comp_H_jacobian_sparsity, p.n_comp, p.n  # type: ignore[operator]
        )
        return G, H, JG, JH

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

    def _eval_standard_con_values(self, x: np.ndarray) -> list[np.ndarray]:
        """
        Evaluate standard constraint *values* only — no Jacobians.

        Use this in ``constraints(x)`` callbacks where the Jacobian is
        not needed.  Calling ``_build_standard_constraints`` from a
        constraints callback unnecessarily invokes the user's Jacobian
        callables and allocates dense ``(n_rows, n)`` blocks that are
        immediately discarded.
        """
        p = self.problem
        parts: list[np.ndarray] = []
        if p.n_ineq > 0:
            parts.append(np.asarray(p.ineq_constraints(x)))  # type: ignore[misc]
        if p.n_eq > 0:
            parts.append(np.asarray(p.eq_constraints(x)))  # type: ignore[misc]
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

    def _build_std_jac_flat(self, x: np.ndarray) -> np.ndarray:
        """
        Return flat 1-D Jacobian values for the standard ``[g, h]`` blocks.

        Sparse blocks contribute their nnz values as-is; dense blocks are
        ravelled in row-major order (matching the ``jac_structure`` layout).
        """
        p = self.problem
        parts: list = []
        if p.n_ineq > 0:
            J = np.asarray(p.ineq_jacobian(x), dtype=float)  # type: ignore[misc, operator]
            parts.append(J if J.ndim == 1 else J.ravel())
        if p.n_eq > 0:
            J = np.asarray(p.eq_jacobian(x), dtype=float)  # type: ignore[misc, operator]
            parts.append(J if J.ndim == 1 else J.ravel())
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

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

    def _comp_residual(self, x: np.ndarray) -> float:
        """Complementarity infeasibility: max_i |G_i * H_i|."""
        p = self.problem
        G = np.asarray(p.comp_G(x))
        H = np.asarray(p.comp_H(x))
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

    def _build_standard_constraints(self, x: np.ndarray):
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

        return parts, jac_rows
