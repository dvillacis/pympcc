"""Internal cyipopt adapters for dense and sparse standard NLPs."""
from __future__ import annotations

from typing import Callable

import cyipopt
import numpy as np


class _HessianMixin:
    """
    Mixin that registers exact Hessian callbacks with cyipopt.

    Must be the first base class in a dynamic subclass of :class:`_DenseNLP`
    or :class:`_SparseNLP`.  The host class is responsible for setting
    ``_hess_fn``, ``_hess_rows``, and ``_hess_cols`` **before**
    ``cyipopt.Problem.__init__`` is called.
    """

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._hess_rows, self._hess_cols  # type: ignore[attr-defined]

    def hessian(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> np.ndarray:
        return np.asarray(self._hess_fn(x, lagrange, obj_factor), dtype=float)  # type: ignore[attr-defined]


class _DenseNLP(cyipopt.Problem):
    """
    Adapts a standard NLP to the cyipopt interface using a **dense** Jacobian.

    Problem form::

        min  f(x)
        s.t. cl <= c(x) <= cu
             xl <= x   <= xu

    Parameters
    ----------
    n : int
        Number of decision variables.
    m : int
        Number of constraints.
    xl, xu : ndarray, shape (n,)
        Variable bounds.
    cl, cu : ndarray, shape (m,)
        Constraint bounds.
    obj_fn : callable
        ``f(x) -> float``
    grad_fn : callable
        ``grad_f(x) -> ndarray, shape (n,)``
    con_fn : callable
        ``c(x) -> ndarray, shape (m,)``
    jac_fn : callable
        ``J(x) -> ndarray, shape (m, n)``  — dense constraint Jacobian
    """

    def __init__(
        self,
        n: int,
        m: int,
        xl: np.ndarray,
        xu: np.ndarray,
        cl: np.ndarray,
        cu: np.ndarray,
        obj_fn: Callable,
        grad_fn: Callable,
        con_fn: Callable,
        jac_fn: Callable,
        hess_fn=None,
        hess_sparsity=None,
    ) -> None:
        self._n = n
        self._m = m
        self._obj_fn = obj_fn
        self._grad_fn = grad_fn
        self._con_fn = con_fn
        self._jac_fn = jac_fn

        # Hessian state — set BEFORE super().__init__ so hessianstructure() is
        # ready if cyipopt probes it during setup.
        self._hess_fn = hess_fn
        self._hess_rows: np.ndarray | None
        self._hess_cols: np.ndarray | None
        if hess_sparsity is not None:
            self._hess_rows = np.asarray(hess_sparsity[0], dtype=np.intp)
            self._hess_cols = np.asarray(hess_sparsity[1], dtype=np.intp)
        else:
            self._hess_rows = self._hess_cols = None

        # Pre-compute Jacobian sparsity pattern (row-major dense) BEFORE
        # super().__init__, which may call jacobianstructure() during setup.
        if m > 0:
            self._rows = np.repeat(np.arange(m), n)
            self._cols = np.tile(np.arange(n), m)
        else:  # pragma: no cover
            self._rows = np.empty(0, dtype=int)
            self._cols = np.empty(0, dtype=int)

        self.n_ipopt_iter: int = 0
        # Restoration-phase tracking — populated by intermediate().
        self.entered_restoration: bool = False
        self.restoration_iter_count: int = 0
        self.last_alg_mod: int = 0
        super().__init__(n=n, m=m, lb=xl, ub=xu, cl=cl, cu=cu)

    # ------------------------------------------------------------------ #
    # cyipopt interface                                                    #
    # ------------------------------------------------------------------ #

    def reset_iter_counters(self) -> None:
        """Zero per-solve diagnostic counters; called by strategies before
        each inner ``nlp.solve`` so counters reflect a single attempt."""
        self.n_ipopt_iter = 0
        self.entered_restoration = False
        self.restoration_iter_count = 0
        self.last_alg_mod = 0

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials) -> bool:
        self.n_ipopt_iter = iter_count + 1
        self.last_alg_mod = int(alg_mod)
        # alg_mod == 1 indicates IPOPT is inside the restoration phase
        # (feasibility-restoration sub-NLP).  Persistent restoration is
        # the strongest signal that the current ε is too tight for the
        # local geometry — strategies use this to force a rollback.
        if alg_mod == 1:
            self.entered_restoration = True
            self.restoration_iter_count += 1
        return True

    def objective(self, x: np.ndarray) -> float:
        return float(self._obj_fn(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._grad_fn(x), dtype=float)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._con_fn(x), dtype=float)

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._rows, self._cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._m == 0:  # pragma: no cover
            return np.empty(0, dtype=float)
        J = np.asarray(self._jac_fn(x), dtype=float)
        return J.ravel()


class _SparseNLP(cyipopt.Problem):
    """
    Adapts a standard NLP to the cyipopt interface using a **sparse** Jacobian.

    When strategies supply a sparse-native ``jac_fn`` (one that returns a flat
    1-D array of nnz values directly), ``jacobian`` passes it through unchanged.
    The fallback path accepts a dense ``(m, n)`` matrix and extracts the declared
    nonzero entries via ``J[jac_rows, jac_cols]``.

    Parameters
    ----------
    jac_rows, jac_cols : ndarray of int
        Row and column indices of the nonzero entries (0-based, matching the
        assembled NLP constraint index order).
    """

    def __init__(
        self,
        n: int,
        m: int,
        xl: np.ndarray,
        xu: np.ndarray,
        cl: np.ndarray,
        cu: np.ndarray,
        obj_fn: Callable,
        grad_fn: Callable,
        con_fn: Callable,
        jac_fn: Callable,
        jac_rows: np.ndarray,
        jac_cols: np.ndarray,
        hess_fn=None,
        hess_sparsity=None,
        linear_solver_fn=None,
    ) -> None:
        self._n = n
        self._m = m
        self._obj_fn = obj_fn
        self._grad_fn = grad_fn
        self._con_fn = con_fn
        self._jac_fn = jac_fn
        # Store before super().__init__ in case jacobianstructure() is called
        # early. Canonicalize to pointer-width ints and row-major order so the
        # native IPOPT backend sees a stable sparse structure regardless of how
        # callers assembled the pattern.
        raw_rows = np.asarray(jac_rows, dtype=np.intp)
        raw_cols = np.asarray(jac_cols, dtype=np.intp)
        order = np.lexsort((raw_cols, raw_rows))
        self._jac_order: np.ndarray | None
        if raw_rows.size and not np.array_equal(order, np.arange(raw_rows.size)):  # pragma: no cover
            self._jac_order = order
            self._jac_rows = raw_rows[order]
            self._jac_cols = raw_cols[order]
        else:
            self._jac_order = None
            self._jac_rows = raw_rows
            self._jac_cols = raw_cols
        # Hessian state — set BEFORE super().__init__.
        self._hess_fn = hess_fn
        self._hess_rows: np.ndarray | None
        self._hess_cols: np.ndarray | None
        if hess_sparsity is not None:
            self._hess_rows = np.asarray(hess_sparsity[0], dtype=np.intp)
            self._hess_cols = np.asarray(hess_sparsity[1], dtype=np.intp)
        else:
            self._hess_rows = self._hess_cols = None
        # Custom linear solver — when set, solve() routes through our C++ bridge
        # instead of cyipopt.  Store bounds for PyTNLP construction.
        self._linear_solver_fn = linear_solver_fn
        if linear_solver_fn is not None:
            self._xl = xl
            self._xu = xu
            self._cl = cl
            self._cu = cu
            self._ipopt_opts: dict = {}
        self.n_ipopt_iter: int = 0
        # Restoration-phase tracking — populated by intermediate().
        self.entered_restoration: bool = False
        self.restoration_iter_count: int = 0
        self.last_alg_mod: int = 0
        super().__init__(n=n, m=m, lb=xl, ub=xu, cl=cl, cu=cu)

    def reset_iter_counters(self) -> None:
        """Zero per-solve diagnostic counters; called by strategies before
        each inner ``nlp.solve`` so counters reflect a single attempt."""
        self.n_ipopt_iter = 0
        self.entered_restoration = False
        self.restoration_iter_count = 0
        self.last_alg_mod = 0

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials) -> bool:
        self.n_ipopt_iter = iter_count + 1
        self.last_alg_mod = int(alg_mod)
        if alg_mod == 1:
            self.entered_restoration = True
            self.restoration_iter_count += 1
        return True

    def objective(self, x: np.ndarray) -> float:
        return float(self._obj_fn(x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._grad_fn(x), dtype=float)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._con_fn(x), dtype=float)

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        return self._jac_rows, self._jac_cols

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._m == 0:  # pragma: no cover
            return np.empty(0, dtype=float)
        J = np.asarray(self._jac_fn(x), dtype=float)
        if J.ndim == 1:
            if self._jac_order is not None:  # pragma: no cover
                return J[self._jac_order]
            return J                              # sparse-native: already nnz values
        return J[self._jac_rows, self._jac_cols]   # legacy: extract from dense matrix  # pragma: no cover

    def add_option(self, key: str, val) -> None:
        if self._linear_solver_fn is not None:
            self._ipopt_opts[key] = val
        super().add_option(key, val)

    def solve(self, x0, lagrange=None, zl=None, zu=None):
        """Solve the NLP.

        When *linear_solver_fn* was supplied at construction, delegates to
        :func:`pympcc.cython._custom_solver.solve_with_preconditioner` (our
        C++ IPOPT bridge).  Otherwise falls through to :meth:`cyipopt.Problem.solve`.
        """
        if lagrange is None:
            lagrange = []
        if zl is None:
            zl = []
        if zu is None:
            zu = []

        if self._linear_solver_fn is None:
            return super().solve(x0, lagrange=lagrange, zl=zl, zu=zu)

        from .cython._custom_solver import PyTNLP, solve_with_preconditioner

        # Hessian wrapper: reorder args from cyipopt convention (x, lagrange, obj_factor)
        # to PyTNLP convention (x, obj_factor, lambda).
        hess_fn = None
        hess_rows = None
        hess_cols = None
        if self._hess_fn is not None:
            def hess_fn(x, obj_factor, lam):  # noqa: E306
                return self.hessian(x, lam, obj_factor)
            hess_rows = (np.asarray(self._hess_rows, dtype=np.int32)
                         if self._hess_rows is not None else None)
            hess_cols = (np.asarray(self._hess_cols, dtype=np.int32)
                         if self._hess_cols is not None else None)

        # Mirror cyipopt: set hessian_approximation when no exact Hessian is provided.
        opts = dict(self._ipopt_opts)
        if hess_fn is None and "hessian_approximation" not in opts:
            opts["hessian_approximation"] = "limited-memory"

        tnlp = PyTNLP(
            n=self._n, m=self._m,
            xl=self._xl, xu=self._xu,
            cl=self._cl, cu=self._cu,
            obj_fn=self.objective,
            grad_fn=self.gradient,
            con_fn=self.constraints,
            jac_fn=self.jacobian,
            jac_rows=np.asarray(self._jac_rows, dtype=np.int32),
            jac_cols=np.asarray(self._jac_cols, dtype=np.int32),
            hess_fn=hess_fn,
            hess_rows=hess_rows,
            hess_cols=hess_cols,
            x0=np.asarray(x0, dtype=np.float64),
            lagrange0=(np.asarray(lagrange, dtype=np.float64)
                       if len(lagrange) > 0 else None),
            zl0=(np.asarray(zl, dtype=np.float64) if len(zl) > 0 else None),
            zu0=(np.asarray(zu, dtype=np.float64) if len(zu) > 0 else None),
        )
        return solve_with_preconditioner(tnlp, self._linear_solver_fn, opts)
