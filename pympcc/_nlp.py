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
        return self._hess_rows, self._hess_cols

    def hessian(
        self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
    ) -> np.ndarray:
        return np.asarray(self._hess_fn(x, lagrange, obj_factor), dtype=float)


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
        else:
            self._rows = np.empty(0, dtype=int)
            self._cols = np.empty(0, dtype=int)

        self.n_ipopt_iter: int = 0
        super().__init__(n=n, m=m, lb=xl, ub=xu, cl=cl, cu=cu)

    # ------------------------------------------------------------------ #
    # cyipopt interface                                                    #
    # ------------------------------------------------------------------ #

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials) -> bool:
        self.n_ipopt_iter = iter_count + 1
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
        if self._m == 0:
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
    ) -> None:
        self._n = n
        self._m = m
        self._obj_fn = obj_fn
        self._grad_fn = grad_fn
        self._con_fn = con_fn
        self._jac_fn = jac_fn
        # Store before super().__init__ in case jacobianstructure() is called early.
        self._jac_rows = np.asarray(jac_rows, dtype=int)
        self._jac_cols = np.asarray(jac_cols, dtype=int)
        # Hessian state — set BEFORE super().__init__.
        self._hess_fn = hess_fn
        if hess_sparsity is not None:
            self._hess_rows = np.asarray(hess_sparsity[0], dtype=np.intp)
            self._hess_cols = np.asarray(hess_sparsity[1], dtype=np.intp)
        else:
            self._hess_rows = self._hess_cols = None
        self.n_ipopt_iter: int = 0
        super().__init__(n=n, m=m, lb=xl, ub=xu, cl=cl, cu=cu)

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                     mu, d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials) -> bool:
        self.n_ipopt_iter = iter_count + 1
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
        if self._m == 0:
            return np.empty(0, dtype=float)
        J = np.asarray(self._jac_fn(x), dtype=float)
        if J.ndim == 1:
            return J                               # sparse-native: already nnz values
        return J[self._jac_rows, self._jac_cols]   # legacy: extract from dense matrix
