"""Internal scipy.optimize adapter for standard NLPs."""
from __future__ import annotations

import numpy as np


class _ScipyAdapter:
    """
    Adapts a standard NLP to the ``scipy.optimize.minimize`` interface.

    Uses ``method='trust-constr'`` which handles general equality and
    inequality constraints with bounds and returns Lagrange multipliers
    that can be used for primal warm-starting in outer-loop strategies.

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
        ``J(x) -> ndarray, shape (m, n)`` (dense) or flat 1-D nnz array (sparse).
    jac_rows, jac_cols : ndarray of int, optional
        COO row/column indices for a sparse *jac_fn*.  When provided, flat nnz
        values are returned to scipy as a CSR matrix.
    solver_options : dict, optional
        Initial options applied via :meth:`add_option`.

    Notes
    -----
    Dual warm-starting (``lagrange``, ``zl``, ``zu``) is accepted by
    :meth:`solve` for API compatibility with the IPOPT backend but is
    not forwarded to scipy — ``trust-constr`` does not support
    multiplier warm-starts.  The primal iterate *x0* is used as the
    warm start on every call.

    Exact Hessian callbacks (``hess_fn``) are intentionally not wired
    through; scipy uses its own quasi-Newton approximation.
    """

    def __init__(
        self,
        n: int,
        m: int,
        xl: np.ndarray,
        xu: np.ndarray,
        cl: np.ndarray,
        cu: np.ndarray,
        obj_fn,
        grad_fn,
        con_fn,
        jac_fn,
        jac_rows=None,
        jac_cols=None,
        solver_options=None,
    ) -> None:
        self._n = n
        self._m = m
        self._obj_fn = obj_fn
        self._grad_fn = grad_fn
        self._con_fn = con_fn
        self._jac_fn = jac_fn
        self._xl = np.asarray(xl, dtype=float)
        self._xu = np.asarray(xu, dtype=float)
        self._cl = np.asarray(cl, dtype=float)
        self._cu = np.asarray(cu, dtype=float)
        self._jac_rows = (
            np.asarray(jac_rows, dtype=np.intp) if jac_rows is not None else None
        )
        self._jac_cols = (
            np.asarray(jac_cols, dtype=np.intp) if jac_cols is not None else None
        )
        self.n_ipopt_iter: int = 0
        self._tol: float | None = None
        self._max_iter: int | None = None
        for key, val in (solver_options or {}).items():
            self.add_option(key, val)

    # ------------------------------------------------------------------ #
    # Public interface (mirrors cyipopt.Problem)                           #
    # ------------------------------------------------------------------ #

    def add_option(self, key: str, val) -> None:
        """
        Set a solver option.

        Recognised keys
        ---------------
        ``"tol"``
            Optimality tolerance forwarded to ``minimize(tol=...)``.
        ``"max_iter"``
            Maximum iterations forwarded to ``options["maxiter"]``.

        All other keys (IPOPT-specific: ``print_level``, ``sb``,
        ``warm_start_init_point``, etc.) are silently ignored.
        """
        if key == "tol":
            self._tol = float(val)
        elif key == "max_iter":
            self._max_iter = int(val)

    def solve(
        self,
        x0: np.ndarray,
        lagrange: np.ndarray | None = None,
        zl: np.ndarray | None = None,
        zu: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Solve the NLP from *x0* and return ``(x, info)``.

        Parameters
        ----------
        x0 : ndarray, shape (n,)
            Initial primal iterate.
        lagrange, zl, zu : ndarray, optional
            Dual warm-start variables accepted for API compatibility with
            the IPOPT backend; not used by this adapter.

        Returns
        -------
        x : ndarray, shape (n,)
            Solution (or best iterate if not converged).
        info : dict
            Keys: ``obj_val``, ``status``, ``status_msg``, ``mult_g``,
            ``mult_x_L``, ``mult_x_U``.  ``status`` uses IPOPT-compatible
            codes: 0 = Solved, -1 = MaxIter, 2 = Infeasible.
        """
        from scipy import sparse
        from scipy.optimize import Bounds, NonlinearConstraint, minimize

        n, m = self._n, self._m
        bounds = Bounds(lb=self._xl, ub=self._xu, keep_feasible=False)

        # Build a Jacobian callable for scipy. Sparse NLP paths return CSR
        # matrices so trust-constr can avoid dense (m, n) allocations.
        jac_rows = self._jac_rows
        jac_cols = self._jac_cols
        raw_jac = self._jac_fn

        if jac_rows is not None:
            # Sparse path: flat 1-D nnz values -> CSR (m, n) matrix.
            def _jac(x):
                vals = np.asarray(raw_jac(x), dtype=float)
                if vals.ndim == 1:
                    return sparse.csr_matrix((vals, (jac_rows, jac_cols)), shape=(m, n))
                return sparse.csr_matrix(vals)
        else:
            def _jac(x):
                J = np.asarray(raw_jac(x), dtype=float)
                if J.ndim == 1:
                    return J.reshape(m, n)
                return J

        constraints = NonlinearConstraint(
            fun=self._con_fn,
            lb=self._cl,
            ub=self._cu,
            jac=_jac,
        )

        options: dict = {}
        if self._max_iter is not None:
            options["maxiter"] = self._max_iter

        result = minimize(
            fun=self._obj_fn,
            x0=x0,
            jac=self._grad_fn,
            method="trust-constr",
            bounds=bounds,
            constraints=constraints,
            tol=self._tol,
            options=options or None,
        )

        self.n_ipopt_iter = result.nit

        # Map scipy OptimizeResult to IPOPT-compatible status codes.
        if result.success:
            status = 0
            msg = "Solve_Succeeded"
        else:
            msg = result.message
            if "maximum" in msg.lower() or "iteration" in msg.lower():
                status = -1   # Maximum_Iterations_Exceeded
            else:
                status = 2    # Infeasible_Problem_Detected

        # Extract constraint multipliers from trust-constr's result.v.
        # result.v is a list of arrays, one per constraint object.
        # With a single NonlinearConstraint, result.v[0] has shape (m,).
        mult_g = np.zeros(m)
        v = getattr(result, "v", None)
        if v is not None:
            if isinstance(v, (list, tuple)) and len(v) > 0 and v[0] is not None:
                candidate = np.asarray(v[0], dtype=float).ravel()
                if candidate.shape[0] == m:
                    mult_g = candidate
            elif hasattr(v, "__len__"):
                candidate = np.asarray(v, dtype=float).ravel()
                if candidate.shape[0] == m:
                    mult_g = candidate

        info = {
            "obj_val": float(result.fun),
            "status": status,
            "status_msg": msg,
            "mult_g": mult_g,
            "mult_x_L": np.zeros(n),
            "mult_x_U": np.zeros(n),
        }
        return result.x, info
