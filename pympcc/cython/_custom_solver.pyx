# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython interface to the pympcc custom IPOPT linear solver bridge.

Public API
----------
PyTNLP
    Holds NLP problem data and callbacks.  Pass to ``solve_with_preconditioner``.

solve_with_preconditioner(py_tnlp, solver_fn, ipopt_options) -> (x, info)
    Solve the NLP using IPOPT with a custom linear solver callable.
"""

from cpython.ref cimport PyObject

cimport numpy as cnp
import numpy as np

from ._ipopt_cpp cimport run_solve

# Ensure NumPy C API is initialised before any PyArray_* call in the C++ layer
cnp.import_array()


# ---------------------------------------------------------------------------
# PyTNLP — Python-visible holder for NLP data and callbacks
# ---------------------------------------------------------------------------

class PyTNLP:
    """
    NLP problem description for :func:`solve_with_preconditioner`.

    Parameters
    ----------
    n, m : int
        Number of variables / constraints.
    xl, xu : array_like, shape (n,)
        Variable bounds.
    cl, cu : array_like, shape (m,)
        Constraint bounds (equality: cl[i] == cu[i]).
    obj_fn : callable  ``f(x) -> float``
    grad_fn : callable  ``f(x) -> ndarray (n,)``
    con_fn : callable  ``f(x) -> ndarray (m,)``
    jac_fn : callable  ``f(x) -> ndarray (nnz_jac_g,)``  flat COO values
    jac_rows, jac_cols : array_like of int32, shape (nnz_jac_g,)
        0-based COO sparsity pattern of the Jacobian.
    hess_fn : callable or None
        ``f(x, obj_factor, lambda) -> ndarray (nnz_h_lag,)`` lower-triangle.
        If None, IPOPT uses L-BFGS.
    hess_rows, hess_cols : array_like of int32 or None
    x0 : array_like (n,)
        Initial primal iterate.  Defaults to zeros if None.
    lagrange0, zl0, zu0 : array_like or None
        Initial dual iterates for warm-starting.
    """

    def __init__(
        self, n, m, xl, xu, cl, cu,
        obj_fn, grad_fn, con_fn,
        jac_fn, jac_rows, jac_cols,
        hess_fn=None, hess_rows=None, hess_cols=None,
        x0=None, lagrange0=None, zl0=None, zu0=None,
    ):
        self.n = int(n)
        self.m = int(m)
        self.nnz_jac_g = int(np.asarray(jac_rows).size)
        self.nnz_h_lag = (int(np.asarray(hess_rows).size)
                          if hess_rows is not None else 0)

        self.xl = np.asarray(xl, dtype=np.float64)
        self.xu = np.asarray(xu, dtype=np.float64)
        self.cl = np.asarray(cl, dtype=np.float64)
        self.cu = np.asarray(cu, dtype=np.float64)

        self.obj_fn  = obj_fn
        self.grad_fn = grad_fn
        self.con_fn  = con_fn
        self.jac_fn  = jac_fn
        self.jac_rows = np.asarray(jac_rows, dtype=np.int32)
        self.jac_cols = np.asarray(jac_cols, dtype=np.int32)

        self.hess_fn   = hess_fn
        self.hess_rows = (np.asarray(hess_rows, dtype=np.int32)
                          if hess_rows is not None else None)
        self.hess_cols = (np.asarray(hess_cols, dtype=np.int32)
                          if hess_cols is not None else None)

        self.x0 = (np.asarray(x0, dtype=np.float64)
                   if x0 is not None else np.zeros(n, dtype=np.float64))
        self.lagrange0 = (np.asarray(lagrange0, dtype=np.float64)
                          if lagrange0 is not None else None)
        self.zl0 = (np.asarray(zl0, dtype=np.float64)
                    if zl0 is not None else None)
        self.zu0 = (np.asarray(zu0, dtype=np.float64)
                    if zu0 is not None else None)

        # Results filled by finalize_solution (via PyTNLPBridge in C++)
        self.x_sol      = None
        self.z_l_sol    = None
        self.z_u_sol    = None
        self.g_sol      = None
        self.mult_g_sol = None
        self.obj_sol    = None
        self.status_sol = None


# ---------------------------------------------------------------------------
# solve_with_preconditioner
# ---------------------------------------------------------------------------

# Map IPOPT ApplicationReturnStatus codes to status message bytes
# (matching cyipopt convention so strategies can call _decode_msg on it)
_IPOPT_STATUS_MSG = {
    0:    b"Solve_Succeeded",
    1:    b"Solved_To_Acceptable_Level",
    2:    b"Infeasible_Problem_Detected",
    3:    b"Search_Direction_Becomes_Too_Small",
    4:    b"Diverging_Iterates",
    5:    b"User_Requested_Stop",
    -1:   b"Maximum_Iterations_Exceeded",
    -2:   b"Restoration_Failed",
    -3:   b"Error_In_Step_Computation",
    -4:   b"Maximum_CpuTime_Exceeded",
    -10:  b"Not_Enough_Degrees_Of_Freedom",
    -11:  b"Invalid_Problem_Definition",
    -12:  b"Invalid_Option",
    -13:  b"Invalid_Number_Detected",
    -100: b"Unrecoverable_Exception",
    -102: b"Insufficient_Memory",
    -199: b"Internal_Error",
}


def solve_with_preconditioner(py_tnlp, solver_fn, dict ipopt_options=None):
    """
    Solve the NLP described by *py_tnlp* with IPOPT using a custom linear solver.

    Parameters
    ----------
    py_tnlp : PyTNLP
        NLP problem description.
    solver_fn : callable
        ``solver_fn(dim, ia, ja, vals, nrhs, rhs) -> (solution, neg_evals)``

        * ``dim`` — int, matrix size
        * ``ia``, ``ja`` — int32 ndarray (nnz,), COO lower-triangle **0-based** indices
        * ``vals`` — float64 ndarray (nnz,), matrix values (duplicate entries must
          be **summed** — IPOPT's triplet format may repeat the same (i,j) position)
        * ``nrhs`` — int, number of right-hand sides (usually 1)
        * ``rhs``  — float64 ndarray (dim * nrhs,)
        * Returns: ``(solution, neg_evals)`` — float64 ndarray (dim*nrhs,) and int

    ipopt_options : dict
        IPOPT options passed as ``{str: int | float | str}``.

    Returns
    -------
    x : ndarray (n,)
        Primal solution.
    info : dict
        ``status``, ``status_msg``, ``obj_val``, ``mult_g``, ``mult_x_L``,
        ``mult_x_U``.  Compatible with cyipopt's info dict so pympcc strategies
        can consume it without modification.
    """
    if ipopt_options is None:
        ipopt_options = {}

    cdef int ret = run_solve(
        <PyObject*>py_tnlp,
        <PyObject*>solver_fn,
        <PyObject*>ipopt_options,
    )

    x = py_tnlp.x_sol
    if x is None:
        x = np.zeros(py_tnlp.n, dtype=np.float64)

    # ret is Ipopt::ApplicationReturnStatus from OptimizeNLP — the same enum that
    # cyipopt exposes and that IPOPTStatus / _IPOPT_STATUS_MSG are keyed to.
    # py_tnlp.status_sol holds Ipopt::SolverReturn from finalize_solution, which
    # has different integer values (e.g. ERROR_IN_STEP_COMPUTATION is 10 in
    # SolverReturn but -3 in ApplicationReturnStatus), so we must NOT use it here.
    status = ret
    info = {
        "status":     status,
        "status_msg": _IPOPT_STATUS_MSG.get(status, b"Unknown"),
        "obj_val":    py_tnlp.obj_sol,
        "mult_g":     py_tnlp.mult_g_sol,
        "mult_x_L":   py_tnlp.z_l_sol,
        "mult_x_U":   py_tnlp.z_u_sol,
        "x":          x,
    }
    return x, info
