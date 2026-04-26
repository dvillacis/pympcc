# Cython extern declarations for pympcc C++ helper types.
#
# Only run_solve() is declared here — all other C++ types are internal
# to _custom_solver_helper.hpp and not exposed to Cython directly.

from cpython.ref cimport PyObject

cdef extern from "_custom_solver_helper.hpp" namespace "pympcc":
    int run_solve(
        PyObject* py_tnlp_obj,
        PyObject* solver_fn_obj,
        PyObject* options_dict,
    ) except +
