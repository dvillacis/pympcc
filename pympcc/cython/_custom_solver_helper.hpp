// pympcc custom linear solver infrastructure.
//
// Provides:
//  1. PyLinearSolverBridge  — SparseSymLinearSolverInterface that calls a
//                             Python solver_fn(dim, ia, ja, vals, nrhs, rhs)
//                             -> (solution, neg_evals).
//  2. CustomAlgBuilder      — AlgorithmBuilder subclass that injects
//                             PyLinearSolverBridge via SymLinearSolverFactory.
//  3. PyTNLPBridge          — TNLP subclass bridging to Python object callbacks.
//  4. run_solve()           — single entry point called from Cython.
//
// GIL policy: run_solve is called WITH the GIL held (matching cyipopt's behaviour).
// All callbacks therefore access Python directly with no extra GIL management.

#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "IpAlgBuilder.hpp"
#include "IpTSymLinearSolver.hpp"
#include "IpSparseSymLinearSolverInterface.hpp"
#include "IpIpoptApplication.hpp"
#include "IpTNLP.hpp"
#include "IpTNLPAdapter.hpp"
#include "IpSmartPtr.hpp"
#include "IpReturnCodes.hpp"
#include "IpIpoptData.hpp"
#include "IpIpoptCalculatedQuantities.hpp"

#include <vector>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace pympcc {

// ============================================================================
// Utility: get contiguous float64 ndarray from a Python attribute
// ============================================================================
static PyObject* get_f64_array(PyObject* obj, const char* name) {
    PyObject* v = PyObject_GetAttrString(obj, name);
    if (!v) throw std::runtime_error(std::string("missing attr: ") + name);
    PyObject* arr = PyArray_FROM_OTF(v, NPY_DOUBLE,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    Py_DECREF(v);
    if (!arr) throw std::runtime_error(std::string("bad array attr: ") + name);
    return arr;   // NEW reference
}

static PyObject* get_i32_array(PyObject* obj, const char* name) {
    PyObject* v = PyObject_GetAttrString(obj, name);
    if (!v) throw std::runtime_error(std::string("missing attr: ") + name);
    PyObject* arr = PyArray_FROM_OTF(v, NPY_INT32,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    Py_DECREF(v);
    if (!arr) throw std::runtime_error(std::string("bad i32 attr: ") + name);
    return arr;
}

static int get_int_attr(PyObject* obj, const char* name) {
    PyObject* v = PyObject_GetAttrString(obj, name);
    if (!v) throw std::runtime_error(std::string("missing int attr: ") + name);
    int r = (int)PyLong_AsLong(v);
    Py_DECREF(v);
    return r;
}


// ============================================================================
// 1. PyLinearSolverBridge
// ============================================================================

class PyLinearSolverBridge : public Ipopt::SparseSymLinearSolverInterface {
public:
    explicit PyLinearSolverBridge(PyObject* solver_fn)
        : solver_fn_(solver_fn), dim_(0), nnz_(0), neg_evals_(0) {
        Py_INCREF(solver_fn_);
    }
    ~PyLinearSolverBridge() override { Py_DECREF(solver_fn_); }

    bool InitializeImpl(const Ipopt::OptionsList&, const std::string&) override {
        return true;
    }

    Ipopt::ESymSolverStatus InitializeStructure(
            Ipopt::Index dim, Ipopt::Index nonzeros,
            const Ipopt::Index* ia, const Ipopt::Index* ja) override {
        dim_ = dim;
        nnz_ = nonzeros;
        // Triplet_Format uses 1-based (MA27-style) indices — convert to 0-based
        ia_.resize(nonzeros);
        ja_.resize(nonzeros);
        for (int i = 0; i < nonzeros; ++i) {
            ia_[i] = ia[i] - 1;
            ja_[i] = ja[i] - 1;
        }
        vals_.resize(nonzeros, 0.0);
        return Ipopt::SYMSOLVER_SUCCESS;
    }

    Ipopt::Number* GetValuesArrayPtr() override { return vals_.data(); }

    Ipopt::ESymSolverStatus MultiSolve(
            bool /*new_matrix*/,
            const Ipopt::Index* /*ia*/, const Ipopt::Index* /*ja*/,
            Ipopt::Index nrhs, Ipopt::Number* rhs_vals,
            bool check_NegEVals, Ipopt::Index numberOfNegEVals) override {
        // Build numpy arrays (no-copy views into our buffers and IPOPT's rhs)
        npy_intp nnz_s  = nnz_;
        npy_intp rhs_s  = dim_ * nrhs;

        PyObject* ia_np  = PyArray_SimpleNewFromData(1, &nnz_s, NPY_INT32,
                                                     ia_.data());
        PyObject* ja_np  = PyArray_SimpleNewFromData(1, &nnz_s, NPY_INT32,
                                                     ja_.data());
        // Give Python a *copy* of vals so it can't corrupt IPOPT's buffer
        PyObject* vals_v = PyArray_SimpleNewFromData(1, &nnz_s, NPY_DOUBLE,
                                                      vals_.data());
        PyObject* vals_c = PyArray_NewCopy((PyArrayObject*)vals_v, NPY_CORDER);
        Py_DECREF(vals_v);

        PyObject* rhs_v  = PyArray_SimpleNewFromData(1, &rhs_s, NPY_DOUBLE,
                                                      rhs_vals);
        PyObject* rhs_c  = PyArray_NewCopy((PyArrayObject*)rhs_v, NPY_CORDER);
        Py_DECREF(rhs_v);

        PyObject* dim_o  = PyLong_FromLong(dim_);
        PyObject* nrhs_o = PyLong_FromLong(nrhs);

        PyObject* ret = PyObject_CallFunctionObjArgs(
            solver_fn_, dim_o, ia_np, ja_np, vals_c, nrhs_o, rhs_c, nullptr);

        Py_DECREF(dim_o); Py_DECREF(nrhs_o);
        Py_DECREF(ia_np); Py_DECREF(ja_np);
        Py_DECREF(vals_c); Py_DECREF(rhs_c);

        if (!ret) {
            // Python raised an exception inside solver_fn — propagate it
            return Ipopt::SYMSOLVER_FATAL_ERROR;
        }
        if (!PyTuple_Check(ret) || PyTuple_GET_SIZE(ret) != 2) {
            Py_DECREF(ret);
            PyErr_SetString(PyExc_ValueError,
                "solver_fn must return (solution_array, neg_evals: int)");
            return Ipopt::SYMSOLVER_FATAL_ERROR;
        }

        PyObject* sol_obj = PyTuple_GET_ITEM(ret, 0);
        PyObject* neg_obj = PyTuple_GET_ITEM(ret, 1);
        neg_evals_ = (int)PyLong_AsLong(neg_obj);

        PyObject* sol_arr = PyArray_FROM_OTF(sol_obj, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        Py_DECREF(ret);
        if (!sol_arr) return Ipopt::SYMSOLVER_FATAL_ERROR;

        double* sp = (double*)PyArray_DATA((PyArrayObject*)sol_arr);
        std::copy(sp, sp + dim_ * nrhs, rhs_vals);
        Py_DECREF(sol_arr);

        if (check_NegEVals && neg_evals_ != numberOfNegEVals)
            return Ipopt::SYMSOLVER_WRONG_INERTIA;
        return Ipopt::SYMSOLVER_SUCCESS;
    }

    Ipopt::Index NumberOfNegEVals() const override { return neg_evals_; }
    bool IncreaseQuality() override { return false; }
    bool ProvidesInertia() const override { return true; }
    EMatrixFormat MatrixFormat() const override { return Triplet_Format; }

private:
    PyObject*           solver_fn_;
    int                 dim_, nnz_;
    std::vector<int>    ia_, ja_;
    std::vector<double> vals_;
    int                 neg_evals_;
};


// ============================================================================
// 2. CustomAlgBuilder
// ============================================================================

class CustomAlgBuilder : public Ipopt::AlgorithmBuilder {
public:
    explicit CustomAlgBuilder(
            Ipopt::SmartPtr<Ipopt::SparseSymLinearSolverInterface> si)
        : solver_interface_(si) {}

    Ipopt::SmartPtr<Ipopt::SymLinearSolver> SymLinearSolverFactory(
            const Ipopt::Journalist&,
            const Ipopt::OptionsList&,
            const std::string&) override {
        return new Ipopt::TSymLinearSolver(solver_interface_, NULL);
    }

private:
    Ipopt::SmartPtr<Ipopt::SparseSymLinearSolverInterface> solver_interface_;
};


// ============================================================================
// 3. PyTNLPBridge
//
// Reads from Python object "py_" which is a pympcc.cython.PyTNLP instance.
// All methods run with the GIL held.
// ============================================================================

class PyTNLPBridge : public Ipopt::TNLP {
public:
    explicit PyTNLPBridge(PyObject* py) : py_(py), n_(0), m_(0) {
        Py_INCREF(py_);
    }
    ~PyTNLPBridge() override { Py_DECREF(py_); }

    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m,
                       Ipopt::Index& nnz_jac_g, Ipopt::Index& nnz_h_lag,
                       IndexStyleEnum& index_style) override {
        try {
            n_         = get_int_attr(py_, "n");
            m_         = get_int_attr(py_, "m");
            nnz_jac_g_ = get_int_attr(py_, "nnz_jac_g");
            nnz_h_lag_ = get_int_attr(py_, "nnz_h_lag");
            n = n_;  m = m_;
            nnz_jac_g = nnz_jac_g_;
            nnz_h_lag = nnz_h_lag_;
            index_style = C_STYLE;
            return true;
        } catch (...) { return false; }
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                          Ipopt::Index m, Ipopt::Number* g_l,
                          Ipopt::Number* g_u) override {
        try {
            auto copy_attr = [&](const char* attr, double* dst, int sz) {
                PyObject* arr = get_f64_array(py_, attr);
                std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                          (double*)PyArray_DATA((PyArrayObject*)arr) + sz, dst);
                Py_DECREF(arr);
            };
            copy_attr("xl", x_l, n); copy_attr("xu", x_u, n);
            copy_attr("cl", g_l, m); copy_attr("cu", g_u, m);
            return true;
        } catch (...) { return false; }
    }

    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                             bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                             Ipopt::Index m, bool init_lambda,
                             Ipopt::Number* lambda) override {
        try {
            auto copy_opt = [&](const char* attr, double* dst, int sz) -> bool {
                PyObject* v = PyObject_GetAttrString(py_, attr);
                if (!v || v == Py_None) { Py_XDECREF(v); return false; }
                PyObject* arr = PyArray_FROM_OTF(v, NPY_DOUBLE,
                    NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
                Py_DECREF(v);
                if (!arr || PyArray_SIZE((PyArrayObject*)arr) != sz) {
                    Py_XDECREF(arr); return false;
                }
                std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                          (double*)PyArray_DATA((PyArrayObject*)arr) + sz, dst);
                Py_DECREF(arr);
                return true;
            };
            if (init_x)      copy_opt("x0",       x,      n);
            if (init_z)    { copy_opt("zl0",      z_L,    n);
                             copy_opt("zu0",      z_U,    n); }
            if (init_lambda) copy_opt("lagrange0", lambda, m);
            return true;
        } catch (...) { return false; }
    }

    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool,
                 Ipopt::Number& obj) override {
        npy_intp sz = n;
        PyObject* x_np = PyArray_SimpleNewFromData(1, &sz, NPY_DOUBLE,
                                                    const_cast<double*>(x));
        PyObject* fn   = PyObject_GetAttrString(py_, "obj_fn");
        PyObject* res  = PyObject_CallOneArg(fn, x_np);
        Py_DECREF(fn); Py_DECREF(x_np);
        if (!res) return false;
        obj = PyFloat_AsDouble(res);
        Py_DECREF(res);
        return !PyErr_Occurred();
    }

    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool,
                      Ipopt::Number* grad) override {
        npy_intp sz = n;
        PyObject* x_np = PyArray_SimpleNewFromData(1, &sz, NPY_DOUBLE,
                                                    const_cast<double*>(x));
        PyObject* fn   = PyObject_GetAttrString(py_, "grad_fn");
        PyObject* res  = PyObject_CallOneArg(fn, x_np);
        Py_DECREF(fn); Py_DECREF(x_np);
        if (!res) return false;
        PyObject* arr = PyArray_FROM_OTF(res, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        Py_DECREF(res);
        if (!arr) return false;
        std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                  (double*)PyArray_DATA((PyArrayObject*)arr) + n, grad);
        Py_DECREF(arr);
        return true;
    }

    bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool,
                 Ipopt::Index m, Ipopt::Number* g) override {
        npy_intp sz = n;
        PyObject* x_np = PyArray_SimpleNewFromData(1, &sz, NPY_DOUBLE,
                                                    const_cast<double*>(x));
        PyObject* fn   = PyObject_GetAttrString(py_, "con_fn");
        PyObject* res  = PyObject_CallOneArg(fn, x_np);
        Py_DECREF(fn); Py_DECREF(x_np);
        if (!res) return false;
        PyObject* arr = PyArray_FROM_OTF(res, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        Py_DECREF(res);
        if (!arr) return false;
        std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                  (double*)PyArray_DATA((PyArrayObject*)arr) + m, g);
        Py_DECREF(arr);
        return true;
    }

    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool,
                     Ipopt::Index, Ipopt::Index nele_jac,
                     Ipopt::Index* iRow, Ipopt::Index* jCol,
                     Ipopt::Number* values) override {
        if (iRow != nullptr) {
            // Structure call
            try {
                PyObject* rows = get_i32_array(py_, "jac_rows");
                PyObject* cols = get_i32_array(py_, "jac_cols");
                int32_t* rp = (int32_t*)PyArray_DATA((PyArrayObject*)rows);
                int32_t* cp = (int32_t*)PyArray_DATA((PyArrayObject*)cols);
                for (int i = 0; i < nele_jac; ++i) {
                    iRow[i] = rp[i];
                    jCol[i] = cp[i];
                }
                Py_DECREF(rows); Py_DECREF(cols);
                return true;
            } catch (...) { return false; }
        }
        // Values call
        npy_intp sz = n;
        PyObject* x_np = PyArray_SimpleNewFromData(1, &sz, NPY_DOUBLE,
                                                    const_cast<double*>(x));
        PyObject* fn   = PyObject_GetAttrString(py_, "jac_fn");
        PyObject* res  = PyObject_CallOneArg(fn, x_np);
        Py_DECREF(fn); Py_DECREF(x_np);
        if (!res) return false;
        PyObject* arr = PyArray_FROM_OTF(res, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        Py_DECREF(res);
        if (!arr) return false;
        std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                  (double*)PyArray_DATA((PyArrayObject*)arr) + nele_jac, values);
        Py_DECREF(arr);
        return true;
    }

    // eval_h: returns false if no hess_fn → IPOPT uses L-BFGS
    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool,
                 Ipopt::Number obj_factor, Ipopt::Index m,
                 const Ipopt::Number* lambda, bool,
                 Ipopt::Index nele_hess, Ipopt::Index* iRow, Ipopt::Index* jCol,
                 Ipopt::Number* values) override {
        PyObject* hfn = PyObject_GetAttrString(py_, "hess_fn");
        bool has_hess = hfn && hfn != Py_None && PyCallable_Check(hfn);
        if (!has_hess) { Py_XDECREF(hfn); return false; }

        if (iRow != nullptr) {
            Py_DECREF(hfn);
            try {
                PyObject* rows = get_i32_array(py_, "hess_rows");
                PyObject* cols = get_i32_array(py_, "hess_cols");
                int32_t* rp = (int32_t*)PyArray_DATA((PyArrayObject*)rows);
                int32_t* cp = (int32_t*)PyArray_DATA((PyArrayObject*)cols);
                for (int i = 0; i < nele_hess; ++i) {
                    iRow[i] = rp[i]; jCol[i] = cp[i];
                }
                Py_DECREF(rows); Py_DECREF(cols);
                return true;
            } catch (...) { return false; }
        }
        npy_intp xsz = n, lsz = m;
        PyObject* x_np = PyArray_SimpleNewFromData(1, &xsz, NPY_DOUBLE,
                                                    const_cast<double*>(x));
        PyObject* l_np = PyArray_SimpleNewFromData(1, &lsz, NPY_DOUBLE,
                                                    const_cast<double*>(lambda));
        PyObject* of   = PyFloat_FromDouble(obj_factor);
        PyObject* res  = PyObject_CallFunctionObjArgs(hfn, x_np, of, l_np, nullptr);
        Py_DECREF(hfn); Py_DECREF(x_np); Py_DECREF(l_np); Py_DECREF(of);
        if (!res) return false;
        PyObject* arr = PyArray_FROM_OTF(res, NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        Py_DECREF(res);
        if (!arr) return false;
        std::copy((double*)PyArray_DATA((PyArrayObject*)arr),
                  (double*)PyArray_DATA((PyArrayObject*)arr) + nele_hess, values);
        Py_DECREF(arr);
        return true;
    }

    void finalize_solution(
            Ipopt::SolverReturn status,
            Ipopt::Index n, const Ipopt::Number* x,
            const Ipopt::Number* z_L, const Ipopt::Number* z_U,
            Ipopt::Index m, const Ipopt::Number* g,
            const Ipopt::Number* lambda, Ipopt::Number obj_value,
            const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) override {
        auto mk = [](const double* p, int sz) -> PyObject* {
            npy_intp s = sz;
            PyObject* a = PyArray_SimpleNew(1, &s, NPY_DOUBLE);
            std::copy(p, p + sz, (double*)PyArray_DATA((PyArrayObject*)a));
            return a;
        };
        auto set_array_attr = [&](const char* name, const double* p, int sz) {
            PyObject* value = mk(p, sz);
            PyObject_SetAttrString(py_, name, value);
            Py_DECREF(value);
        };
        set_array_attr("x_sol",      x,      n);
        set_array_attr("z_l_sol",    z_L,    n);
        set_array_attr("z_u_sol",    z_U,    n);
        set_array_attr("g_sol",      g,      m);
        set_array_attr("mult_g_sol", lambda, m);
        PyObject* ov = PyFloat_FromDouble(obj_value);
        PyObject* st = PyLong_FromLong((long)status);
        PyObject_SetAttrString(py_, "obj_sol",    ov);
        PyObject_SetAttrString(py_, "status_sol", st);
        Py_DECREF(ov); Py_DECREF(st);
    }

private:
    PyObject* py_;
    int n_, m_, nnz_jac_g_, nnz_h_lag_;
};


// ============================================================================
// 4. run_solve — single entry point called from Cython WITH the GIL held.
//
//   py_tnlp_obj  : PyTNLP Cython object
//   solver_fn_obj: Python callable matching PyLinearSolverBridge's expected sig
//   options_dict : Python dict {str: int|float|str}
//
// Returns ApplicationReturnStatus as int.
// ============================================================================
inline int run_solve(
        PyObject* py_tnlp_obj,
        PyObject* solver_fn_obj,
        PyObject* options_dict) {

    // Create bridge objects
    Ipopt::SmartPtr<PyLinearSolverBridge> bridge =
        new PyLinearSolverBridge(solver_fn_obj);
    Ipopt::SmartPtr<CustomAlgBuilder> builder =
        new CustomAlgBuilder(bridge);
    Ipopt::SmartPtr<PyTNLPBridge> tnlp_bridge =
        new PyTNLPBridge(py_tnlp_obj);
    Ipopt::SmartPtr<Ipopt::NLP> nlp =
        new Ipopt::TNLPAdapter(tnlp_bridge);

    // Create application (IpoptApplicationFactory is extern "C", not Ipopt::)
    Ipopt::SmartPtr<Ipopt::IpoptApplication> app =
        IpoptApplicationFactory();

    // Set options from Python dict
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(options_dict, &pos, &key, &value)) {
        const char* k = PyUnicode_AsUTF8(key);
        if (!k) continue;
        std::string ks(k);
        if (PyBool_Check(value)) {
            // bool before int check (bool is subclass of int)
            app->Options()->SetStringValue(ks,
                value == Py_True ? "yes" : "no", true, true);
        } else if (PyLong_Check(value)) {
            app->Options()->SetIntegerValue(ks,
                (int)PyLong_AsLong(value), true, true);
        } else if (PyFloat_Check(value)) {
            app->Options()->SetNumericValue(ks,
                PyFloat_AsDouble(value), true, true);
        } else if (PyUnicode_Check(value)) {
            const char* vs = PyUnicode_AsUTF8(value);
            if (vs)
                app->Options()->SetStringValue(ks, std::string(vs),
                                               true, true);
        }
    }

    // Initialize
    Ipopt::ApplicationReturnStatus init_stat = app->Initialize();
    if (init_stat != Ipopt::Solve_Succeeded) {
        return (int)Ipopt::Internal_Error;
    }

    // Solve with custom builder
    Ipopt::SmartPtr<Ipopt::AlgorithmBuilder> builder_base(
        static_cast<Ipopt::AlgorithmBuilder*>(Ipopt::GetRawPtr(builder)));
    return (int)app->OptimizeNLP(nlp, builder_base);
}

} // namespace pympcc
