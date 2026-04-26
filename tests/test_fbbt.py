"""Tests for FBBT (Feasibility-Based Bound Tightening)."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._presolve import _fbbt, presolve


# ======================================================================= #
# Builders                                                                  #
# ======================================================================= #

def _stub_comps(n, n_comp=1):
    """Return COO-sparse comp_G/H for a trivially feasible problem."""
    G  = lambda x: np.array([x[0]] * n_comp)
    Gj = lambda x: np.ones(n_comp)
    Gs = (np.arange(n_comp), np.zeros(n_comp, dtype=int))
    H  = lambda x: np.array([x[1]] * n_comp)
    Hj = lambda x: np.ones(n_comp)
    Hs = (np.arange(n_comp), np.ones(n_comp, dtype=int))
    return G, Gj, Gs, H, Hj, Hs


def _problem(n, *, x0=None, xl=None, xu=None,
             A_ineq=None, b_ineq=None, A_eq=None, b_eq=None,
             nonlinear_ineq=None):
    """Build an MPCC with linear g(x) = A_ineq·x − b_ineq ≤ 0 and
    h(x) = A_eq·x − b_eq = 0.  Sparsity patterns reflect ``A_*``.
    Optionally append a single nonlinear inequality ``x[0]² − 1 ≤ 0``
    (used to check linearity discrimination).
    """
    if x0 is None:
        x0 = np.zeros(n)
    G, Gj, Gs, H, Hj, Hs = _stub_comps(n)

    rows_ineq, cols_ineq, vals_ineq = [], [], []
    if A_ineq is not None:
        for i, row in enumerate(np.atleast_2d(A_ineq)):
            for j, v in enumerate(row):
                if v != 0:
                    rows_ineq.append(i); cols_ineq.append(j); vals_ineq.append(float(v))
    n_lin_ineq = 0 if A_ineq is None else int(np.atleast_2d(A_ineq).shape[0])

    if nonlinear_ineq:
        # A trailing row x[0]² − 1 ≤ 0; Jacobian entry = 2 x[0].
        nl_row = n_lin_ineq
        rows_ineq.append(nl_row); cols_ineq.append(0); vals_ineq.append(0.0)  # placeholder
    n_ineq = n_lin_ineq + (1 if nonlinear_ineq else 0)

    A_lin = (np.atleast_2d(A_ineq).astype(float) if A_ineq is not None
             else np.zeros((0, n)))
    b_lin = (np.asarray(b_ineq, dtype=float) if b_ineq is not None
             else np.zeros(0))

    def ineq_fn(x):
        out = np.zeros(n_ineq)
        if n_lin_ineq:
            out[:n_lin_ineq] = A_lin @ x - b_lin
        if nonlinear_ineq:
            out[-1] = x[0] ** 2 - 1.0
        return out

    rows_arr = np.asarray(rows_ineq, dtype=int)
    cols_arr = np.asarray(cols_ineq, dtype=int)
    val_arr  = np.asarray(vals_ineq, dtype=float)

    def ineq_jac(x):
        out = val_arr.copy()
        if nonlinear_ineq:
            # last entry is for the nonlinear row: 2*x[0]
            out[-1] = 2.0 * x[0]
        return out

    eq_args = {}
    if A_eq is not None:
        A_eq_arr = np.atleast_2d(A_eq).astype(float)
        b_eq_arr = np.asarray(b_eq, dtype=float)
        n_eq = A_eq_arr.shape[0]
        rows_eq, cols_eq, vals_eq = [], [], []
        for i in range(n_eq):
            for j in range(n):
                if A_eq_arr[i, j] != 0:
                    rows_eq.append(i); cols_eq.append(j); vals_eq.append(float(A_eq_arr[i, j]))
        eq_args = dict(
            n_eq=n_eq,
            eq_constraints=lambda x: A_eq_arr @ x - b_eq_arr,
            eq_jacobian=lambda x: np.asarray(vals_eq, dtype=float),
            eq_jacobian_sparsity=(np.asarray(rows_eq, dtype=int),
                                  np.asarray(cols_eq, dtype=int)),
        )

    kw = {}
    if xl is not None: kw["xl"] = np.asarray(xl, dtype=float)
    if xu is not None: kw["xu"] = np.asarray(xu, dtype=float)

    return pympcc.MPCCProblem(
        n=n, n_comp=1, x0=np.asarray(x0, dtype=float),
        objective=lambda x: float(np.sum(x ** 2)),
        gradient=lambda x: 2.0 * x,
        comp_G=G, comp_G_jacobian=Gj, comp_G_jacobian_sparsity=Gs,
        comp_H=H, comp_H_jacobian=Hj, comp_H_jacobian_sparsity=Hs,
        n_ineq=n_ineq,
        ineq_constraints=ineq_fn if n_ineq else None,
        ineq_jacobian=ineq_jac if n_ineq else None,
        ineq_jacobian_sparsity=(rows_arr, cols_arr) if n_ineq else None,
        **eq_args, **kw,
    )


# ======================================================================= #
# Single-row tightening                                                     #
# ======================================================================= #

class TestSingleRow:
    def test_singleton_ineq_tightens_upper_bound(self):
        # x[0] ≤ 0  →  xu[0] should drop from +inf to 0.
        p = _problem(n=2, x0=np.array([-0.5, 0.0]),
                     A_ineq=[[1.0, 0.0]], b_ineq=[0.0])
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        assert xu[0] == pytest.approx(0.0, abs=1e-9)
        # Other variable untouched.
        assert not np.isfinite(xu[1])
        assert not np.isfinite(xl[1])

    def test_singleton_ineq_tightens_lower_bound(self):
        # −x[0] ≤ 0  →  x[0] ≥ 0  →  xl[0] from −inf to 0.
        p = _problem(n=2, x0=np.array([0.5, 0.0]),
                     A_ineq=[[-1.0, 0.0]], b_ineq=[0.0])
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        assert xl[0] == pytest.approx(0.0, abs=1e-9)

    def test_eq_singleton_pins_variable(self):
        # x[0] = 5  →  both bounds collapse to 5.
        p = _problem(n=2, x0=np.array([5.0, 0.0]),
                     A_eq=[[1.0, 0.0]], b_eq=[5.0])
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        assert xl[0] == pytest.approx(5.0, abs=1e-9)
        assert xu[0] == pytest.approx(5.0, abs=1e-9)


# ======================================================================= #
# Multi-row / iterative                                                     #
# ======================================================================= #

class TestIterative:
    def test_chained_tightening(self):
        # Both vars non-negative.
        # Row 0:  x[0] ≤ 5            →  xu[0] from +inf to 5.
        # Row 1: -x[0] + x[1] ≤ 0     →  given xu[0]=5 and a_0=-1 (so the
        #                                worst case for x[1] uses xu[0]),
        #                                xu[1] ≤ 5.  Two sweeps needed when
        #                                row 1 is visited before row 0.
        p = _problem(
            n=2, x0=np.array([0.0, 0.0]),
            xl=[0.0, 0.0], xu=[100.0, 100.0],
            A_ineq=[[1.0, 0.0], [-1.0, 1.0]], b_ineq=[5.0, 0.0],
        )
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        assert xu[0] == pytest.approx(5.0, abs=1e-9)
        assert xu[1] == pytest.approx(5.0, abs=1e-9)


# ======================================================================= #
# Infeasibility                                                             #
# ======================================================================= #

class TestInfeasibility:
    def test_constant_violation_detected(self):
        # x[0] ≤ 0 with xl[0] = 1, xu[0] = 2.  Min g = 1 > 0 → infeasible.
        # x0 must satisfy bounds even though constraint is violated; FBBT
        # only inspects bounds, not x0-feasibility, for this check.
        p = _problem(n=2, x0=np.array([1.0, 0.0]),
                     xl=[1.0, -10.0], xu=[2.0, 10.0],
                     A_ineq=[[1.0, 0.0]], b_ineq=[0.0])
        _, _, infeas = _fbbt(p)
        assert infeas

    def test_presolve_warns_and_falls_back(self):
        p = _problem(n=2, x0=np.array([1.0, 0.0]),
                     xl=[1.0, -10.0], xu=[2.0, 10.0],
                     A_ineq=[[1.0, 0.0]], b_ineq=[0.0])
        with pytest.warns(UserWarning, match="infeasible"):
            reduced, pmap = presolve(p)
        assert reduced is p
        assert pmap.is_identity


# ======================================================================= #
# Composition with pinned-var elimination                                   #
# ======================================================================= #

class TestComposition:
    def test_fbbt_pins_then_eliminates(self):
        # Equality x[2] = 3 should be detected, then pinned-var pass
        # eliminates column 2.
        p = _problem(n=3, x0=np.array([0.0, 0.0, 3.0]),
                     A_eq=[[0.0, 0.0, 1.0]], b_eq=[3.0])
        reduced, pmap = presolve(p)
        assert pmap.fixed_vars.tolist() == [2]
        assert pmap.fixed_vals.tolist() == [3.0]
        assert reduced.n == 2


# ======================================================================= #
# Robustness                                                                #
# ======================================================================= #

class TestRobustness:
    def test_nonlinear_row_ignored(self):
        # x[0]² ≤ 1.  FBBT must not derive xl/xu changes from this.
        p = _problem(n=2, x0=np.array([0.5, 0.0]), nonlinear_ineq=True)
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        assert not np.isfinite(xu[0])
        assert not np.isfinite(xl[0])

    def test_no_constraints_no_op(self):
        p = _problem(n=2, x0=np.array([0.5, 0.5]))
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        np.testing.assert_array_equal(xl, p.xl)
        np.testing.assert_array_equal(xu, p.xu)

    def test_dense_jacobian_no_op(self):
        # No sparsity provided → FBBT must skip (we don't probe dense jacs).
        G  = lambda x: np.array([x[0]])
        Gj = lambda x: np.array([[1.0, 0.0]])
        H  = lambda x: np.array([x[1]])
        Hj = lambda x: np.array([[0.0, 1.0]])
        p = pympcc.MPCCProblem(
            n=2, n_comp=1, x0=np.array([0.5, 0.5]),
            objective=lambda x: float(np.sum(x ** 2)),
            gradient=lambda x: 2.0 * x,
            comp_G=G, comp_G_jacobian=Gj,
            comp_H=H, comp_H_jacobian=Hj,
            n_ineq=1,
            ineq_constraints=lambda x: np.array([x[0] - 1.0]),
            ineq_jacobian=lambda x: np.array([[1.0, 0.0]]),
        )
        xl, xu, infeas = _fbbt(p)
        assert not infeas
        np.testing.assert_array_equal(xl, p.xl)
        np.testing.assert_array_equal(xu, p.xu)


# ======================================================================= #
# End-to-end through solver                                                 #
# ======================================================================= #

class TestSolverIntegration:
    def test_pinning_via_fbbt_round_trip(self):
        # Equality x[2] = 0.5 is detected by FBBT; presolve pins and
        # eliminates; the solver lifts back to n=3 on return.
        n = 3
        # Pair 0: G=x[0], H=x[1]; standard.
        G  = lambda x: np.array([x[0]])
        Gj = lambda x: np.array([1.0])
        Gs = (np.array([0]), np.array([0]))
        H  = lambda x: np.array([x[1]])
        Hj = lambda x: np.array([1.0])
        Hs = (np.array([0]), np.array([1]))
        # Equality x[2] − 0.5 = 0.
        p = pympcc.MPCCProblem(
            n=n, n_comp=1, x0=np.array([0.5, 0.5, 0.5]),
            objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + x[2] ** 2),
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0),
                                         2.0 * (x[1] - 1.0),
                                         2.0 * x[2]]),
            comp_G=G, comp_G_jacobian=Gj, comp_G_jacobian_sparsity=Gs,
            comp_H=H, comp_H_jacobian=Hj, comp_H_jacobian_sparsity=Hs,
            n_eq=1,
            eq_constraints=lambda x: np.array([x[2] - 0.5]),
            eq_jacobian=lambda x: np.array([1.0]),
            eq_jacobian_sparsity=(np.array([0]), np.array([2])),
        )
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 50, "tol": 1e-7},
                              presolve=True)
        assert result.success
        assert len(result.x) == n
        assert result.x[2] == pytest.approx(0.5, abs=1e-9)
