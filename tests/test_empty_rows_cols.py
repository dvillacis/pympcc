"""Tests for A4 — empty-row and empty-column removal."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._presolve import (
    _detect_empty_cols,
    _detect_empty_rows,
    presolve,
)


# ======================================================================= #
# Builders                                                                  #
# ======================================================================= #

def _stub_comps(n_comp=1):
    G  = lambda x: np.full(n_comp, x[0])
    Gj = lambda x: np.ones(n_comp)
    Gs = (np.arange(n_comp), np.zeros(n_comp, dtype=int))
    H  = lambda x: np.full(n_comp, x[1])
    Hj = lambda x: np.ones(n_comp)
    Hs = (np.arange(n_comp), np.ones(n_comp, dtype=int))
    return G, Gj, Gs, H, Hj, Hs


def _problem(
    n, *, x0=None, xl=None, xu=None,
    n_ineq=0, ineq_fn=None, ineq_jac=None, ineq_sp=None,
    n_eq=0, eq_fn=None, eq_jac=None, eq_sp=None,
    objective=None, gradient=None,
):
    """Direct builder so empty-row/col tests can fully control sparsities."""
    if x0 is None:
        x0 = np.zeros(n)
    G, Gj, Gs, H, Hj, Hs = _stub_comps()
    if objective is None:
        objective = lambda x: float(np.dot(x, x))
        gradient  = lambda x: 2.0 * np.asarray(x, dtype=float)
    kw = {}
    if xl is not None: kw["xl"] = np.asarray(xl, dtype=float)
    if xu is not None: kw["xu"] = np.asarray(xu, dtype=float)
    return pympcc.MPCCProblem(
        n=n, n_comp=1, x0=np.asarray(x0, dtype=float),
        objective=objective, gradient=gradient,
        comp_G=G, comp_G_jacobian=Gj, comp_G_jacobian_sparsity=Gs,
        comp_H=H, comp_H_jacobian=Hj, comp_H_jacobian_sparsity=Hs,
        n_ineq=n_ineq, ineq_constraints=ineq_fn,
        ineq_jacobian=ineq_jac, ineq_jacobian_sparsity=ineq_sp,
        n_eq=n_eq, eq_constraints=eq_fn,
        eq_jacobian=eq_jac, eq_jacobian_sparsity=eq_sp,
        **kw,
    )


# ======================================================================= #
# Empty-row detection                                                       #
# ======================================================================= #

class TestEmptyRowDetection:
    def test_redundant_constant_ineq_dropped(self):
        # Two ineq rows: row 0 is x[0] ≤ 0 (live), row 1 has empty
        # sparsity and evaluates to −1 at x0 (trivially satisfied).
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def g(x):
            return np.array([x[0], -1.0])
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2,
            n_ineq=2, ineq_fn=g, ineq_jac=J, ineq_sp=(rows, cols),
        )
        di, de, infeas = _detect_empty_rows(p)
        assert not infeas
        assert di.tolist() == [False, True]
        assert de.size == 0

    def test_violated_constant_ineq_flags_infeasible(self):
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def g(x):
            return np.array([x[0], +5.0])  # row 1 = 5 > 0 → infeasible
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2,
            n_ineq=2, ineq_fn=g, ineq_jac=J, ineq_sp=(rows, cols),
        )
        di, de, infeas = _detect_empty_rows(p)
        assert infeas

    def test_satisfied_constant_eq_dropped(self):
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def h(x):
            return np.array([x[0], 0.0])  # row 1 ≡ 0
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2,
            n_eq=2, eq_fn=h, eq_jac=J, eq_sp=(rows, cols),
        )
        di, de, infeas = _detect_empty_rows(p)
        assert not infeas
        assert de.tolist() == [False, True]

    def test_violated_constant_eq_flags_infeasible(self):
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def h(x):
            return np.array([x[0], 0.5])  # |0.5| > tol
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2,
            n_eq=2, eq_fn=h, eq_jac=J, eq_sp=(rows, cols),
        )
        di, de, infeas = _detect_empty_rows(p)
        assert infeas


# ======================================================================= #
# Empty-column detection                                                    #
# ======================================================================= #

class TestEmptyColDetection:
    def test_free_var_pinned_to_zero(self):
        # n=3; only x[0] and x[1] appear in comp_G/H; x[2] absent and
        # objective independent of x[2] (gradient zero in that slot).
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2)
        def grad(x):
            g = np.zeros(3)
            g[0] = 2 * x[0]; g[1] = 2 * x[1]
            return g
        p = _problem(n=3, x0=np.array([0.1, 0.2, 0.3]),
                     objective=obj, gradient=grad)
        free = _detect_empty_cols(p, np.full(3, -np.inf), np.full(3, np.inf))
        assert free.tolist() == [2]

    def test_var_in_constraint_not_pinned(self):
        # x[2] participates in an inequality → must NOT be flagged free.
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2)
        def grad(x):
            g = np.zeros(3)
            g[0] = 2 * x[0]; g[1] = 2 * x[1]
            return g
        rows = np.array([0], dtype=int)
        cols = np.array([2], dtype=int)
        def gineq(x):
            return np.array([x[2] - 1.0])
        def Jineq(x):
            return np.array([1.0])
        p = _problem(n=3, x0=np.array([0.1, 0.2, 0.0]),
                     objective=obj, gradient=grad,
                     n_ineq=1, ineq_fn=gineq, ineq_jac=Jineq,
                     ineq_sp=(rows, cols))
        free = _detect_empty_cols(p, np.full(3, -np.inf), np.full(3, np.inf))
        assert free.size == 0

    def test_var_in_gradient_not_pinned(self):
        # x[2] is absent from constraints but has nonzero gradient.
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2 + 3.0 * x[2])
        def grad(x):
            g = np.zeros(3)
            g[0] = 2 * x[0]; g[1] = 2 * x[1]; g[2] = 3.0
            return g
        p = _problem(n=3, x0=np.array([0.1, 0.2, 0.0]),
                     objective=obj, gradient=grad)
        free = _detect_empty_cols(p, np.full(3, -np.inf), np.full(3, np.inf))
        assert free.size == 0

    def test_clip_to_finite_bound(self):
        # Free var with bounds [5, 10] → fix should be 5 (closest to 0).
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2)
        def grad(x):
            g = np.zeros(3)
            g[0] = 2 * x[0]; g[1] = 2 * x[1]
            return g
        p = _problem(n=3, x0=np.array([0.0, 0.0, 7.0]),
                     xl=[-np.inf, -np.inf, 5.0],
                     xu=[ np.inf,  np.inf, 10.0],
                     objective=obj, gradient=grad)
        reduced, pmap = presolve(p)
        # x[2] should be pinned to 5.0 (clipped from 0).
        assert 2 in pmap.fixed_vars.tolist()
        idx = list(pmap.fixed_vars).index(2)
        assert pmap.fixed_vals[idx] == pytest.approx(5.0, abs=1e-12)


# ======================================================================= #
# End-to-end through presolve()                                             #
# ======================================================================= #

class TestPresolveIntegration:
    def test_empty_row_drops_through_presolve(self):
        # n=2; one live ineq (x[0] ≤ 0) and one empty-row ineq satisfied.
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def g(x):
            return np.array([x[0], -1.0])
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2, x0=np.array([-0.5, 0.0]),
            n_ineq=2, ineq_fn=g, ineq_jac=J, ineq_sp=(rows, cols),
        )
        reduced, pmap = presolve(p)
        assert reduced.n_ineq == 1
        assert pmap.keep_ineq is not None
        assert pmap.keep_ineq.tolist() == [0]

    def test_free_var_eliminated_through_presolve(self):
        # n=3; x[2] absent + objective independent → pinned + eliminated.
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2)
        def grad(x):
            g = np.zeros(3)
            g[0] = 2 * x[0]; g[1] = 2 * x[1]
            return g
        p = _problem(n=3, x0=np.array([0.1, 0.2, 0.3]),
                     objective=obj, gradient=grad)
        reduced, pmap = presolve(p)
        assert reduced.n == 2
        assert 2 in pmap.fixed_vars.tolist()
        idx = list(pmap.fixed_vars).index(2)
        # 0 is in [-inf, +inf] → clipped to 0.
        assert pmap.fixed_vals[idx] == pytest.approx(0.0, abs=1e-12)

    def test_no_op_when_nothing_to_eliminate(self):
        # Simple feasible MPCC with all rows live and all vars used.
        def obj(x):
            return float(x[0] ** 2 + x[1] ** 2)
        def grad(x):
            return 2.0 * np.asarray(x, dtype=float)
        p = _problem(n=2, x0=np.array([0.1, 0.2]),
                     objective=obj, gradient=grad)
        reduced, pmap = presolve(p)
        assert pmap.is_identity

    def test_multiplier_padding_at_dropped_ineq(self):
        # After dropping a redundant ineq row, expand_result must zero-pad
        # mult_g at that index.
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        def g(x):
            return np.array([x[0], -1.0])
        def J(x):
            return np.array([1.0])
        p = _problem(
            n=2, x0=np.array([-0.5, 0.0]),
            n_ineq=2, ineq_fn=g, ineq_jac=J, ineq_sp=(rows, cols),
        )
        reduced, pmap = presolve(p)
        assert reduced.n_ineq == 1

        from pympcc.result import MPCCResult
        # Simulate a reduced result. Layout: [ineq | eq | comp_G | comp_H].
        red_mult = np.array([3.0,    # the surviving ineq multiplier
                             0.5, 0.5])  # comp_G | comp_H scalars (n_comp=1)
        result = MPCCResult(
            x=np.array([0.0, 0.0]),
            obj=0.0, status=0, message="ok",
            G=np.array([0.0]), H=np.array([0.0]),
            comp_residual=0.0, comp_residual_mean=0.0,
            success=True, strategy="test",
            mult_g=red_mult,
        )
        expanded = pmap.expand_result(result, p)
        # mult_g layout in original problem: [ineq(2) | eq(0) | G(1) | H(1)]
        assert expanded.mult_g.tolist() == [3.0, 0.0, 0.5, 0.5]
