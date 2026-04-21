"""
Tests for StructuredMPCC — the mixed linear/nonlinear constraint model.

Three canonical problems are used:

1. **lin_only** — only linear equalities (bard1 KKT, no nonlinear eq).
   Shows that StructuredMPCC reproduces MPCCProblem when n_nl_eq = 0.

2. **nl_only** — only nonlinear equalities (Dempe-style).
   Shows the nonlinear-only path.

3. **mixed** — BOTH linear and nonlinear equalities in the same model.
   This is the primary use-case StructuredMPCC was designed for.

   Problem::

       min  x[0]^2 + x[1]^2 + x[2]^2
       s.t. x[0] + x[1] = 1              (linear equality)
            x[0]^2 + x[2] = 1            (nonlinear equality)
            x[0] >= 0,  x[1] >= 0,  x[2] >= 0
            0 <= x[0]  complementary  x[1] >= 0

   Analysis:
     Branch x[0]=0: x[1]=1 (lin.eq.), x[2]=1 (nl.eq.), f=0+1+1=2
     Branch x[1]=0: x[0]=1 (lin.eq.), x[2]=0 (nl.eq.), f=1+0+0=1  ← global
   Optimal: x*=(1, 0, 0), f*=1.

4. **mixed_ineq** — linear AND nonlinear inequality constraints alongside
   complementarity, no equalities.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.models import StructuredMPCC
from pympcc.problem import MPCCProblem

# ======================================================================= #
# Problem builders                                                          #
# ======================================================================= #

def _make_lin_only() -> StructuredMPCC:
    """
    bard1 reformulated with StructuredMPCC (linear KKT equality only).

    Variables: [x, y, l1, l2, l3],  n=5, n_comp=3
    Linear equality: -1.5x + 2y + l1 - 0.5*l2 + l3 = 2
    Optimal: x=1, y=0, l1=3.5, l2=l3=0, f=17
    """
    return StructuredMPCC(
        n=5, n_comp=3,
        x0=np.array([2.0, 2.0, 1.0, 1.0, 1.0]),
        xl=np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf]),
        objective=lambda x: (x[0]-5)**2 + (2*x[1]+1)**2,
        gradient=lambda x: np.array([
            2*(x[0]-5), 4*(2*x[1]+1), 0.0, 0.0, 0.0
        ]),
        # Linear KKT equality:  -1.5x + 2y + l1 - 0.5*l2 + l3 = 2
        A_eq=np.array([[-1.5, 2.0, 1.0, -0.5, 1.0]]),
        b_eq=np.array([2.0]),
        comp_G=lambda x: np.array([
            3*x[0]-x[1]-3, -x[0]+0.5*x[1]+4, -x[0]-x[1]+7
        ]),
        comp_G_jacobian=lambda x: np.array([
            [ 3., -1., 0., 0., 0.],
            [-1.,  .5, 0., 0., 0.],
            [-1., -1., 0., 0., 0.],
        ]),
        comp_H=lambda x: x[2:5].copy(),
        comp_H_jacobian=lambda x: np.eye(3, 5, k=2),
    )


def _make_nl_only() -> StructuredMPCC:
    """
    scholtes1 reformulated with StructuredMPCC (nonlinear equality only).

    Variables: [x, y1, y2],  n=3, n_comp=1
    Nonlinear equality: none in scholtes1 (uses only complementarity).

    We use a modified scholtes1 where we pin y2 via a nonlinear equality:
        exp(y2) - 1 = 0  →  y2 = 0  (nonlinear)

    With y2 forced to 0:
        G = -exp(x) + y1 - 1 >= 0
    At x=0: y1 >= 2, min (y1-2.5)^2 at y1=2.5, f=1+0+1=2
    """
    import math

    return StructuredMPCC(
        n=3, n_comp=1,
        x0=np.array([1.0, 1.0, 0.0]),
        xl=np.array([0.0, -np.inf, 0.0]),
        objective=lambda x: (x[0]+1)**2 + (x[1]-2.5)**2 + (x[2]+1)**2,
        gradient=lambda x: np.array([
            2*(x[0]+1), 2*(x[1]-2.5), 2*(x[2]+1)
        ]),
        # Nonlinear equality: exp(y2) - 1 = 0  → forces y2 = 0
        n_nl_eq=1,
        eq_nl=lambda x: np.array([math.exp(x[2]) - 1.0]),
        jac_eq_nl=lambda x: np.array([[0.0, 0.0, math.exp(x[2])]]),
        comp_G=lambda x: np.array([-math.exp(x[0]) + x[1] - math.exp(x[2])]),
        comp_G_jacobian=lambda x: np.array([
            [-math.exp(x[0]), 1.0, -math.exp(x[2])]
        ]),
        comp_H=lambda x: np.array([x[0]]),
        comp_H_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
    )


def _make_mixed() -> StructuredMPCC:
    """
    Primary StructuredMPCC test case: BOTH linear and nonlinear equalities.

        min   x[0]^2 + x[1]^2 + x[2]^2
        s.t.  x[0] + x[1] = 1          (linear equality)
              x[0]^2 + x[2] = 1        (nonlinear equality)
              x[0] >= 0, x[1] >= 0, x[2] >= 0
              0 <= x[0]  complementary  x[1] >= 0

    Global optimum: x* = (1, 0, 0), f* = 1.
    """
    return StructuredMPCC(
        n=3, n_comp=1,
        x0=np.array([0.5, 0.5, 0.75]),
        xl=np.zeros(3),
        objective=lambda x: float(x @ x),
        gradient=lambda x: 2*x,
        # Linear equality:  x[0] + x[1] = 1
        A_eq=np.array([[1.0, 1.0, 0.0]]),
        b_eq=np.array([1.0]),
        # Nonlinear equality:  x[0]^2 + x[2] - 1 = 0
        n_nl_eq=1,
        eq_nl=lambda x: np.array([x[0]**2 + x[2] - 1.0]),
        jac_eq_nl=lambda x: np.array([[2*x[0], 0.0, 1.0]]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
    )


def _make_mixed_ineq() -> StructuredMPCC:
    """
    StructuredMPCC with linear AND nonlinear inequality constraints.

        min   (x[0]-2)^2 + (x[1]-1)^2
        s.t.  x[0] + x[1] <= 4           (linear inequality)
              x[0]^2 + x[1]^2 <= 9       (nonlinear inequality)
              x[0] >= 0, x[1] >= 0
              0 <= x[0]  complementary  x[1] >= 0

    The two inequality constraints are inactive at the global optimum
    x* = (2, 0), f* = 1  (same as the unconstrained 'simple' problem).
    """
    return StructuredMPCC(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        xl=np.zeros(2),
        objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
        gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
        # Linear inequality:  x[0] + x[1] - 4 <= 0
        A_ineq=np.array([[1.0, 1.0]]),
        b_ineq=np.array([4.0]),
        # Nonlinear inequality:  x[0]^2 + x[1]^2 - 9 <= 0
        n_nl_ineq=1,
        ineq_nl=lambda x: np.array([x[0]**2 + x[1]**2 - 9.0]),
        jac_ineq_nl=lambda x: np.array([[2*x[0], 2*x[1]]]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# Construction and derived properties                                       #
# ======================================================================= #

class TestDerivedDimensions:
    def test_n_lin_eq_from_A_eq(self):
        m = _make_lin_only()
        assert m.n_lin_eq == 1
        assert m.n_nl_eq == 0
        assert m.n_eq == 1

    def test_n_nl_eq_from_field(self):
        m = _make_nl_only()
        assert m.n_lin_eq == 0
        assert m.n_nl_eq == 1
        assert m.n_eq == 1

    def test_mixed_eq_dimensions(self):
        m = _make_mixed()
        assert m.n_lin_eq == 1
        assert m.n_nl_eq == 1
        assert m.n_eq == 2

    def test_ineq_dimensions(self):
        m = _make_mixed_ineq()
        assert m.n_lin_ineq == 1
        assert m.n_nl_ineq == 1
        assert m.n_ineq == 2

    def test_no_constraints_gives_zero(self):
        m = StructuredMPCC(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: float(x @ x),
            gradient=lambda x: 2*x,
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        assert m.n_lin_eq == 0
        assert m.n_nl_eq == 0
        assert m.n_eq == 0
        assert m.n_lin_ineq == 0
        assert m.n_nl_ineq == 0
        assert m.n_ineq == 0


# ======================================================================= #
# Validation errors                                                         #
# ======================================================================= #

class TestValidation:
    def _base(self):
        return dict(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: float(x @ x),
            gradient=lambda x: 2*x,
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )

    def test_A_eq_without_b_eq_raises(self):
        kw = self._base()
        kw["A_eq"] = np.array([[1.0, 0.0]])
        # b_eq is None — must raise
        with pytest.raises(ValueError, match="b_eq"):
            StructuredMPCC(**kw)

    def test_b_eq_without_A_eq_raises(self):
        kw = self._base()
        kw["b_eq"] = np.array([1.0])
        with pytest.raises(ValueError, match="b_eq"):
            StructuredMPCC(**kw)

    def test_A_eq_wrong_columns_raises(self):
        kw = self._base()
        kw["A_eq"] = np.array([[1.0, 0.0, 0.0]])  # 3 cols, n=2
        kw["b_eq"] = np.array([1.0])
        with pytest.raises(ValueError, match="A_eq"):
            StructuredMPCC(**kw)

    def test_b_eq_wrong_length_raises(self):
        kw = self._base()
        kw["A_eq"] = np.array([[1.0, 0.0]])
        kw["b_eq"] = np.array([1.0, 2.0])  # 2 rows, but A_eq has 1
        with pytest.raises(ValueError, match="b_eq"):
            StructuredMPCC(**kw)

    def test_n_nl_eq_without_callable_raises(self):
        kw = self._base()
        kw["n_nl_eq"] = 1
        # eq_nl and jac_eq_nl are None — must raise
        with pytest.raises(ValueError, match="eq_nl"):
            StructuredMPCC(**kw)

    def test_eq_nl_wrong_shape_raises(self):
        kw = self._base()
        kw["n_nl_eq"] = 1
        kw["eq_nl"] = lambda x: np.array([x[0], x[1]])  # returns 2, expects 1
        kw["jac_eq_nl"] = lambda x: np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="eq_nl"):
            StructuredMPCC(**kw)

    def test_A_ineq_without_b_ineq_raises(self):
        kw = self._base()
        kw["A_ineq"] = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="b_ineq"):
            StructuredMPCC(**kw)

    def test_n_nl_ineq_without_callable_raises(self):
        kw = self._base()
        kw["n_nl_ineq"] = 1
        with pytest.raises(ValueError, match="ineq_nl"):
            StructuredMPCC(**kw)

    def test_n_comp_zero_raises(self):
        kw = self._base()
        kw["n_comp"] = 0
        with pytest.raises(ValueError, match="n_comp"):
            StructuredMPCC(**kw)

    def test_xl_gt_xu_raises(self):
        kw = self._base()
        kw["xl"] = np.array([1.0, 0.0])
        kw["xu"] = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="xl"):
            StructuredMPCC(**kw)


# ======================================================================= #
# to_mpcc_problem() — structural checks                                     #
# ======================================================================= #

class TestToMPCCProblem:
    def test_returns_mpcc_problem(self):
        p = _make_mixed().to_mpcc_problem()
        assert isinstance(p, MPCCProblem)

    def test_dimensions_propagated(self):
        m = _make_mixed()
        p = m.to_mpcc_problem()
        assert p.n == m.n
        assert p.n_comp == m.n_comp
        assert p.n_eq == m.n_eq        # 2 = 1 lin + 1 nl
        assert p.n_ineq == m.n_ineq    # 0

    def test_lin_only_eq_matches_manual(self):
        """Linear equality via StructuredMPCC must equal A@x - b directly."""
        m = _make_lin_only()
        p = m.to_mpcc_problem()
        x = np.array([1.0, 2.0, 0.5, 0.5, 0.5])
        val_struct = p.eq_constraints(x)
        val_manual = m.A_eq @ x - m.b_eq
        np.testing.assert_allclose(val_struct, val_manual)

    def test_lin_only_jacobian_is_constant(self):
        """Jacobian of a purely linear equality must equal A_eq at any x."""
        m = _make_lin_only()
        p = m.to_mpcc_problem()
        for x in [np.ones(5), np.zeros(5), np.random.default_rng(0).random(5)]:
            np.testing.assert_allclose(p.eq_jacobian(x), m.A_eq)

    def test_mixed_eq_stacks_linear_first(self):
        """
        Combined equality must be [A_eq @ x - b_eq ; h_nl(x)], in that order.
        """
        m = _make_mixed()
        p = m.to_mpcc_problem()
        x = np.array([0.5, 0.5, 0.75])

        lin_part = m.A_eq @ x - m.b_eq          # (1,)
        nl_part  = np.array([x[0]**2 + x[2] - 1.0])   # (1,)
        expected = np.concatenate([lin_part, nl_part])  # (2,)

        np.testing.assert_allclose(p.eq_constraints(x), expected)

    def test_mixed_jacobian_stacks_A_eq_first(self):
        """Jacobian rows must be [A_eq ; jac_h_nl(x)]."""
        m = _make_mixed()
        p = m.to_mpcc_problem()
        x = np.array([0.5, 0.5, 0.75])

        J = p.eq_jacobian(x)
        np.testing.assert_allclose(J[0], m.A_eq[0])                  # linear row
        np.testing.assert_allclose(J[1], [2*x[0], 0.0, 1.0])          # nl row

    def test_ineq_stacks_linear_first(self):
        m = _make_mixed_ineq()
        p = m.to_mpcc_problem()
        x = np.array([1.0, 1.0])

        lin_part = m.A_ineq @ x - m.b_ineq           # (1,)
        nl_part  = np.array([x[0]**2 + x[1]**2 - 9.0])  # (1,)
        expected = np.concatenate([lin_part, nl_part])

        np.testing.assert_allclose(p.ineq_constraints(x), expected)

    def test_to_mpcc_problem_does_not_mutate_x0(self):
        m = _make_mixed()
        x0_before = m.x0.copy()
        _ = m.to_mpcc_problem()
        np.testing.assert_array_equal(m.x0, x0_before)


# ======================================================================= #
# Solver integration                                                        #
# ======================================================================= #

class TestSolverIntegration:
    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_lin_only_bard1(self, strategy):
        """StructuredMPCC with linear equality only reproduces bard1 (f*=17)."""
        result = pympcc.solve(_make_lin_only(), strategy=strategy)
        assert result.success
        assert result.comp_residual < 1e-4
        assert abs(result.obj - 17.0) < 1e-3

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_nl_only(self, strategy):
        """StructuredMPCC with nonlinear equality only (modified scholtes1, f*=2)."""
        result = pympcc.solve(_make_nl_only(), strategy=strategy)
        assert result.success
        assert result.comp_residual < 1e-4
        assert abs(result.obj - 2.0) < 1e-3

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_mixed_eq(self, strategy):
        """Mixed linear+nonlinear equalities: x*=(1,0,0), f*=1."""
        result = pympcc.solve(_make_mixed(), strategy=strategy)
        assert result.success
        assert result.comp_residual < 1e-4
        assert abs(result.obj - 1.0) < 1e-3

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_mixed_ineq(self, strategy):
        """Mixed linear+nonlinear inequalities: x*=(2,0), f*=1."""
        result = pympcc.solve(_make_mixed_ineq(), strategy=strategy)
        assert result.success
        assert result.comp_residual < 1e-4
        assert abs(result.obj - 1.0) < 1e-3

    def test_accepts_structured_in_mpcc_solver(self):
        """MPCCSolver must accept StructuredMPCC directly."""
        solver = pympcc.MPCCSolver(_make_mixed(), strategy="scholtes")
        result = solver.solve()
        assert result.success

    def test_solve_is_equivalent_to_manual_to_mpcc_problem(self):
        """solve(structured) == solve(structured.to_mpcc_problem())."""
        m = _make_mixed()
        r1 = pympcc.solve(m, strategy="scholtes")
        r2 = pympcc.solve(m.to_mpcc_problem(), strategy="scholtes")
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-6)
        assert abs(r1.obj - r2.obj) < 1e-8
