"""Tests for §6.2 — condition-number diagnostics on solved MPCCs.

Covers ``MPCCResult.jac_condition`` and ``MPCCResult.hessian_condition_estimate``,
populated by ``MPCCSolver._attach_diagnostics`` from
``pympcc._diagnostics.jac_condition_number`` and ``pympcc._sosc.sosc_check``
respectively.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc import MPCCProblem
from pympcc._diagnostics import jac_condition_number
from pympcc._sosc import sosc_check
from pympcc.result import MPCCResult


def _simple_problem(x0=(0.5, 0.5)):
    """min (x-1)^2 + (y-1)^2  s.t. x>=0 ⊥ y>=0.  Optima at (1,0) / (0,1)."""
    return MPCCProblem(
        n=2, n_comp=1, x0=np.array(x0, dtype=float),
        objective=lambda x: (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2 * (x[0] - 1.0), 2 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# --------------------------------------------------------------------------- #
# Default (no diagnostics)                                                    #
# --------------------------------------------------------------------------- #

class TestDefaultsNoDiagnostics:
    def test_fields_none_without_diagnostics(self):
        result = pympcc.solve(_simple_problem(), strategy="scholtes")
        assert result.success
        assert result.jac_condition is None
        assert result.hessian_condition_estimate is None


# --------------------------------------------------------------------------- #
# Diagnostics ON                                                              #
# --------------------------------------------------------------------------- #

class TestDiagnosticsOn:
    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_jac_condition_populated(self, strategy):
        result = pympcc.solve(_simple_problem(), strategy=strategy,
                              diagnostics=True)
        assert result.success
        assert result.jac_condition is not None
        assert result.jac_condition > 0.0
        assert np.isfinite(result.jac_condition)

    def test_jac_condition_matches_direct_call(self):
        problem = _simple_problem()
        result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
        direct = jac_condition_number(result, problem)
        assert direct is not None
        assert result.jac_condition == pytest.approx(direct, rel=1e-12)

    def test_hessian_condition_matches_sosc_cond_W(self):
        problem = _simple_problem()
        result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
        sc = sosc_check(result, problem)
        if sc["cond_W"] is None:
            assert result.hessian_condition_estimate is None
        else:
            assert result.hessian_condition_estimate == pytest.approx(
                sc["cond_W"], rel=1e-10
            )

    def test_hessian_condition_positive_when_sosc_holds(self):
        problem = _simple_problem()
        result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
        if result.sosc is True and result.hessian_condition_estimate is not None:
            assert result.hessian_condition_estimate >= 1.0


# --------------------------------------------------------------------------- #
# Direct-API edge cases                                                       #
# --------------------------------------------------------------------------- #

class TestJacConditionEdgeCases:
    def test_returns_none_when_result_failed(self):
        # Construct a non-converged result manually; the helper must short-circuit.
        problem = _simple_problem()
        x = np.array([0.0, 0.0])
        G = np.array([0.0])
        H = np.array([0.0])
        result = MPCCResult(
            x=x, obj=0.0, status=-1, message="failed",
            G=G, H=H, comp_residual=0.0, comp_residual_mean=0.0,
            success=False, strategy="test",
        )
        assert jac_condition_number(result, problem) is None

    def test_returns_none_when_active_matrix_empty(self):
        # No active constraints, no active bounds → empty matrix → None.
        problem = MPCCProblem(
            n=2, n_comp=1, x0=np.array([2.0, 3.0]),
            objective=lambda x: 0.0,
            gradient=lambda x: np.zeros(2),
            comp_G=lambda x: np.array([x[0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        result = MPCCResult(
            x=np.array([2.0, 3.0]), obj=0.0, status=0, message="ok",
            G=np.array([2.0]), H=np.array([3.0]),
            comp_residual=6.0, comp_residual_mean=6.0,
            success=True, strategy="test",
        )
        assert jac_condition_number(result, problem) is None

    def test_finite_for_well_conditioned_active_set(self):
        # At the LICQ point x*=(1,0) the active matrix has one row (∇H=[0,1]);
        # condition number must be 1.0 for a single unit row.
        problem = _simple_problem()
        x = np.array([1.0, 0.0])
        result = MPCCResult(
            x=x, obj=0.0, status=0, message="ok",
            G=np.array([1.0]), H=np.array([0.0]),
            comp_residual=0.0, comp_residual_mean=0.0,
            success=True, strategy="test",
        )
        cond = jac_condition_number(result, problem)
        assert cond is not None
        assert cond == pytest.approx(1.0, rel=1e-12)
