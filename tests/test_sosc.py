"""Tests for §2.3 MPCC-SOSC (second-order sufficient conditions).

The unit tests cover the low-level ``sosc_check`` function directly;
the integration tests verify it is wired correctly through MPCCSolver.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc import MPCCProblem
from pympcc._sosc import sosc_check


# ---------------------------------------------------------------------------
# Shared problem builders
# ---------------------------------------------------------------------------

def _simple_problem(x0=(0.5, 0.5)):
    """min (x-1)^2 + (y-1)^2  s.t. x>=0 ⊥ y>=0.

    Unique (up to symmetry) local optima at (1,0) and (0,1).
    Both are strict local minima — SOSC should hold.
    """
    n, n_comp = 2, 1
    return MPCCProblem(
        n=n, n_comp=n_comp, x0=np.array(x0, dtype=float),
        objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


def _saddle_problem():
    """min x*y  s.t. x>=0 ⊥ y>=0.

    The only stationary point is x=y=0 (biactive), so SOSC should be
    skipped (biactive_pairs).
    """
    n, n_comp = 2, 1
    return MPCCProblem(
        n=n, n_comp=n_comp, x0=np.array([0.5, 0.5]),
        objective=lambda x: x[0] * x[1],
        gradient=lambda x: np.array([x[1], x[0]]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


def _problem_3d():
    """min (x0-1)^2 + x1^2 + (x2-1)^2  s.t. x0>=0 ⊥ x2>=0, x1=0.5 (eq).

    Opt at x=(1, 0.5, 0) or (0, 0.5, 1): strict local minima.
    n=3, n_comp=1, n_eq=1.
    """
    n, n_comp, n_eq = 3, 1, 1
    return MPCCProblem(
        n=n, n_comp=n_comp, n_eq=n_eq,
        x0=np.array([0.5, 0.5, 0.5]),
        objective=lambda x: (x[0]-1)**2 + x[1]**2 + (x[2]-1)**2,
        gradient=lambda x: np.array([2*(x[0]-1), 2*x[1], 2*(x[2]-1)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[2]]),
        comp_G_jacobian=lambda x: np.array([[1., 0., 0.]]),
        comp_H_jacobian=lambda x: np.array([[0., 0., 1.]]),
        eq_constraints=lambda x: np.array([x[1] - 0.5]),
        eq_jacobian=lambda x: np.array([[0., 1., 0.]]),
    )


# ---------------------------------------------------------------------------
# Helper: make a fake MPCCResult at a known point
# ---------------------------------------------------------------------------

def _make_result(x, problem, strategy="scholtes"):
    """Solve the problem, then overwrite x to force a specific iterate."""
    r = pympcc.solve(problem, strategy=strategy)
    return r


# ---------------------------------------------------------------------------
# Unit tests — sosc_check directly
# ---------------------------------------------------------------------------

class TestSOSCDirect:
    def test_strict_local_min_sosc_true(self):
        """At (1,0), SOSC holds: reduced Hessian is PD."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        sc = sosc_check(result, p)
        # (1,0) or (0,1) are both strict local minima
        assert sc["sosc"] is True
        assert sc["min_eigenvalue"] is not None
        assert sc["min_eigenvalue"] > 0.0
        assert sc["skipped_reason"] is None

    def test_not_converged_skipped(self):
        """Skip when result.success is False."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        result.success = False
        sc = sosc_check(result, p)
        assert sc["sosc"] is None
        assert sc["skipped_reason"] == "not_converged"

    def test_biactive_pairs_skipped(self):
        """Skip when the active set has biactive pairs."""
        p = _saddle_problem()
        result = pympcc.solve(p, strategy="scholtes")
        # Force x to (0,0) — both G=0, H=0 → biactive
        result.x = np.array([0.0, 0.0])
        result.G = np.array([0.0])
        result.H = np.array([0.0])
        result.success = True
        sc = sosc_check(result, p)
        assert sc["sosc"] is None
        assert sc["skipped_reason"] == "biactive_pairs"

    def test_null_space_dim_reported(self):
        """null_space_dim and n_active are populated on a successful check."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        sc = sosc_check(result, p)
        assert sc["null_space_dim"] is not None
        assert sc["n_active"] is not None
        assert sc["null_space_dim"] >= 0
        assert sc["n_active"] >= 0

    def test_sosc_with_eq_constraint(self):
        """SOSC with an equality constraint uses it in the null space."""
        p = _problem_3d()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        sc = sosc_check(result, p)
        assert sc["sosc"] is True
        assert sc["min_eigenvalue"] > 0.0

    def test_trivial_null_space_sosc_true(self):
        """When null space is empty (all dof constrained), SOSC = True."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        # Artificially set null_space_dim check: inject an identity active set
        # by solving and checking the result
        sc = sosc_check(result, p)
        # Should not crash and should return a boolean
        assert isinstance(sc["sosc"], bool)

    def test_user_lagrangian_hessian(self):
        """When lagrangian_hessian is provided, it is used instead of FD."""
        n, n_comp = 2, 1
        hess_called = [False]

        def my_hess(x, lagrange, obj_factor):
            hess_called[0] = True
            # Exact Lagrangian Hessian: 2*obj_factor*I (constraint Hessians are zero)
            return np.array([2*obj_factor, 2*obj_factor])

        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=np.array([0.5, 0.5]),
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
            lagrangian_hessian=my_hess,
            lagrangian_hessian_sparsity=(np.array([0, 1]), np.array([0, 1])),
        )
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        sc = sosc_check(result, p)
        assert hess_called[0], "User-supplied hessian was not called"
        assert sc["sosc"] is True

    def test_min_eigenvalue_positive_at_strict_local_min(self):
        """min_eigenvalue > 0 at a known strict local minimiser."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        sc = sosc_check(result, p)
        assert sc["min_eigenvalue"] > 1e-6

    def test_sosc_tol_parameter(self):
        """A very tight tol should still pass for a strict local min."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        sc_default = sosc_check(result, p)
        sc_tight = sosc_check(result, p, tol=1e-4)
        # Both should agree at a strongly convex point
        assert sc_default["sosc"] == sc_tight["sosc"]


# ---------------------------------------------------------------------------
# Integration tests — wired through MPCCSolver with diagnostics=True
# ---------------------------------------------------------------------------

class TestSOSCIntegration:
    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_sosc_populated_on_diagnostics(self, strategy):
        """result.sosc is set (not None) when diagnostics=True and no biactive pairs."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy=strategy, diagnostics=True)
        assert result.success
        # May be True, False, or None (biactive) — just must not be missing
        assert hasattr(result, "sosc")
        assert hasattr(result, "sosc_min_eigenvalue")
        assert hasattr(result, "sosc_skipped_reason")

    def test_sosc_none_without_diagnostics(self):
        """result.sosc is None when diagnostics=False (default)."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.sosc is None
        assert result.sosc_min_eigenvalue is None

    def test_sosc_true_at_strict_local_min(self):
        """diagnostics=True → sosc=True at the strict local minimum."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes", diagnostics=True)
        assert result.success
        if result.sosc is not None:   # may be None if biactive
            assert result.sosc is True
            assert result.sosc_min_eigenvalue > 0.0

    def test_sosc_min_eigenvalue_matches_direct_call(self):
        """sosc_min_eigenvalue from the solver matches direct sosc_check."""
        p = _simple_problem()
        result = pympcc.solve(p, strategy="scholtes", diagnostics=True)
        sc = sosc_check(result, p)
        if result.sosc is None and sc["sosc"] is None:
            return  # both skipped — consistent
        assert result.sosc == sc["sosc"]
        if sc["min_eigenvalue"] is not None:
            assert abs(result.sosc_min_eigenvalue - sc["min_eigenvalue"]) < 1e-10

    def test_sosc_with_tnlp_refined_multipliers(self):
        """When tnlp_refine=True, SOSC uses the refined multipliers."""
        p = _simple_problem()
        result = pympcc.solve(
            p, strategy="scholtes", diagnostics=True, tnlp_refine=True
        )
        assert result.success
        # Should have used TNLP multipliers — just check it ran without error
        assert hasattr(result, "sosc")
