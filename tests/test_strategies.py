"""
Unit tests for individual strategy behaviours and result structure.

These tests use the tiny 'simple' problem (n=2, n_comp=1, f*=1) to keep
runtime short while exercising API surface, option handling, and diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from .macmpec_problems import PROBLEM_NAMES

try:
    import scipy  # noqa: F401
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_skip_no_scipy = pytest.mark.skipif(not _HAS_SCIPY, reason="scipy not installed")

SIMPLE = PROBLEM_NAMES["simple"]

# Sparse version of SIMPLE: comp_G/H Jacobians return 1-D nnz values
SIMPLE_SPARSE = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([1.0]),
    comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([1.0]),
    comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
)


# ======================================================================= #
# Result object contract                                                    #
# ======================================================================= #

class TestResultFields:
    def test_fields_present(self):
        result = pympcc.solve(SIMPLE.problem)
        assert hasattr(result, "x")
        assert hasattr(result, "obj")
        assert hasattr(result, "G")
        assert hasattr(result, "H")
        assert hasattr(result, "comp_residual")
        assert hasattr(result, "success")
        assert hasattr(result, "strategy")
        assert hasattr(result, "history")

    def test_x_shape(self):
        result = pympcc.solve(SIMPLE.problem)
        assert result.x.shape == (SIMPLE.problem.n,)

    def test_G_H_shape(self):
        result = pympcc.solve(SIMPLE.problem)
        assert result.G.shape == (SIMPLE.problem.n_comp,)
        assert result.H.shape == (SIMPLE.problem.n_comp,)

    def test_strategy_name_recorded(self):
        for strategy in ("direct", "scholtes", "smoothing", "lin_fukushima",
                         "augmented_lagrangian"):
            result = pympcc.solve(SIMPLE.problem, strategy=strategy)
            assert result.strategy == strategy

    def test_repr(self):
        result = pympcc.solve(SIMPLE.problem)
        r = repr(result)
        assert "MPCCResult" in r
        assert "obj=" in r


# ======================================================================= #
# Direct strategy                                                           #
# ======================================================================= #

class TestDirectStrategy:
    def test_no_history(self):
        result = pympcc.solve(SIMPLE.problem, strategy="direct")
        assert result.history == []

    def test_converges(self):
        result = pympcc.solve(SIMPLE.problem, strategy="direct")
        assert result.success

    def test_comp_feasible(self):
        result = pympcc.solve(SIMPLE.problem, strategy="direct")
        assert result.comp_residual < 1e-4


# ======================================================================= #
# Scholtes strategy                                                         #
# ======================================================================= #

class TestScholtesStrategy:
    def test_dual_warmstart_default_true(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="scholtes")
        assert solver._strategy.dual_warmstart is True

    def test_dual_warmstart_can_disable(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="scholtes",
                                   dual_warmstart=False)
        assert solver._strategy.dual_warmstart is False

    def test_history_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              epsilon_0=1.0, reduction=0.1, max_iter=5)
        assert len(result.history) == 5

    def test_epsilon_decreases(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              epsilon_0=1.0, reduction=0.1, max_iter=6)
        eps_values = [it.epsilon for it in result.history]
        for a, b in zip(eps_values, eps_values[1:]):
            assert b < a

    def test_comp_residual_decreases_overall(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              epsilon_0=1.0, reduction=0.1, max_iter=8)
        first = result.history[0].comp_residual
        last  = result.history[-1].comp_residual
        assert last <= first + 1e-6  # last must be no worse than first

    def test_custom_epsilon_0(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes", epsilon_0=0.1)
        assert result.success

    def test_epsilon_min_stops_loop(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes",
                              epsilon_0=1.0, reduction=0.1,
                              epsilon_min=1e-2, max_iter=100)
        # Should stop after ~3 iterations (1, 0.1, 0.01 < 1e-2)
        assert len(result.history) <= 5


# ======================================================================= #
# Smoothing strategy                                                        #
# ======================================================================= #

class TestSmoothingStrategy:
    def test_dual_warmstart_default_true(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="smoothing")
        assert solver._strategy.dual_warmstart is True

    def test_dual_warmstart_can_disable(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="smoothing",
                                   dual_warmstart=False)
        assert solver._strategy.dual_warmstart is False

    def test_history_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="smoothing",
                              epsilon_0=1.0, reduction=0.1, max_iter=5)
        assert len(result.history) == 5

    def test_epsilon_decreases(self):
        result = pympcc.solve(SIMPLE.problem, strategy="smoothing",
                              epsilon_0=1.0, reduction=0.1, max_iter=6)
        eps_values = [it.epsilon for it in result.history]
        for a, b in zip(eps_values, eps_values[1:]):
            assert b < a

    def test_custom_reduction(self):
        result = pympcc.solve(SIMPLE.problem, strategy="smoothing",
                              reduction=0.01, max_iter=5)
        assert result.success

    def test_comp_feasible(self):
        result = pympcc.solve(SIMPLE.problem, strategy="smoothing")
        assert result.comp_residual < 1e-4


# ======================================================================= #
# Lin-Fukushima strategy                                                    #
# ======================================================================= #

class TestLinFukushimaStrategy:
    def test_dual_warmstart_default_true(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="lin_fukushima")
        assert solver._strategy.dual_warmstart is True

    def test_dual_warmstart_can_disable(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="lin_fukushima",
                                   dual_warmstart=False)
        assert solver._strategy.dual_warmstart is False

    def test_history_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima",
                              epsilon_0=1.0, reduction=0.1, max_iter=5)
        assert len(result.history) == 5

    def test_epsilon_decreases(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima",
                              epsilon_0=1.0, reduction=0.1, max_iter=6)
        eps_values = [it.epsilon for it in result.history]
        for a, b in zip(eps_values, eps_values[1:]):
            assert b < a

    def test_comp_feasible(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima")
        assert result.comp_residual < 1e-4

    def test_converges(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima")
        assert result.success

    def test_objective(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima")
        assert abs(result.obj - 1.0) < 1e-3

    def test_stationarity_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima")
        assert result.stationarity in {
            "S-stationary", "M-stationary", "C-stationary",
            "W-stationary", "unknown", "not stationary",
        }

    def test_epsilon_min_stops_loop(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima",
                              epsilon_0=1.0, reduction=0.1,
                              epsilon_min=1e-2, max_iter=100)
        assert len(result.history) <= 5

    def test_mult_g_shape(self):
        result = pympcc.solve(SIMPLE.problem, strategy="lin_fukushima")
        p = SIMPLE.problem
        assert result.mult_g is not None
        # Layout: [ineq | eq | G | H | G*H | G+H]
        expected = p.n_ineq + p.n_eq + 4 * p.n_comp
        assert result.mult_g.shape == (expected,)


# ======================================================================= #
# MPCCSolver class API                                                      #
# ======================================================================= #

class TestMPCCSolverAPI:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            pympcc.MPCCSolver(SIMPLE.problem, strategy="bad_strategy")

    def test_ipopt_options_forwarded(self):
        solver = pympcc.MPCCSolver(
            SIMPLE.problem,
            strategy="scholtes",
            ipopt_options={"max_iter": 200, "tol": 1e-6},
        )
        result = solver.solve()
        assert result.success

    def test_solver_reusable(self):
        """Calling solve() twice on the same solver should both succeed."""
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="scholtes")
        r1 = solver.solve()
        r2 = solver.solve()
        assert r1.success and r2.success
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-6)


# ======================================================================= #
# MPCCProblem validation                                                    #
# ======================================================================= #

class TestProblemValidation:
    def _base_kwargs(self):
        return dict(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: float(x[0]**2 + x[1]**2),
            gradient=lambda x: np.array([2*x[0], 2*x[1]]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )

    def test_bad_x0_shape(self):
        kw = self._base_kwargs()
        kw["x0"] = np.array([0.5])  # wrong shape
        with pytest.raises(ValueError, match="x0"):
            pympcc.MPCCProblem(**kw)

    def test_bad_comp_G_shape(self):
        kw = self._base_kwargs()
        kw["comp_G"] = lambda x: np.array([x[0], x[1]])  # returns 2 instead of 1
        with pytest.raises(ValueError, match="comp_G"):
            pympcc.MPCCProblem(**kw)

    def test_bad_comp_G_jacobian_shape(self):
        kw = self._base_kwargs()
        kw["comp_G_jacobian"] = lambda x: np.array([[1.0]])  # (1,1) instead of (1,2)
        with pytest.raises(ValueError, match="comp_G_jacobian"):
            pympcc.MPCCProblem(**kw)

    def test_n_comp_zero_raises(self):
        kw = self._base_kwargs()
        kw["n_comp"] = 0
        with pytest.raises(ValueError, match="n_comp"):
            pympcc.MPCCProblem(**kw)

    def test_ineq_without_jacobian_raises(self):
        kw = self._base_kwargs()
        kw["n_ineq"] = 1
        kw["ineq_constraints"] = lambda x: np.array([x[0] - 1])
        # ineq_jacobian is None — should raise
        with pytest.raises(ValueError, match="ineq_jacobian"):
            pympcc.MPCCProblem(**kw)

    def test_bounds_default_to_unbounded(self):
        kw = self._base_kwargs()
        p = pympcc.MPCCProblem(**kw)
        assert np.all(np.isinf(p.xl) & (p.xl < 0))
        assert np.all(np.isinf(p.xu) & (p.xu > 0))

    def test_xl_gt_xu_raises(self):
        kw = self._base_kwargs()
        kw["xl"] = np.array([1.0, 0.0])
        kw["xu"] = np.array([0.0, 1.0])  # xl[0] > xu[0]
        with pytest.raises(ValueError, match="xl must be"):
            pympcc.MPCCProblem(**kw)


# ======================================================================= #
# Dual warm-starting                                                        #
# ======================================================================= #

class TestDualWarmstart:
    @pytest.mark.parametrize("strategy",
                             ["scholtes", "smoothing", "lin_fukushima",
                              "augmented_lagrangian"])
    def test_default_enabled(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy, max_iter=8)
        assert result.success

    @pytest.mark.parametrize("strategy",
                             ["scholtes", "smoothing", "lin_fukushima",
                              "augmented_lagrangian"])
    def test_disabled_converges(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy,
                              dual_warmstart=False, max_iter=8)
        assert result.success

    @pytest.mark.parametrize("strategy",
                             ["scholtes", "smoothing", "lin_fukushima",
                              "augmented_lagrangian"])
    def test_warm_cold_same_solution(self, strategy):
        r_warm = pympcc.solve(SIMPLE.problem, strategy=strategy, max_iter=10)
        r_cold = pympcc.solve(SIMPLE.problem, strategy=strategy,
                              dual_warmstart=False, max_iter=10)
        np.testing.assert_allclose(r_warm.x, r_cold.x, atol=1e-4)


# ======================================================================= #
# Sparse Jacobian support                                                   #
# ======================================================================= #

class TestSparseJacobian:
    @pytest.mark.parametrize("strategy",
                             ["direct", "scholtes", "smoothing", "lin_fukushima",
                              "augmented_lagrangian"])
    def test_sparse_matches_dense(self, strategy):
        """Sparse and dense formulations must converge to the same solution."""
        r_d = pympcc.solve(SIMPLE.problem, strategy=strategy)
        r_s = pympcc.solve(SIMPLE_SPARSE,  strategy=strategy)
        np.testing.assert_allclose(r_d.x, r_s.x, atol=1e-4)

    def test_is_sparse_flag(self):
        assert SIMPLE_SPARSE.is_sparse is True
        assert SIMPLE.problem.is_sparse is False

    def test_bad_sparsity_row_out_of_range(self):
        with pytest.raises(ValueError, match="row indices out of range"):
            pympcc.MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: 0.0,
                gradient=lambda x: np.zeros(2),
                comp_G=lambda x: np.array([x[0]]),
                comp_G_jacobian=lambda x: np.array([1.0]),
                comp_G_jacobian_sparsity=(np.array([5]), np.array([0])),  # row ≥ n_comp
                comp_H=lambda x: np.array([x[1]]),
                comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
            )

    def test_bad_sparsity_col_out_of_range(self):
        with pytest.raises(ValueError, match="col indices out of range"):
            pympcc.MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: 0.0,
                gradient=lambda x: np.zeros(2),
                comp_G=lambda x: np.array([x[0]]),
                comp_G_jacobian=lambda x: np.array([1.0]),
                comp_G_jacobian_sparsity=(np.array([0]), np.array([9])),  # col ≥ n
                comp_H=lambda x: np.array([x[1]]),
                comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
            )

    def test_bad_sparse_callable_shape(self):
        with pytest.raises(ValueError, match="must return shape"):
            pympcc.MPCCProblem(
                n=2, n_comp=1, x0=np.array([0.5, 0.5]),
                objective=lambda x: 0.0,
                gradient=lambda x: np.zeros(2),
                comp_G=lambda x: np.array([x[0]]),
                comp_G_jacobian=lambda x: np.array([1.0, 0.0]),  # 2 values, nnz=1
                comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
                comp_H=lambda x: np.array([x[1]]),
                comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
            )


# ======================================================================= #
# Augmented Lagrangian strategy                                             #
# ======================================================================= #

class TestAugmentedLagrangianStrategy:
    def test_dual_warmstart_default_true(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="augmented_lagrangian")
        assert solver._strategy.dual_warmstart is True

    def test_dual_warmstart_can_disable(self):
        solver = pympcc.MPCCSolver(SIMPLE.problem, strategy="augmented_lagrangian",
                                   dual_warmstart=False)
        assert solver._strategy.dual_warmstart is False

    def test_history_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                              max_iter=5, comp_tol=1e-20)
        assert len(result.history) == 5

    def test_rho_nondecreasing(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                              max_iter=6, comp_tol=1e-20)
        rho_values = [it.epsilon for it in result.history]
        for a, b in zip(rho_values, rho_values[1:]):
            assert b >= a - 1e-12

    def test_comp_feasible(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian")
        assert result.comp_residual < 1e-4

    def test_converges(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian")
        assert result.success

    def test_objective(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian")
        assert abs(result.obj - 1.0) < 1e-3

    def test_stationarity_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian")
        assert result.stationarity in {
            "S-stationary", "M-stationary", "C-stationary",
            "W-stationary", "unknown", "not stationary",
        }

    def test_mult_g_shape(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian")
        p = SIMPLE.problem
        assert result.mult_g is not None
        # Constraint layout: [ineq | eq | G | H]  — no G*H block
        expected = p.n_ineq + p.n_eq + 2 * p.n_comp
        assert result.mult_g.shape == (expected,)

    def test_comp_tol_stops_early(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                              comp_tol=1e-2, max_iter=100)
        # Should stop well before 100 iterations on the simple problem
        assert len(result.history) < 100

    def test_custom_rho_0(self):
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                              rho_0=100.0)
        assert result.success

    def test_rho_max_caps_penalty(self):
        """rho must never exceed rho_max regardless of tau."""
        result = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                              rho_0=1.0, tau=1000.0, rho_max=50.0,
                              max_iter=10, comp_tol=1e-20)
        rho_values = [it.epsilon for it in result.history]
        assert all(r <= 50.0 + 1e-12 for r in rho_values)

    def test_eta_controls_rho_growth(self):
        """eta=0.0 (grow rho every iteration) must yield a higher final rho than eta=0.9."""
        # eta=0.0 → condition is comp_residual > 0, true until convergence → rho grows every step.
        # eta=0.9 → condition is comp_residual > 0.9 * prev; when the solver makes fast
        #            progress (>10% reduction per step), rho stays flat.
        result_aggressive = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                                         rho_0=1.0, eta=0.0, tau=10.0,
                                         max_iter=6, comp_tol=1e-20)
        result_cautious = pympcc.solve(SIMPLE.problem, strategy="augmented_lagrangian",
                                       rho_0=1.0, eta=0.9, tau=10.0,
                                       max_iter=6, comp_tol=1e-20)
        rho_aggressive = [it.epsilon for it in result_aggressive.history]
        rho_cautious   = [it.epsilon for it in result_cautious.history]
        assert rho_aggressive[-1] >= rho_cautious[-1]


# ======================================================================= #
# Slack (lifting) strategy                                                  #
# ======================================================================= #

class TestSlackStrategy:
    def test_solves_basic(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack")
        assert result.success
        assert abs(result.obj - 1.0) < 1e-3

    def test_comp_feasible(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack")
        assert result.comp_residual < 1e-4

    def test_history_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack",
                              epsilon_0=1.0, reduction=0.1, max_iter=5)
        assert len(result.history) == 5

    def test_epsilon_decreases(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack",
                              epsilon_0=1.0, reduction=0.1, max_iter=6)
        eps_values = [it.epsilon for it in result.history]
        for a, b in zip(eps_values, eps_values[1:]):
            assert b < a

    def test_result_fields(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack")
        assert result.success
        assert result.comp_residual < 1e-4
        assert len(result.history) > 0
        assert result.strategy == "slack"

    def test_dual_warmstart_false(self):
        r_warm = pympcc.solve(SIMPLE.problem, strategy="slack", max_iter=10)
        r_cold = pympcc.solve(SIMPLE.problem, strategy="slack",
                              dual_warmstart=False, max_iter=10)
        np.testing.assert_allclose(r_warm.x, r_cold.x, atol=1e-4)

    def test_matches_scholtes(self):
        r_slack   = pympcc.solve(SIMPLE.problem, strategy="slack")
        r_scholtes = pympcc.solve(SIMPLE.problem, strategy="scholtes")
        assert abs(r_slack.obj - r_scholtes.obj) < 1e-3

    def test_x_block_zero_in_comp_rows(self):
        """Jacobian rows for s_G·s_H must have zero entries in x-columns."""
        from pympcc.strategies.slack import SlackStrategy
        p = SIMPLE.problem
        strategy = SlackStrategy(p, {})
        rows, cols = strategy._build_lifted_jac_structure()
        row_comp = p.n_ineq + p.n_eq + 2 * p.n_comp
        comp_mask = rows >= row_comp
        # All column indices in the comp rows must be ≥ n (slack columns)
        assert not np.any(cols[comp_mask] < p.n)

    def test_comp_row_nnz_independent_of_n(self):
        """Comp rows have exactly 2*n_comp entries regardless of n."""
        from pympcc.strategies.slack import SlackStrategy
        # Build a larger problem: n=20, n_comp=3
        n, n_c = 20, 3
        problem = pympcc.MPCCProblem(
            n=n, n_comp=n_c,
            x0=np.zeros(n),
            objective=lambda x: float(np.sum(x**2)),
            gradient=lambda x: 2.0 * x,
            comp_G=lambda x: x[:n_c].copy(),
            comp_G_jacobian=lambda x: np.eye(n_c, n),
            comp_H=lambda x: x[n_c:2*n_c].copy(),
            comp_H_jacobian=lambda x: np.hstack([np.zeros((n_c, n_c)),
                                                  np.eye(n_c),
                                                  np.zeros((n_c, n - 2*n_c))]),
        )
        strategy = SlackStrategy(problem, {})
        rows, cols = strategy._build_lifted_jac_structure()
        row_comp = problem.n_ineq + problem.n_eq + 2 * n_c
        comp_mask = rows >= row_comp
        assert np.sum(comp_mask) == 2 * n_c

    def test_stationarity_populated(self):
        result = pympcc.solve(SIMPLE.problem, strategy="slack")
        assert result.stationarity in {
            "S-stationary", "unknown",
        }

    def test_sparse_jacobians_respected(self):
        """Slack strategy uses sparse *_jacobian_sparsity for pinning rows."""
        r_dense  = pympcc.solve(SIMPLE.problem,  strategy="slack")
        r_sparse = pympcc.solve(SIMPLE_SPARSE, strategy="slack")
        np.testing.assert_allclose(r_dense.x, r_sparse.x, atol=1e-4)


# ======================================================================= #
# Verbose mode, comp_tol, misc API                                          #
# ======================================================================= #

class TestVerboseMode:
    def test_prints_header(self, capsys):
        pympcc.solve(SIMPLE.problem, strategy="scholtes", verbose=True, max_iter=2)
        out = capsys.readouterr().out
        assert "pympcc" in out
        assert "iter" in out

    def test_prints_kkt_col(self, capsys):
        pympcc.solve(SIMPLE.problem, strategy="scholtes", verbose=True, max_iter=2)
        out = capsys.readouterr().out
        assert "kkt_res" in out

    def test_verbose_shows_rows(self, capsys):
        pympcc.solve(SIMPLE.problem, strategy="smoothing", verbose=True, max_iter=3)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        # header + separator + 3 data rows
        assert len(lines) >= 4

    def test_verbose_suppressed_when_callback_provided(self, capsys):
        calls = []
        pympcc.solve(SIMPLE.problem, strategy="scholtes", max_iter=2,
                     verbose=True,
                     callback=lambda k, info: calls.append(k))
        out = capsys.readouterr().out
        # verbose=True is suppressed because callback was given
        assert "iter" not in out
        assert len(calls) == 2

    def test_verbose_preamble_includes_strategy(self, capsys):
        pympcc.solve(SIMPLE.problem, strategy="lin_fukushima", verbose=True, max_iter=1)
        out = capsys.readouterr().out
        assert "lin_fukushima" in out


class TestCompTolIterative:
    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_stops_before_max_iter(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy,
                              comp_tol=1e-2, max_iter=100)
        assert len(result.history) < 100

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_kkt_residual_is_float(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy, max_iter=3)
        for info in result.history:
            assert info.kkt_residual is None or isinstance(info.kkt_residual, float)


class TestMiscBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            pympcc.solve(SIMPLE.problem, backend="bad_backend")

    def test_filtersqp_import_error(self):
        with pytest.raises(ImportError, match="pyfiltersqp"):
            pympcc.solve(SIMPLE.problem, backend="filterSQP")

    def test_comp_residual_helper(self):
        s = pympcc.MPCCSolver(SIMPLE.problem, strategy="scholtes")._strategy
        res = s._comp_residual(SIMPLE.problem.x0)
        assert isinstance(res, float) and res >= 0.0

    def test_manual_hessian_path(self):
        """A user-supplied Lagrangian Hessian is passed to IPOPT."""
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        # Inject a diagonal Lagrangian Hessian (identity scales)
        rows = np.array([0, 1], dtype=np.intp)
        cols = np.array([0, 1], dtype=np.intp)
        p.lagrangian_hessian = lambda x, lam, obj_factor: obj_factor * np.array([2.0, 2.0])
        p.lagrangian_hessian_sparsity = (rows, cols)
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        assert abs(result.obj - 1.0) < 1e-2


class TestSparseWithConstraints:
    """Sparse Jacobians for ineq/eq constraints — covers _build_std_jac_flat
    and _build_standard_constraints sparse paths."""

    @staticmethod
    def _make_problem_with_sparse_ineq():
        """n=3, n_comp=1, one inequality g(x) = x[2] - 1 <= 0 with sparse jac."""
        return pympcc.MPCCProblem(
            n=3, n_comp=1,
            x0=np.array([0.5, 0.5, 0.5]),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0), 0.0]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
            n_ineq=1,
            ineq_constraints=lambda x: np.array([x[2] - 1.0]),
            ineq_jacobian=lambda x: np.array([1.0]),          # sparse: 1 nnz
            ineq_jacobian_sparsity=(np.array([0]), np.array([2])),
        )

    @staticmethod
    def _make_problem_with_sparse_eq():
        """n=3, n_comp=1, one equality h(x) = x[2] - 0.5 = 0 with sparse jac."""
        return pympcc.MPCCProblem(
            n=3, n_comp=1,
            x0=np.array([0.5, 0.5, 0.5]),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0), 0.0]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0, 0.0]]),
            n_eq=1,
            eq_constraints=lambda x: np.array([x[2] - 0.5]),
            eq_jacobian=lambda x: np.array([1.0]),
            eq_jacobian_sparsity=(np.array([0]), np.array([2])),
        )

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_sparse_ineq_jacobian_converges(self, strategy):
        p = self._make_problem_with_sparse_ineq()
        result = pympcc.solve(p, strategy=strategy)
        assert result.success

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "augmented_lagrangian"])
    def test_sparse_eq_jacobian_converges(self, strategy):
        p = self._make_problem_with_sparse_eq()
        result = pympcc.solve(p, strategy=strategy)
        assert result.success

    def test_stationarity_sparse_jac(self):
        """KKT residual with sparse ineq Jacobian."""
        p = self._make_problem_with_sparse_ineq()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.stationarity in {
            "S-stationary", "M-stationary", "C-stationary",
            "W-stationary", "unknown", "not stationary",
        }


# ======================================================================= #
# scipy backend                                                             #
# ======================================================================= #

@_skip_no_scipy
class TestScipyBackend:
    """Tests for backend='scipy' (trust-constr) across strategies."""

    @pytest.mark.parametrize("strategy", [
        "direct", "scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian",
    ])
    def test_converges(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy, backend="scipy")
        assert result.success

    @pytest.mark.parametrize("strategy", [
        "direct", "scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian",
    ])
    def test_comp_feasible(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy, backend="scipy")
        assert result.comp_residual < 1e-4

    @pytest.mark.parametrize("strategy", [
        "direct", "scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian",
    ])
    def test_objective(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy, backend="scipy")
        assert abs(result.obj - 1.0) < 1e-2

    def test_result_fields_present(self):
        result = pympcc.solve(SIMPLE.problem, backend="scipy")
        assert result.x.shape == (SIMPLE.problem.n,)
        assert result.G.shape == (SIMPLE.problem.n_comp,)
        assert result.H.shape == (SIMPLE.problem.n_comp,)
        assert isinstance(result.obj, float)
        assert isinstance(result.success, bool)

    def test_history_populated_iterative(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes", backend="scipy",
                              epsilon_0=1.0, reduction=0.1, max_iter=5)
        assert len(result.history) == 5

    def test_n_iter_tracked(self):
        result = pympcc.solve(SIMPLE.problem, strategy="direct", backend="scipy")
        assert result.history == []   # direct has no outer history
        assert result.n_ipopt_iter if hasattr(result, "n_ipopt_iter") else True

    def test_sparse_problem(self):
        result = pympcc.solve(SIMPLE_SPARSE, strategy="scholtes", backend="scipy")
        assert result.success
        assert result.comp_residual < 1e-4

    def test_matches_ipopt_solution(self):
        """scipy and IPOPT backends must agree on the solution to within 1e-2."""
        r_ipopt = pympcc.solve(SIMPLE.problem, strategy="scholtes", backend="ipopt")
        r_scipy = pympcc.solve(SIMPLE.problem, strategy="scholtes", backend="scipy")
        np.testing.assert_allclose(r_ipopt.x, r_scipy.x, atol=1e-2)
