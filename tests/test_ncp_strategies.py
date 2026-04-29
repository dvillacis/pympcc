"""Tests for §3.5 NCP-function reformulation strategies.

Covers: SmoothMinStrategy, ChenChenKanzowStrategy, KanzowSchwartzStrategy.
Uses the tiny 'simple' benchmark (n=2, n_comp=1, f*=1, x*=(0,1) or (2,0))
for speed, plus a few option-validation checks.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc

from .macmpec_problems import PROBLEM_NAMES

SIMPLE = PROBLEM_NAMES["simple"]

_NCP_STRATEGIES = ["smooth_min", "chen_chen_kanzow", "kanzow_schwartz"]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _solve(strategy: str, **kwargs) -> pympcc.MPCCResult:
    return pympcc.solve(SIMPLE.problem, strategy=strategy, **kwargs)


# --------------------------------------------------------------------------- #
# Convergence: each strategy must converge and satisfy comp residual           #
# --------------------------------------------------------------------------- #

class TestNcpConvergence:
    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_converges(self, strategy):
        result = _solve(strategy)
        assert result.success, f"{strategy}: did not converge"

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_comp_residual_small(self, strategy):
        result = _solve(strategy)
        assert result.comp_residual < 1e-4, (
            f"{strategy}: comp_residual={result.comp_residual:.2e} ≥ 1e-4"
        )

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_objective_near_optimal(self, strategy):
        result = _solve(strategy)
        assert abs(result.obj - SIMPLE.f_opt) <= 1e-2 * max(1.0, abs(SIMPLE.f_opt)), (
            f"{strategy}: obj={result.obj:.6f} far from f*={SIMPLE.f_opt}"
        )

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_result_has_history(self, strategy):
        result = _solve(strategy)
        assert result.history is not None and len(result.history) > 0

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_strategy_name_in_result(self, strategy):
        result = _solve(strategy)
        assert result.strategy == strategy


# --------------------------------------------------------------------------- #
# CCK: lam parameter                                                           #
# --------------------------------------------------------------------------- #

class TestCCKOptions:
    def test_lam_default_converges(self):
        result = _solve("chen_chen_kanzow")
        assert result.success

    def test_lam_1_is_fb(self):
        result_cck = _solve("chen_chen_kanzow", lam=1.0)
        result_fb  = _solve("smoothing")
        assert result_cck.success
        assert abs(result_cck.obj - result_fb.obj) < 1e-3

    def test_lam_near_zero(self):
        result = _solve("chen_chen_kanzow", lam=0.05)
        assert result.success

    def test_lam_out_of_range_raises(self):
        with pytest.raises(ValueError, match="lam"):
            pympcc.solve(SIMPLE.problem, strategy="chen_chen_kanzow", lam=0.0)

    def test_lam_above_1_raises(self):
        with pytest.raises(ValueError, match="lam"):
            pympcc.solve(SIMPLE.problem, strategy="chen_chen_kanzow", lam=1.5)


# --------------------------------------------------------------------------- #
# KS: lam parameter                                                            #
# --------------------------------------------------------------------------- #

class TestKSOptions:
    def test_lam_default_converges(self):
        result = _solve("kanzow_schwartz")
        assert result.success

    def test_lam_0_is_fb_like(self):
        result_ks = _solve("kanzow_schwartz", lam=0.0)
        result_fb = _solve("smoothing")
        assert result_ks.success
        assert abs(result_ks.obj - result_fb.obj) < 1e-3

    def test_lam_high(self):
        result = _solve("kanzow_schwartz", lam=0.9)
        assert result.success

    def test_lam_1_raises(self):
        with pytest.raises(ValueError, match="lam"):
            pympcc.solve(SIMPLE.problem, strategy="kanzow_schwartz", lam=1.0)

    def test_lam_negative_raises(self):
        with pytest.raises(ValueError, match="lam"):
            pympcc.solve(SIMPLE.problem, strategy="kanzow_schwartz", lam=-0.1)


# --------------------------------------------------------------------------- #
# G/H non-negativity at solution                                               #
# --------------------------------------------------------------------------- #

class TestNcpFeasibility:
    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_G_nonneg(self, strategy):
        result = _solve(strategy)
        assert np.all(result.G >= -1e-6), f"{strategy}: G has negative components"

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_H_nonneg(self, strategy):
        result = _solve(strategy)
        assert np.all(result.H >= -1e-6), f"{strategy}: H has negative components"


# --------------------------------------------------------------------------- #
# Sparse problem path                                                           #
# --------------------------------------------------------------------------- #

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


class TestNcpSparse:
    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_sparse_converges(self, strategy):
        result = pympcc.solve(SIMPLE_SPARSE, strategy=strategy)
        assert result.success

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_sparse_comp_residual(self, strategy):
        result = pympcc.solve(SIMPLE_SPARSE, strategy=strategy)
        assert result.comp_residual < 1e-4


# --------------------------------------------------------------------------- #
# Epsilon continuation options forwarded correctly                             #
# --------------------------------------------------------------------------- #

class TestNcpEpsilonOptions:
    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_custom_epsilon_0(self, strategy):
        result = _solve(strategy, epsilon_0=0.1)
        assert result.success

    @pytest.mark.parametrize("strategy", _NCP_STRATEGIES)
    def test_max_iter_1_runs(self, strategy):
        result = _solve(strategy, max_iter=1)
        assert result.history is not None
        assert len(result.history) == 1
