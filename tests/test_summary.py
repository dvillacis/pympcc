"""Tests for MPCCResult.summary() formatter."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.result import IterationInfo, MPCCResult


def _minimal_result(**overrides) -> MPCCResult:
    base = dict(
        x=np.array([1.0, 0.0]),
        obj=-6.25,
        status=0,
        message="Solve_Succeeded",
        G=np.array([1.0]),
        H=np.array([0.0]),
        comp_residual=0.0,
        comp_residual_mean=0.0,
        success=True,
        strategy="scholtes",
    )
    base.update(overrides)
    return MPCCResult(**base)


# ======================================================================= #
# Verbosity 0 — headline                                                    #
# ======================================================================= #

class TestVerbosityZero:
    def test_headline_minimal(self):
        r = _minimal_result()
        s = r.summary(verbosity=0)
        assert "scholtes" in s
        assert "success=True" in s
        assert "obj=-6.25" in s
        assert "comp=" in s
        assert "\n" not in s

    def test_headline_includes_kkt_when_set(self):
        r = _minimal_result(kkt_residual=1.23e-7)
        s = r.summary(verbosity=0)
        assert "kkt=" in s

    def test_headline_omits_kkt_when_none(self):
        r = _minimal_result(kkt_residual=None)
        s = r.summary(verbosity=0)
        assert "kkt=" not in s


# ======================================================================= #
# Verbosity 1 — block (default)                                             #
# ======================================================================= #

class TestVerbosityOne:
    def test_default_verbosity_is_one(self):
        r = _minimal_result()
        assert r.summary() == r.summary(verbosity=1)

    def test_block_has_core_sections(self):
        r = _minimal_result(solve_time=0.034)
        s = r.summary()
        assert "Status: 0 (SOLVED)" in s
        assert "Solution:" in s
        assert "objective" in s
        assert "comp_residual" in s
        assert "Performance:" in s
        assert "0.034s" in s

    def test_unknown_status_code(self):
        r = _minimal_result(status=999)
        s = r.summary()
        assert "999 (UNKNOWN)" in s

    def test_stationarity_section_present_when_class_known(self):
        r = _minimal_result(stationarity="S-stationary")
        s = r.summary()
        assert "Stationarity:" in s
        assert "S-stationary" in s

    def test_stationarity_section_skipped_when_unknown(self):
        r = _minimal_result(stationarity="unknown")
        s = r.summary()
        assert "Stationarity:" not in s

    def test_cq_block_with_rank_deficit(self):
        r = _minimal_result(
            stationarity="S-stationary",
            cq="MPCC-LICQ", cq_rank_deficit=0,
            cq_active_set_sizes={"g": 0, "h": 0, "G": 1, "H": 0,
                                 "biactive": 0, "xL": 0, "xU": 0},
        )
        s = r.summary()
        assert "MPCC-LICQ" in s
        assert "rank deficit 0" in s
        assert "Active set:" in s
        assert "|I_G|=1" in s
        assert "|I_biactive|=0" in s

    def test_b_stationary_with_descent(self):
        r = _minimal_result(
            b_stationary="B-stationary",
            b_stationary_min_descent=1.2e-9,
        )
        s = r.summary()
        assert "B-stationary    = B-stationary" in s
        assert "min descent 1.200e-09" in s

    def test_b_stationary_without_descent(self):
        r = _minimal_result(b_stationary="intractable")
        s = r.summary()
        assert "B-stationary    = intractable" in s
        assert "min descent" not in s

    def test_history_iter_count_in_performance(self):
        hist = [
            IterationInfo(epsilon=1e-1, x=np.array([0.0]), obj=-6.0, status=0,
                          message="ok", comp_residual=1e-2,
                          comp_residual_mean=1e-2, n_ipopt_iter=5,
                          iter_time=0.003),
            IterationInfo(epsilon=1e-2, x=np.array([0.0]), obj=-6.2, status=0,
                          message="ok", comp_residual=1e-4,
                          comp_residual_mean=1e-4, n_ipopt_iter=4,
                          iter_time=0.002),
        ]
        r = _minimal_result(history=hist, solve_time=0.005)
        s = r.summary()
        assert "outer iters     = 2" in s
        assert "IPOPT iters: 9" in s

    def test_history_with_restoration(self):
        hist = [
            IterationInfo(epsilon=1e-1, x=np.array([0.0]), obj=-6.0, status=0,
                          message="ok", comp_residual=1e-2,
                          comp_residual_mean=1e-2, n_ipopt_iter=5,
                          iter_time=0.003,
                          restoration_iter_count=2, entered_restoration=True),
        ]
        r = _minimal_result(history=hist)
        s = r.summary()
        assert "restoration: 2" in s

    def test_cleanup_section_when_populated(self):
        r = _minimal_result(
            cleanup_status=0, cleanup_n_iter=3, cleanup_obj=-6.2501,
            cleanup_accepted=True,
        )
        s = r.summary()
        assert "Cleanup:" in s
        assert "accepted=True" in s
        assert "n_iter=3" in s

    def test_cleanup_section_omitted_when_none(self):
        r = _minimal_result()
        s = r.summary()
        assert "Cleanup:" not in s


# ======================================================================= #
# Verbosity 2 — per-iteration table                                         #
# ======================================================================= #

class TestVerbosityTwo:
    def test_history_table_present(self):
        hist = [
            IterationInfo(epsilon=1e-1, x=np.array([0.0]), obj=-6.0, status=0,
                          message="ok", comp_residual=1e-2,
                          comp_residual_mean=1e-2, n_ipopt_iter=5,
                          iter_time=0.003, kkt_residual=2e-3),
            IterationInfo(epsilon=1e-2, x=np.array([0.0]), obj=-6.2, status=0,
                          message="ok", comp_residual=1e-4,
                          comp_residual_mean=1e-4, n_ipopt_iter=4,
                          iter_time=0.002, kkt_residual=2e-5),
        ]
        r = _minimal_result(history=hist)
        s = r.summary(verbosity=2)
        assert "Per-iteration history:" in s
        assert "epsilon" in s
        # one row per iteration
        for k in (0, 1):
            assert f"\n    {k} " in s or f"    {k}    " in s

    def test_table_handles_missing_kkt(self):
        hist = [
            IterationInfo(epsilon=1e-1, x=np.array([0.0]), obj=-6.0, status=0,
                          message="ok", comp_residual=1e-2,
                          comp_residual_mean=1e-2, n_ipopt_iter=5,
                          iter_time=0.003, kkt_residual=None),
        ]
        r = _minimal_result(history=hist)
        s = r.summary(verbosity=2)
        assert "Per-iteration history:" in s

    def test_no_history_no_table(self):
        r = _minimal_result()
        s = r.summary(verbosity=2)
        assert "Per-iteration history:" not in s


# ======================================================================= #
# End-to-end: real solve                                                    #
# ======================================================================= #

class TestEndToEnd:
    def test_summary_after_real_solve(self):
        problem = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        result = pympcc.solve(problem, strategy="scholtes")
        s0 = result.summary(verbosity=0)
        s1 = result.summary(verbosity=1)
        s2 = result.summary(verbosity=2)
        assert isinstance(s0, str) and isinstance(s1, str) and isinstance(s2, str)
        assert len(s2) >= len(s1) >= len(s0)
        assert "scholtes" in s0
        assert "Performance:" in s1
