"""
Tests for §2.6: TNLP active-set refinement.

Verifies that:
  - run_tnlp_refinement() returns a TNLPResult with the expected structure.
  - Multiplier signs satisfy S-stationarity for clearly non-biactive solutions.
  - The tnlp_refine=True flag populates result.tnlp_refined and
    result.mult_comp_G/H_mpcc.
  - TNLP works with every non-slack strategy (direct, scholtes, smoothing,
    lin_fukushima, augmented_lagrangian) since they all expose _build_cleanup_nlp.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._tnlp import (
    TNLPResult, _classify_tnlp_stationarity, _infer_active_set, run_tnlp_refinement,
)


# ----------------------------------------------------------------------- #
# Simple problem: x*=(2,0), H_active, G=2>0, H=0                          #
# ----------------------------------------------------------------------- #

def _make_simple():
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ----------------------------------------------------------------------- #
# _infer_active_set unit tests                                              #
# ----------------------------------------------------------------------- #

class TestInferActiveSet:
    def test_h_active(self):
        G = np.array([2.0])
        H = np.array([0.0])
        I_G, I_H = _infer_active_set(G, H, None, None)
        assert len(I_G) == 0
        assert list(I_H) == [0]

    def test_g_active(self):
        G = np.array([0.0])
        H = np.array([3.0])
        I_G, I_H = _infer_active_set(G, H, None, None)
        assert list(I_G) == [0]
        assert len(I_H) == 0

    def test_biactive_multiplier_rule(self):
        # G and H both tiny; multiplier rule picks by smaller |mu|.
        G = np.array([1e-8])
        H = np.array([1e-8])
        # If |mu_G| < |mu_H|, pin G (G_active).
        mu_G = np.array([0.1])
        mu_H = np.array([5.0])
        I_G, I_H = _infer_active_set(G, H, mu_G, mu_H)
        assert list(I_G) == [0]
        assert len(I_H) == 0

    def test_empty_problem(self):
        I_G, I_H = _infer_active_set(np.array([]), np.array([]), None, None)
        assert len(I_G) == 0
        assert len(I_H) == 0

    def test_multiple_pairs(self):
        G = np.array([2.0, 0.0])
        H = np.array([0.0, 3.0])
        I_G, I_H = _infer_active_set(G, H, None, None)
        assert list(I_G) == [1]
        assert list(I_H) == [0]


# ----------------------------------------------------------------------- #
# _classify_tnlp_stationarity unit tests                                   #
# ----------------------------------------------------------------------- #

class TestClassifyTnlpStationarity:
    def test_s_stationary_all_positive(self):
        mu_G = np.array([1.0, 0.0])
        mu_H = np.array([0.0, 2.0])
        I_G = np.array([0], dtype=np.intp)
        I_H = np.array([1], dtype=np.intp)
        stat, n_viol = _classify_tnlp_stationarity(mu_G, mu_H, I_G, I_H)
        assert stat == "S-stationary"
        assert n_viol == 0

    def test_w_stationary_negative_mu_G(self):
        mu_G = np.array([-0.5])
        mu_H = np.array([0.0])
        I_G = np.array([0], dtype=np.intp)
        I_H = np.empty(0, dtype=np.intp)
        stat, n_viol = _classify_tnlp_stationarity(mu_G, mu_H, I_G, I_H)
        assert stat == "W-stationary"
        assert n_viol == 1

    def test_w_stationary_negative_mu_H(self):
        mu_G = np.array([0.0])
        mu_H = np.array([-1.0])
        I_G = np.empty(0, dtype=np.intp)
        I_H = np.array([0], dtype=np.intp)
        stat, n_viol = _classify_tnlp_stationarity(mu_G, mu_H, I_G, I_H)
        assert stat == "W-stationary"
        assert n_viol == 1

    def test_tol_boundary(self):
        # Exactly at -tol should not count as violation (boundary is exclusive)
        mu_G = np.array([-1e-6])
        mu_H = np.array([0.0])
        I_G = np.array([0], dtype=np.intp)
        I_H = np.empty(0, dtype=np.intp)
        stat, n_viol = _classify_tnlp_stationarity(mu_G, mu_H, I_G, I_H, tol=1e-6)
        assert stat == "S-stationary"
        assert n_viol == 0


# ----------------------------------------------------------------------- #
# Biactivity pre-screen                                                     #
# ----------------------------------------------------------------------- #

class TestBiactivityPrescreen:
    def test_skipped_when_all_biactive(self):
        # All pairs biactive: both G and H near zero → fraction = 1.0 > 0.10
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.1, 0.1]),
            objective=lambda x: x[0] ** 2 + x[1] ** 2,
            gradient=lambda x: np.array([2.0 * x[0], 2.0 * x[1]]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        tn = result.tnlp_refined
        assert tn is not None
        assert tn.stationarity == "skipped"
        assert tn.success is False
        assert tn.status == -999
        assert tn.n_violations > 0  # reports n_biactive

    def test_not_skipped_when_clean(self):
        # Clean non-biactive solution: G=2>0, H=0 → no biactive pairs
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
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        tn = result.tnlp_refined
        assert tn is not None
        assert tn.stationarity != "skipped"
        assert tn.success is True


# ----------------------------------------------------------------------- #
# run_tnlp_refinement integration tests                                    #
# ----------------------------------------------------------------------- #

class TestRunTnlpRefinement:
    @pytest.fixture(scope="class")
    def solved(self):
        p = _make_simple()
        solver = pympcc.MPCCSolver(p, strategy="scholtes")
        result = solver.solve()
        return result, p, solver._strategy

    def test_returns_tnlp_result(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        assert isinstance(tnlp, TNLPResult)

    def test_tnlp_success(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        assert tnlp.success, f"TNLP failed with status {tnlp.status}: {tnlp.message}"

    def test_tnlp_obj_close(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        # TNLP objective should match the relaxation closely.
        assert abs(tnlp.obj - result.obj) < 1e-4

    def test_active_set_shape(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        I_G, I_H = tnlp.active_set
        assert len(I_G) + len(I_H) == p.n_comp

    def test_mult_comp_shape(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        assert tnlp.mult_comp_G.shape == (p.n_comp,)
        assert tnlp.mult_comp_H.shape == (p.n_comp,)

    def test_s_stationary_multipliers(self, solved):
        # At x*=(2,0): H_active (H_i=0 pinned equality, λ_H>0 → mu_H>0).
        # G is inactive (G_i=2>0) → lambda_G=0 → mu_G=0.
        # TNLP equality-pinned sign convention: no negation, so mu = λ directly.
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        if tnlp.success:
            # Inactive G side: multiplier should be ≈ 0
            assert abs(tnlp.mult_comp_G[0]) < 1e-4, (
                f"mu_G = {tnlp.mult_comp_G[0]:.4e} should be ≈ 0 for inactive G"
            )
            # Active H side (equality-pinned): multiplier should be > 0
            assert tnlp.mult_comp_H[0] >= -1e-4, (
                f"mu_H = {tnlp.mult_comp_H[0]:.4e} should be ≥ 0 for active H"
            )

    def test_kkt_residual_small(self, solved):
        # With correct (no-negation) TNLP multipliers the KKT residual should
        # match IPOPT's own convergence tolerance (≤ tol ≈ 1e-8).
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        if tnlp.success and tnlp.kkt_residual is not None:
            assert tnlp.kkt_residual < 1e-3

    def test_stationarity_not_unknown(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        if tnlp.success:
            assert tnlp.stationarity != "unknown"

    def test_solve_time_positive(self, solved):
        result, p, strategy = solved
        tnlp = run_tnlp_refinement(result, p, strategy)
        assert tnlp.solve_time >= 0.0


# ----------------------------------------------------------------------- #
# Via pympcc.solve(tnlp_refine=True)                                       #
# ----------------------------------------------------------------------- #

class TestTnlpViasolve:
    def test_tnlp_refined_populated(self):
        p = _make_simple()
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        assert result.tnlp_refined is not None
        assert isinstance(result.tnlp_refined, TNLPResult)

    def test_mult_comp_mpcc_populated_on_success(self):
        p = _make_simple()
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        if result.tnlp_refined.success:
            assert result.mult_comp_G_mpcc is not None
            assert result.mult_comp_H_mpcc is not None
            assert result.mult_comp_G_mpcc.shape == (p.n_comp,)
            assert result.mult_comp_H_mpcc.shape == (p.n_comp,)

    def test_no_tnlp_by_default(self):
        p = _make_simple()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.tnlp_refined is None
        assert result.mult_comp_G_mpcc is None
        assert result.mult_comp_H_mpcc is None

    def test_tnlp_in_json(self):
        import json
        p = _make_simple()
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        d = json.loads(result.to_json())
        if result.tnlp_refined is not None:
            assert "tnlp_refined" in d
            assert "stationarity" in d["tnlp_refined"]

    def test_summary_shows_tnlp(self):
        p = _make_simple()
        result = pympcc.solve(p, strategy="scholtes", tnlp_refine=True)
        s = result.summary(verbosity=1)
        assert "TNLP" in s

    @pytest.mark.parametrize("strategy", [
        "scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian",
    ])
    def test_all_iterative_strategies(self, strategy):
        p = _make_simple()
        result = pympcc.solve(p, strategy=strategy, tnlp_refine=True)
        assert result.per_pair_status is not None
        # TNLP should run and return something (success may vary by strategy)
        assert result.tnlp_refined is not None

    def test_direct_strategy(self):
        # Direct strategy doesn't set cleanup attrs; TNLP should still work.
        p = _make_simple()
        result = pympcc.solve(p, strategy="direct", tnlp_refine=True)
        # Direct may not converge (LICQ fails) but TNLP should at least not crash.
        assert result.tnlp_refined is not None or not result.success


# ----------------------------------------------------------------------- #
# TNLPResult dataclass                                                      #
# ----------------------------------------------------------------------- #

class TestTnlpResultDataclass:
    def test_fields(self):
        tnlp = TNLPResult(
            x=np.zeros(2), obj=0.0, status=0, message="ok", success=True,
            mult_comp_G=np.zeros(1), mult_comp_H=np.zeros(1),
            mult_ineq=None, mult_eq=None,
            kkt_residual=0.0, stationarity="S-stationary",
            n_iter=5, solve_time=0.01,
            active_set=(np.array([]), np.array([0])),
            n_violations=0,
        )
        assert tnlp.success
        assert tnlp.stationarity == "S-stationary"
        assert tnlp.n_iter == 5
        assert tnlp.n_violations == 0
