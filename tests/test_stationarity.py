"""
Tests for MPCC stationarity classification.

Unit tests use mocked MPCCResult with controlled mult_g values to verify the
classification logic directly.  Integration tests run the full solve pipeline
and check that result.stationarity is populated and consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._stationarity import classify_stationarity
from tests.macmpec_problems import PROBLEM_NAMES

SIMPLE = PROBLEM_NAMES["simple"]
BARD1  = PROBLEM_NAMES["bard1"]
KTH1   = PROBLEM_NAMES["kth1"]

VALID_LEVELS = {
    "S-stationary", "M-stationary", "C-stationary",
    "W-stationary", "unknown", "not stationary",
}


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _make_mock(
    G, H,
    mu_G_lit, mu_H_lit,
    n_ineq: int = 0,
    n_eq: int = 0,
    success: bool = True,
):
    """
    Build a minimal (MPCCResult, MPCCProblem) pair with hand-crafted
    multipliers for unit testing.

    Multipliers are given in *literature* convention (μ ≥ 0 at S-stationary).
    They are negated internally to match IPOPT's sign convention before
    being stored in mult_g.
    """
    G_arr = np.asarray(G, dtype=float)
    H_arr = np.asarray(H, dtype=float)
    n_comp = len(G_arr)
    # IPOPT stores λ = -μ_lit at active lower-bound constraints
    mu_G_ipopt = -np.asarray(mu_G_lit, dtype=float)
    mu_H_ipopt = -np.asarray(mu_H_lit, dtype=float)
    # Layout: [ineq(n_ineq) | eq(n_eq) | G(n_comp) | H(n_comp) | product(n_comp)]
    mult_g = np.concatenate([
        np.zeros(n_ineq),
        np.zeros(n_eq),
        mu_G_ipopt,
        mu_H_ipopt,
        np.zeros(n_comp),   # product/smoothing block — ignored by classifier
    ])
    n_vars = 2
    # Minimal MPCCProblem that only serves as a dimension container.
    problem = pympcc.MPCCProblem(
        n=n_vars, n_comp=n_comp,
        x0=np.zeros(n_vars),
        objective=lambda x: 0.0,
        gradient=lambda x: np.zeros(n_vars),
        comp_G=lambda x: G_arr,
        comp_G_jacobian=lambda x: np.zeros((n_comp, n_vars)),
        comp_H=lambda x: H_arr,
        comp_H_jacobian=lambda x: np.zeros((n_comp, n_vars)),
        n_ineq=n_ineq,
        ineq_constraints=lambda x: np.zeros(n_ineq) if n_ineq else None,
        ineq_jacobian=lambda x: np.zeros((n_ineq, n_vars)) if n_ineq else None,
        n_eq=n_eq,
        eq_constraints=lambda x: np.zeros(n_eq) if n_eq else None,
        eq_jacobian=lambda x: np.zeros((n_eq, n_vars)) if n_eq else None,
    )
    result = pympcc.MPCCResult(
        x=np.zeros(n_vars),
        obj=0.0,
        status=0,
        message="",
        G=G_arr,
        H=H_arr,
        comp_residual=0.0,
        comp_residual_mean=0.0,
        success=success,
        strategy="mock",
        mult_g=mult_g,
    )
    return result, problem


# ------------------------------------------------------------------ #
# 1. Unit tests: classification logic                                  #
# ------------------------------------------------------------------ #

class TestClassifyLogic:
    """Each test uses a single biactive pair (G=0, H=0) unless noted."""

    def test_both_positive_is_S(self):
        r, p = _make_mock([0.0], [0.0], [1.0], [1.0])
        assert classify_stationarity(r, p) == "S-stationary"

    def test_one_zero_one_positive_is_S(self):
        r, p = _make_mock([0.0], [0.0], [0.0], [1.0])
        assert classify_stationarity(r, p) == "S-stationary"

    def test_both_zero_is_S(self):
        r, p = _make_mock([0.0], [0.0], [0.0], [0.0])
        assert classify_stationarity(r, p) == "S-stationary"

    def test_zero_negative_is_M(self):
        # (0, -1): product=0 → C holds; NOT(both<0) → M holds; one<0 → S fails
        r, p = _make_mock([0.0], [0.0], [0.0], [-1.0])
        assert classify_stationarity(r, p) == "M-stationary"

    def test_negative_zero_is_M(self):
        r, p = _make_mock([0.0], [0.0], [-1.0], [0.0])
        assert classify_stationarity(r, p) == "M-stationary"

    def test_both_negative_is_C(self):
        # product = (-1)*(-1) = 1 > 0 → C holds; but NOT(both<0) fails → M fails
        r, p = _make_mock([0.0], [0.0], [-1.0], [-1.0])
        assert classify_stationarity(r, p) == "C-stationary"

    def test_positive_negative_is_W(self):
        # product = 1*(-1) = -1 < 0 → C fails; M fails; W is weakest → W
        r, p = _make_mock([0.0], [0.0], [1.0], [-1.0])
        assert classify_stationarity(r, p) == "W-stationary"

    def test_negative_positive_is_W(self):
        r, p = _make_mock([0.0], [0.0], [-1.0], [1.0])
        assert classify_stationarity(r, p) == "W-stationary"

    def test_no_biactive_is_S(self):
        # G=1.0 >> tol, H=1.0 >> tol → I_00 is empty → vacuously S
        r, p = _make_mock([1.0], [1.0], [-999.0], [-999.0])
        assert classify_stationarity(r, p) == "S-stationary"

    def test_mult_g_none_is_unknown(self):
        r, p = _make_mock([0.0], [0.0], [1.0], [1.0])
        r.mult_g = None
        assert classify_stationarity(r, p) == "unknown"

    def test_not_success_is_not_stationary(self):
        r, p = _make_mock([0.0], [0.0], [1.0], [1.0], success=False)
        assert classify_stationarity(r, p) == "not stationary"

    def test_mixed_pairs_worst_wins(self):
        # Two biactive pairs: pair 0 is S-stat (both positive), pair 1 is W (opposite signs)
        r, p = _make_mock([0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, -1.0])
        # Pair 1 has mu_G=1, mu_H=-1: product=-1 < 0 → C fails → W
        assert classify_stationarity(r, p) == "W-stationary"

    def test_multiple_pairs_all_S(self):
        r, p = _make_mock([0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [0.5, 0.1])
        assert classify_stationarity(r, p) == "S-stationary"

    def test_multiple_pairs_mixed_S_C(self):
        # Pair 0: S-stat (both positive); Pair 1: C-stat (both negative)
        r, p = _make_mock([0.0, 0.0], [0.0, 0.0], [1.0, -1.0], [1.0, -1.0])
        assert classify_stationarity(r, p) == "C-stationary"

    def test_tol_controls_biactive_detection(self):
        # G=5e-7 is within tol=1e-6 → biactive; within tol=1e-8 → NOT biactive
        r, p = _make_mock([5e-7], [0.0], [1.0], [1.0])
        assert classify_stationarity(r, p, tol=1e-6) == "S-stationary"
        # With very tight tol, pair is NOT biactive → vacuously S regardless
        assert classify_stationarity(r, p, tol=1e-8) == "S-stationary"

    def test_tol_controls_biactive_with_bad_multipliers(self):
        # G=5e-7, negative multipliers. With tol=1e-6 (biactive), C-stat expected.
        # With tol=1e-8 (not biactive), vacuously S.
        r, p = _make_mock([5e-7], [0.0], [-1.0], [-1.0])
        assert classify_stationarity(r, p, tol=1e-6) == "C-stationary"
        assert classify_stationarity(r, p, tol=1e-8) == "S-stationary"

    def test_with_ineq_and_eq_offset(self):
        # n_ineq=2, n_eq=1: G multipliers start at offset 3 in mult_g
        r, p = _make_mock([0.0], [0.0], [1.0], [1.0], n_ineq=2, n_eq=1)
        assert classify_stationarity(r, p) == "S-stationary"

    def test_with_ineq_and_eq_offset_bad(self):
        r, p = _make_mock([0.0], [0.0], [-1.0], [-1.0], n_ineq=2, n_eq=1)
        assert classify_stationarity(r, p) == "C-stationary"


# ------------------------------------------------------------------ #
# 2. Integration tests                                                 #
# ------------------------------------------------------------------ #

class TestStationarityIntegration:
    @pytest.mark.parametrize("strategy", ["direct", "scholtes", "smoothing"])
    def test_stationarity_field_populated(self, strategy):
        result = pympcc.solve(SIMPLE.problem, strategy=strategy)
        assert result.stationarity in VALID_LEVELS

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_classify_matches_result_field(self, strategy):
        """classify_stationarity() must return the same value as result.stationarity."""
        result = pympcc.solve(SIMPLE.problem, strategy=strategy)
        assert pympcc.classify_stationarity(result, SIMPLE.problem) == result.stationarity

    def test_mult_g_shape_simple(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes")
        p = SIMPLE.problem
        assert result.mult_g is not None
        assert isinstance(result.mult_g, np.ndarray)
        expected_len = p.n_ineq + p.n_eq + 3 * p.n_comp
        assert result.mult_g.shape == (expected_len,)

    def test_bard1_stationarity_valid(self):
        result = pympcc.solve(BARD1.problem, strategy="scholtes")
        assert result.stationarity in VALID_LEVELS

    def test_kth1_biactive_classified(self):
        """kth1 has x*=(0,0) — biactive pair — and must return a meaningful level."""
        result = pympcc.solve(KTH1.problem, strategy="scholtes")
        assert result.stationarity not in ("unknown", "not stationary")
        assert result.stationarity in VALID_LEVELS

    def test_stationarity_not_unknown_for_converged(self):
        """A successful solve must never return 'unknown'."""
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes")
        assert result.success
        assert result.stationarity != "unknown"

    def test_tol_parameter_valid(self):
        result = pympcc.solve(SIMPLE.problem, strategy="scholtes")
        for tol in [1e-8, 1e-6, 1e-3, 1e-1]:
            s = pympcc.classify_stationarity(result, SIMPLE.problem, tol=tol)
            assert s in VALID_LEVELS

    def test_classify_stationarity_exported(self):
        """classify_stationarity must be importable from the top-level pympcc namespace."""
        assert hasattr(pympcc, "classify_stationarity")
        assert callable(pympcc.classify_stationarity)
