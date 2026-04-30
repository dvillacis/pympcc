"""Tests for §2.7 — PATH-style multi-merit & degeneracy diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pympcc
from pympcc._diagnostics import (
    degeneracy_report,
    initial_point_statistics,
    jac_norms,
    merit_cross_check,
)
from pympcc.result import MPCCResult


# ======================================================================= #
# Helpers                                                                   #
# ======================================================================= #

def _make_result(x, G, H, *, success=True):
    x = np.asarray(x, dtype=float)
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)
    return MPCCResult(
        x=x, obj=0.0, status=0, message="ok",
        G=G, H=H,
        comp_residual=float(np.max(np.abs(G * H))) if G.size else 0.0,
        comp_residual_mean=float(np.mean(np.abs(G * H))) if G.size else 0.0,
        success=success, strategy="test",
    )


def _simple_problem():
    """Tiny well-behaved MPCC: min (x-2)² + (y-1)² s.t. x ≥ 0 ⊥ y ≥ 0."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# Multi-merit cross-check                                                   #
# ======================================================================= #

class TestMeritCrossCheck:
    def test_zero_at_exact_complementarity(self):
        # x* = (2, 0): G_active.  All three merits vanish on G·H, FB, min.
        r = _make_result([2.0, 0.0], [2.0], [0.0])
        m = merit_cross_check(r, _simple_problem())
        assert m["fb_max"] < 1e-12
        assert m["min_map_max"] < 1e-12
        assert m["inner_product_max"] < 1e-12

    def test_disagreement_ratio_unit_when_all_zero(self):
        # All merits identically zero → disagreement = max/min with floor = 1.
        r = _make_result([2.0, 0.0], [2.0], [0.0])
        m = merit_cross_check(r, _simple_problem())
        # max = 0, denom floored at 1e-16 → ratio is 0; treat as ≤ 1.
        assert m["disagreement_ratio"] <= 1.0 + 1e-9

    def test_inner_product_matches_comp_residual(self):
        r = _make_result([1.0, 2.0], [1.0], [2.0])
        m = merit_cross_check(r, _simple_problem())
        assert m["inner_product_max"] == pytest.approx(2.0)
        assert m["inner_product_max"] == pytest.approx(r.comp_residual)

    def test_fb_formula(self):
        # φ_FB(3, 4) = 3 + 4 − √(9 + 16) = 2.  |.| = 2.
        r = _make_result([3.0, 4.0], [3.0], [4.0])
        m = merit_cross_check(r, _simple_problem())
        assert m["fb_max"] == pytest.approx(2.0)

    def test_min_map_takes_smaller(self):
        r = _make_result([5.0, 0.7], [5.0], [0.7])
        m = merit_cross_check(r, _simple_problem())
        assert m["min_map_max"] == pytest.approx(0.7)

    def test_empty_problem(self):
        r = _make_result([1.0], [], [])
        m = merit_cross_check(r, _simple_problem())
        assert m["fb_max"] == 0.0
        assert m["min_map_max"] == 0.0
        assert m["inner_product_max"] == 0.0
        assert m["disagreement_ratio"] == 1.0


# ======================================================================= #
# Jacobian row/col norms                                                    #
# ======================================================================= #

class TestJacNorms:
    def test_returns_empty_on_failed_solve(self):
        r = _make_result([0.0, 0.0], [0.0], [0.0], success=False)
        out = jac_norms(r, _simple_problem())
        assert out["row"]["max"] is None
        assert out["col"]["max"] is None

    def test_no_active_constraints_at_interior(self):
        # Interior: G > 0 and H > 0, no bounds active → empty active matrix.
        r = _make_result([2.0, 3.0], [2.0], [3.0])
        out = jac_norms(r, _simple_problem())
        assert out["row"]["n_rows"] == 0
        assert out["col"]["n_cols"] == 0

    def test_picks_up_active_G_row(self):
        # G_active at x=(0, 1): I_G = {0}, x_L is also active on x[0]=0.
        r = _make_result([0.0, 1.0], [0.0], [1.0])
        # Need bounds for I_xL detection.
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            objective=lambda x: float((x[0] - 2) ** 2 + (x[1] - 1) ** 2),
            gradient=lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 1)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        out = jac_norms(r, p)
        assert out["row"]["n_rows"] >= 1   # at least the G row picked up
        assert out["row"]["max"] >= 1.0


# ======================================================================= #
# Initial-point statistics                                                  #
# ======================================================================= #

class TestInitialPointStatistics:
    def test_simple(self):
        p = _simple_problem()
        s = initial_point_statistics(p)
        # x0 = (0.5, 0.5) → G·H = 0.25, min = 0.5, no bound or constraint viol.
        assert s["comp_residual"] == pytest.approx(0.25)
        assert s["min_map_residual"] == pytest.approx(0.5)
        assert s["max_bound_violation"] == 0.0
        assert s["ineq_residual"] == 0.0
        assert s["eq_residual"] == 0.0

    def test_bound_violation(self):
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([-1.0, 0.5]),
            xl=np.array([0.0, 0.0]),
            xu=np.array([10.0, 10.0]),
            objective=lambda x: float(x @ x),
            gradient=lambda x: 2 * x,
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        s = initial_point_statistics(p)
        assert s["max_bound_violation"] == pytest.approx(1.0)


# ======================================================================= #
# Degeneracy report                                                         #
# ======================================================================= #

class TestDegeneracyReport:
    def test_returns_none_when_failed(self):
        r = _make_result([0.0, 0.0], [0.0], [0.0], success=False)
        d = degeneracy_report(r, _simple_problem())
        assert d["n_biactive"] is None

    def test_biactive_count(self):
        r = _make_result([0.0, 0.0], [0.0], [0.0])
        d = degeneracy_report(r, _simple_problem())
        assert d["n_biactive"] == 1

    def test_no_biactive_at_strict(self):
        r = _make_result([2.0, 0.0], [2.0], [0.0])
        d = degeneracy_report(r, _simple_problem())
        assert d["n_biactive"] == 0


# ======================================================================= #
# End-to-end through pympcc.solve(diagnostics=True)                         #
# ======================================================================= #

class TestEndToEnd:
    def test_diagnostics_populated(self):
        p = _simple_problem()
        r = pympcc.solve(p, strategy="scholtes", diagnostics=True)
        assert r.success
        assert r.merit_cross_check is not None
        assert r.jac_row_norms is not None
        assert r.jac_col_norms is not None
        assert r.degeneracy_report is not None
        assert r.initial_point_stats is not None
        # FB and min-map should agree with comp_residual to within a few orders.
        m = r.merit_cross_check
        assert m["fb_max"] < 1e-3
        assert m["min_map_max"] < 1e-3

    def test_diagnostics_off_by_default(self):
        p = _simple_problem()
        r = pympcc.solve(p, strategy="scholtes")
        assert r.merit_cross_check is None
        assert r.jac_row_norms is None
        assert r.degeneracy_report is None
        assert r.initial_point_stats is None

    def test_json_roundtrip_with_diagnostics(self):
        p = _simple_problem()
        r = pympcc.solve(p, strategy="scholtes", diagnostics=True)
        d = json.loads(r.to_json())
        assert "merit_cross_check" in d
        assert "jac_row_norms" in d
        assert "jac_col_norms" in d
        assert "degeneracy_report" in d
        assert "initial_point_stats" in d
        # Schemas survive JSON.
        assert "fb_max" in d["merit_cross_check"]
        assert "n_biactive" in d["degeneracy_report"]
        assert "comp_residual" in d["initial_point_stats"]

    def test_summary_renders_diagnostics(self):
        p = _simple_problem()
        r = pympcc.solve(p, strategy="scholtes", diagnostics=True)
        s = r.summary(verbosity=1)
        assert "Merit cross-check" in s
        assert "Degeneracy" in s
