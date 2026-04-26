"""Tests for B-stationarity verification via LPCC enumeration.

Constructs candidate (problem, result) pairs by hand so the active-set
partition is controlled directly and the LP outcomes are predictable.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._stationarity import verify_b_stationarity
from pympcc.result import MPCCResult


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


def _affine_problem(grad_vec, JG, JH, n_comp):
    """Build a 2-D MPCC with linear G, H and a chosen objective gradient.

    f(x) = grad_vec @ x  →  ∇f = grad_vec  (constant).
    G(x) = JG @ x        →  ∇G = JG.
    H(x) = JH @ x        →  ∇H = JH.
    """
    grad_vec = np.asarray(grad_vec, dtype=float)
    JG = np.asarray(JG, dtype=float)
    JH = np.asarray(JH, dtype=float)
    n = grad_vec.size
    return pympcc.MPCCProblem(
        n=n, n_comp=n_comp,
        x0=np.zeros(n),
        objective=lambda x: float(grad_vec @ x),
        gradient=lambda x: grad_vec.copy(),
        comp_G=lambda x: JG @ x,
        comp_G_jacobian=lambda x: JG.copy(),
        comp_H=lambda x: JH @ x,
        comp_H_jacobian=lambda x: JH.copy(),
    )


# ======================================================================= #
# Trivial / early-exit branches                                             #
# ======================================================================= #

class TestEarlyExit:
    def test_unsuccessful_returns_unknown(self):
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([0.0, 0.0], [0.0], [0.0], success=False)
        out = verify_b_stationarity(r, p)
        assert out["status"] == "unknown"
        assert out["n_branches_checked"] == 0

    def test_empty_biactive_is_b_stationary(self):
        # G(x*)=2>0, H(x*)=1>0 — nothing biactive. With grad=0 the
        # linearised problem has min 0 trivially, so B-stat.
        p = _affine_problem([0.0, 0.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([2.0, 1.0], [2.0], [1.0])
        out = verify_b_stationarity(r, p)
        assert out["status"] == "B-stationary"
        assert out["n_biactive"] == 0
        assert out["n_branches_checked"] == 1
        assert out["witness_d"] is None

    def test_intractable_above_cap(self):
        n_comp = 12
        JG = np.zeros((n_comp, 2)); JG[:, 0] = 1.0
        JH = np.zeros((n_comp, 2)); JH[:, 1] = 1.0
        p = _affine_problem([1.0, 1.0], JG, JH, n_comp)
        r = _make_result([0.0, 0.0], np.zeros(n_comp), np.zeros(n_comp))
        out = verify_b_stationarity(r, p, max_biactive=10)
        assert out["status"] == "intractable"
        assert out["n_biactive"] == n_comp
        assert out["n_branches_checked"] == 0


# ======================================================================= #
# Single biactive pair                                                      #
# ======================================================================= #

class TestSingleBiactive:
    def test_b_stationary_at_origin(self):
        # f = x[0] + x[1], G = x[0], H = x[1], at x*=(0,0).
        # G branch: d[1]=0, d[0]≥0 → min d[0]+d[1] = 0.
        # H branch: d[0]=0, d[1]≥0 → min = 0. Both ≥ 0 ⇒ B-stat.
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([0.0, 0.0], [0.0], [0.0])
        out = verify_b_stationarity(r, p)
        assert out["status"] == "B-stationary"
        assert out["n_biactive"] == 1
        assert out["n_branches_checked"] == 2
        assert out["witness_d"] is None
        assert out["min_descent"] == pytest.approx(0.0, abs=1e-9)

    def test_not_b_stationary_witness(self):
        # f = -x[0] + x[1], G = x[0], H = x[1], at x*=(0,0).
        # H branch (H stays active → ∇H·d=0, ∇G·d≥0): d[1]=0, d[0]≥0.
        # Min -d[0] = -1 at d=(1,0) — descent witness via the H branch.
        p = _affine_problem([-1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([0.0, 0.0], [0.0], [0.0])
        out = verify_b_stationarity(r, p)
        assert out["status"] == "not B-stationary"
        assert out["min_descent"] == pytest.approx(-1.0, abs=1e-6)
        assert out["witness_branch"] == ("H",)
        d = out["witness_d"]
        assert d is not None
        assert d[0] == pytest.approx(1.0, abs=1e-6)
        assert d[1] == pytest.approx(0.0, abs=1e-6)


# ======================================================================= #
# Two biactive pairs (4 branches)                                           #
# ======================================================================= #

class TestMultipleBiactive:
    def test_two_pairs_b_stationary(self):
        # n=4, two independent comp pairs, f = sum(x). All branches min = 0.
        JG = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        JH = np.array([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        p = _affine_problem([1.0, 1.0, 1.0, 1.0], JG, JH, 2)
        r = _make_result([0.0, 0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        out = verify_b_stationarity(r, p)
        assert out["status"] == "B-stationary"
        assert out["n_biactive"] == 2
        assert out["n_branches_checked"] == 4

    def test_two_pairs_one_descent_branch(self):
        # f = -x[0] + x[1] + x[2] + x[3]. Only branch (G,*) on pair 0
        # admits descent on x[0]; pair 1 is B-stat in any branch.
        JG = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
        JH = np.array([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
        p = _affine_problem([-1.0, 1.0, 1.0, 1.0], JG, JH, 2)
        r = _make_result([0.0, 0.0, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        out = verify_b_stationarity(r, p, max_biactive=4)
        assert out["status"] == "not B-stationary"
        assert out["min_descent"] == pytest.approx(-1.0, abs=1e-6)
        assert out["witness_branch"] is not None
        assert out["witness_branch"][0] == "H"
