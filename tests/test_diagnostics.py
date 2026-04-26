"""Tests for §2.1 — MPCC-LICQ / MPCC-MFCQ classification."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._diagnostics import active_sets, classify_cq
from pympcc.result import MPCCResult


# ======================================================================= #
# Builders                                                                  #
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


def _affine_problem(
    grad_vec, JG, JH, n_comp,
    *, xl=None, xu=None,
    n_ineq=0, ineq_fn=None, ineq_jac=None, ineq_sp=None,
    n_eq=0, eq_fn=None, eq_jac=None, eq_sp=None,
):
    """Build a small MPCC with linear G, H and constant ∇f."""
    grad_vec = np.asarray(grad_vec, dtype=float)
    JG = np.asarray(JG, dtype=float)
    JH = np.asarray(JH, dtype=float)
    n = grad_vec.size
    n_rows_G = JG.shape[0]
    n_rows_H = JH.shape[0]
    G_rows, G_cols = np.where(JG != 0.0)
    H_rows, H_cols = np.where(JH != 0.0)
    G_vals = JG[G_rows, G_cols].copy()
    H_vals = JH[H_rows, H_cols].copy()
    kw = {}
    if xl is not None: kw["xl"] = np.asarray(xl, dtype=float)
    if xu is not None: kw["xu"] = np.asarray(xu, dtype=float)
    return pympcc.MPCCProblem(
        n=n, n_comp=n_comp,
        x0=np.zeros(n),
        objective=lambda x: float(grad_vec @ x),
        gradient=lambda x: grad_vec.copy(),
        comp_G=lambda x: JG @ x,
        comp_G_jacobian=lambda x: G_vals.copy(),
        comp_G_jacobian_sparsity=(G_rows.astype(int), G_cols.astype(int)),
        comp_H=lambda x: JH @ x,
        comp_H_jacobian=lambda x: H_vals.copy(),
        comp_H_jacobian_sparsity=(H_rows.astype(int), H_cols.astype(int)),
        n_ineq=n_ineq, ineq_constraints=ineq_fn,
        ineq_jacobian=ineq_jac, ineq_jacobian_sparsity=ineq_sp,
        n_eq=n_eq, eq_constraints=eq_fn,
        eq_jacobian=eq_jac, eq_jacobian_sparsity=eq_sp,
        **kw,
    )


# ======================================================================= #
# Active-set detection                                                      #
# ======================================================================= #

class TestActiveSets:
    def test_no_active_at_interior(self):
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([2.0, 1.0], [2.0], [1.0])
        sets = active_sets(r, p)
        assert sets["I_G"].size == 0
        assert sets["I_H"].size == 0
        assert sets["I_00"].size == 0
        assert sets["I_xL"].size == 0
        assert sets["I_xU"].size == 0

    def test_biactive_partition(self):
        # G=x[0], H=x[1] biactive at (0,0); third pair has G>0,H>0.
        JG = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        JH = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p = _affine_problem([1.0, 0.0, 0.0], JG, JH, 2,
                            xl=[-np.inf, -np.inf, -np.inf],
                            xu=[ np.inf,  np.inf,  np.inf])
        # Manual G, H: pair 0 = (0,0), pair 1 = (0.5, 0.5)
        r = _make_result([0.0, 0.0, 0.5], [0.0, 0.5], [0.0, 0.5])
        sets = active_sets(r, p)
        assert sets["I_G"].tolist() == [0]
        assert sets["I_H"].tolist() == [0]
        assert sets["I_00"].tolist() == [0]

    def test_active_bound(self):
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1,
                            xl=[0.0, -np.inf], xu=[np.inf, np.inf])
        r = _make_result([0.0, 1.0], [0.0], [1.0])
        sets = active_sets(r, p)
        assert sets["I_xL"].tolist() == [0]
        assert sets["I_xU"].size == 0


# ======================================================================= #
# MPCC-LICQ                                                                 #
# ======================================================================= #

class TestLICQ:
    def test_S_point_with_one_active_comp(self):
        # x*=(2,0): H active, G inactive. ∇H = [0,1] alone → full rank.
        p = _affine_problem([0.0, 0.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([2.0, 0.0], [2.0], [0.0])
        out = classify_cq(r, p)
        assert out["cq"] == "MPCC-LICQ"
        assert out["rank_deficit"] == 0
        assert out["active_set_sizes"]["G"] == 0
        assert out["active_set_sizes"]["H"] == 1

    def test_biactive_orthogonal_gradients(self):
        # G=x[0], H=x[1] biactive at origin: ∇G=[1,0], ∇H=[0,1] → rank 2.
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([0.0, 0.0], [0.0], [0.0])
        out = classify_cq(r, p)
        assert out["cq"] == "MPCC-LICQ"
        assert out["rank_deficit"] == 0
        assert out["active_set_sizes"]["biactive"] == 1

    def test_unsuccessful_returns_unknown(self):
        p = _affine_problem([1.0, 1.0], [[1.0, 0.0]], [[0.0, 1.0]], 1)
        r = _make_result([0.0, 0.0], [0.0], [0.0], success=False)
        out = classify_cq(r, p)
        assert out["cq"] == "unknown"
        assert out["rank_deficit"] is None


# ======================================================================= #
# MPCC-MFCQ (LICQ fails, MFCQ holds)                                        #
# ======================================================================= #

class TestMFCQ:
    def test_redundant_active_inequality(self):
        # n=3: G=x[0], H=x[1] biactive; ineqs g1=x[2], g2=2*x[2] both active.
        # ∇g1=[0,0,1], ∇g2=[0,0,2] are parallel → LICQ fails (deficit 1),
        # but a direction d=(0,0,-1), t=1 strictly satisfies both → MFCQ holds.
        JG = np.array([[1.0, 0.0, 0.0]])
        JH = np.array([[0.0, 1.0, 0.0]])
        rows = np.array([0, 1], dtype=int)
        cols = np.array([2, 2], dtype=int)
        ineq_fn  = lambda x: np.array([x[2], 2.0 * x[2]])
        ineq_jac = lambda x: np.array([1.0, 2.0])
        p = _affine_problem(
            [1.0, 1.0, 1.0], JG, JH, 1,
            xl=[-np.inf]*3, xu=[np.inf]*3,
            n_ineq=2, ineq_fn=ineq_fn, ineq_jac=ineq_jac,
            ineq_sp=(rows, cols),
        )
        r = _make_result([0.0, 0.0, 0.0], [0.0], [0.0])
        out = classify_cq(r, p)
        assert out["cq"] == "MPCC-MFCQ"
        assert out["rank_deficit"] >= 1


# ======================================================================= #
# Neither LICQ nor MFCQ                                                     #
# ======================================================================= #

class TestNone:
    def test_active_ineq_in_eq_kernel(self):
        # G=x[0], H=x[1] biactive; ineq g=x[0] active.
        # Eq-block kernel = {(0,0,*) | d[0]=d[1]=0}. ∇g·d = d[0] = 0 in
        # the kernel → no strict descent direction → MFCQ fails.
        JG = np.array([[1.0, 0.0]])
        JH = np.array([[0.0, 1.0]])
        rows = np.array([0], dtype=int)
        cols = np.array([0], dtype=int)
        ineq_fn  = lambda x: np.array([x[0]])
        ineq_jac = lambda x: np.array([1.0])
        p = _affine_problem(
            [1.0, 1.0], JG, JH, 1,
            xl=[-np.inf, -np.inf], xu=[np.inf, np.inf],
            n_ineq=1, ineq_fn=ineq_fn, ineq_jac=ineq_jac,
            ineq_sp=(rows, cols),
        )
        r = _make_result([0.0, 0.0], [0.0], [0.0])
        out = classify_cq(r, p)
        assert out["cq"] == "none"
        assert out["rank_deficit"] >= 1


# ======================================================================= #
# Solver integration                                                        #
# ======================================================================= #

class TestSolverIntegration:
    def test_diagnostics_off_by_default(self):
        problem = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([1.0]),
            comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
        )
        result = pympcc.solve(problem, strategy="scholtes")
        assert result.cq is None
        assert result.b_stationary is None

    def test_diagnostics_on_textbook_problem(self):
        # Textbook MPCC without redundant xl=0 (H≥0 already enforces x[1]≥0).
        # Stacking both makes the active-gradient matrix rank-deficient at
        # x*=(2,0) — a famous MPCC-MFCQ failure mode.
        problem = pympcc.MPCCProblem(
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
        result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
        assert result.success
        # x* ≈ (2, 0): H is active, no other active rows → MPCC-LICQ.
        assert result.cq == "MPCC-LICQ"
        assert result.cq_active_set_sizes["G"] == 0
        assert result.cq_active_set_sizes["H"] == 1
        assert result.cq_rank_deficit == 0
        assert result.b_stationary == "B-stationary"
        assert result.b_stationary_min_descent == pytest.approx(0.0, abs=1e-6)

    def test_diagnostics_flags_classical_mfcq_failure(self):
        # README's textbook formulation: H=x[1]≥0 *and* xl[1]=0 active at x*.
        # Both rows project to e_1 — rank deficit 1, MFCQ also fails because
        # the eq-block kernel has d[1]=0 forced and the bound row needs d[1]>0.
        problem = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            xl=np.zeros(2),
            objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([1.0]),
            comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
        )
        result = pympcc.solve(problem, strategy="scholtes", diagnostics=True)
        assert result.success
        assert result.cq == "none"
        assert result.cq_rank_deficit >= 1
