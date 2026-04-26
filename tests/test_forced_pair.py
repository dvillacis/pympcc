"""Tests for B2 — forced-pair pruning (G_i / H_i promotion to ineq)."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._presolve import _detect_forced, presolve


# ======================================================================= #
# Builder                                                                   #
# ======================================================================= #

def _build_problem(
    n, n_comp, *,
    G_fn, G_jac, G_sp,
    H_fn, H_jac, H_sp,
    x0=None, xl=None, xu=None,
    n_ineq=0, ineq_fn=None, ineq_jac=None, ineq_sp=None,
):
    if x0 is None:
        x0 = np.zeros(n)
    kw = {}
    if xl is not None: kw["xl"] = np.asarray(xl, dtype=float)
    if xu is not None: kw["xu"] = np.asarray(xu, dtype=float)
    return pympcc.MPCCProblem(
        n=n, n_comp=n_comp,
        x0=np.asarray(x0, dtype=float),
        objective=lambda x: float(np.dot(x, x)),
        gradient=lambda x: 2.0 * np.asarray(x, dtype=float),
        comp_G=G_fn, comp_G_jacobian=G_jac, comp_G_jacobian_sparsity=G_sp,
        comp_H=H_fn, comp_H_jacobian=H_jac, comp_H_jacobian_sparsity=H_sp,
        n_ineq=n_ineq,
        ineq_constraints=ineq_fn, ineq_jacobian=ineq_jac,
        ineq_jacobian_sparsity=ineq_sp,
        **kw,
    )


# ======================================================================= #
# Detection                                                                 #
# ======================================================================= #

class TestDetection:
    def test_promote_G_when_H_structurally_zero(self):
        # n_comp = 2, both G_i depend on x; only pair 0 has H_i ≡ 0
        # (empty H sparsity row + value 0 at x0).
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0], x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        # H_i: pair 0 ≡ 0 (no Jac entry for row 0); pair 1 = x[1]
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        dead = np.zeros(n_comp, dtype=bool)
        promote_G, promote_H, extra_dead = _detect_forced(p, dead)
        assert promote_G.tolist() == [0]
        assert promote_H.size == 0
        assert extra_dead.size == 0

    def test_promote_H_when_G_structurally_zero(self):
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([0.0, x[1]])
        G_jac = lambda x: np.array([1.0])
        G_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        H_fn  = lambda x: np.array([x[0], x[1]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        promote_G, promote_H, extra_dead = _detect_forced(
            p, np.zeros(n_comp, dtype=bool))
        assert promote_G.size == 0
        assert promote_H.tolist() == [0]
        assert extra_dead.size == 0

    def test_both_constant_zero_marks_extra_dead(self):
        # n_comp = 2; pair 0 has G ≡ 0 AND H ≡ 0; pair 1 is live.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([0.0, x[1]])
        G_jac = lambda x: np.array([1.0])
        G_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        promote_G, promote_H, extra_dead = _detect_forced(
            p, np.zeros(n_comp, dtype=bool))
        assert promote_G.size == 0
        assert promote_H.size == 0
        assert extra_dead.tolist() == [0]

    def test_skips_pair_already_dead(self):
        # If B1 marks pair 0 dead, _detect_forced must skip it.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([0.0, x[1]])  # G_0 ≡ 0 (empty Jac row 0)
        G_jac = lambda x: np.array([1.0])
        G_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        H_fn  = lambda x: np.array([x[0], x[1]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        dead = np.array([True, False])
        promote_G, promote_H, extra_dead = _detect_forced(p, dead)
        # Pair 0 was already dead so promote_H should NOT include it.
        assert promote_H.size == 0


# ======================================================================= #
# End-to-end through presolve()                                             #
# ======================================================================= #

class TestPresolveIntegration:
    def test_promote_G_appears_in_reduced_ineq(self):
        # n=2, n_comp=2; pair 0 has H ≡ 0 → G_0 promoted to ineq.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0], x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        reduced, pmap = presolve(p)
        assert pmap.promote_G.tolist() == [0]
        assert reduced.n_comp == 1
        assert reduced.n_ineq == 1
        # The promoted ineq evaluates -G_0(x) = -x[0]
        x_red = np.array([0.7, 0.3])  # = x_full since no pinning
        np.testing.assert_allclose(reduced.ineq_constraints(x_red),
                                   [-0.7], atol=1e-12)
        # Jacobian of -x[0] w.r.t. (x0, x1) is (-1, 0).
        # COO: row 0 col 0 = -1.
        rows, cols = reduced.ineq_jacobian_sparsity
        vals = reduced.ineq_jacobian(x_red)
        # Build dense
        J = np.zeros((1, 2))
        for r, c, v in zip(rows, cols, vals):
            J[r, c] += v
        np.testing.assert_allclose(J, [[-1.0, 0.0]], atol=1e-12)

    def test_promote_with_existing_ineq(self):
        # n_comp=2; orig ineq x[0]-1 ≤ 0; pair 0 promoted (H_0 ≡ 0).
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0], x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        ineq_fn  = lambda x: np.array([x[0] - 1.0])
        ineq_jac = lambda x: np.array([1.0])
        ineq_sp  = (np.array([0], dtype=int), np.array([0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.5]),
                           n_ineq=1, ineq_fn=ineq_fn, ineq_jac=ineq_jac,
                           ineq_sp=ineq_sp)
        reduced, pmap = presolve(p)
        assert pmap.promote_G.tolist() == [0]
        assert reduced.n_comp == 1
        assert reduced.n_ineq == 2  # 1 orig + 1 promoted
        # Verify both rows evaluate as expected at x = (0.2, 0.5):
        # row 0: x[0] - 1 = -0.8
        # row 1: -G_0(x) = -x[0] = -0.2
        np.testing.assert_allclose(
            reduced.ineq_constraints(np.array([0.2, 0.5])),
            [-0.8, -0.2], atol=1e-12,
        )

    def test_promote_with_no_orig_ineq(self):
        # n_comp=2, no orig ineq; pair 0 promotes (H_0 ≡ 0).
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] - 0.1, x[1] - 0.2])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.3, 0.4]))
        reduced, pmap = presolve(p)
        assert pmap.promote_G.tolist() == [0]
        assert reduced.n_comp == 1
        assert reduced.n_ineq == 1
        # -G_0 at x = (0.3, 0.4): -(0.3 - 0.1) = -0.2
        np.testing.assert_allclose(
            reduced.ineq_constraints(np.array([0.3, 0.4])),
            [-0.2], atol=1e-12,
        )


# ======================================================================= #
# Result expansion                                                          #
# ======================================================================= #

class TestExpandResult:
    def test_promoted_multipliers_dropped_comp_zero_padded(self):
        # Same setup as test_promote_G_appears_in_reduced_ineq.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0], x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([0.0, x[1]])
        H_jac = lambda x: np.array([1.0])
        H_sp  = (np.array([1], dtype=int), np.array([1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]))
        reduced, pmap = presolve(p)
        assert reduced.n_comp == 1
        assert reduced.n_ineq == 1

        # Reduced layout: [ineq(1) | eq(0) | G(1) | H(1)]
        from pympcc.result import MPCCResult
        red_mult = np.array([2.5,    # promoted G_0 multiplier (will be discarded)
                             0.7,    # G_1 multiplier
                             1.1])   # H_1 multiplier
        result = MPCCResult(
            x=np.array([0.4, 0.6]),
            obj=0.0, status=0, message="ok",
            G=np.array([0.4]), H=np.array([0.6]),
            comp_residual=0.0, comp_residual_mean=0.0,
            success=True, strategy="test",
            mult_g=red_mult,
        )
        expanded = pmap.expand_result(result, p)
        # Original layout: [ineq(0) | eq(0) | G(2) | H(2)]
        # Expanded mult_g should have 4 entries:
        # ineq_full = []
        # eq_full = []
        # G_full = [0, 0.7]    (zero at promoted pair 0)
        # H_full = [0, 1.1]
        assert expanded.mult_g.tolist() == [0.0, 0.7, 0.0, 1.1]
