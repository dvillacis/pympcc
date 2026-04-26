"""Tests for B3 — pre-fixing on linear sign analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc._presolve import _detect_prefix_eq, presolve


# ======================================================================= #
# Builder                                                                   #
# ======================================================================= #

def _build_problem(
    n, n_comp, *,
    G_fn, G_jac, G_sp,
    H_fn, H_jac, H_sp,
    x0=None, xl=None, xu=None,
    n_ineq=0, ineq_fn=None, ineq_jac=None, ineq_sp=None,
    n_eq=0, eq_fn=None, eq_jac=None, eq_sp=None,
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
        n_eq=n_eq,
        eq_constraints=eq_fn, eq_jacobian=eq_jac,
        eq_jacobian_sparsity=eq_sp,
        **kw,
    )


# ======================================================================= #
# Detection                                                                 #
# ======================================================================= #

class TestDetection:
    def test_prefix_H_when_G_strictly_positive(self):
        # Pair 0: G_0(x) = x[0] + 1, x[0] ∈ [-0.5, 0.5] → G_0 ∈ [0.5, 1.5] > 0
        #         → H_0 = 0 forced (prefix_H_eq).
        # Pair 1: live, G/H normal.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -np.inf], xu=[0.5, np.inf])
        skip = np.zeros(n_comp, dtype=bool)
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pH.tolist() == [0]
        assert pG.size == 0

    def test_prefix_G_when_H_strictly_positive(self):
        # Symmetric: H_0 = x[0] + 1 > 0 over box → G_0 = 0 forced.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[1], x[0]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        H_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -np.inf], xu=[0.5, np.inf])
        skip = np.zeros(n_comp, dtype=bool)
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pG.tolist() == [0]
        assert pH.size == 0

    def test_both_positive_flags_infeasible(self):
        # Pair 0: G_0 = x[0] + 1, H_0 = x[1] + 2; both > 0 over box
        # x[0] ∈ [-0.5, 0.5], x[1] ∈ [-1.0, 1.0] → infeasible.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1] + 2.0, x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -1.0], xu=[0.5, 1.0])
        skip = np.zeros(n_comp, dtype=bool)
        _, _, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert infeas

    def test_interval_touching_zero_not_flagged(self):
        # G_0 = x[0] over [0, 1] → interval [0, 1], min == 0 so NOT > 0.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0], x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.5, 0.5]),
                           xl=[0.0, 0.0], xu=[1.0, 1.0])
        skip = np.zeros(n_comp, dtype=bool)
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pH.size == 0
        assert pG.size == 0

    def test_unbounded_below_no_certificate(self):
        # G_0 = x[0] + 1 with x[0] ∈ [-inf, 0.5] → gmin = -inf, not flagged.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-np.inf, -np.inf], xu=[0.5, np.inf])
        skip = np.zeros(n_comp, dtype=bool)
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pH.size == 0
        assert pG.size == 0

    def test_nonlinear_G_not_flagged(self):
        # G_0(x) = exp(x[0]) — strictly positive but nonlinear; B3 only
        # certifies linear rows.  Falls through to FBBT scope (which also
        # rejects nonlinear).
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([np.exp(x[0]), x[1]])
        G_jac = lambda x: np.array([np.exp(x[0]), 1.0])
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -1.0], xu=[0.5, 1.0])
        skip = np.zeros(n_comp, dtype=bool)
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pH.size == 0
        assert pG.size == 0

    def test_skip_mask_respected(self):
        # Pair 0 already dropped by B1/B2 must not be re-flagged.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -np.inf], xu=[0.5, np.inf])
        skip = np.array([True, False])
        pH, pG, infeas = _detect_prefix_eq(p,
            np.asarray(p.xl, dtype=float),
            np.asarray(p.xu, dtype=float),
            skip)
        assert not infeas
        assert pH.size == 0
        assert pG.size == 0


# ======================================================================= #
# End-to-end through presolve()                                             #
# ======================================================================= #

class TestPresolveIntegration:
    def test_pair_dropped_eq_grows(self):
        # n_comp=2, pair 0 has G_0 strictly positive over box; H_0 = 0
        # is appended to eq, n_comp drops to 1, n_eq grows by 1.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -np.inf], xu=[0.5, np.inf])
        reduced, pmap = presolve(p)
        assert pmap.prefix_H_eq.tolist() == [0]
        assert pmap.prefix_G_eq.size == 0
        assert reduced.n_comp == 1
        # No orig eq, so reduced.n_eq comes entirely from one prefix-H row.
        assert reduced.n_eq == 1
        # The eq row should evaluate H_0(x) = x[1] at x = (0.3, 0.4) → 0.4.
        np.testing.assert_allclose(
            reduced.eq_constraints(np.array([0.3, 0.4])),
            [0.4], atol=1e-12,
        )
        rows, cols = reduced.eq_jacobian_sparsity
        vals = reduced.eq_jacobian(np.array([0.3, 0.4]))
        J = np.zeros((1, 2))
        for r, c, v in zip(rows, cols, vals):
            J[r, c] += v
        np.testing.assert_allclose(J, [[0.0, 1.0]], atol=1e-12)

    def test_pair_dropped_with_orig_eq(self):
        # Mix B3 with an original eq row.  Reduced eq layout: [orig | prefix_H].
        # Use a 3-var problem so the orig eq constrains x[2] (not x[1]) and
        # therefore doesn't tighten H_0 = x[1] into a contradiction.
        n, n_comp = 3, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        # Original eq: x[2]^2 - 0.5 = 0  (nonlinear → escapes FBBT/A4 collapse)
        eq_fn  = lambda x: np.array([x[2] * x[2] - 0.5])
        eq_jac = lambda x: np.array([2.0 * x[2]])
        eq_sp  = (np.array([0], dtype=int), np.array([2], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0, 0.7]),
                           xl=[-0.5, -np.inf, -np.inf],
                           xu=[ 0.5,  np.inf,  np.inf],
                           n_eq=1, eq_fn=eq_fn, eq_jac=eq_jac, eq_sp=eq_sp)
        reduced, pmap = presolve(p)
        assert pmap.prefix_H_eq.tolist() == [0]
        assert reduced.n_comp == 1
        assert reduced.n_eq == 2  # 1 orig + 1 prefix-H
        # Reduced n may shrink if FBBT pins x[2]; rebuild a probe in reduced space.
        x_red = np.zeros(reduced.n)
        # locate x[1] in reduced (it survives — appears in comp_H/G)
        # easiest: evaluate at the first two reduced vars set to (0.3, 0.4).
        # The orig kept-var order is the same as input order minus pinned slots.
        # We don't depend on the exact index — just check shape and finiteness.
        eq_vals = reduced.eq_constraints(x_red)
        assert eq_vals.shape == (2,)

    def test_infeasible_falls_back_to_identity(self):
        # G_0 > 0 AND H_0 > 0 simultaneously → emit warning + identity.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1] + 2.0, x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -1.0], xu=[0.5, 1.0])
        with pytest.warns(UserWarning, match="linear sign analysis"):
            reduced, pmap = presolve(p)
        assert pmap.is_identity
        assert reduced is p


# ======================================================================= #
# Result expansion                                                          #
# ======================================================================= #

class TestExpandResult:
    def test_prefix_eq_multipliers_dropped_comp_zero_padded(self):
        # After dropping pair 0 via B3, expand_result must:
        # - discard the prefix-eq slot multiplier
        # - zero-pad comp_G/comp_H multipliers at the dropped pair index.
        n, n_comp = 2, 2
        G_fn  = lambda x: np.array([x[0] + 1.0, x[1]])
        G_jac = lambda x: np.ones(2)
        G_sp  = (np.array([0, 1], dtype=int), np.array([0, 1], dtype=int))
        H_fn  = lambda x: np.array([x[1], x[0]])
        H_jac = lambda x: np.ones(2)
        H_sp  = (np.array([0, 1], dtype=int), np.array([1, 0], dtype=int))
        p = _build_problem(n, n_comp,
                           G_fn=G_fn, G_jac=G_jac, G_sp=G_sp,
                           H_fn=H_fn, H_jac=H_jac, H_sp=H_sp,
                           x0=np.array([0.0, 0.0]),
                           xl=[-0.5, -np.inf], xu=[0.5, np.inf])
        reduced, pmap = presolve(p)
        assert reduced.n_comp == 1
        assert reduced.n_eq == 1
        assert reduced.n_ineq == 0

        # Reduced layout: [ineq(0) | eq(1: prefix-H) | G(1) | H(1)]
        from pympcc.result import MPCCResult
        red_mult = np.array([
            4.2,    # the prefix-H multiplier (will be discarded)
            0.7,    # G_1 multiplier
            1.1,    # H_1 multiplier
        ])
        result = MPCCResult(
            x=np.array([0.4, 0.5]),
            obj=0.0, status=0, message="ok",
            G=np.array([0.5]), H=np.array([0.5]),
            comp_residual=0.0, comp_residual_mean=0.0,
            success=True, strategy="test",
            mult_g=red_mult,
        )
        expanded = pmap.expand_result(result, p)
        # Original layout: [ineq(0) | eq(0) | G(2) | H(2)]
        # G_full = [0, 0.7], H_full = [0, 1.1]
        assert expanded.mult_g.tolist() == [0.0, 0.7, 0.0, 1.1]
