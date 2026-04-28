"""Tests for the exact-Hessian wiring in the active-set cleanup NLP.

Cleanup previously forced ``hessian_approximation=limited-memory`` on
every strategy.  For large problems (n ≫ 1e3) L-BFGS is dramatically
slower per iter than the exact Hessian the strategy already builds —
the run-time gap is what made the post-continuation cleanup pass burn
≥10 minutes on TV-MPCC.

The cleanup Lagrangian for Scholtes is the strategy Lagrangian with
``λ_GH = 0``, so we wrap whichever Hessian callback the strategy has
(manual or JAX) and pad the multipliers with ``n_c`` zeros.  These
tests exercise the wrapper directly (call shape, padding) and
end-to-end through ``cleanup="auto"``.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.strategies.lin_fukushima import LinFukushimaStrategy
from pympcc.strategies.scholtes import ScholtesStrategy
from pympcc.strategies.slack import SlackStrategy
from pympcc.strategies.smoothing import SmoothingStrategy


def _make_problem():
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


# ======================================================================= #
# Unit: wrapper math                                                        #
# ======================================================================= #

class TestWrapperPadding:
    def test_pads_lam_with_zero_GH_block(self):
        """Cleanup hess wraps the strategy hess with ``λ_GH = 0``."""
        p = _make_problem()
        # Manual Hessian whose values *depend on lam* so we can verify the
        # last n_c entries are passed as zero by the wrapper.
        rows = np.array([0, 1], dtype=np.intp)
        cols = np.array([0, 1], dtype=np.intp)
        seen: list[np.ndarray] = []

        def hess(x, lam, obj_factor):  # noqa: ARG001
            seen.append(np.asarray(lam).copy())
            return obj_factor * np.array([2.0, 2.0])

        p.lagrangian_hessian = hess
        p.lagrangian_hessian_sparsity = (rows, cols)

        strat = ScholtesStrategy(p, ipopt_options={})
        cleanup_hess, sp = strat._build_cleanup_hessian()
        assert sp == (rows, cols)

        # cleanup layout: lam_cu = [g, h, G, H] (size n_g + n_h + 2*n_c = 2)
        lam_cu = np.array([7.0, 11.0])
        out = cleanup_hess(np.array([0.5, 0.5]), lam_cu, 1.0)
        # Wrapper should have padded to size 3 (n_g + n_h + 3*n_c = 0+0+3*1 = 3)
        assert seen, "base hess was not invoked"
        lam_full = seen[-1]
        assert lam_full.shape == (3,)
        assert lam_full[0] == pytest.approx(7.0)   # G mult passes through
        assert lam_full[1] == pytest.approx(11.0)  # H mult passes through
        assert lam_full[2] == pytest.approx(0.0)   # GH block zeroed
        # Output unchanged from base hess (Hessian doesn't depend on lam here)
        np.testing.assert_allclose(out, [2.0, 2.0])

    def test_returns_none_without_problem_hessian(self):
        """Strategies fall back to L-BFGS when no exact Hessian is available."""
        strat = ScholtesStrategy(_make_problem(), ipopt_options={})
        hess_fn, sp = strat._build_cleanup_hessian()
        assert hess_fn is None
        assert sp is None

    def test_lin_fukushima_pads_two_blocks(self):
        """Lin-Fukushima layout is [g, h, G, H, G·H, G+H] — pad 2·n_c zeros."""
        p = _make_problem()
        rows = np.array([0, 1], dtype=np.intp)
        cols = np.array([0, 1], dtype=np.intp)
        seen: list[np.ndarray] = []

        def hess(x, lam, obj_factor):  # noqa: ARG001
            seen.append(np.asarray(lam).copy())
            return obj_factor * np.array([2.0, 2.0])

        p.lagrangian_hessian = hess
        p.lagrangian_hessian_sparsity = (rows, cols)

        strat = LinFukushimaStrategy(p, ipopt_options={})
        cleanup_hess, sp = strat._build_cleanup_hessian()
        assert sp == (rows, cols)

        # cleanup layout: lam_cu = [g, h, G, H], size 0+0+2*1 = 2
        lam_cu = np.array([3.0, 5.0])
        cleanup_hess(np.array([0.5, 0.5]), lam_cu, 1.0)
        # full layout: m = 0+0+4*1 = 4
        lam_full = seen[-1]
        assert lam_full.shape == (4,)
        assert lam_full[0] == pytest.approx(3.0)   # G
        assert lam_full[1] == pytest.approx(5.0)   # H
        assert lam_full[2] == pytest.approx(0.0)   # G·H
        assert lam_full[3] == pytest.approx(0.0)   # G+H

    def test_other_strategies_default_none_without_jax(self):
        """Smoothing/slack inherit the base default — None when no JAX."""
        for cls in (SmoothingStrategy, SlackStrategy):
            strat = cls(_make_problem(), ipopt_options={})
            hess_fn, sp = strat._build_cleanup_hessian()
            assert hess_fn is None
            assert sp is None


# ======================================================================= #
# Integration: cleanup uses exact Hessian end-to-end                        #
# ======================================================================= #

class TestCleanupEndToEnd:
    def test_cleanup_with_exact_hessian_solves(self):
        """Manual-Hessian problem solves through cleanup="auto"."""
        p = _make_problem()
        rows = np.array([0, 1], dtype=np.intp)
        cols = np.array([0, 1], dtype=np.intp)
        p.lagrangian_hessian = (
            lambda x, lam, obj_factor: obj_factor * np.array([2.0, 2.0])
        )
        p.lagrangian_hessian_sparsity = (rows, cols)

        result = pympcc.solve(p, strategy="scholtes", cleanup="auto")
        assert result.success
        # Cleanup should have run (comp_residual < 1e-2 after continuation).
        assert result.cleanup_status is not None
        # And it should have converged on the smooth fixed-active-set NLP.
        assert result.cleanup_accepted

    def test_cleanup_lbfgs_fallback_still_works(self):
        """No exact Hessian → cleanup still runs via L-BFGS path."""
        result = pympcc.solve(
            _make_problem(), strategy="scholtes", cleanup="auto",
        )
        assert result.success
        assert result.cleanup_status is not None
