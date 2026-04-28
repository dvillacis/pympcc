"""Tests for the auto pair-scaling detector (§4.3)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc._autoscale import autoscale_comp_pairs


# ======================================================================= #
# Builders                                                                  #
# ======================================================================= #

def _balanced_problem():
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


def _imbalanced_problem(scale_G=1e-6, scale_H=1.0):
    """One pair where G is scaled down by `scale_G` and H by `scale_H`."""
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x, sG=scale_G: np.array([sG * x[0]]),
        comp_G_jacobian=lambda x, sG=scale_G: np.array([[sG, 0.0]]),
        comp_H=lambda x, sH=scale_H: np.array([sH * x[1]]),
        comp_H_jacobian=lambda x, sH=scale_H: np.array([[0.0, sH]]),
    )


def _mixed_pairs_problem():
    """Two pairs: pair 0 balanced, pair 1 imbalanced (G ~1e-5, H ~1)."""
    return pympcc.MPCCProblem(
        n=4, n_comp=2,
        x0=np.array([0.5, 0.5, 0.5, 0.5]),
        objective=lambda x: float(np.sum((x - np.array([2.0, 1.0, 2.0, 1.0])) ** 2)),
        gradient=lambda x: 2.0 * (x - np.array([2.0, 1.0, 2.0, 1.0])),
        comp_G=lambda x: np.array([x[0], 1e-5 * x[2]]),
        comp_G_jacobian=lambda x: np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1e-5, 0.0]]
        ),
        comp_H=lambda x: np.array([x[1], x[3]]),
        comp_H_jacobian=lambda x: np.array(
            [[0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        ),
    )


# ======================================================================= #
# Detector behaviour                                                        #
# ======================================================================= #

class TestAutoscaleDetector:
    def test_balanced_returns_unit_scales(self):
        s_G, s_H = autoscale_comp_pairs(_balanced_problem())
        np.testing.assert_array_equal(s_G, np.ones(1))
        np.testing.assert_array_equal(s_H, np.ones(1))

    def test_imbalanced_pair_rescaled(self):
        p = _imbalanced_problem(scale_G=1e-6, scale_H=1.0)
        s_G, s_H = autoscale_comp_pairs(p)
        # G typical magnitude ≈ 1e-6 · 0.5 ⇒ s_G ~ O(1e6)
        assert s_G[0] > 1e5
        # H is well-conditioned at this scale, but pair flagged imbalanced ⇒
        # both sides get equilibrated to ~1.
        assert s_H[0] == pytest.approx(1.0 / 0.5, rel=0.5)

    def test_mixed_pairs_only_imbalanced_rescaled(self):
        s_G, s_H = autoscale_comp_pairs(_mixed_pairs_problem())
        # Pair 0 (balanced) untouched
        assert s_G[0] == 1.0
        assert s_H[0] == 1.0
        # Pair 1 rescaled
        assert s_G[1] > 1.0
        assert s_H[1] != 1.0 or s_G[1] != 1.0  # at least G changes

    def test_threshold_controls_aggressiveness(self):
        p = _imbalanced_problem(scale_G=1e-2, scale_H=1.0)  # ratio ~ 100
        s_G_loose, _ = autoscale_comp_pairs(p, threshold=1e3)
        s_G_strict, _ = autoscale_comp_pairs(p, threshold=10.0)
        # Loose threshold (1e3) > observed ratio (100) ⇒ no rescale
        assert s_G_loose[0] == 1.0
        # Strict threshold (10) < observed ratio ⇒ rescale
        assert s_G_strict[0] != 1.0

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold must be > 1"):
            autoscale_comp_pairs(_balanced_problem(), threshold=1.0)

    def test_invalid_n_probes(self):
        with pytest.raises(ValueError, match="n_probes must be >= 0"):
            autoscale_comp_pairs(_balanced_problem(), n_probes=-1)

    def test_zero_probes_uses_x0_only(self):
        s_G, s_H = autoscale_comp_pairs(_balanced_problem(), n_probes=0)
        # Still well-defined for balanced problem
        np.testing.assert_array_equal(s_G, np.ones(1))
        np.testing.assert_array_equal(s_H, np.ones(1))

    def test_seed_determinism(self):
        p = _imbalanced_problem(scale_G=1e-6, scale_H=1.0)
        sG1, sH1 = autoscale_comp_pairs(p, seed=42)
        sG2, sH2 = autoscale_comp_pairs(p, seed=42)
        np.testing.assert_array_equal(sG1, sG2)
        np.testing.assert_array_equal(sH1, sH2)

    def test_returns_strictly_positive(self):
        p = _imbalanced_problem(scale_G=1e-15, scale_H=1.0)
        s_G, s_H = autoscale_comp_pairs(p, floor=1e-10)
        # Floor caps the divisor; result still positive and finite.
        assert np.all(s_G > 0)
        assert np.all(s_H > 0)
        assert np.all(np.isfinite(s_G))
        assert np.all(np.isfinite(s_H))


# ======================================================================= #
# Solver integration                                                        #
# ======================================================================= #

class TestSolverIntegration:
    def test_autoscale_flag_balanced_no_warning(self):
        # Balanced problem ⇒ no rescale ⇒ no warning emitted.
        p = _balanced_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = pympcc.solve(p, strategy="scholtes", autoscale=True)
        assert result.success
        # Scales not populated (nothing was rescaled)
        assert p.comp_G_scale is None
        assert p.comp_H_scale is None

    def test_autoscale_flag_imbalanced_emits_warning(self):
        p = _imbalanced_problem(scale_G=1e-6, scale_H=1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pympcc.solve(p, strategy="scholtes", autoscale=True)
        msgs = [str(w.message) for w in caught
                if "autoscale" in str(w.message)]
        assert len(msgs) == 1
        assert "1/1" in msgs[0]
        assert result.success
        # Scales were populated on the problem
        assert p.comp_G_scale is not None
        assert p.comp_H_scale is not None
        # Result also carries them (for unscale_multipliers)
        assert result.comp_G_scale is not None
        assert result.comp_H_scale is not None

    def test_autoscale_off_by_default(self):
        p = _imbalanced_problem(scale_G=1e-6, scale_H=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pympcc.solve(p, strategy="scholtes")
        assert p.comp_G_scale is None

    def test_user_scale_wins_over_autoscale(self):
        p = _imbalanced_problem(scale_G=1e-6, scale_H=1.0)
        manual = np.array([2.5])
        p.comp_G_scale = manual
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no autoscale warning expected
            pympcc.solve(p, strategy="scholtes", autoscale=True)
        np.testing.assert_array_equal(p.comp_G_scale, manual)

    def test_autoscale_helps_imbalanced_solve(self):
        # Sanity: with autoscale, the imbalanced problem still converges
        # to the correct optimum.
        p = _imbalanced_problem(scale_G=1e-8, scale_H=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pympcc.solve(p, strategy="scholtes", autoscale=True)
        assert r.success
        # Optimum is x=(2,0) with f=1
        assert abs(r.obj - 1.0) < 1e-2
