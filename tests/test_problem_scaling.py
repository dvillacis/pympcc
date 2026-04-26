"""Tests for per-pair complementarity rescaling.

Covers MPCCProblem validation, identity-scale invariance, Jacobian-row
scaling correctness for both dense and sparse paths, and
unscale_multipliers post-processing.
"""

from __future__ import annotations

import numpy as np
import pytest

import pympcc


def _make_simple_dense(s_G=None, s_H=None):
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        comp_G_scale=s_G,
        comp_H_scale=s_H,
    )


def _make_simple_sparse(s_G=None, s_H=None):
    return pympcc.MPCCProblem(
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
        comp_G_scale=s_G,
        comp_H_scale=s_H,
    )


# ======================================================================= #
# Validation                                                                #
# ======================================================================= #

class TestScaleValidation:
    def test_default_none(self):
        p = _make_simple_dense()
        assert p.comp_G_scale is None
        assert p.comp_H_scale is None
        assert p.has_comp_scale is False

    def test_accepts_positive_array(self):
        p = _make_simple_dense(s_G=np.array([2.5]), s_H=np.array([0.1]))
        assert p.has_comp_scale is True
        np.testing.assert_array_equal(p.comp_G_scale, [2.5])
        np.testing.assert_array_equal(p.comp_H_scale, [0.1])

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            _make_simple_dense(s_G=np.array([1.0, 2.0]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="positive"):
            _make_simple_dense(s_G=np.array([-1.0]))

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _make_simple_dense(s_H=np.array([0.0]))

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError, match="non-finite"):
            _make_simple_dense(s_G=np.array([np.inf]))


# ======================================================================= #
# Identity-scale invariance                                                #
# ======================================================================= #

class TestScaleInvariance:
    """Setting s_G = s_H = 1 must reproduce the unscaled solve bit-for-bit."""

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_identity_scale_dense(self, strategy):
        p_plain = _make_simple_dense()
        p_scaled = _make_simple_dense(s_G=np.array([1.0]), s_H=np.array([1.0]))
        r1 = pympcc.solve(p_plain, strategy=strategy)
        r2 = pympcc.solve(p_scaled, strategy=strategy)
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-8)
        np.testing.assert_allclose(r1.obj, r2.obj, atol=1e-8)

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing"])
    def test_identity_scale_sparse(self, strategy):
        p_plain = _make_simple_sparse()
        p_scaled = _make_simple_sparse(s_G=np.array([1.0]), s_H=np.array([1.0]))
        r1 = pympcc.solve(p_plain, strategy=strategy)
        r2 = pympcc.solve(p_scaled, strategy=strategy)
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-8)


# ======================================================================= #
# Strategy-level Jacobian scaling                                          #
# ======================================================================= #

class TestJacobianScaling:
    """Verify that row-scaling is correctly applied at the BaseStrategy
    funnel point, for both dense and sparse Jacobian paths."""

    def test_dense_jacobian_rows_scaled(self):
        s_G = np.array([2.0])
        s_H = np.array([5.0])
        p = _make_simple_dense(s_G=s_G, s_H=s_H)
        from pympcc.strategies.scholtes import ScholtesStrategy
        strat = ScholtesStrategy(p, ipopt_options={})
        x = np.array([0.7, 0.3])
        vG, vH = strat._eval_comp_jac_raw(x)
        # Original dense JG = [[1, 0]]; scaled = [[2, 0]].
        np.testing.assert_allclose(vG, [[2.0, 0.0]])
        np.testing.assert_allclose(vH, [[0.0, 5.0]])

    def test_dense_values_scaled(self):
        s_G = np.array([3.0])
        s_H = np.array([0.25])
        p = _make_simple_dense(s_G=s_G, s_H=s_H)
        from pympcc.strategies.scholtes import ScholtesStrategy
        strat = ScholtesStrategy(p, ipopt_options={})
        x = np.array([0.4, 0.8])
        G, H = strat._eval_comp_values(x)
        np.testing.assert_allclose(G, [3.0 * 0.4])
        np.testing.assert_allclose(H, [0.25 * 0.8])

    def test_sparse_jacobian_values_scaled(self):
        s_G = np.array([4.0])
        s_H = np.array([0.5])
        p = _make_simple_sparse(s_G=s_G, s_H=s_H)
        from pympcc.strategies.scholtes import ScholtesStrategy
        strat = ScholtesStrategy(p, ipopt_options={})
        x = np.array([0.7, 0.3])
        vG, vH = strat._eval_comp_jac_raw(x)
        # Sparse 1-D values: [1.0] each, scaled by row -> [4.0] and [0.5].
        np.testing.assert_allclose(vG, [4.0])
        np.testing.assert_allclose(vH, [0.5])

    def test_multi_pair_per_row_scaling(self):
        """Three comp pairs with three different scales — verify each row
        is multiplied by the corresponding scale."""
        n_c = 3
        problem = pympcc.MPCCProblem(
            n=3, n_comp=n_c,
            x0=np.full(3, 0.5),
            objective=lambda x: float(np.sum(x ** 2)),
            gradient=lambda x: 2.0 * x,
            comp_G=lambda x: x.copy(),
            comp_G_jacobian=lambda x: np.eye(3),
            comp_H=lambda x: 1.0 - x,
            comp_H_jacobian=lambda x: -np.eye(3),
            comp_G_scale=np.array([1.0, 2.0, 4.0]),
            comp_H_scale=np.array([0.5, 0.25, 0.125]),
        )
        from pympcc.strategies.scholtes import ScholtesStrategy
        strat = ScholtesStrategy(problem, ipopt_options={})
        x = np.array([0.3, 0.6, 0.9])
        G, H = strat._eval_comp_values(x)
        np.testing.assert_allclose(G, [0.3, 1.2, 3.6])     # x_i * s_G_i
        np.testing.assert_allclose(H, [0.35, 0.1, 0.0125])  # (1-x_i) * s_H_i
        vG, vH = strat._eval_comp_jac_raw(x)
        # Each row of the dense Jacobian scales by its row factor.
        np.testing.assert_allclose(vG, np.diag([1.0, 2.0, 4.0]))
        np.testing.assert_allclose(vH, -np.diag([0.5, 0.25, 0.125]))


# ======================================================================= #
# Result-side scale propagation                                            #
# ======================================================================= #

class TestResultScales:
    def test_scales_propagate_to_result(self):
        s_G = np.array([2.0])
        s_H = np.array([3.0])
        p = _make_simple_dense(s_G=s_G, s_H=s_H)
        result = pympcc.solve(p, strategy="scholtes")
        np.testing.assert_array_equal(result.comp_G_scale, s_G)
        np.testing.assert_array_equal(result.comp_H_scale, s_H)

    def test_no_scale_yields_none_on_result(self):
        p = _make_simple_dense()
        result = pympcc.solve(p, strategy="scholtes")
        assert result.comp_G_scale is None
        assert result.comp_H_scale is None

    def test_unscale_multipliers_roundtrip(self):
        s_G = np.array([2.0])
        s_H = np.array([3.0])
        p = _make_simple_dense(s_G=s_G, s_H=s_H)
        result = pympcc.solve(p, strategy="scholtes")
        scaled_mu_G = np.array([0.7])
        scaled_mu_H = np.array([0.1])
        mu_G, mu_H = pympcc.unscale_multipliers(result, scaled_mu_G, scaled_mu_H)
        np.testing.assert_allclose(mu_G, s_G * scaled_mu_G)
        np.testing.assert_allclose(mu_H, s_H * scaled_mu_H)

    def test_unscale_with_no_scale_is_identity(self):
        p = _make_simple_dense()
        result = pympcc.solve(p, strategy="scholtes")
        mu_G_in = np.array([0.42])
        mu_H_in = np.array([0.13])
        mu_G, mu_H = pympcc.unscale_multipliers(result, mu_G_in, mu_H_in)
        np.testing.assert_array_equal(mu_G, mu_G_in)
        np.testing.assert_array_equal(mu_H, mu_H_in)
