"""Tests for §4.5 variable-paired complementarity (MCP form).

comp_var_pairs lets users declare x[j] >= 0 ⊥ h(x) >= 0 at the problem
level without writing comp_G / comp_H manually.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc import MPCCProblem


# ---------------------------------------------------------------------------
# Helpers — simple MPCC problems used across tests
# ---------------------------------------------------------------------------

def _simple_problem_manual():
    """min (x-1)^2 + (y-1)^2   s.t. x>=0 ⊥ y>=0  (standard form, x0=(0.5,0.5))."""
    n, n_comp = 2, 1
    x0 = np.array([0.5, 0.5])
    return MPCCProblem(
        n=n, n_comp=n_comp, x0=x0,
        objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


def _simple_problem_var_pairs():
    """Same problem declared via comp_var_pairs (all-var-pairs mode)."""
    n, n_comp = 2, 1
    x0 = np.array([0.5, 0.5])
    return MPCCProblem(
        n=n, n_comp=n_comp, x0=x0,
        objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
        comp_var_pairs=[
            (0, lambda x: np.array([x[1]]), lambda x: np.array([0.0, 1.0])),
        ],
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_all_var_pairs_builds_comp_callables(self):
        """comp_G and comp_H are synthesized from comp_var_pairs."""
        p = _simple_problem_var_pairs()
        x0 = np.array([0.5, 0.5])
        assert p.comp_G is not None
        assert p.comp_H is not None
        np.testing.assert_allclose(p.comp_G(x0), [0.5])
        np.testing.assert_allclose(p.comp_H(x0), [0.5])

    def test_g_jacobian_is_exact_identity_row(self):
        """G Jacobian for var-pair rows is exact (identity at var_idx)."""
        p = _simple_problem_var_pairs()
        x0 = np.array([0.3, 0.7])
        J = p.comp_G_jacobian(x0)
        assert J.shape == (1, 2)
        np.testing.assert_array_equal(J, [[1.0, 0.0]])

    def test_h_jacobian_user_provided(self):
        """User-supplied h_jac_fn is used verbatim for the H Jacobian row."""
        p = _simple_problem_var_pairs()
        x0 = np.array([0.3, 0.7])
        J = p.comp_H_jacobian(x0)
        assert J.shape == (1, 2)
        np.testing.assert_array_equal(J, [[0.0, 1.0]])

    def test_h_jacobian_fd_fallback(self):
        """When h_jac_fn is None, fd is used for the H Jacobian row."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]])),  # no h_jac_fn → fd
            ],
        )
        J = p.comp_H_jacobian(x0)
        assert J.shape == (1, 2)
        np.testing.assert_allclose(J[0, 1], 1.0, atol=1e-6)
        np.testing.assert_allclose(J[0, 0], 0.0, atol=1e-6)

    def test_lower_bound_clamped(self):
        """xl[var_idx] is forced to max(xl[var_idx], 0.0)."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            xl=np.array([-5.0, -5.0]),
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([0.0, 1.0])),
            ],
        )
        assert p.xl[0] == 0.0     # clamped
        assert p.xl[1] == -5.0    # untouched

    def test_already_nonneg_bound_unchanged(self):
        """xl[var_idx] is not changed when it is already >= 0."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            xl=np.array([2.0, 0.0]),
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([0.0, 1.0])),
            ],
        )
        assert p.xl[0] == 2.0     # unchanged (already >= 0)

    def test_multiple_var_pairs(self):
        """Multiple var-pair entries produce correct stacked G/H and Jacobians."""
        n, n_comp = 3, 2
        x0 = np.array([0.5, 0.5, 0.5])
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: float(np.sum((x - 1.0)**2)),
            gradient=lambda x: 2.0 * (x - 1.0),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([0.0, 1.0, 0.0])),
                (1, lambda x: np.array([x[2]]), lambda x: np.array([0.0, 0.0, 1.0])),
            ],
        )
        x = x0
        G = p.comp_G(x)
        H = p.comp_H(x)
        assert G.shape == (2,)
        assert H.shape == (2,)
        np.testing.assert_allclose(G, [x[0], x[1]])
        np.testing.assert_allclose(H, [x[1], x[2]])

        JG = p.comp_G_jacobian(x)
        assert JG.shape == (2, 3)
        np.testing.assert_array_equal(JG, [[1, 0, 0], [0, 1, 0]])

        JH = p.comp_H_jacobian(x)
        assert JH.shape == (2, 3)
        np.testing.assert_array_equal(JH, [[0, 1, 0], [0, 0, 1]])


# ---------------------------------------------------------------------------
# Mixed mode tests
# ---------------------------------------------------------------------------

class TestMixedMode:
    def _mixed_problem(self):
        """2 base pairs + 1 var-pair: n_comp=3, n=4.

        base G0=x[2], H0=x[3];  var-pair: x[0] ⊥ x[1]
        """
        n, n_comp = 4, 3
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        return MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: float(np.sum((x - 1.0)**2)),
            gradient=lambda x: 2.0 * (x - 1.0),
            comp_G=lambda x: np.array([x[2]]),
            comp_H=lambda x: np.array([x[3]]),
            comp_G_jacobian=lambda x: np.array([[0., 0., 1., 0.]]),
            comp_H_jacobian=lambda x: np.array([[0., 0., 0., 1.]]),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([0., 1., 0., 0.])),
                (1, lambda x: np.array([x[3]]), lambda x: np.array([0., 0., 0., 1.])),
            ],
        )

    def test_mixed_comp_G_shape(self):
        p = self._mixed_problem()
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        assert p.comp_G(x0).shape == (3,)

    def test_mixed_comp_H_shape(self):
        p = self._mixed_problem()
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        assert p.comp_H(x0).shape == (3,)

    def test_mixed_g_jacobian_identity_rows_appended(self):
        p = self._mixed_problem()
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        JG = p.comp_G_jacobian(x0)
        assert JG.shape == (3, 4)
        # base row
        np.testing.assert_array_equal(JG[0], [0., 0., 1., 0.])
        # var-pair identity rows
        np.testing.assert_array_equal(JG[1], [1., 0., 0., 0.])
        np.testing.assert_array_equal(JG[2], [0., 1., 0., 0.])

    def test_mixed_h_jacobian(self):
        p = self._mixed_problem()
        x0 = np.array([0.5, 0.5, 0.5, 0.5])
        JH = p.comp_H_jacobian(x0)
        assert JH.shape == (3, 4)
        np.testing.assert_array_equal(JH[0], [0., 0., 0., 1.])  # base row
        np.testing.assert_array_equal(JH[1], [0., 1., 0., 0.])  # var-pair row 0
        np.testing.assert_array_equal(JH[2], [0., 0., 0., 1.])  # var-pair row 1

    def test_mixed_requires_base_jacobians(self):
        """Mixed mode errors when base comp_G_jacobian is missing."""
        n, n_comp = 3, 2
        x0 = np.ones(n) * 0.5
        with pytest.raises(ValueError, match="comp_G_jacobian"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(np.sum(x**2)),
                gradient=lambda x: 2.0 * x,
                comp_G=lambda x: np.array([x[2]]),
                comp_H=lambda x: np.array([x[1]]),
                # no comp_G_jacobian / comp_H_jacobian → None (no derivatives default)
                comp_var_pairs=[(0, lambda x: np.array([x[1]]))],
            )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_invalid_var_idx_too_large(self):
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="out of range"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[(5, lambda x: np.array([x[1]]))],
            )

    def test_invalid_var_idx_negative(self):
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="out of range"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[(-1, lambda x: np.array([x[1]]))],
            )

    def test_n_comp_mismatch_all_var_pairs(self):
        """n_comp=2 but only 1 var-pair entry in all-var-pairs mode."""
        n, n_comp = 3, 2
        x0 = np.ones(n) * 0.5
        with pytest.raises(ValueError, match="n_comp"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(np.sum(x**2)),
                gradient=lambda x: 2.0 * x,
                comp_var_pairs=[(0, lambda x: np.array([x[1]]))],
            )

    def test_bad_tuple_length(self):
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises((ValueError, TypeError)):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[(0,)],  # tuple too short
            )

    def test_no_comp_G_and_no_var_pairs_raises(self):
        """Providing neither comp_G nor comp_var_pairs must raise."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="comp_G"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
            )


# ---------------------------------------------------------------------------
# Solve equivalence: var-pairs vs manual comp_G/comp_H
# ---------------------------------------------------------------------------

class TestSolveEquivalence:
    """The var-pairs formulation must produce the same optimal solution as the manual form."""

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_solve_matches_manual(self, strategy):
        """min (x-1)^2 + (y-1)^2 s.t. x>=0, y>=0, x*y=0 → opt x=1,y=0 or x=0,y=1."""
        manual_result = pympcc.solve(_simple_problem_manual(), strategy=strategy)
        var_result = pympcc.solve(_simple_problem_var_pairs(), strategy=strategy)

        assert manual_result.success
        assert var_result.success
        np.testing.assert_allclose(var_result.obj, manual_result.obj, atol=1e-4)
        np.testing.assert_allclose(var_result.comp_residual,
                                   manual_result.comp_residual, atol=1e-4)

    def test_derivatives_fd_mode(self):
        """var-pairs with derivatives='fd' (no explicit gradient or h_jac_fn) should solve."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p = MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
                derivatives="fd",
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]])),  # no h_jac_fn → fd
                ],
            )
        result = pympcc.solve(p, strategy="scholtes")
        assert result.success
        assert result.comp_residual < 1e-4
