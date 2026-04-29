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


# ---------------------------------------------------------------------------
# Sparse 4-tuple form
# ---------------------------------------------------------------------------

def _simple_problem_var_pairs_sparse():
    """Same simple problem declared via the sparse 4-tuple form."""
    n, n_comp = 2, 1
    x0 = np.array([0.5, 0.5])
    return MPCCProblem(
        n=n, n_comp=n_comp, x0=x0,
        objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
        gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
        comp_var_pairs=[
            (0, lambda x: np.array([x[1]]), lambda x: np.array([1.0]), [1]),
        ],
    )


class TestSparseVarPairs:
    """4-tuple form (var_idx, h_fn, h_jac_fn, h_cols) emits sparse Jacobians."""

    def test_sparsity_patterns_emitted(self):
        p = _simple_problem_var_pairs_sparse()
        assert p.comp_G_jacobian_sparsity is not None
        assert p.comp_H_jacobian_sparsity is not None
        gr, gc = p.comp_G_jacobian_sparsity
        hr, hc = p.comp_H_jacobian_sparsity
        np.testing.assert_array_equal(gr, [0])
        np.testing.assert_array_equal(gc, [0])
        np.testing.assert_array_equal(hr, [0])
        np.testing.assert_array_equal(hc, [1])

    def test_sparse_jacobians_return_values_only(self):
        p = _simple_problem_var_pairs_sparse()
        x = np.array([0.3, 0.7])
        vG = p.comp_G_jacobian(x)
        vH = p.comp_H_jacobian(x)
        assert vG.ndim == 1 and vG.size == 1
        assert vH.ndim == 1 and vH.size == 1
        np.testing.assert_allclose(vG, [1.0])
        np.testing.assert_allclose(vH, [1.0])

    def test_is_sparse_property(self):
        p = _simple_problem_var_pairs_sparse()
        assert p.is_sparse

    def test_multiple_sparse_pairs_concat(self):
        n, n_comp = 4, 2
        x0 = np.full(n, 0.5)
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: float(np.sum((x - 1.0)**2)),
            gradient=lambda x: 2.0 * (x - 1.0),
            comp_var_pairs=[
                # H_0(x) = x[1] + 2*x[2]   (cols 1, 2)
                (0, lambda x: np.array([x[1] + 2*x[2]]),
                    lambda x: np.array([1.0, 2.0]), [1, 2]),
                # H_1(x) = 3*x[3]          (col 3)
                (1, lambda x: np.array([3*x[3]]),
                    lambda x: np.array([3.0]), [3]),
            ],
        )
        gr, gc = p.comp_G_jacobian_sparsity
        hr, hc = p.comp_H_jacobian_sparsity
        np.testing.assert_array_equal(gr, [0, 1])
        np.testing.assert_array_equal(gc, [0, 1])
        np.testing.assert_array_equal(hr, [0, 0, 1])
        np.testing.assert_array_equal(hc, [1, 2, 3])

        x = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(p.comp_G_jacobian(x), [1.0, 1.0])
        np.testing.assert_allclose(p.comp_H_jacobian(x), [1.0, 2.0, 3.0])

    def test_user_supplied_sparsity_not_overwritten(self):
        """When the user pre-sets *_jacobian_sparsity, the auto-emit is a no-op."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        custom_G_sp = (np.array([0]), np.array([0]))
        custom_H_sp = (np.array([0]), np.array([1]))
        p = MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_G_jacobian_sparsity=custom_G_sp,
            comp_H_jacobian_sparsity=custom_H_sp,
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([1.0]), [1]),
            ],
        )
        # Identity check: the auto-emit code does not overwrite when already set.
        assert p.comp_G_jacobian_sparsity is custom_G_sp
        assert p.comp_H_jacobian_sparsity is custom_H_sp

    def test_value_count_mismatch_raises(self):
        """Mismatch between h_jac_fn return size and len(h_cols) is caught at validate."""
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="h_cols"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
                gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
                comp_var_pairs=[
                    # h_jac_fn returns 2 values but h_cols has length 1
                    (0, lambda x: np.array([x[1]]),
                        lambda x: np.array([1.0, 0.0]), [1]),
                ],
            )

    def test_mixing_sparse_and_dense_raises(self):
        n, n_comp = 3, 2
        x0 = np.full(n, 0.5)
        with pytest.raises(ValueError, match="all-sparse"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(np.sum(x**2)),
                gradient=lambda x: 2.0 * x,
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]),
                        lambda x: np.array([1.0]), [1]),  # sparse
                    (1, lambda x: np.array([x[2]]),
                        lambda x: np.array([0.0, 0.0, 1.0])),  # dense
                ],
            )

    def test_h_cols_out_of_range_raises(self):
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="h_cols"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]),
                        lambda x: np.array([1.0]), [5]),
                ],
            )

    def test_4tuple_requires_h_jac_fn(self):
        n, n_comp = 2, 1
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="h_jac_fn"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]), None, [1]),
                ],
            )

    @pytest.mark.parametrize("strategy", ["scholtes", "smoothing", "lin_fukushima"])
    def test_sparse_solve_matches_manual(self, strategy):
        """Sparse var-pair form solves to the same optimum as the manual sparse form."""
        manual = MPCCProblem(
            n=2, n_comp=1, x0=np.array([0.5, 0.5]),
            objective=lambda x: (x[0] - 1.0)**2 + (x[1] - 1.0)**2,
            gradient=lambda x: np.array([2*(x[0]-1), 2*(x[1]-1)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_H_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
            comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
        )
        sparse = _simple_problem_var_pairs_sparse()
        rm = pympcc.solve(manual, strategy=strategy)
        rs = pympcc.solve(sparse, strategy=strategy)
        assert rm.success and rs.success
        np.testing.assert_allclose(rs.obj, rm.obj, atol=1e-4)
        np.testing.assert_allclose(rs.comp_residual, rm.comp_residual, atol=1e-4)


class TestMixedModeSparse:
    """Mixed mode sparse: base comp_G/comp_H sparse + 4-tuple var-pair tail."""

    def _mixed_sparse_problem(self):
        """2 base pairs + 1 sparse var-pair: n_comp=2, n=4.

        base G0=x[2], H0=x[3];  var-pair: x[0] ⊥ x[1]
        Both base and tail are sparse.
        """
        n, n_comp = 4, 2
        x0 = np.full(n, 0.5)
        return MPCCProblem(
            n=n, n_comp=n_comp, x0=x0,
            objective=lambda x: float(np.sum((x - 1.0)**2)),
            gradient=lambda x: 2.0 * (x - 1.0),
            comp_G=lambda x: np.array([x[2]]),
            comp_H=lambda x: np.array([x[3]]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_H_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=(np.array([0]), np.array([2])),
            comp_H_jacobian_sparsity=(np.array([0]), np.array([3])),
            comp_var_pairs=[
                (0, lambda x: np.array([x[1]]), lambda x: np.array([1.0]), [1]),
            ],
        )

    def test_mixed_sparse_patterns_concatenated(self):
        p = self._mixed_sparse_problem()
        gr, gc = p.comp_G_jacobian_sparsity
        hr, hc = p.comp_H_jacobian_sparsity
        np.testing.assert_array_equal(gr, [0, 1])
        np.testing.assert_array_equal(gc, [2, 0])
        np.testing.assert_array_equal(hr, [0, 1])
        np.testing.assert_array_equal(hc, [3, 1])

    def test_mixed_sparse_jacobian_values(self):
        p = self._mixed_sparse_problem()
        x = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(p.comp_G_jacobian(x), [1.0, 1.0])
        np.testing.assert_allclose(p.comp_H_jacobian(x), [1.0, 1.0])

    def test_mixed_sparse_requires_base_sparsity(self):
        """4-tuple in mixed mode without base sparsity must error."""
        n, n_comp = 4, 2
        x0 = np.full(n, 0.5)
        with pytest.raises(ValueError, match="comp_G_jacobian_sparsity"):
            MPCCProblem(
                n=n, n_comp=n_comp, x0=x0,
                objective=lambda x: float(np.sum((x - 1.0)**2)),
                gradient=lambda x: 2.0 * (x - 1.0),
                comp_G=lambda x: np.array([x[2]]),
                comp_H=lambda x: np.array([x[3]]),
                comp_G_jacobian=lambda x: np.array([[0., 0., 1., 0.]]),
                comp_H_jacobian=lambda x: np.array([[0., 0., 0., 1.]]),
                # no base sparsity → cannot append sparse tail
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]),
                        lambda x: np.array([1.0]), [1]),
                ],
            )


# ---------------------------------------------------------------------------
# Bulk form (single vectorized callable for all k var-pairs)
# ---------------------------------------------------------------------------

def _bulk_problem(k=3):
    """min sum (x_i - 1)^2  s.t. x[0..k-1] >= 0  ⊥  x[k..2k-1] >= 0.

    Built via comp_var_pairs_bulk: one vectorized h_fn returning all k H values.
    """
    n = 2 * k
    x0 = np.full(n, 0.5)
    var_idxs = np.arange(k, dtype=np.intp)

    def h_bulk(x):
        return np.asarray(x, dtype=float)[k:2 * k]

    # H Jacobian is identity rows at columns k..2k-1: nnz=k, 1 per row.
    H_rows = np.arange(k, dtype=np.intp)
    H_cols = np.arange(k, 2 * k, dtype=np.intp)
    H_vals_const = np.ones(k)

    def h_bulk_jac(x, _v=H_vals_const):
        return _v

    return MPCCProblem(
        n=n, n_comp=k, x0=x0,
        objective=lambda x: float(np.sum((np.asarray(x) - 1.0) ** 2)),
        gradient=lambda x: 2.0 * (np.asarray(x) - 1.0),
        comp_var_pairs_bulk=(var_idxs, h_bulk, h_bulk_jac, (H_rows, H_cols)),
    )


class TestBulkForm:
    def test_construction_emits_sparsity(self):
        p = _bulk_problem(k=3)
        assert p.comp_G_jacobian_sparsity is not None
        assert p.comp_H_jacobian_sparsity is not None
        gr, gc = p.comp_G_jacobian_sparsity
        np.testing.assert_array_equal(gr, [0, 1, 2])
        np.testing.assert_array_equal(gc, [0, 1, 2])

    def test_g_and_h_values(self):
        p = _bulk_problem(k=3)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_allclose(p.comp_G(x), [0.1, 0.2, 0.3])
        np.testing.assert_allclose(p.comp_H(x), [0.4, 0.5, 0.6])

    def test_g_jacobian_is_constant_ones(self):
        p = _bulk_problem(k=4)
        x = np.linspace(0.1, 0.8, 8)
        vG = p.comp_G_jacobian(x)
        assert vG.ndim == 1 and vG.size == 4
        np.testing.assert_allclose(vG, [1.0, 1.0, 1.0, 1.0])

    def test_lower_bounds_clamped(self):
        n, k = 4, 2
        x0 = np.full(n, 0.5)
        p = MPCCProblem(
            n=n, n_comp=k, x0=x0,
            xl=np.array([-1.0, -1.0, -1.0, -1.0]),
            objective=lambda x: float(np.sum(x ** 2)),
            gradient=lambda x: 2.0 * np.asarray(x),
            comp_var_pairs_bulk=(
                np.array([0, 1], dtype=np.intp),
                lambda x: np.asarray(x)[2:],
                lambda x: np.ones(2),
                (np.array([0, 1], dtype=np.intp), np.array([2, 3], dtype=np.intp)),
            ),
        )
        np.testing.assert_array_equal(p.xl[:2], [0.0, 0.0])
        np.testing.assert_array_equal(p.xl[2:], [-1.0, -1.0])

    def test_solve_matches_per_row_form(self):
        """Bulk-form MCP solves to the same optimum as the equivalent 4-tuple form."""
        bulk_result = pympcc.solve(_bulk_problem(k=3), strategy="scholtes")
        # Equivalent per-row 4-tuple form
        n, k = 6, 3
        x0 = np.full(n, 0.5)
        per_row = MPCCProblem(
            n=n, n_comp=k, x0=x0,
            objective=lambda x: float(np.sum((np.asarray(x) - 1.0) ** 2)),
            gradient=lambda x: 2.0 * (np.asarray(x) - 1.0),
            comp_var_pairs=[
                (i, (lambda x, _i=i: np.array([x[k + _i]])),
                    (lambda x: np.array([1.0])), [k + i])
                for i in range(k)
            ],
        )
        per_row_result = pympcc.solve(per_row, strategy="scholtes")
        assert bulk_result.success
        assert per_row_result.success
        np.testing.assert_allclose(bulk_result.obj, per_row_result.obj, atol=1e-4)

    def test_mutual_exclusion_with_var_pairs(self):
        n = 2
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="mutually exclusive"):
            MPCCProblem(
                n=n, n_comp=1, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]),
                        lambda x: np.array([1.0]), [1]),
                ],
                comp_var_pairs_bulk=(
                    np.array([0], dtype=np.intp),
                    lambda x: np.array([x[1]]),
                    lambda x: np.array([1.0]),
                    (np.array([0], dtype=np.intp), np.array([1], dtype=np.intp)),
                ),
            )

    def test_var_idxs_out_of_range(self):
        n = 2
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="var_idxs"):
            MPCCProblem(
                n=n, n_comp=1, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs_bulk=(
                    np.array([5], dtype=np.intp),  # out of range
                    lambda x: np.array([x[1]]),
                    lambda x: np.array([1.0]),
                    (np.array([0], dtype=np.intp), np.array([1], dtype=np.intp)),
                ),
            )

    def test_n_comp_mismatch(self):
        n = 4
        x0 = np.full(n, 0.5)
        with pytest.raises(ValueError, match="n_comp"):
            MPCCProblem(
                n=n, n_comp=2, x0=x0,  # claims 2 pairs
                objective=lambda x: float(np.sum(x ** 2)),
                gradient=lambda x: 2.0 * np.asarray(x),
                comp_var_pairs_bulk=(
                    np.array([0], dtype=np.intp),  # but only 1 var_idx
                    lambda x: np.array([x[1]]),
                    lambda x: np.array([1.0]),
                    (np.array([0], dtype=np.intp), np.array([1], dtype=np.intp)),
                ),
            )

    def test_bad_tuple_shape(self):
        n = 2
        x0 = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="4-tuple"):
            MPCCProblem(
                n=n, n_comp=1, x0=x0,
                objective=lambda x: float(x[0]),
                gradient=lambda x: np.array([1.0, 0.0]),
                comp_var_pairs_bulk=(np.array([0], dtype=np.intp),),  # too short
            )


# ---------------------------------------------------------------------------
# Large-MCP fd guard
# ---------------------------------------------------------------------------

class TestFdGuard:
    def test_fd_rejected_for_large_mcp(self):
        """2-tuple form with k * (n+1) > 1e7 must raise with a clear message."""
        n = 1000
        k = 50
        x0 = np.full(n, 0.5)
        # n_fd_rows * (n+1) = 50 * 1001 = 50_050 → not large enough; need bigger
        # Choose k=100, n=200_000 → 100 * 200_001 ≈ 2e7 > 1e7
        n2 = 200_000
        k2 = 100
        x0_2 = np.full(n2, 0.5)
        pairs = [(i, (lambda x, _i=i: np.array([x[_i + 1]]))) for i in range(k2)]
        with pytest.raises(ValueError, match="finite-difference"):
            MPCCProblem(
                n=n2, n_comp=k2, x0=x0_2,
                objective=lambda x: float(np.sum(np.asarray(x) ** 2)),
                gradient=lambda x: 2.0 * np.asarray(x),
                comp_var_pairs=pairs,
            )

    def test_fd_allowed_for_small_mcp(self):
        """Small MCP using fd fallback still works."""
        n = 10
        k = 1
        x0 = np.full(n, 0.5)
        p = MPCCProblem(
            n=n, n_comp=k, x0=x0,
            objective=lambda x: float(np.sum(np.asarray(x) ** 2)),
            gradient=lambda x: 2.0 * np.asarray(x),
            comp_var_pairs=[(0, lambda x: np.array([x[1]]))],  # 2-tuple → fd
        )
        assert p.comp_G_jacobian is not None  # construction succeeded
