"""Tests for pympcc._kernels — correctness of all three hot-path functions."""
import numpy as np
import pytest

from pympcc._kernels import (
    HAS_NUMBA,
    eval_weighted_union,
    scatter_add,
    weighted_row_sum,
)


def test_has_numba_is_bool():
    assert isinstance(HAS_NUMBA, bool)


# ------------------------------------------------------------------ #
# eval_weighted_union                                                  #
# ------------------------------------------------------------------ #

class TestEvalWeightedUnion:
    """
    Reference implementation:
        out[k] = alpha[r_u[k]] * v_G[map1[k]]   (when map1[k] >= 0)
               + beta[r_u[k]]  * v_H[map2[k]]   (when map2[k] >= 0)
    """

    def _ref(self, v_G, v_H, alpha, beta, r_u, map1, map2):
        out = np.zeros(len(r_u))
        for k in range(len(r_u)):
            row = r_u[k]
            if map1[k] >= 0:
                out[k] += alpha[row] * v_G[map1[k]]
            if map2[k] >= 0:
                out[k] += beta[row] * v_H[map2[k]]
        return out

    def test_both_maps_present(self):
        """All union entries come from both v_G and v_H."""
        rng = np.random.default_rng(0)
        n_rows, nnz = 4, 6
        v_G = rng.standard_normal(nnz)
        v_H = rng.standard_normal(nnz)
        alpha = rng.standard_normal(n_rows)
        beta  = rng.standard_normal(n_rows)
        r_u   = np.array([0, 0, 1, 2, 3, 3], dtype=np.intp)
        map1  = np.array([0, 1, 2, 3, 4, 5], dtype=np.intp)
        map2  = np.array([0, 1, 2, 3, 4, 5], dtype=np.intp)
        out = np.empty(nnz)
        eval_weighted_union(v_G, v_H, alpha, beta, r_u, map1, map2, out)
        expected = self._ref(v_G, v_H, alpha, beta, r_u, map1, map2)
        np.testing.assert_allclose(out, expected)

    def test_sentinel_minus_one(self):
        """map1 == -1 means only v_H contributes; map2 == -1 means only v_G."""
        v_G = np.array([1.0, 2.0])
        v_H = np.array([3.0, 4.0])
        alpha = np.array([1.0, 1.0])
        beta  = np.array([1.0, 1.0])
        r_u   = np.array([0, 1], dtype=np.intp)
        map1  = np.array([-1, 0], dtype=np.intp)   # entry 0: only v_H
        map2  = np.array([0, -1], dtype=np.intp)   # entry 1: only v_G
        out = np.empty(2)
        eval_weighted_union(v_G, v_H, alpha, beta, r_u, map1, map2, out)
        assert out[0] == pytest.approx(beta[0] * v_H[0])
        assert out[1] == pytest.approx(alpha[1] * v_G[0])

    def test_writes_into_pre_allocated_buffer(self):
        """Result must land in the supplied output array (in-place, returns None)."""
        v_G = np.ones(3)
        v_H = np.ones(3)
        alpha = np.ones(2)
        beta  = np.ones(2)
        r_u   = np.array([0, 0, 1], dtype=np.intp)
        map1  = np.array([0, 1, 2], dtype=np.intp)
        map2  = np.array([0, 1, 2], dtype=np.intp)
        out = np.zeros(3)
        eval_weighted_union(v_G, v_H, alpha, beta, r_u, map1, map2, out)
        np.testing.assert_allclose(out, 2.0)

    def test_zero_weights(self):
        v_G = np.array([5.0])
        v_H = np.array([7.0])
        alpha = np.array([0.0])
        beta  = np.array([0.0])
        r_u   = np.array([0], dtype=np.intp)
        map1  = np.array([0], dtype=np.intp)
        map2  = np.array([0], dtype=np.intp)
        out = np.empty(1)
        eval_weighted_union(v_G, v_H, alpha, beta, r_u, map1, map2, out)
        assert out[0] == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# weighted_row_sum                                                     #
# ------------------------------------------------------------------ #

class TestWeightedRowSum:
    """out[i,j] = alpha[i]*A[i,j] + beta[i]*B[i,j]"""

    def test_matches_numpy(self):
        rng = np.random.default_rng(1)
        n_rows, n_cols = 5, 8
        alpha = rng.standard_normal(n_rows)
        beta  = rng.standard_normal(n_rows)
        A = rng.standard_normal((n_rows, n_cols))
        B = rng.standard_normal((n_rows, n_cols))
        out = np.empty((n_rows, n_cols))
        weighted_row_sum(alpha, A, beta, B, out)
        expected = alpha[:, None] * A + beta[:, None] * B
        np.testing.assert_allclose(out, expected)

    def test_writes_inplace(self):
        alpha = np.array([2.0, 3.0])
        beta  = np.array([1.0, 0.0])
        A = np.ones((2, 3))
        B = np.ones((2, 3))
        out = np.zeros((2, 3))
        weighted_row_sum(alpha, A, beta, B, out)
        np.testing.assert_allclose(out[0], 3.0)
        np.testing.assert_allclose(out[1], 3.0)

    def test_single_row(self):
        alpha = np.array([2.0])
        beta  = np.array([3.0])
        A = np.array([[1.0, 2.0]])
        B = np.array([[4.0, 5.0]])
        out = np.empty((1, 2))
        weighted_row_sum(alpha, A, beta, B, out)
        np.testing.assert_allclose(out, [[2 + 12, 4 + 15]])


# ------------------------------------------------------------------ #
# scatter_add                                                          #
# ------------------------------------------------------------------ #

class TestScatterAdd:
    """Equivalent to np.add.at(out, indices, values)."""

    def test_basic(self):
        out = np.zeros(5)
        indices = np.array([0, 2, 2, 4], dtype=np.intp)
        values  = np.array([1.0, 3.0, 2.0, 5.0])
        scatter_add(out, indices, values)
        np.testing.assert_allclose(out, [1, 0, 5, 0, 5])

    def test_matches_add_at(self):
        rng = np.random.default_rng(2)
        out_ref = np.zeros(10)
        out_jit = np.zeros(10)
        indices = rng.integers(0, 10, size=20).astype(np.intp)
        values  = rng.standard_normal(20)
        np.add.at(out_ref, indices, values)
        scatter_add(out_jit, indices, values)
        np.testing.assert_allclose(out_jit, out_ref)

    def test_empty_inputs(self):
        out = np.zeros(3)
        scatter_add(out, np.array([], dtype=np.intp), np.array([]))
        np.testing.assert_allclose(out, 0.0)

    def test_accumulates_into_existing_values(self):
        out = np.array([10.0, 20.0, 30.0])
        scatter_add(out, np.array([0, 2], dtype=np.intp), np.array([1.0, 2.0]))
        np.testing.assert_allclose(out, [11.0, 20.0, 32.0])
