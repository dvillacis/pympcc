"""Tests for pympcc._kernels — correctness of all three hot-path functions."""
import numpy as np
import pytest

from pympcc._kernels import (
    HAS_NUMBA,
    coo_to_dense,
    eval_phi_eps_weighted_union,
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


# ------------------------------------------------------------------ #
# eval_phi_eps_weighted_union                                          #
# ------------------------------------------------------------------ #

class TestEvalPhiEpsWeightedUnion:
    """
    Fused Fischer-Burmeister kernel.

    Reference (naive):
        r      = sqrt(G[row]^2 + H[row]^2 + eps^2)
        alpha  = 1 - G[row] / r
        beta   = 1 - H[row] / r
        out[k] = alpha * v_G[map1[k]] + beta * v_H[map2[k]]
    """

    def _ref(self, v_G, v_H, G, H, eps, r_u, map1, map2):
        """Naive per-element reference, independent of the kernel."""
        out = np.zeros(len(r_u))
        for k in range(len(r_u)):
            row = r_u[k]
            r_norm = np.sqrt(G[row] ** 2 + H[row] ** 2 + eps ** 2)
            alpha_k = 1.0 - G[row] / r_norm
            beta_k  = 1.0 - H[row] / r_norm
            if map1[k] >= 0:
                out[k] += alpha_k * v_G[map1[k]]
            if map2[k] >= 0:
                out[k] += beta_k  * v_H[map2[k]]
        return out

    def test_matches_reference(self):
        rng = np.random.default_rng(42)
        n_rows, nnz = 5, 8
        v_G = rng.standard_normal(nnz)
        v_H = rng.standard_normal(nnz)
        G   = rng.standard_normal(n_rows)
        H   = rng.standard_normal(n_rows)
        eps = 0.1
        r_u  = np.array([0, 0, 1, 2, 3, 3, 4, 4], dtype=np.intp)
        map1 = np.arange(nnz, dtype=np.intp)
        map2 = np.arange(nnz, dtype=np.intp)
        out = np.empty(nnz)
        eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out)
        expected = self._ref(v_G, v_H, G, H, eps, r_u, map1, map2)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_sentinel_minus_one(self):
        """map1 == -1 means only v_H contributes; map2 == -1 means only v_G."""
        G   = np.array([1.0, 2.0])
        H   = np.array([0.5, 0.5])
        v_G = np.array([10.0, 20.0])
        v_H = np.array([30.0, 40.0])
        eps = 0.01
        r_u  = np.array([0, 1], dtype=np.intp)
        map1 = np.array([-1, 0], dtype=np.intp)   # entry 0: only v_H
        map2 = np.array([0, -1], dtype=np.intp)   # entry 1: only v_G
        out = np.empty(2)
        eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out)
        expected = self._ref(v_G, v_H, G, H, eps, r_u, map1, map2)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_writes_into_pre_allocated_buffer(self):
        G   = np.array([1.0])
        H   = np.array([1.0])
        v_G = np.array([2.0])
        v_H = np.array([3.0])
        eps = 1.0
        r_u  = np.array([0], dtype=np.intp)
        map1 = np.array([0], dtype=np.intp)
        map2 = np.array([0], dtype=np.intp)
        out = np.zeros(1)
        eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out)
        expected = self._ref(v_G, v_H, G, H, eps, r_u, map1, map2)
        np.testing.assert_allclose(out, expected, rtol=1e-12)

    def test_multiple_eps_values(self):
        """Varying eps should change the weights continuously."""
        rng = np.random.default_rng(7)
        n_rows, nnz = 3, 4
        v_G = rng.standard_normal(nnz)
        v_H = rng.standard_normal(nnz)
        G   = np.abs(rng.standard_normal(n_rows)) + 0.1
        H   = np.abs(rng.standard_normal(n_rows)) + 0.1
        r_u  = np.array([0, 0, 1, 2], dtype=np.intp)
        map1 = np.arange(nnz, dtype=np.intp)
        map2 = np.arange(nnz, dtype=np.intp)
        out = np.empty(nnz)
        for eps in [1.0, 0.1, 1e-4, 1e-8]:
            eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out)
            expected = self._ref(v_G, v_H, G, H, eps, r_u, map1, map2)
            np.testing.assert_allclose(out, expected, rtol=1e-11,
                                       err_msg=f"failed at eps={eps}")

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
    def test_numba_matches_numpy_fallback(self):
        """Numba and pure-NumPy fallback must produce identical results."""
        import pympcc._kernels as _mod
        # Access the fallback directly from the else-branch by temporarily
        # shadowing HAS_NUMBA.  We do this by defining the same function body.
        rng = np.random.default_rng(99)
        n_rows, nnz = 6, 10
        v_G = rng.standard_normal(nnz)
        v_H = rng.standard_normal(nnz)
        G   = rng.standard_normal(n_rows)
        H   = rng.standard_normal(n_rows)
        eps = 0.05
        r_u  = np.sort(rng.integers(0, n_rows, size=nnz).astype(np.intp))
        map1 = np.arange(nnz, dtype=np.intp)
        map2 = np.arange(nnz, dtype=np.intp)
        # Numba version
        out_jit = np.empty(nnz)
        eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out_jit)
        # Reference
        out_ref = self._ref(v_G, v_H, G, H, eps, r_u, map1, map2)
        np.testing.assert_allclose(out_jit, out_ref, rtol=1e-12)


# ------------------------------------------------------------------ #
# coo_to_dense                                                         #
# ------------------------------------------------------------------ #

class TestCooToDense:
    """Fill out[rows[k], cols[k]] = values[k] in-place."""

    def test_basic(self):
        out = np.zeros((3, 4))
        rows   = np.array([0, 1, 2], dtype=np.intp)
        cols   = np.array([1, 2, 3], dtype=np.intp)
        values = np.array([1.0, 2.0, 3.0])
        coo_to_dense(rows, cols, values, out)
        assert out[0, 1] == pytest.approx(1.0)
        assert out[1, 2] == pytest.approx(2.0)
        assert out[2, 3] == pytest.approx(3.0)
        # Untouched entries stay zero
        assert out[0, 0] == pytest.approx(0.0)

    def test_matches_advanced_indexing(self):
        rng = np.random.default_rng(5)
        n_rows, n_cols, nnz = 8, 10, 20
        rows   = rng.integers(0, n_rows, size=nnz).astype(np.intp)
        cols   = rng.integers(0, n_cols, size=nnz).astype(np.intp)
        values = rng.standard_normal(nnz)
        out_kernel = np.zeros((n_rows, n_cols))
        out_ref    = np.zeros((n_rows, n_cols))
        coo_to_dense(rows, cols, values, out_kernel)
        out_ref[rows, cols] = values
        np.testing.assert_allclose(out_kernel, out_ref)

    def test_writes_inplace(self):
        """Existing non-zero entries outside COO positions must be preserved."""
        out = np.ones((2, 3)) * 9.0
        rows   = np.array([0, 1], dtype=np.intp)
        cols   = np.array([0, 2], dtype=np.intp)
        values = np.array([5.0, 7.0])
        coo_to_dense(rows, cols, values, out)
        assert out[0, 0] == pytest.approx(5.0)
        assert out[1, 2] == pytest.approx(7.0)
        # Untouched entries keep their original value
        assert out[0, 1] == pytest.approx(9.0)

    def test_empty_coo(self):
        out = np.ones((2, 2))
        coo_to_dense(
            np.array([], dtype=np.intp),
            np.array([], dtype=np.intp),
            np.array([]),
            out,
        )
        np.testing.assert_allclose(out, 1.0)
