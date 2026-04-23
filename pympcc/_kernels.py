"""
Optional Numba-JIT kernels for hot-path operations.

When ``numba`` is installed the functions are compiled on first use and
cached to disk (subsequent imports are instant).  When it is not installed
the module exposes identical pure-NumPy fallbacks — there is no API
difference for callers.

Exported functions
------------------
eval_weighted_union(v_G, v_H, alpha, beta, r_u, map1, map2, out) -> None
    Fill *out[k] = alpha[r_u[k]]*v_G[map1[k]] + beta[r_u[k]]*v_H[map2[k]]*.
    Entries where map1[k] or map2[k] == -1 contribute zero.
    Writes into *out* in-place (shape: nnz_union,).

eval_phi_eps_weighted_union(v_G, v_H, G, H, eps, r_u, map1, map2, out) -> None
    Fused Fischer-Burmeister kernel for the smoothing strategy Jacobian.
    Computes the phi_eps weights inline — no temporary (n_comp,) arrays:
        r      = sqrt(G[row]^2 + H[row]^2 + eps^2)
        alpha  = 1 - G[row] / r
        beta   = 1 - H[row] / r
        out[k] = alpha * v_G[map1[k]] + beta * v_H[map2[k]]

weighted_row_sum(alpha, A, beta, B, out) -> None
    Fill *out[i,j] = alpha[i]*A[i,j] + beta[i]*B[i,j]* in-place.
    Used for ∂(G·H)/∂x = H·JG + G·JH in dense Jacobian paths.

scatter_add(out, indices, values) -> None
    Equivalent to ``np.add.at(out, indices, values)`` but faster when
    Numba is available (avoids Python-level loop overhead in np.add.at).

coo_to_dense(rows, cols, values, out) -> None
    Fill *out[rows[k], cols[k]] = values[k]* in-place.
    Faster than NumPy advanced indexing assignment for large sparse arrays.

HAS_NUMBA : bool
    True when Numba is available in the current environment.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit as _njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:   # pragma: no cover
    @_njit(cache=True)
    def eval_weighted_union(
        v_G: np.ndarray,
        v_H: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        r_u: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Fill *out* in-place with the weighted union evaluation.

        For each union entry k::

            out[k] = alpha[r_u[k]] * v_G[map1[k]]   (if map1[k] >= 0)
                   + beta[r_u[k]]  * v_H[map2[k]]   (if map2[k] >= 0)

        Called on every sparse Jacobian callback.  No heap allocation.
        """
        for k in range(out.shape[0]):
            row = r_u[k]
            v = 0.0
            if map1[k] >= 0:
                v += alpha[row] * v_G[map1[k]]
            if map2[k] >= 0:
                v += beta[row] * v_H[map2[k]]
            out[k] = v

    @_njit(cache=True)
    def weighted_row_sum(
        alpha: np.ndarray,
        A: np.ndarray,
        beta: np.ndarray,
        B: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Fill *out[i,j] = alpha[i]*A[i,j] + beta[i]*B[i,j]* in-place.

        Equivalent to ``out[:] = alpha[:,None]*A + beta[:,None]*B`` but
        without intermediate (n_comp, n) temporaries.  Called on every
        dense Jacobian callback for the G·H block.
        """
        n_rows = A.shape[0]
        n_cols = A.shape[1]
        for i in range(n_rows):
            ai = alpha[i]
            bi = beta[i]
            for j in range(n_cols):
                out[i, j] = ai * A[i, j] + bi * B[i, j]

    @_njit(cache=True)
    def scatter_add(
        out: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """
        Equivalent to ``np.add.at(out, indices, values)`` but JIT-compiled.

        Used to scatter sparse gradient contributions into the full
        gradient vector.
        """
        for i in range(len(indices)):
            out[indices[i]] += values[i]

    @_njit(cache=True)
    def eval_phi_eps_weighted_union(
        v_G: np.ndarray,
        v_H: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
        r_u: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Fused Fischer-Burmeister kernel for the smoothing Jacobian.

        Computes phi_eps weights inline — no (n_comp,) temporaries::

            r      = sqrt(G[row]^2 + H[row]^2 + eps^2)
            alpha  = 1 - G[row] / r
            beta   = 1 - H[row] / r
            out[k] = alpha * v_G[map1[k]] + beta * v_H[map2[k]]

        Called on every sparse Jacobian callback.  No heap allocation.
        """
        for k in range(out.shape[0]):
            row = r_u[k]
            r_norm = (G[row] * G[row] + H[row] * H[row] + eps * eps) ** 0.5
            alpha_k = 1.0 - G[row] / r_norm
            beta_k  = 1.0 - H[row] / r_norm
            v = 0.0
            if map1[k] >= 0:
                v += alpha_k * v_G[map1[k]]
            if map2[k] >= 0:
                v += beta_k  * v_H[map2[k]]
            out[k] = v

    @_njit(cache=True)
    def coo_to_dense(
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """
        Fill ``out[rows[k], cols[k]] = values[k]`` in-place.

        Faster than NumPy advanced indexing for large sparse arrays due
        to avoided Python-level dispatch overhead.
        """
        for k in range(len(values)):
            out[rows[k], cols[k]] = values[k]

else:
    def eval_weighted_union(    # type: ignore[misc]
        v_G: np.ndarray,
        v_H: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        r_u: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        out: np.ndarray,
    ) -> None:
        out[:] = 0.0
        m1 = map1 >= 0
        if m1.any():
            out[m1] += alpha[r_u[m1]] * v_G[map1[m1]]
        m2 = map2 >= 0
        if m2.any():
            out[m2] += beta[r_u[m2]] * v_H[map2[m2]]

    def weighted_row_sum(       # type: ignore[misc]
        alpha: np.ndarray,
        A: np.ndarray,
        beta: np.ndarray,
        B: np.ndarray,
        out: np.ndarray,
    ) -> None:
        # One temporary (n_comp, n) instead of three
        np.multiply(A, alpha[:, None], out=out)
        out += B * beta[:, None]

    def scatter_add(            # type: ignore[misc]
        out: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> None:
        np.add.at(out, indices, values)

    def eval_phi_eps_weighted_union(    # type: ignore[misc]
        v_G: np.ndarray,
        v_H: np.ndarray,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
        r_u: np.ndarray,
        map1: np.ndarray,
        map2: np.ndarray,
        out: np.ndarray,
    ) -> None:
        g_r = G[r_u]
        h_r = H[r_u]
        r = np.sqrt(g_r * g_r + h_r * h_r + eps * eps)
        alpha = 1.0 - g_r / r
        beta  = 1.0 - h_r / r
        out[:] = 0.0
        m1 = map1 >= 0
        if m1.any():
            out[m1] += alpha[m1] * v_G[map1[m1]]
        m2 = map2 >= 0
        if m2.any():
            out[m2] += beta[m2] * v_H[map2[m2]]

    def coo_to_dense(                   # type: ignore[misc]
        rows: np.ndarray,
        cols: np.ndarray,
        values: np.ndarray,
        out: np.ndarray,
    ) -> None:
        out[rows, cols] = values
