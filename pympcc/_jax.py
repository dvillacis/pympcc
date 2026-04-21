"""
Optional JAX-autodiff backend for Jacobians and Hessians.

When ``jax`` is installed the functions below replace hand-coded derivative
callables with JIT-compiled exact derivatives.  Sparsity is **auto-detected**
once at construction time (by evaluating the dense Jacobian at three nearby
points and thresholding), then a sparse extractor is compiled.  All downstream
strategy code is unchanged — the produced callables look identical to the
manually-coded sparse Jacobians.

When JAX is not installed a safe ``ImportError`` is raised if any ``"jax"``
sentinel is actually used.

Exported
--------
HAS_JAX : bool
    True when JAX is importable.
jax_gradient(f, n, x0, tol) -> Callable
    JIT-compiled ``jax.grad(f)`` wrapped to accept/return NumPy arrays.
jax_jacobian(fn, n_out, n, x0, tol) -> (Callable, (rows, cols))
    Sparse-native Jacobian callable + auto-detected COO sparsity pattern.
jax_hessian_lagrangian(lagrangian_fn, n, x0, m, tol) -> (Callable, (rows, cols))
    Lower-triangular Lagrangian Hessian callable + COO sparsity pattern.
    ``lagrangian_fn(x, lam, obj_factor) -> scalar``
"""
from __future__ import annotations

from typing import Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:  # pragma: no cover
    HAS_JAX = False


# ------------------------------------------------------------------ #
# Gradient                                                             #
# ------------------------------------------------------------------ #

def jax_gradient(
    f: Callable,
    n: int,
    x0: np.ndarray,
    tol: float = 1e-12,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a JIT-compiled gradient callable using ``jax.grad``.

    Parameters
    ----------
    f : callable
        Scalar-valued JAX-differentiable function ``f(x) -> float``.
    n : int
        Number of variables (unused at runtime; kept for API symmetry with
        ``fd_gradient``).
    x0 : ndarray, shape (n,)
        Initial point (unused; kept for API symmetry).
    tol : float
        Unused for gradient (no sparsity to detect).

    Returns
    -------
    callable
        ``grad(x) -> ndarray, shape (n,)``
    """
    grad_f = jax.jit(jax.grad(f))

    def gradient(x: np.ndarray) -> np.ndarray:
        return np.asarray(grad_f(jnp.asarray(x, dtype=float)), dtype=float)

    return gradient


# ------------------------------------------------------------------ #
# Jacobian with auto-sparsity detection                               #
# ------------------------------------------------------------------ #

def jax_jacobian(
    fn: Callable,
    n_out: int,
    n: int,
    x0: np.ndarray,
    tol: float = 1e-12,
) -> tuple[Callable[[np.ndarray], np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Return a sparse-native Jacobian callable and its auto-detected COO pattern.

    Sparsity is detected by evaluating the full dense Jacobian at three nearby
    points (union), which guards against structural zeros that happen to vanish
    exactly at ``x0``.

    Parameters
    ----------
    fn : callable
        Vector-valued JAX-differentiable function ``fn(x) -> (n_out,)``.
    n_out : int
        Output dimension.
    n : int
        Input dimension.
    x0 : ndarray, shape (n,)
        Initial point used for sparsity detection.
    tol : float
        Entries with absolute value ``<= tol`` across all probe points are
        treated as structural zeros (default 1e-12).

    Returns
    -------
    jac_callable : callable
        ``jac(x) -> ndarray, shape (nnz,)`` — sparse-native, JIT-compiled.
    sparsity : (rows, cols)
        Auto-detected COO pattern, dtype ``np.intp``.
    """
    x0_j = jnp.asarray(x0, dtype=float)
    # Three probe points — union protects against accidental zeros at x0
    probe_pts = [x0_j, x0_j + 1e-3, x0_j * 1.1 + 1e-3]
    union = np.zeros((n_out, n), dtype=float)
    _jac_dense = jax.jit(jax.jacfwd(fn))
    for xk in probe_pts:
        union = np.maximum(union, np.abs(np.asarray(_jac_dense(xk), dtype=float)))
    rows, cols = np.where(union > tol)
    rows = rows.astype(np.intp)
    cols = cols.astype(np.intp)
    r_j = jnp.array(rows)
    c_j = jnp.array(cols)

    @jax.jit
    def _sparse_jac(x):
        J = jax.jacfwd(fn)(x)
        return J[r_j, c_j]

    def jac_callable(x: np.ndarray) -> np.ndarray:
        return np.asarray(_sparse_jac(jnp.asarray(x, dtype=float)), dtype=float)

    return jac_callable, (rows, cols)


# ------------------------------------------------------------------ #
# Dense Jacobian (no sparsity detection — used by StructuredMPCC)      #
# ------------------------------------------------------------------ #

def jax_jacobian_dense(
    fn: Callable,
    n_out: int,
    n: int,
    x0: np.ndarray,
    tol: float = 1e-12,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a JIT-compiled dense Jacobian callable using ``jax.jacfwd``.

    Unlike :func:`jax_jacobian`, this does **not** detect sparsity and
    returns a full ``(n_out, n)`` matrix.  Used for ``StructuredMPCC``
    nonlinear Jacobians where the combined linear+nonlinear callable
    is always dense.

    Parameters
    ----------
    fn : callable
        JAX-differentiable function ``fn(x) -> (n_out,)``.
    n_out, n : int
        Output and input dimensions (unused at runtime; kept for API symmetry).
    x0 : ndarray
        Unused; kept for API symmetry with :func:`jax_jacobian`.
    tol : float
        Unused; kept for API symmetry.

    Returns
    -------
    callable
        ``jac(x) -> ndarray, shape (n_out, n)``
    """
    jac_fn = jax.jit(jax.jacfwd(fn))

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.asarray(jac_fn(jnp.asarray(x, dtype=float)), dtype=float)

    return jacobian


# ------------------------------------------------------------------ #
# Lagrangian Hessian                                                   #
# ------------------------------------------------------------------ #

def jax_hessian_lagrangian(
    lagrangian_fn: Callable,
    n: int,
    x0: np.ndarray,
    m: int,
    tol: float = 1e-12,
) -> tuple[Callable, tuple[np.ndarray, np.ndarray]]:
    """
    Return a sparse lower-triangular Lagrangian Hessian callable and its pattern.

    The Lagrangian Hessian is symmetric; only the lower triangular part is
    returned (cyipopt convention).  Sparsity is detected once at construction
    from the dense Hessian evaluated at ``(x0, ones_m, 1.0)``.

    Parameters
    ----------
    lagrangian_fn : callable
        ``lagrangian(x, lam, obj_factor) -> float`` — JAX-differentiable.
    n : int
        Number of (primal) variables.
    x0 : ndarray, shape (n,)
        Initial point for sparsity detection.
    m : int
        Number of constraints (length of ``lam``).
    tol : float
        Sparsity threshold (default 1e-12).

    Returns
    -------
    hess_fn : callable
        ``hess(x, lam, obj_factor) -> ndarray, shape (nnz_hess,)``
    hess_sparsity : (rows, cols)
        Lower-triangular COO pattern, dtype ``np.intp``.
    """
    x0_j   = jnp.asarray(x0, dtype=float)
    lam0_j = jnp.ones(m, dtype=float)

    # Detect sparsity from dense Hessian at (x0, ones, 1.0)
    def _lag_x(x):
        return lagrangian_fn(x, lam0_j, 1.0)

    H_dense = np.asarray(jax.hessian(_lag_x)(x0_j), dtype=float)
    mask    = np.tril(np.abs(H_dense)) > tol
    rows, cols = np.where(mask)
    rows = rows.astype(np.intp)
    cols = cols.astype(np.intp)
    r_j  = jnp.array(rows)
    c_j  = jnp.array(cols)

    @jax.jit
    def _hess_sparse(x, lam, obj_factor):
        def _lag(xk):
            return lagrangian_fn(xk, lam, obj_factor)
        H = jax.hessian(_lag)(x)
        return H[r_j, c_j]

    def hess_fn(
        x: np.ndarray,
        lam: np.ndarray,
        obj_factor: float,
    ) -> np.ndarray:
        return np.asarray(
            _hess_sparse(
                jnp.asarray(x, dtype=float),
                jnp.asarray(lam, dtype=float),
                float(obj_factor),
            ),
            dtype=float,
        )

    return hess_fn, (rows, cols)
