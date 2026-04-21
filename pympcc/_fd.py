"""
Finite-difference Jacobian and gradient factories.

Two modes are supported:

* ``"forward"`` — first-order accurate, O(h) error, costs n+1 function
  evaluations per call.  Optimal step: h = sqrt(eps_machine) ≈ 1.49e-8.
* ``"central"`` — second-order accurate, O(h²) error, costs 2n function
  evaluations per call.  The same default step is used for simplicity;
  users who want optimal central-difference accuracy can pass
  h = eps_machine^(1/3) ≈ 6e-6 explicitly.

Both factories return plain Python closures.  They hold no mutable state
and are safe to call from multiple threads simultaneously.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Literal

_DEFAULT_H: float = float(np.sqrt(np.finfo(float).eps))  # ≈ 1.4901e-8
FDMode = Literal["forward", "central"]


def fd_gradient(
    f: Callable[[np.ndarray], float],
    n: int,
    h: float = _DEFAULT_H,
    mode: str = "forward",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable ``x → (n,)`` that approximates the gradient of ``f``
    using finite differences.

    Parameters
    ----------
    f : callable
        Scalar-valued function ``f(x) -> float``.
    n : int
        Length of the input vector ``x``.
    h : float
        Step size (default: ``sqrt(machine_epsilon)`` ≈ 1.49e-8).
    mode : {"forward", "central"}
        Difference scheme.

    Returns
    -------
    callable
        ``grad(x) -> ndarray, shape (n,)``
    """
    def wrapper(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.empty(n, dtype=float)
        ei = np.zeros(n, dtype=float)
        if mode == "forward":
            f0 = float(f(x))
            for i in range(n):
                ei[i] = h
                g[i] = (float(f(x + ei)) - f0) / h
                ei[i] = 0.0
        else:  # central
            for i in range(n):
                ei[i] = h
                g[i] = (float(f(x + ei)) - float(f(x - ei))) / (2.0 * h)
                ei[i] = 0.0
        return g

    return wrapper


def fd_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    n_out: int,
    n: int,
    h: float = _DEFAULT_H,
    mode: str = "forward",
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a callable ``x → (n_out, n)`` that approximates the Jacobian of
    ``fn`` using finite differences.

    Parameters
    ----------
    fn : callable
        Vector-valued function ``fn(x) -> ndarray, shape (n_out,)``.
    n_out : int
        Output dimension of ``fn``.
    n : int
        Input dimension (length of ``x``).
    h : float
        Step size (default: ``sqrt(machine_epsilon)`` ≈ 1.49e-8).
    mode : {"forward", "central"}
        Difference scheme.

    Returns
    -------
    callable
        ``jac(x) -> ndarray, shape (n_out, n)``
    """
    def wrapper(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        J = np.empty((n_out, n), dtype=float)
        ei = np.zeros(n, dtype=float)
        if mode == "forward":
            f0 = np.asarray(fn(x), dtype=float)
            for i in range(n):
                ei[i] = h
                J[:, i] = (np.asarray(fn(x + ei), dtype=float) - f0) / h
                ei[i] = 0.0
        else:  # central
            for i in range(n):
                ei[i] = h
                J[:, i] = (
                    np.asarray(fn(x + ei), dtype=float)
                    - np.asarray(fn(x - ei), dtype=float)
                ) / (2.0 * h)
                ei[i] = 0.0
        return J

    return wrapper
