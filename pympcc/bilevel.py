"""
Bilevel KKT-emitter frontend.

A bilevel program

    min_{x, y}  F(x, y)
    s.t.        y ∈ argmin_y { f(x, y) : g(x, y) ≤ 0,  h(x, y) = 0 }

becomes an MPCC by replacing the lower-level argmin with its KKT system::

    ∇_y f(x, y) + Σ λ_i ∇_y g_i(x, y) + Σ μ_k ∇_y h_k(x, y) = 0     (stationarity)
    h(x, y)  =  0                                                     (lower-eq)
    λ ≥ 0,    -g(x, y) ≥ 0,    λ ⊥ -g(x, y)                         (complementarity)

The :func:`from_lower_level` emitter does this rewrite automatically and
returns a :class:`~pympcc.problem.MPCCProblem` ready for
:func:`pympcc.solve`.

Variable layout of the emitted MPCC::

    z = [ x_upper (n_x) | y_lower (n_y) | λ (n_g_lower) | μ (n_h_lower) ]

KKT validity assumes lower-level convexity (or at least a stationary
optimum); for non-convex lower-level problems the resulting MPCC
characterises stationary points of the lower problem rather than its
global argmin.

Composes naturally with §4.5 (variable-paired complementarity) — λ enters
as a non-negative slice of ``z`` and the complementarity row ``H_i`` is
``−g_i(x, y)``.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .problem import MPCCProblem

__all__ = ["from_lower_level"]


def from_lower_level(
    *,
    n_x: int,
    n_y: int,
    x0: np.ndarray,
    y0: np.ndarray,
    f_upper: Callable[[np.ndarray, np.ndarray], float],
    f_lower: Callable[[np.ndarray, np.ndarray], float],
    n_g_lower: int = 0,
    g_lower: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    n_h_lower: int = 0,
    h_lower: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    derivatives: str = "jax",
    xl: Optional[np.ndarray] = None,
    xu: Optional[np.ndarray] = None,
    yl: Optional[np.ndarray] = None,
    yu: Optional[np.ndarray] = None,
    lambda0: Optional[np.ndarray] = None,
    mu0: Optional[np.ndarray] = None,
) -> MPCCProblem:
    """Emit an :class:`~pympcc.problem.MPCCProblem` from a bilevel program.

    Parameters
    ----------
    n_x, n_y : int
        Sizes of the upper-level and lower-level decision-variable blocks.
    x0, y0 : array-like
        Initial guesses for ``x`` and ``y``.
    f_upper : callable
        Upper-level objective ``F(x, y) -> float``.
    f_lower : callable
        Lower-level objective ``f(x, y) -> float``.
    n_g_lower : int
        Number of lower-level inequality constraints ``g(x, y) ≤ 0``.
        Must be ``≥ 1`` (a bilevel without lower-level inequalities is not
        an MPCC; solve it as a regular NLP).
    g_lower : callable
        ``g(x, y) -> ndarray, shape (n_g_lower,)``.  Required when
        ``n_g_lower > 0``.
    n_h_lower : int
        Number of lower-level equality constraints ``h(x, y) = 0`` (default 0).
    h_lower : callable, optional
        ``h(x, y) -> ndarray, shape (n_h_lower,)``.  Required when
        ``n_h_lower > 0``.
    derivatives : {"jax", "fd"}, default "jax"
        Backend used both to form the stationarity rows and to fill every
        Jacobian on the emitted MPCC.  ``"jax"`` requires the user-supplied
        callables to be ``jax.numpy``-traceable and is dramatically more
        accurate (single-level autodiff) than the ``"fd"`` fallback (nested
        finite differences).
    xl, xu, yl, yu : array-like, optional
        Bounds on the upper- and lower-level variable blocks (default ``±inf``).
        The λ block is automatically lower-bounded at ``0``; μ stays free.
    lambda0, mu0 : array-like, optional
        Initial multipliers for ``λ`` and ``μ`` (default zeros).

    Returns
    -------
    MPCCProblem
        With ``n = n_x + n_y + n_g_lower + n_h_lower``,
        ``n_comp = n_g_lower``, ``n_eq = n_y + n_h_lower``.

    Notes
    -----
    No lower-level inequality slacks are introduced; the complementarity
    pair lives directly between ``λ`` and ``-g(x, y)``.  The stationarity
    rows depend on ``y, λ, μ`` through the lower-level Lagrangian's
    second-order partials, so the MPCC's own Jacobian computation involves
    second derivatives of ``f_lower``, ``g_lower``, and ``h_lower``.  With
    ``derivatives="jax"`` JAX takes those derivatives exactly; with
    ``"fd"`` they are computed by nested finite differences (acceptable for
    prototyping, noisy on tight tolerances).
    """
    if derivatives not in ("jax", "fd"):
        raise ValueError(
            f"derivatives must be 'jax' or 'fd', got {derivatives!r}"
        )
    if n_g_lower < 1:
        raise ValueError(
            "n_g_lower must be >= 1; a bilevel program without lower-level "
            "inequality constraints is not an MPCC — solve it as a regular NLP."
        )
    if g_lower is None:
        raise ValueError("g_lower is required when n_g_lower > 0")
    if n_h_lower < 0:
        raise ValueError("n_h_lower must be >= 0")
    if (n_h_lower > 0) != (h_lower is not None):
        raise ValueError(
            "h_lower and n_h_lower must agree: pass both or neither"
        )

    n_lam = int(n_g_lower)
    n_mu = int(n_h_lower)
    n = n_x + n_y + n_lam + n_mu

    x0_arr = np.asarray(x0, dtype=float).ravel()
    y0_arr = np.asarray(y0, dtype=float).ravel()
    if x0_arr.shape != (n_x,):
        raise ValueError(f"x0 must have shape ({n_x},), got {x0_arr.shape}")
    if y0_arr.shape != (n_y,):
        raise ValueError(f"y0 must have shape ({n_y},), got {y0_arr.shape}")

    if lambda0 is None:
        lam0 = np.zeros(n_lam)
    else:
        lam0 = np.asarray(lambda0, dtype=float).ravel()
        if lam0.shape != (n_lam,):
            raise ValueError(
                f"lambda0 must have shape ({n_lam},), got {lam0.shape}"
            )
        if np.any(lam0 < 0.0):
            raise ValueError("lambda0 must be non-negative (λ ≥ 0)")

    if mu0 is None:
        mu0_arr = np.zeros(n_mu)
    else:
        mu0_arr = np.asarray(mu0, dtype=float).ravel()
        if mu0_arr.shape != (n_mu,):
            raise ValueError(
                f"mu0 must have shape ({n_mu},), got {mu0_arr.shape}"
            )

    z0 = np.concatenate([x0_arr, y0_arr, lam0, mu0_arr])

    def _resolve_bound(b: Optional[np.ndarray], default: float, m: int,
                       name: str) -> np.ndarray:
        if b is None:
            return np.full(m, default)
        arr = np.asarray(b, dtype=float).ravel()
        if arr.shape != (m,):
            raise ValueError(f"{name} must have shape ({m},), got {arr.shape}")
        return arr

    xl_full = np.concatenate([
        _resolve_bound(xl, -np.inf, n_x, "xl"),
        _resolve_bound(yl, -np.inf, n_y, "yl"),
        np.zeros(n_lam),
        np.full(n_mu, -np.inf),
    ])
    xu_full = np.concatenate([
        _resolve_bound(xu, np.inf, n_x, "xu"),
        _resolve_bound(yu, np.inf, n_y, "yu"),
        np.full(n_lam, np.inf),
        np.full(n_mu, np.inf),
    ])

    sx0, sx1 = 0, n_x
    sy0, sy1 = n_x, n_x + n_y
    sl0, sl1 = n_x + n_y, n_x + n_y + n_lam
    sm0, sm1 = n_x + n_y + n_lam, n

    def _split(z):
        return z[sx0:sx1], z[sy0:sy1], z[sl0:sl1], z[sm0:sm1]

    def objective(z: np.ndarray) -> float:
        x, y, _, _ = _split(z)
        return f_upper(x, y)

    def comp_G(z: np.ndarray) -> np.ndarray:
        return z[sl0:sl1]

    def comp_H(z: np.ndarray) -> np.ndarray:
        x, y, _, _ = _split(z)
        return -g_lower(x, y)

    if derivatives == "jax":
        from ._jax import HAS_JAX

        if not HAS_JAX:
            raise ImportError(
                "derivatives='jax' requires JAX. Install with "
                "`pip install pympcc[jax]` or pass derivatives='fd'."
            )
        import jax
        import jax.numpy as jnp

        def _lagrangian_y(x, y, lam, mu):
            L = f_lower(x, y)
            if n_lam > 0:
                L = L + jnp.dot(lam, g_lower(x, y))
            if n_mu > 0:
                L = L + jnp.dot(mu, h_lower(x, y))
            return L

        _grad_y_L = jax.grad(_lagrangian_y, argnums=1)

        def stationarity(z):
            x, y, lam, mu = _split(z)
            return _grad_y_L(x, y, lam, mu)

        def eq_constraints(z):
            x, y, _lam, _mu = _split(z)
            stat = stationarity(z)
            if n_mu > 0:
                return jnp.concatenate([stat, h_lower(x, y)])
            return stat

    else:
        from ._fd import _DEFAULT_H, fd_gradient

        def stationarity(z):
            x, y, lam, mu = _split(z)
            x_a = np.asarray(x, dtype=float)
            lam_a = np.asarray(lam, dtype=float)
            mu_a = np.asarray(mu, dtype=float)

            def _scalar_L_of_y(yy):
                v = float(f_lower(x_a, yy))
                if n_lam > 0:
                    v += float(np.dot(lam_a, np.asarray(g_lower(x_a, yy))))
                if n_mu > 0:
                    v += float(np.dot(mu_a, np.asarray(h_lower(x_a, yy))))
                return v

            grad_fn = fd_gradient(_scalar_L_of_y, n_y, h=_DEFAULT_H,
                                  mode="forward")
            return grad_fn(np.asarray(y, dtype=float))

        def eq_constraints(z):
            x, y, _lam, _mu = _split(z)
            stat = np.asarray(stationarity(z), dtype=float)
            if n_mu > 0:
                hh = np.asarray(h_lower(x, y), dtype=float)
                return np.concatenate([stat, hh])
            return stat

    n_eq_total = n_y + n_mu

    return MPCCProblem(
        n=n,
        n_comp=n_lam,
        x0=z0,
        xl=xl_full,
        xu=xu_full,
        objective=objective,
        comp_G=comp_G,
        comp_H=comp_H,
        n_eq=n_eq_total,
        eq_constraints=eq_constraints,
        derivatives=derivatives,
    )
