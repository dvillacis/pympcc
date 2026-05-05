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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ._typing import Derivatives
from .problem import MPCCProblem

__all__ = ["from_lower_level", "Leader", "LowerLevel", "from_epec"]


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
    derivatives: Derivatives = "jax",
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


# ============================================================================ #
# EPEC multi-leader-common-follower emitter (§5.5)                              #
# ============================================================================ #

@dataclass
class Leader:
    """One leader in a multi-leader-common-follower EPEC.

    Each leader chooses its own block ``x_i ∈ ℝ^{n_x}`` to minimise its
    upper-level objective ``F_i(x, y)``, where ``x`` is the concatenated
    leader-decision vector across *all* leaders (``x = (x_1, …, x_N)``) and
    ``y`` is the shared lower-level decision.  Each ``F_i`` typically reads
    its own block ``x[i_self_slice]`` and the rivals' blocks ``x_{-i}``.

    Parameters
    ----------
    n_x : int
        Size of this leader's decision block.
    F : callable
        Upper-level objective ``F_i(x_full, y) -> scalar`` where ``x_full``
        is the concatenated leader decision vector.  Must be JAX-traceable
        when ``derivatives='jax'``.
    x0 : array-like, optional
        Initial guess for this leader's block (default zeros).
    """

    n_x: int
    F: Callable[[np.ndarray, np.ndarray], float]
    x0: Optional[np.ndarray] = None


@dataclass
class LowerLevel:
    """Common lower-level optimisation shared across all leaders.

    Defines ``y ∈ argmin_y { f(x, y) : g(x, y) ≤ 0,  h(x, y) = 0 }``,
    where ``x`` is the concatenated leader-decision vector.

    Parameters
    ----------
    n_y : int
        Size of the lower-level decision block ``y``.
    f : callable
        Lower-level objective ``f(x_full, y) -> scalar``.  Must be
        JAX-traceable when ``derivatives='jax'``.
    n_g : int
        Number of inequality constraints ``g(x, y) ≤ 0``.  Must be ``≥ 1``;
        EPECs without a lower-level inequality block are not MPCCs.
    g : callable
        ``g(x_full, y) -> ndarray, shape (n_g,)``.
    y0 : array-like, optional
        Initial guess for ``y`` (default zeros).
    n_h : int, default 0
        Number of equality constraints ``h(x, y) = 0``.
    h : callable, optional
        ``h(x_full, y) -> ndarray, shape (n_h,)``.  Required iff ``n_h > 0``.
    """

    n_y: int
    f: Callable[[np.ndarray, np.ndarray], float]
    n_g: int
    g: Callable[[np.ndarray, np.ndarray], np.ndarray]
    y0: Optional[np.ndarray] = None
    n_h: int = 0
    h: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None


def from_epec(
    *,
    leaders: list[Leader],
    common_lower: LowerLevel,
    derivatives: Derivatives = "jax",
) -> MPCCProblem:
    r"""Emit an :class:`MPCCProblem` from a multi-leader-common-follower EPEC.

    Each leader ``i = 1, …, N`` solves an MPCC

    .. math::

        \min_{x_i}\ F_i(x, y)
        \quad\text{s.t.}\quad
        y \in \mathrm{argmin}_y\{f(x,y): g(x,y)\le 0,\ h(x,y)=0\}

    with ``x = (x_1, …, x_N)``.  A Nash equilibrium of the EPEC is a fixed
    point of best-response: each ``x_i^*`` is optimal for leader ``i``'s
    MPCC given ``x_{-i}^*``.  This emitter stacks each leader's KKT
    conditions (with the lower-level KKT system as constraints) into a
    single MPCC; under MPCC-MFCQ at the equilibrium, the stacked KKT
    system is a necessary characterisation of the Nash point.

    Variable layout
    ---------------
    ``z = [x_1, …, x_N | y | λ_lo | μ_lo | (ξ_i^y, ξ_i^h, θ_i, ν_i)_{i=1}^N]``

    where ``λ_lo``, ``μ_lo`` are the shared lower-level multipliers for
    ``g, h``, and ``ξ_i^y``, ``ξ_i^h``, ``θ_i``, ``ν_i`` are leader ``i``'s
    multipliers on the lower-level KKT rows that act as constraints in
    leader ``i``'s MPCC.

    Equality rows
    -------------
    For each leader ``i``: stationarity of the leader Lagrangian w.r.t.
    ``x_i`` (``n_{x,i}`` rows), ``y`` (``n_y``), ``λ_lo`` (``n_g``), and
    ``μ_lo`` (``n_h``).  Plus shared lower-level KKT eq: ``∇_y L_lo = 0``
    (``n_y``) and ``h_lo(x, y) = 0`` (``n_h``).

    Complementarity rows  (``n_comp = (1 + 2N)·n_g``)
    --------------------
    * shared:  ``λ_lo ⊥ −g_lo(x, y)``  (``n_g`` pairs)
    * per leader ``i``:
      ``θ_i ⊥ −g_lo(x, y)`` and ``ν_i ⊥ λ_lo``  (``2·n_g`` pairs each)

    Parameters
    ----------
    leaders : list of :class:`Leader`
        ``len(leaders) >= 2``.
    common_lower : :class:`LowerLevel`
    derivatives : {"jax"}, default "jax"
        Only ``"jax"`` is supported in v1; the leader Lagrangian involves
        the gradient of the lower-level Lagrangian w.r.t. ``y``, so the
        emitted equality block needs second-order autodiff.  Pass
        ``derivatives="fd"`` to receive an explicit ``NotImplementedError``.

    Returns
    -------
    MPCCProblem
        Objective is the constant zero — EPEC equilibrium is a feasibility
        problem; complementarity + stationarity rows pin the equilibrium
        without help from the upper objective.

    Notes
    -----
    * No private leader constraints in v1.  Every leader sees the same
      lower-level KKT system as its only constraint set; their upper
      objectives ``F_i`` may, of course, differ.
    * The ``ν_i`` block is a primal variable of the emitted MPCC even
      though leader ``i``'s KKT determines it from ``ξ_i^y`` via
      ``ν_i = (∇_y g_lo)·ξ_i^y``.  Carrying ``ν_i`` explicitly keeps the
      Jacobian sparse and the complementarity row ``ν_i ⊥ λ_lo`` linear.
    * Initial multipliers are zero; this is feasible w.r.t. the
      complementarity rows since ``θ_i = ν_i = 0`` and ``λ_lo = 0`` start
      satisfies all comp pairs.
    """
    if derivatives == "fd":
        raise NotImplementedError(
            "from_epec v1 supports derivatives='jax' only.  The emitted "
            "equality rows differentiate the lower-level Lagrangian "
            "gradient w.r.t. (x, y, λ, μ), so a second-order autodiff "
            "backend is required.  FD fallback is planned for v2."
        )
    if derivatives != "jax":
        raise ValueError(
            f"derivatives must be 'jax' (got {derivatives!r})."
        )

    if not isinstance(leaders, (list, tuple)) or len(leaders) < 2:
        raise ValueError(
            "from_epec requires at least 2 leaders; for a single-leader "
            "bilevel program use from_lower_level instead."
        )

    cl = common_lower
    if cl.n_g < 1:
        raise ValueError(
            "common_lower.n_g must be >= 1; an EPEC without lower-level "
            "inequality constraints is not an MPCC."
        )
    if cl.g is None:
        raise ValueError("common_lower.g is required when n_g > 0")
    if cl.n_h < 0:
        raise ValueError("common_lower.n_h must be >= 0")
    if (cl.n_h > 0) != (cl.h is not None):
        raise ValueError(
            "common_lower.h and common_lower.n_h must agree: pass both or neither"
        )

    from ._jax import HAS_JAX
    if not HAS_JAX:
        raise ImportError(
            "from_epec requires JAX. Install with `pip install pympcc[jax]`."
        )
    import jax
    import jax.numpy as jnp

    N = len(leaders)
    n_x_per = [int(L.n_x) for L in leaders]
    if any(s < 1 for s in n_x_per):
        raise ValueError("each Leader must have n_x >= 1")
    n_x_total = int(sum(n_x_per))
    n_y = int(cl.n_y)
    n_g = int(cl.n_g)
    n_h = int(cl.n_h)

    # z layout offsets
    x_offsets = np.cumsum([0] + n_x_per).tolist()
    off_y = n_x_total
    off_lam = off_y + n_y
    off_mu = off_lam + n_g
    leader_dual_size = n_y + n_h + 2 * n_g
    leader_off = [off_mu + n_h + i * leader_dual_size for i in range(N + 1)]
    n_total = leader_off[-1]

    def _split(z):
        x_full = z[:n_x_total]
        y = z[off_y:off_lam]
        lam_lo = z[off_lam:off_mu]
        mu_lo = z[off_mu:off_mu + n_h]
        leader_duals = []
        for i in range(N):
            base = leader_off[i]
            xi_y  = z[base                     : base + n_y]
            xi_h  = z[base + n_y               : base + n_y + n_h]
            theta = z[base + n_y + n_h         : base + n_y + n_h + n_g]
            nu    = z[base + n_y + n_h + n_g   : base + n_y + n_h + 2 * n_g]
            leader_duals.append((xi_y, xi_h, theta, nu))
        return x_full, y, lam_lo, mu_lo, leader_duals

    f_lo = cl.f
    g_lo = cl.g
    h_lo = cl.h

    # Lower-level Lagrangian gradient w.r.t. y — used as a stationarity row
    # in both the shared lower-level KKT block and inside each leader's
    # Lagrangian (where it acts as an = 0 constraint with multiplier ξ_i^y).
    def _stat_lo(x_full, y, lam_lo, mu_lo):
        def _lag(yy):
            v = f_lo(x_full, yy)
            if n_g > 0:
                v = v + jnp.dot(lam_lo, g_lo(x_full, yy))
            if n_h > 0:
                v = v + jnp.dot(mu_lo, h_lo(x_full, yy))
            return v
        return jax.grad(_lag)(y)

    def _leader_lagrangian(i_int: int, z):
        x_full, y, lam_lo, mu_lo, leader_duals = _split(z)
        xi_y, xi_h, theta, nu = leader_duals[i_int]
        L = leaders[i_int].F(x_full, y)
        L = L + jnp.dot(xi_y, _stat_lo(x_full, y, lam_lo, mu_lo))
        if n_h > 0:
            L = L + jnp.dot(xi_h, h_lo(x_full, y))  # type: ignore[misc]
        L = L + jnp.dot(theta, g_lo(x_full, y)) - jnp.dot(nu, lam_lo)
        return L

    _leader_grads = [
        jax.grad(lambda z, _i=i: _leader_lagrangian(_i, z))
        for i in range(N)
    ]

    def eq_constraints(z):
        x_full, y, lam_lo, mu_lo, _ = _split(z)
        rows: list = []
        for i in range(N):
            grad_z = _leader_grads[i](z)
            xi, xi1 = x_offsets[i], x_offsets[i + 1]
            rows.append(grad_z[xi:xi1])                    # ∂L_i/∂x_i
            rows.append(grad_z[off_y:off_lam])             # ∂L_i/∂y
            rows.append(grad_z[off_lam:off_mu])            # ∂L_i/∂λ_lo
            if n_h > 0:
                rows.append(grad_z[off_mu:off_mu + n_h])   # ∂L_i/∂μ_lo
        rows.append(_stat_lo(x_full, y, lam_lo, mu_lo))    # ∇_y L_lo = 0
        if n_h > 0:
            rows.append(h_lo(x_full, y))                   # h_lo(x, y) = 0
        return jnp.concatenate(rows)

    def comp_G(z):
        _, _, lam_lo, _, leader_duals = _split(z)
        parts = [lam_lo]
        for _xi_y, _xi_h, theta, nu in leader_duals:
            parts.append(theta)
            parts.append(nu)
        return jnp.concatenate(parts)

    def comp_H(z):
        x_full, y, lam_lo, _, _ = _split(z)
        neg_g = -g_lo(x_full, y)
        parts = [neg_g]
        for _ in range(N):
            parts.append(neg_g)
            parts.append(lam_lo)
        return jnp.concatenate(parts)

    def objective(_z):
        return 0.0

    # x0
    x0_parts = []
    for L in leaders:
        if L.x0 is None:
            x0_parts.append(np.zeros(L.n_x))
        else:
            arr = np.asarray(L.x0, dtype=float).ravel()
            if arr.shape != (L.n_x,):
                raise ValueError(
                    f"Leader.x0 must have shape ({L.n_x},), got {arr.shape}"
                )
            x0_parts.append(arr)
    if cl.y0 is None:
        y0 = np.zeros(n_y)
    else:
        y0 = np.asarray(cl.y0, dtype=float).ravel()
        if y0.shape != (n_y,):
            raise ValueError(f"common_lower.y0 must have shape ({n_y},), got {y0.shape}")

    z0 = np.concatenate([
        *x0_parts, y0,
        np.zeros(n_g), np.zeros(n_h),
        *[np.zeros(leader_dual_size) for _ in range(N)],
    ])

    # Bounds: λ_lo ≥ 0, θ_i ≥ 0, ν_i ≥ 0; everything else free.
    xl_full = np.full(n_total, -np.inf)
    xu_full = np.full(n_total, np.inf)
    xl_full[off_lam:off_mu] = 0.0
    for i in range(N):
        base = leader_off[i]
        theta_off = base + n_y + n_h
        nu_off    = theta_off + n_g
        xl_full[theta_off:theta_off + n_g] = 0.0
        xl_full[nu_off   :nu_off   + n_g] = 0.0

    n_eq_total  = n_x_total + N * (n_y + n_g + n_h) + n_y + n_h
    n_comp_total = (1 + 2 * N) * n_g

    return MPCCProblem(
        n=n_total,
        n_comp=n_comp_total,
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
