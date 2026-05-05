"""JAX-differentiable MPCC solve via ``jax.custom_vjp`` (§5.6).

Differentiates through a converged MPCC solution by solving the adjoint
KKT system once (sIPOPT-style implicit differentiation).  The forward
pass runs :func:`pympcc.solve` with ``tnlp_refine=True`` to obtain a
certified active set; the backward pass solves
``K · [u; w] = [v; 0]`` and contracts ``u``, ``w`` with
``∂(∇_xL)/∂θ`` and ``∂c/∂θ`` (computed via :func:`jax.vjp`) to produce
the ``θ`` cotangent.

Usage
-----
>>> import jax, jax.numpy as jnp
>>> from pympcc import ParametricMPCC, solve_jax
>>> pmpcc = ParametricMPCC(
...     n=2, n_comp=1,
...     objective=lambda x, theta: 0.5 * jnp.sum((x - theta) ** 2),
...     comp_G=lambda x, theta: x[:1],
...     comp_H=lambda x, theta: x[1:],
... )
>>> def loss(theta):
...     x_star = solve_jax(pmpcc, theta)
...     return jnp.sum(x_star)
>>> grad = jax.grad(loss)(jnp.array([1.0, 2.0]))

Caveats
-------
* **Not jittable.** :func:`pympcc.solve` is opaque NumPy + IPOPT code;
  invoke ``jax.grad(loss)(theta)`` outside any :func:`jax.jit` wrapper.
* **No θ-dependent bounds.** ``xl`` / ``xu`` are constants of the solve;
  no derivative flows through them.  Use soft constraints for parametric
  bounds.
* **Forward-mode deferred.** Only ``custom_vjp`` is registered;
  ``jax.jvp`` / ``jax.jacfwd`` are not yet supported.

Adjoint identity
----------------
At a TNLP-refined solution the implicit-function theorem gives::

    K [dx_dθ;  dλ_dθ] = -[∂(∇_xL)/∂θ;  ∂c/∂θ]

For a cotangent ``v`` on ``x*`` the VJP becomes (``K`` is symmetric)::

    K [u; w] = [v; 0]
    θ̄ = -(u^T ∂(∇_xL)/∂θ + w^T ∂c/∂θ)

Both contractions are obtained for any ``θ`` pytree shape via two
:func:`jax.vjp` calls, so a 1-D ``θ`` and a more elaborate pytree work
through the same code path.

Skipped (returns zero ``θ``-cotangent with a ``UserWarning``):
* solve did not converge,
* TNLP refinement skipped (large biactive set, IPOPT restoration
  failure, etc.),
* biactive pairs at ``x*`` (IFT prerequisites fail).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ._constants import BIACTIVE_TOL as _BIACTIVE_TOL
from ._diagnostics import _stack_active_gradient_matrix, active_sets
from ._jax import HAS_JAX
from ._sosc import _build_hessian
from ._typing import StrategyName
from .problem import MPCCProblem
from .result import MPCCResult

__all__ = ["ParametricMPCC", "solve_jax"]


@dataclass
class ParametricMPCC:
    """JAX-traceable MPCC parameterised by an external vector ``θ``.

    Mirrors :class:`pympcc.MPCCProblem` but every callable takes
    ``(x, θ)`` instead of just ``x``.  All callables must be
    JAX-differentiable (use :mod:`jax.numpy` operations exclusively).

    Attributes
    ----------
    n : int
        Number of decision variables.
    n_comp : int
        Number of complementarity pairs.
    objective : callable
        ``f(x, θ) -> scalar``.
    comp_G, comp_H : callable
        ``G(x, θ) -> (n_comp,)`` / ``H(x, θ) -> (n_comp,)``.
    eq_constraints, ineq_constraints : callable, optional
        ``h(x, θ) -> (n_eq,)`` / ``g(x, θ) -> (n_ineq,)``.
    n_eq, n_ineq : int
        Constraint counts (default ``0``).
    xl, xu : array-like, optional
        Variable bounds; treated as constants by the autodiff pass.
    """

    n: int
    n_comp: int
    objective: Callable
    comp_G: Callable
    comp_H: Callable
    eq_constraints: Optional[Callable] = None
    ineq_constraints: Optional[Callable] = None
    n_eq: int = 0
    n_ineq: int = 0
    xl: Optional[np.ndarray] = None
    xu: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("ParametricMPCC: n must be >= 1")
        if self.n_comp < 1:
            raise ValueError("ParametricMPCC: n_comp must be >= 1")
        if self.n_eq and self.eq_constraints is None:
            raise ValueError("eq_constraints required when n_eq > 0")
        if self.n_ineq and self.ineq_constraints is None:
            raise ValueError("ineq_constraints required when n_ineq > 0")

    def materialise(
        self,
        theta_np: np.ndarray,
        x0: np.ndarray,
    ) -> MPCCProblem:
        """Build a numeric :class:`MPCCProblem` with ``θ`` baked into closures.

        Sets ``derivatives="jax"`` so all gradients/Jacobians are produced
        by :mod:`jax`; the closure over ``theta_np`` stays JAX-traceable
        in the ``x`` argument because ``theta_np`` is a constant.
        """
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for ParametricMPCC.materialise(); "
                "install with `pip install pympcc[jax]`"
            )
        import jax.numpy as jnp

        theta_jnp = jnp.asarray(theta_np, dtype=float)

        def _obj(x, _t=theta_jnp):
            return self.objective(x, _t)

        def _G(x, _t=theta_jnp):
            return self.comp_G(x, _t)

        def _H(x, _t=theta_jnp):
            return self.comp_H(x, _t)

        eq_fn = None
        if self.eq_constraints is not None:
            def _eq(x, _t=theta_jnp):
                return self.eq_constraints(x, _t)
            eq_fn = _eq

        ineq_fn = None
        if self.ineq_constraints is not None:
            def _ineq(x, _t=theta_jnp):
                return self.ineq_constraints(x, _t)
            ineq_fn = _ineq

        return MPCCProblem(
            n=self.n,
            n_comp=self.n_comp,
            x0=np.asarray(x0, dtype=float),
            objective=_obj,
            comp_G=_G,
            comp_H=_H,
            n_eq=self.n_eq,
            n_ineq=self.n_ineq,
            eq_constraints=eq_fn,
            ineq_constraints=ineq_fn,
            xl=self.xl,
            xu=self.xu,
            derivatives="jax",
        )


def solve_jax(
    parametric: ParametricMPCC,
    theta: Any,
    *,
    x0: Optional[np.ndarray] = None,
    strategy: StrategyName = "scholtes",
    **solve_kwargs,
):
    """Solve a parametric MPCC; returns ``x*`` differentiable in ``θ``.

    Wraps :func:`pympcc.solve` in a :func:`jax.custom_vjp` whose backward
    rule is the sIPOPT-style adjoint solve.  ``tnlp_refine=True`` is
    forced (overridden if explicitly supplied) so the bwd pass has access
    to certified TNLP multipliers.

    Parameters
    ----------
    parametric : ParametricMPCC
        JAX-traceable problem definition.
    theta : array-like (or pytree of arrays)
        Parameter input.  Differentiable.
    x0 : array-like, optional
        Initial primal point; defaults to ``np.zeros(parametric.n)``.
        Treated as a constant by the autodiff.
    strategy : str
        Strategy forwarded to :func:`pympcc.solve` (default ``"scholtes"``).
    **solve_kwargs
        Additional kwargs forwarded to :func:`pympcc.solve`
        (e.g. ``ipopt_options``, ``presolve``).  Not differentiated.

    Returns
    -------
    jnp.ndarray, shape (n,)
        Optimal primal vector ``x*``.

    Raises
    ------
    ImportError
        When :mod:`jax` is not installed.

    Notes
    -----
    Not jittable: :func:`pympcc.solve` runs IPOPT under the hood.  Call
    inside ``jax.grad(...)`` directly, never inside ``jax.jit(...)``.
    Forward-mode autodiff is not yet supported.
    """
    if not HAS_JAX:
        raise ImportError(
            "JAX is required for solve_jax; install with `pip install pympcc[jax]`"
        )
    import jax
    import jax.numpy as jnp

    from .solver import solve as _solve

    if x0 is None:
        x0_np = np.zeros(parametric.n, dtype=float)
    else:
        x0_np = np.asarray(x0, dtype=float)

    # Force certified MPCC multipliers — the adjoint pass needs them.
    solve_kwargs = dict(solve_kwargs)
    solve_kwargs["tnlp_refine"] = True

    # JAX requires residuals to be JAX-compatible pytrees.  ``MPCCProblem`` /
    # ``MPCCResult`` are not, so stash them via a closure list and only thread
    # ``theta`` through the residual pytree.  Each call to ``solve_jax``
    # constructs a fresh ``_solve_op`` so concurrent calls don't clobber.
    _state: dict = {}

    @jax.custom_vjp
    def _solve_op(theta_):
        return _fwd(theta_)[0]

    def _fwd(theta_):
        theta_np = np.asarray(theta_, dtype=float)
        problem = parametric.materialise(theta_np, x0_np)
        result = _solve(problem, strategy=strategy, **solve_kwargs)
        _state["problem"] = problem
        _state["result"] = result
        x_star = jnp.asarray(result.x, dtype=float)
        return x_star, theta_

    def _bwd(theta_res, v):
        problem = _state["problem"]
        result = _state["result"]
        v_np = np.asarray(v, dtype=float)
        theta_bar = _theta_cotangent(parametric, theta_res, problem, result, v_np)
        return (theta_bar,)

    _solve_op.defvjp(_fwd, _bwd)
    return _solve_op(jnp.asarray(theta, dtype=float))


# --------------------------------------------------------------------------- #
# Backward pass — adjoint KKT solve + jax.vjp for the parametric pjacs        #
# --------------------------------------------------------------------------- #

def _theta_cotangent(
    parametric: ParametricMPCC,
    theta_jnp: Any,
    problem: MPCCProblem,
    result: MPCCResult,
    v: np.ndarray,
) -> Any:
    """Return ``θ̄ = (dx*/dθ)^T v`` via the adjoint KKT solve."""
    import jax
    import jax.numpy as jnp

    zero_theta = jax.tree_util.tree_map(
        lambda a: jnp.zeros_like(jnp.asarray(a, dtype=float)),
        theta_jnp,
    )

    if not result.success:
        warnings.warn(
            "solve_jax: forward solve did not converge; θ-gradient is zero.",
            UserWarning,
            stacklevel=4,
        )
        return zero_theta

    x_star = np.asarray(result.x, dtype=float)
    sets = active_sets(result, problem, tol=_BIACTIVE_TOL)

    if sets["I_00"].size > 0:
        warnings.warn(
            "solve_jax: biactive pairs at x*; θ-gradient is zero. "
            "Implicit differentiation requires MPCC-LICQ (no biactive set).",
            UserWarning,
            stacklevel=4,
        )
        return zero_theta

    # ----- multipliers: TNLP-refined (preferred) → zeros (conservative) ----- #
    n_g, n_h, n_c = problem.n_ineq, problem.n_eq, problem.n_comp
    lam_g = np.zeros(n_g)
    lam_h = np.zeros(n_h)
    lam_G = np.zeros(n_c)
    lam_H = np.zeros(n_c)
    tnlp = getattr(result, "tnlp_refined", None)
    if tnlp is not None and getattr(tnlp, "success", False):
        lam_G = np.asarray(tnlp.mult_comp_G, dtype=float)
        lam_H = np.asarray(tnlp.mult_comp_H, dtype=float)
        if tnlp.mult_ineq is not None:
            lam_g = np.asarray(tnlp.mult_ineq, dtype=float)
        if tnlp.mult_eq is not None:
            lam_h = np.asarray(tnlp.mult_eq, dtype=float)
    else:
        warnings.warn(
            "solve_jax: TNLP refinement unavailable; using zero multipliers in "
            "the adjoint Hessian. The gradient is exact only when the active "
            "constraints are linear in x.",
            UserWarning,
            stacklevel=4,
        )

    # ----- KKT matrix and adjoint solve ------------------------------------ #
    Jc, _ = _stack_active_gradient_matrix(problem, x_star, sets)
    n_active = int(Jc.shape[0])
    H_lag = _build_hessian(x_star, problem, lam_g, lam_h, lam_G, lam_H, problem.fd_h)
    K = np.block([
        [H_lag, Jc.T],
        [Jc,    np.zeros((n_active, n_active))],
    ])
    rhs = np.concatenate([v, np.zeros(n_active)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        warnings.warn(
            "solve_jax: KKT matrix singular; using Tikhonov-regularised "
            "pseudoinverse for the adjoint solve.",
            UserWarning,
            stacklevel=4,
        )
        K_reg = K + 1e-12 * np.eye(K.shape[0])
        sol, *_ = np.linalg.lstsq(K_reg, rhs, rcond=None)
    u = sol[: problem.n]
    w = sol[problem.n:]

    # ----- VJPs over parametric callables at (x*, θ) ----------------------- #
    x_star_jnp = jnp.asarray(x_star, dtype=float)
    lam_g_jnp = jnp.asarray(lam_g, dtype=float)
    lam_h_jnp = jnp.asarray(lam_h, dtype=float)
    lam_G_jnp = jnp.asarray(lam_G, dtype=float)
    lam_H_jnp = jnp.asarray(lam_H, dtype=float)

    def _lagrangian(x, theta):
        L = parametric.objective(x, theta)
        L = L + jnp.sum(lam_G_jnp * parametric.comp_G(x, theta))
        L = L + jnp.sum(lam_H_jnp * parametric.comp_H(x, theta))
        if parametric.n_eq:
            L = L + jnp.sum(lam_h_jnp * parametric.eq_constraints(x, theta))
        if parametric.n_ineq:
            L = L + jnp.sum(lam_g_jnp * parametric.ineq_constraints(x, theta))
        return L

    def _grad_x_L_of_theta(theta):
        return jax.grad(_lagrangian, argnums=0)(x_star_jnp, theta)

    I_g = sets["I_g"]
    I_G = sets["I_G"]
    I_H = sets["I_H"]
    n_bnd = int(np.union1d(sets["I_xL"], sets["I_xU"]).size)

    def _active_constraints(theta):
        parts = []
        if parametric.n_eq:
            parts.append(parametric.eq_constraints(x_star_jnp, theta))
        if I_G.size:
            parts.append(parametric.comp_G(x_star_jnp, theta)[jnp.asarray(I_G)])
        if I_H.size:
            parts.append(parametric.comp_H(x_star_jnp, theta)[jnp.asarray(I_H)])
        if I_g.size:
            parts.append(parametric.ineq_constraints(x_star_jnp, theta)[jnp.asarray(I_g)])
        if n_bnd:
            # Bounds are constants of the solve (no θ dependence).
            parts.append(jnp.zeros(n_bnd))
        if not parts:
            return jnp.zeros(0)
        return jnp.concatenate(parts)

    u_jnp = jnp.asarray(u, dtype=float)
    _, vjp_grad_L = jax.vjp(_grad_x_L_of_theta, theta_jnp)
    contrib_L = vjp_grad_L(u_jnp)[0]

    if n_active > 0:
        w_jnp = jnp.asarray(w, dtype=float)
        _, vjp_c = jax.vjp(_active_constraints, theta_jnp)
        contrib_c = vjp_c(w_jnp)[0]
    else:
        contrib_c = zero_theta

    return jax.tree_util.tree_map(lambda a, b: -(a + b), contrib_L, contrib_c)
