"""Derivative-resolution helpers for :class:`~pympcc.MPCCProblem`.

Translates ``derivatives="jax"`` / ``"fd"`` sentinels into actual
callables, and reports a clear error if any required derivative is
still missing after resolution.  All functions mutate the supplied
``MPCCProblem`` in place.

These are called from :meth:`MPCCProblem.__post_init__` in this order::

    apply_derivatives_default(p)   # broadcast `derivatives` sentinel to None fields
    resolve_jax_fields(p)          # "jax" sentinel → jax.grad / jax.jacfwd
    resolve_fd_fields(p)           # "fd"  sentinel → fd_gradient / fd_jacobian
    # ... var-pair / box-pair normalisation runs in between ...
    check_derivatives_resolved(p)  # final guard: every required slot is callable
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .problem import MPCCProblem


def apply_derivatives_default(p: "MPCCProblem") -> None:
    """Auto-fill ``None`` derivative fields with the ``derivatives`` sentinel.

    ``derivatives="jax"`` and ``derivatives="fd"`` provide a single
    keyword opt-in into autodiff or finite differences for every
    derivative field the user hasn't supplied.  Fields that already
    hold a callable or an explicit sentinel are left untouched.
    """
    if p.derivatives is None:
        return
    if p.derivatives not in ("jax", "fd"):
        raise ValueError(
            f"derivatives must be None, 'jax', or 'fd'; got {p.derivatives!r}"
        )
    sentinel = p.derivatives

    if p.gradient is None:
        p.gradient = sentinel
    if p.comp_G_jacobian is None:
        p.comp_G_jacobian = sentinel
    if p.comp_H_jacobian is None:
        p.comp_H_jacobian = sentinel
    if p.n_ineq > 0 and p.ineq_constraints is not None and p.ineq_jacobian is None:
        p.ineq_jacobian = sentinel
    if p.n_eq > 0 and p.eq_constraints is not None and p.eq_jacobian is None:
        p.eq_jacobian = sentinel


def resolve_jax_fields(p: "MPCCProblem") -> None:
    """Replace any ``"jax"`` sentinel with a JAX-autodiff callable."""
    from ._jax import HAS_JAX

    # Collect which fields hold the "jax" sentinel.
    _sentinel_fields = [
        p.gradient,
        p.comp_G_jacobian, p.comp_H_jacobian,
        p.ineq_jacobian, p.eq_jacobian,
    ]
    jax_fields = [v for v in _sentinel_fields if v == "jax"]

    if not jax_fields:
        return   # nothing to do — fast path for most users

    if not HAS_JAX:
        raise ImportError(
            "JAX is required for the 'jax' sentinel but is not installed. "
            "Install it with:  pip install 'pympcc[jax]'"
        )

    from ._jax import jax_gradient, jax_jacobian

    tol = p.jax_sparsity_tol
    jax_used: list[str] = []

    if p.gradient == "jax":
        p.gradient = jax_gradient(p.objective, p.n, p.x0, tol)
        jax_used.append("gradient")

    for attr, fn_attr, sparsity_attr, n_out in [
        ("comp_G_jacobian", "comp_G", "comp_G_jacobian_sparsity", p.n_comp),
        ("comp_H_jacobian", "comp_H", "comp_H_jacobian_sparsity", p.n_comp),
        ("ineq_jacobian", "ineq_constraints", "ineq_jacobian_sparsity", p.n_ineq),
        ("eq_jacobian", "eq_constraints", "eq_jacobian_sparsity", p.n_eq),
    ]:
        if getattr(p, attr) != "jax":
            continue
        fn = getattr(p, fn_attr)
        # Defer comp_G/H Jacobian resolution to _normalize_var_pairs when comp_G is None
        if fn is None and attr in ("comp_G_jacobian", "comp_H_jacobian"):
            continue
        if fn is None or n_out == 0:
            raise ValueError(
                f"{attr}='jax' requires the corresponding function "
                f"({fn_attr}) to be set and n_out > 0"
            )
        jac_fn, sparsity = jax_jacobian(fn, n_out, p.n, p.x0, tol)
        setattr(p, attr, jac_fn)
        setattr(p, sparsity_attr, sparsity)
        jax_used.append(attr)

    if jax_used:
        warnings.warn(
            f"JAX autodiff active for: {', '.join(jax_used)}. "
            "Ensure all primal callables are JAX-differentiable.",
            UserWarning,
            stacklevel=3,
        )


def resolve_fd_fields(p: "MPCCProblem") -> None:
    """Replace any ``"fd"`` sentinel with a finite-difference callable."""
    from ._fd import fd_gradient, fd_jacobian

    if p.fd_mode not in ("forward", "central"):
        raise ValueError(
            f"fd_mode must be 'forward' or 'central', got {p.fd_mode!r}"
        )
    h, mode = p.fd_h, p.fd_mode
    fd_used: list[str] = []

    if p.gradient == "fd":
        p.gradient = fd_gradient(p.objective, p.n, h, mode)
        fd_used.append("gradient")
    if p.comp_G_jacobian == "fd":
        if p.comp_G is not None:
            p.comp_G_jacobian = fd_jacobian(
                p.comp_G, p.n_comp, p.n, h, mode
            )
            fd_used.append("comp_G_jacobian")
        # else: comp_G is None (var-pairs only) — _normalize_var_pairs builds exact G Jac
    if p.comp_H_jacobian == "fd":
        if p.comp_H is not None:
            p.comp_H_jacobian = fd_jacobian(
                p.comp_H, p.n_comp, p.n, h, mode
            )
            fd_used.append("comp_H_jacobian")
        # else: comp_H is None (var-pairs only) — _normalize_var_pairs builds H Jac (fd per row)
    if p.ineq_jacobian == "fd":
        if p.n_ineq == 0 or p.ineq_constraints is None:
            raise ValueError(
                "ineq_jacobian='fd' requires n_ineq > 0 and ineq_constraints"
            )
        p.ineq_jacobian = fd_jacobian(
            p.ineq_constraints, p.n_ineq, p.n, h, mode
        )
        fd_used.append("ineq_jacobian")
    if p.eq_jacobian == "fd":
        if p.n_eq == 0 or p.eq_constraints is None:
            raise ValueError(
                "eq_jacobian='fd' requires n_eq > 0 and eq_constraints"
            )
        p.eq_jacobian = fd_jacobian(
            p.eq_constraints, p.n_eq, p.n, h, mode
        )
        fd_used.append("eq_jacobian")

    if fd_used:
        warnings.warn(
            f"Finite-difference Jacobian active for: {', '.join(fd_used)}. "
            "Suitable for prototyping; use exact Jacobians in production.",
            UserWarning,
            stacklevel=3,
        )


def check_derivatives_resolved(p: "MPCCProblem") -> None:
    """Raise a clear error when a required derivative is still missing.

    Runs after :func:`resolve_jax_fields` and :func:`resolve_fd_fields`
    so any sentinel that survived resolution (e.g. JAX missing) has
    already been reported.
    """
    missing: list[str] = []
    if p.gradient is None or isinstance(p.gradient, str):
        missing.append("gradient")
    if p.comp_G_jacobian is None or isinstance(p.comp_G_jacobian, str):
        missing.append("comp_G_jacobian")
    if p.comp_H_jacobian is None or isinstance(p.comp_H_jacobian, str):
        missing.append("comp_H_jacobian")
    if p.n_ineq > 0 and (
        p.ineq_jacobian is None or isinstance(p.ineq_jacobian, str)
    ):
        missing.append("ineq_jacobian")
    if p.n_eq > 0 and (
        p.eq_jacobian is None or isinstance(p.eq_jacobian, str)
    ):
        missing.append("eq_jacobian")
    if missing:
        raise ValueError(
            f"Missing derivative callable(s): {missing}. "
            "Pass each as a callable, set the field to 'jax' or 'fd', "
            "or pass derivatives='jax' / derivatives='fd' to fill all "
            "unset derivative fields at once."
        )


__all__ = [
    "apply_derivatives_default",
    "resolve_jax_fields",
    "resolve_fd_fields",
    "check_derivatives_resolved",
]
