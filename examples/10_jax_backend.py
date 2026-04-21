"""
Example 10 — JAX autodiff backend
==================================

Demonstrates the ``"jax"`` sentinel that replaces hand-coded Jacobians with
JIT-compiled exact derivatives and auto-detects COO sparsity at construction
time.  Also shows ``use_jax_hessian=True``, which supplies IPOPT with the
exact Lagrangian Hessian instead of the default L-BFGS approximation.

Three ways to define the same problem
--------------------------------------

1. **Exact (hand-coded)** — traditional approach, full control.
2. **JAX Jacobians** — ``"jax"`` sentinel; sparsity auto-detected from three
   probe evaluations near ``x0``.
3. **JAX Jacobians + exact Hessian** — adds ``use_jax_hessian=True``; each
   inner IPOPT solve receives the exact Lagrangian Hessian.

Problem (Scholtes 1997, example 1)
------------------------------------

    min  (x0 - 1)^2 + (x1 - 0.5)^2 + x2
    s.t. x0^2 + x1^2 - 4 = 0          (equality)
         G(x) = x0 + x1 - 1  ≥ 0
         H(x) = x2            ≥ 0
         G · H = 0

    Global optimum: G* ≈ 0, H* = 0, f* ≈ 2.

All three callables use JAX-compatible arithmetic (``jnp``), so the
``"jax"`` sentinel and ``use_jax_hessian=True`` can both trace through them.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import pympcc


# ------------------------------------------------------------------ #
# Problem callables (JAX-compatible arithmetic)                       #
# ------------------------------------------------------------------ #

X0 = np.array([1.0, 0.5, 1.0])
XL = np.array([-np.inf, -np.inf, 0.0])


def objective(x):
    return (x[0] - 1.0) ** 2 + (x[1] - 0.5) ** 2 + x[2]


def gradient(x):
    return np.array([2.0 * (x[0] - 1.0), 2.0 * (x[1] - 0.5), 1.0])


def eq_con(x):
    return jnp.array([x[0] ** 2 + x[1] ** 2 - 4.0])


def comp_G(x):
    return jnp.array([x[0] + x[1] - 1.0])


def comp_H(x):
    return jnp.array([x[2]])


# ------------------------------------------------------------------ #
# Problem builders                                                     #
# ------------------------------------------------------------------ #

def build_exact() -> pympcc.MPCCProblem:
    """Hand-coded Jacobians — the traditional approach."""
    return pympcc.MPCCProblem(
        n=3, n_comp=1, n_eq=1,
        x0=X0.copy(), xl=XL.copy(),
        objective=objective,
        gradient=gradient,
        eq_constraints=eq_con,
        eq_jacobian=lambda x: np.array([[2.0 * x[0], 2.0 * x[1], 0.0]]),
        comp_G=comp_G,
        comp_G_jacobian=lambda x: np.array([[1.0, 1.0, 0.0]]),
        comp_H=comp_H,
        comp_H_jacobian=lambda x: np.array([[0.0, 0.0, 1.0]]),
    )


def build_jax() -> pympcc.MPCCProblem:
    """JAX autodiff — no hand-coded Jacobians."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pympcc.MPCCProblem(
            n=3, n_comp=1, n_eq=1,
            x0=X0.copy(), xl=XL.copy(),
            objective=objective,
            gradient="jax",
            eq_constraints=eq_con,
            eq_jacobian="jax",
            comp_G=comp_G,
            comp_G_jacobian="jax",
            comp_H=comp_H,
            comp_H_jacobian="jax",
        )


def build_jax_with_hessian() -> pympcc.MPCCProblem:
    """JAX autodiff + exact Lagrangian Hessian supplied to IPOPT."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pympcc.MPCCProblem(
            n=3, n_comp=1, n_eq=1,
            x0=X0.copy(), xl=XL.copy(),
            objective=objective,
            gradient="jax",
            eq_constraints=eq_con,
            eq_jacobian="jax",
            comp_G=comp_G,
            comp_G_jacobian="jax",
            comp_H=comp_H,
            comp_H_jacobian="jax",
            use_jax_hessian=True,
        )


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    print("JAX autodiff backend  (n=3, n_comp=1)")
    print("=" * 60)

    # --- show auto-detected sparsity ---------------------------------
    p_jax = build_jax()
    sp_G = p_jax.comp_G_jacobian_sparsity
    sp_H = p_jax.comp_H_jacobian_sparsity
    sp_eq = p_jax.eq_jacobian_sparsity

    print()
    print("Auto-detected sparsity (COO, 0-based):")
    print(f"  comp_G_jacobian:  rows={sp_G[0].tolist()}, cols={sp_G[1].tolist()}")
    print(f"  comp_H_jacobian:  rows={sp_H[0].tolist()}, cols={sp_H[1].tolist()}")
    if sp_eq is not None:
        print(f"  eq_jacobian:      rows={sp_eq[0].tolist()}, cols={sp_eq[1].tolist()}")
    print(f"  is_sparse: {p_jax.is_sparse}")

    # --- compare all three formulations across two strategies --------
    strategies = ("scholtes", "smoothing")
    builders = [
        ("exact (hand-coded)", build_exact),
        ("jax (auto-sparsity)", build_jax),
        ("jax + hessian",       build_jax_with_hessian),
    ]

    print()
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        print(f"  {'formulation':<24}  {'f*':>8}  {'comp':>10}  {'status':>6}")
        print(f"  {'─'*24}  {'─'*8}  {'─'*10}  {'─'*6}")
        for label, builder in builders:
            p = builder()
            r = pympcc.solve(p, strategy=strategy)
            print(
                f"  {label:<24}  {r.obj:>8.5f}"
                f"  {r.comp_residual:>10.2e}"
                f"  {'ok' if r.success else 'fail':>6}"
            )
        print()

    # --- verify JAX and exact solutions agree ------------------------
    r_exact = pympcc.solve(build_exact(), strategy="scholtes")
    r_jax   = pympcc.solve(build_jax(),   strategy="scholtes")
    r_hess  = pympcc.solve(build_jax_with_hessian(), strategy="scholtes")

    tol = 1e-4
    assert abs(r_jax.obj  - r_exact.obj) < tol, "JAX result diverged from exact"
    assert abs(r_hess.obj - r_exact.obj) < tol, "JAX+Hessian result diverged"
    print("All three formulations agree within solver tolerance.")


if __name__ == "__main__":
    main()
