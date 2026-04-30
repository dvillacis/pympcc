"""Parametric sensitivity analysis for MPCC solutions (§5.1).

At a converged MPCC solution where MPCC-LICQ holds (no biactive pairs and
the active constraint Jacobian has full row rank), the implicit function
theorem applies to the KKT system of the **tightened NLP** (TNLP) — the
NLP obtained by fixing each comp pair's active side as an equality:

    ∇_x f(x; p) + J_c(x; p)^T λ = 0
    c(x; p) = 0

Differentiating with respect to a parameter vector ``p`` yields the saddle-
point linear system::

    ┌ ∇²_xx L   J_c^T ┐ ┌ dx*/dp ┐     ┌ ∂(∇_x L)/∂p ┐
    │                 │ │        │ = − │              │
    └ J_c       0     ┘ └ dλ*/dp ┘     └ ∂c/∂p       ┘

where ``c`` collects the equality, equality-pinned-G, equality-pinned-H,
active-inequality, and active-bound constraints (in that order — see
:func:`active_row_labels`).

This module exposes the linear-solve primitive only.  Callers supply the
two RHS blocks evaluated at ``x*`` directly; the higher-level convenience
that builds those blocks from JAX-traceable callables is part of §5.6.

Parameter-vector convention
---------------------------
``∂(∇_x L)/∂p`` is the partial derivative of the *Lagrangian gradient*
holding ``(x, λ)`` fixed.  When parameters enter only the objective, this
reduces to ``∂(∇_x f)/∂p``.  When constraints are also parametric, the
caller must include the cross-term ``Σ_i λ_i · ∂(∇_x c_i)/∂p``.

Multiplier convention follows :mod:`pympcc._sosc`: the Lagrangian Hessian
builder consumes raw IPOPT-convention multipliers, so callers do not need
to negate.  TNLP-refined multipliers are picked up automatically when
:func:`pympcc.solve` was invoked with ``tnlp_refine=True``; otherwise the
fallback is the zero vector with a ``UserWarning``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._diagnostics import _stack_active_gradient_matrix, active_sets
from ._sosc import _build_hessian
from .problem import MPCCProblem
from .result import MPCCResult

__all__ = ["SensitivityResult", "sensitivity", "active_row_labels"]


_SKIP_NOT_CONVERGED = "not_converged"
_SKIP_BIACTIVE = "biactive_pairs"
_SKIP_HESSIAN = "no_hessian_callable_and_fd_failed"


@dataclass
class SensitivityResult:
    """Result of a parametric sensitivity solve.

    Attributes
    ----------
    dx_dp : ndarray, shape (n, n_p)
        Parametric derivative of the primal solution.  Shape ``(n, 0)``
        when the check was skipped.
    dlam_dp : ndarray, shape (n_active, n_p)
        Parametric derivative of the Lagrange multipliers on the active
        constraint rows, in the order described by ``active_row_labels``.
    active_row_labels : list[tuple]
        Per-row labels for the active constraint Jacobian.  Each entry
        is one of ``("h", k)``, ``("G", i)``, ``("H", i)``, ``("g", j)``,
        ``("xL", j)``, or ``("xU", j)``.  The caller's ``dc_dp`` rows
        must match this order.
    rank_deficit : int
        ``0`` when the KKT matrix is non-singular.  Positive only when
        the regularised lstsq fallback was used.
    used_pseudoinverse : bool
        ``True`` when the dense ``solve`` failed and ``lstsq`` with
        Tikhonov regularisation was used instead.
    kkt_condition : float or None
        ``np.linalg.cond`` of the KKT matrix; ``None`` when the matrix
        was empty or the check was skipped.
    skipped_reason : str or None
        Set when prerequisites for IFT were not met:
        ``"not_converged"`` (``result.success`` was ``False``);
        ``"biactive_pairs"`` (``I_00 ≠ ∅`` — IFT requires LICQ);
        ``"no_hessian_callable_and_fd_failed"`` (FD fallback raised).
    """

    dx_dp: np.ndarray
    dlam_dp: np.ndarray
    active_row_labels: list
    rank_deficit: int = 0
    used_pseudoinverse: bool = False
    kkt_condition: Optional[float] = None
    skipped_reason: Optional[str] = None


def _build_active_row_labels(problem: MPCCProblem, sets: dict) -> list:
    """Per-row labels matching :func:`_stack_active_gradient_matrix` row order.

    Order: ``h`` (all eq) | ``G_{I_G}`` | ``H_{I_H}`` | ``g_{I_g}`` |
    ``bnd`` (sorted ``I_xL ∪ I_xU``).  When a variable appears in both
    ``I_xL`` and ``I_xU`` (pinned), prefers ``"xL"`` since the underlying
    matrix produces a single identity row per variable in the union.
    """
    labels: list = []
    for k in range(problem.n_eq):
        labels.append(("h", int(k)))
    for i in sets["I_G"]:
        labels.append(("G", int(i)))
    for i in sets["I_H"]:
        labels.append(("H", int(i)))
    for j in sets["I_g"]:
        labels.append(("g", int(j)))
    bound_idx = np.union1d(sets["I_xL"], sets["I_xU"]).astype(np.intp)
    I_xL_set = {int(j) for j in sets["I_xL"]}
    for j in bound_idx:
        labels.append(("xL" if int(j) in I_xL_set else "xU", int(j)))
    return labels


def active_row_labels(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    active_tol: float = 1e-6,
) -> list:
    """Return active-row labels in the order :func:`sensitivity` expects ``dc_dp``.

    Useful for assembling the ``dc_dp`` matrix without first calling
    :func:`sensitivity`.

    Parameters
    ----------
    result, problem : MPCCResult, MPCCProblem
        Solved result and its source problem.
    active_tol : float
        Constraint tolerance for active-set detection (default ``1e-6``).

    Returns
    -------
    list of (str, int)
        Per-row labels.  See :class:`SensitivityResult.active_row_labels`.
    """
    sets = active_sets(result, problem, tol=active_tol)
    return _build_active_row_labels(problem, sets)


def _gather_multipliers(
    result: MPCCResult, problem: MPCCProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Pick the best available Lagrangian multipliers for Hessian construction.

    Prefers TNLP-refined multipliers (signed and certified by the §2.6
    re-solve); falls back to zero vectors when no TNLP refinement is
    attached.  Returns ``(lam_g, lam_h, lam_G, lam_H, used_tnlp)``.
    """
    p = problem
    lam_g = np.zeros(p.n_ineq)
    lam_h = np.zeros(p.n_eq)
    lam_G = np.zeros(p.n_comp)
    lam_H = np.zeros(p.n_comp)

    tnlp = getattr(result, "tnlp_refined", None)
    if tnlp is not None and getattr(tnlp, "success", False):
        lam_G = np.asarray(tnlp.mult_comp_G, dtype=float)
        lam_H = np.asarray(tnlp.mult_comp_H, dtype=float)
        if tnlp.mult_ineq is not None:
            lam_g = np.asarray(tnlp.mult_ineq, dtype=float)
        if tnlp.mult_eq is not None:
            lam_h = np.asarray(tnlp.mult_eq, dtype=float)
        return lam_g, lam_h, lam_G, lam_H, True

    return lam_g, lam_h, lam_G, lam_H, False


def sensitivity(
    result: MPCCResult,
    problem: MPCCProblem,
    *,
    dgrad_L_dp,
    dc_dp,
    fd_h: Optional[float] = None,
    regularize_eps: float = 1e-12,
    active_tol: float = 1e-6,
) -> SensitivityResult:
    """Compute ``dx*/dp`` via implicit differentiation through the KKT system.

    Solves the saddle-point system

        [[H,  J_c^T], [J_c, 0]] · [dx; dλ] = − [dgrad_L_dp; dc_dp]

    where ``H = ∇²_xx L(x*, λ*)`` and ``J_c`` is the active constraint
    Jacobian.  Skips with ``skipped_reason`` set when MPCC-LICQ
    prerequisites fail (non-converged result, or biactive pairs present).

    Parameters
    ----------
    result : MPCCResult
        Converged solve result.  When ``tnlp_refine=True`` was used, the
        TNLP-refined multipliers are picked up automatically; otherwise
        a zero-multiplier fallback is used and a ``UserWarning`` is
        emitted (only when no analytic Hessian is available).
    problem : MPCCProblem
        Source problem (original space, not presolve-reduced).
    dgrad_L_dp : array-like, shape (n,) or (n, n_p)
        ``∂(∇_x L)/∂p`` at ``x*`` with ``(x, λ)`` held fixed.  When
        parameters enter only ``f``, this is ``∂(∇_x f)/∂p``; otherwise
        callers must include the cross-term
        ``Σ_i λ_i · ∂(∇_x c_i)/∂p``.  A 1-D input is treated as a single
        parameter column.
    dc_dp : array-like, shape (n_active,) or (n_active, n_p)
        ``∂c/∂p`` at ``x*``, with rows ordered to match
        :func:`active_row_labels`.  See :class:`SensitivityResult` for
        the per-row label format.
    fd_h : float, optional
        Step size for the central-difference Hessian fallback when
        ``problem.lagrangian_hessian`` is not provided.  Defaults to
        ``problem.fd_h``.
    regularize_eps : float
        Tikhonov ridge added to the KKT diagonal when the dense solve
        fails.  Default ``1e-12``.
    active_tol : float
        Active-set detection tolerance (default ``1e-6``).

    Returns
    -------
    SensitivityResult
    """
    p = problem
    n = p.n
    n_p = _infer_n_p(dgrad_L_dp, dc_dp)

    def _empty(reason: str) -> SensitivityResult:
        return SensitivityResult(
            dx_dp=np.zeros((n, n_p)),
            dlam_dp=np.zeros((0, n_p)),
            active_row_labels=[],
            skipped_reason=reason,
        )

    if not result.success:
        return _empty(_SKIP_NOT_CONVERGED)

    x = np.asarray(result.x, dtype=float)
    sets = active_sets(result, p, tol=active_tol)
    if sets["I_00"].size > 0:
        return _empty(_SKIP_BIACTIVE)

    Jc, _ = _stack_active_gradient_matrix(p, x, sets)
    labels = _build_active_row_labels(p, sets)
    n_active = int(Jc.shape[0])

    dgrad_L_dp_arr = _validate_rhs(
        dgrad_L_dp, expected_rows=n, n_p=n_p, name="dgrad_L_dp",
    )
    dc_dp_arr = _validate_rhs(
        dc_dp, expected_rows=n_active, n_p=n_p, name="dc_dp",
    )

    lam_g, lam_h, lam_G, lam_H, used_tnlp = _gather_multipliers(result, p)
    if not used_tnlp and p.lagrangian_hessian is None:
        warnings.warn(
            "sensitivity: no TNLP-refined multipliers available; the "
            "FD-Hessian fallback is using zero multipliers, which is "
            "exact only when constraints are linear in x. Pass "
            "tnlp_refine=True to pympcc.solve() for accurate sensitivity.",
            UserWarning,
            stacklevel=2,
        )
    _fd_h = fd_h if fd_h is not None else p.fd_h
    try:
        H = _build_hessian(x, p, lam_g, lam_h, lam_G, lam_H, _fd_h)
    except Exception:
        return _empty(_SKIP_HESSIAN)

    K = np.block([
        [H,  Jc.T],
        [Jc, np.zeros((n_active, n_active))],
    ])
    rhs = -np.vstack([dgrad_L_dp_arr, dc_dp_arr])

    cond = float(np.linalg.cond(K)) if K.size else None

    used_pinv = False
    rank_def = 0
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        used_pinv = True
        K_reg = K + regularize_eps * np.eye(K.shape[0])
        sol, _, rank, _ = np.linalg.lstsq(K_reg, rhs, rcond=None)
        rank_def = K.shape[0] - int(rank)

    return SensitivityResult(
        dx_dp=sol[:n],
        dlam_dp=sol[n:],
        active_row_labels=labels,
        rank_deficit=rank_def,
        used_pseudoinverse=used_pinv,
        kkt_condition=cond,
        skipped_reason=None,
    )


def _infer_n_p(dgrad_L_dp, dc_dp) -> int:
    """Infer the parameter count from either RHS block, treating 1-D as 1."""
    arr = np.asarray(dgrad_L_dp)
    if arr.ndim == 1:
        return 1
    if arr.ndim == 2:
        return arr.shape[1]
    raise ValueError(
        f"dgrad_L_dp must be 1-D or 2-D; got ndim={arr.ndim}"
    )


def _validate_rhs(arr, *, expected_rows: int, n_p: int, name: str) -> np.ndarray:
    """Coerce ``arr`` to shape ``(expected_rows, n_p)`` or raise."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got ndim={a.ndim}")
    if a.shape[0] != expected_rows:
        raise ValueError(
            f"{name} has {a.shape[0]} rows; expected {expected_rows} "
            f"(rows must follow active_row_labels order)"
        )
    if a.shape[1] != n_p:
        raise ValueError(
            f"{name} has {a.shape[1]} parameter columns; expected {n_p}"
        )
    return a
