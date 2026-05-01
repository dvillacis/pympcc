"""
TNLP refinement: re-solve MPCC with fixed complementarity active set.

After a relaxation strategy (Scholtes, smoothing, etc.) converges, this module
fixes the inferred complementarity active set (I_G_active, I_H_active) and
re-solves the resulting equality/inequality NLP to obtain proper MPCC-stationarity
multipliers with correct signs.

This is the same postprocessing step that KNITRO's ``mpec_finalize`` and
FilterMPEC use to certify multiplier signs and confirm B-stationarity when
the biactive set is too large for LP enumeration.

Constraint layout of the tightened NLP:
    [g (n_ineq), h (n_eq), G (n_comp), H (n_comp)]

where ``G_i = 0`` is enforced (cl=cu=0) for i ∈ I_G_active and ``H_i = 0``
is enforced for i ∈ I_H_active.  Other G_i / H_i rows remain as ``≥ 0``
inequalities.

Multiplier sign convention (IPOPT vs. literature):
    IPOPT: KKT ∇f + Jᵀλ = 0  →  λ_i ≤ 0 at an active lower bound.
    Literature: μ = −λ ≥ 0 at S-stationary points.
All multipliers returned here use the **literature convention** (μ = −λ).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ._constants import BIACTIVE_TOL_FLOOR

if TYPE_CHECKING:
    from .problem import MPCCProblem
    from .result import MPCCResult

__all__ = ["TNLPResult", "run_tnlp_refinement"]


@dataclass
class TNLPResult:
    """Result of a TNLP active-set refinement re-solve (§2.6).

    Attributes
    ----------
    x : ndarray
        Solution of the tightened NLP.
    obj : float
        Objective value at ``x``.
    status : int
        IPOPT exit code (0 = Solve_Succeeded, 1 = Acceptable, etc.).
    message : str
        Human-readable IPOPT status string.
    success : bool
        ``True`` when ``status ∈ {0, 1, 3}``.
    mult_comp_G : ndarray, shape (n_comp,)
        MPCC multipliers ``μ_G = −λ_G`` (literature sign convention).
        Non-negative at S-stationary points for pairs with ``G_i = 0``.
    mult_comp_H : ndarray, shape (n_comp,)
        MPCC multipliers ``μ_H = −λ_H``.
        Non-negative at S-stationary points for pairs with ``H_i = 0``.
    mult_ineq : ndarray or None
        Inequality-constraint multipliers (IPOPT sign convention, ≤ 0).
    mult_eq : ndarray or None
        Equality-constraint multipliers (IPOPT sign convention, free).
    kkt_residual : float or None
        ∞-norm of the MPCC stationarity residual ``‖∇f + Jᵀμ‖`` at the
        TNLP solution.
    stationarity : str
        Stationarity classification at the TNLP solution.
    n_iter : int
        IPOPT iterations in the TNLP solve.
    solve_time : float
        Wall-clock seconds for the TNLP solve.
    active_set : tuple
        ``(I_G_active, I_H_active)`` — integer index arrays of comp pairs
        with ``G_i`` pinned to 0 and ``H_i`` pinned to 0 respectively.
    n_violations : int
        Number of pairs whose equality-side MPCC multiplier was negative
        (< ``-1e-6``) in the final TNLP solve.  Zero at an S-stationary point.
        Non-zero indicates pairs where the active-set assignment was wrong
        (typically from a poorly converged relaxation iterate).
    """

    x: np.ndarray
    obj: float
    status: int
    message: str
    success: bool
    mult_comp_G: np.ndarray
    mult_comp_H: np.ndarray
    mult_ineq: Optional[np.ndarray]
    mult_eq: Optional[np.ndarray]
    kkt_residual: Optional[float]
    stationarity: str
    n_iter: int
    solve_time: float
    active_set: tuple
    n_violations: int = 0


def _classify_tnlp_stationarity(
    mu_G: np.ndarray,
    mu_H: np.ndarray,
    I_G_active: np.ndarray,
    I_H_active: np.ndarray,
    *,
    tol: float = 1e-6,
) -> tuple[str, int]:
    """Stationarity classification for a TNLP result.

    In the TNLP, every comp pair is split into one equality-pinned side
    (G_i = 0 or H_i = 0) and one free side (inactive inequality).
    S-stationarity holds when the equality-side MPCC multiplier is ≥ 0
    (inactive sides have multiplier 0 by complementarity slackness).

    Returns ``(stat, n_violations)`` where ``n_violations`` is the count of
    pairs with a negative equality-side multiplier.
    """
    n_viol = 0
    for i in I_G_active:
        if mu_G[i] < -tol:
            n_viol += 1
    for i in I_H_active:
        if mu_H[i] < -tol:
            n_viol += 1
    # In the TNLP all pairs are non-biactive (each pinned to one side).
    # C/M/S all require mu >= 0 for non-biactive pairs; W does not.
    # Any violation therefore certifies W-stationarity (not "not S-stationary").
    stat = "W-stationary" if n_viol > 0 else "S-stationary"
    return stat, n_viol


def _flip_wrong_pairs(
    mu_G_orig: np.ndarray,
    mu_H_orig: np.ndarray,
    I_G_active: np.ndarray,
    I_H_active: np.ndarray,
    *,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Swap pairs whose equality-side multiplier is negative.

    Pairs in ``I_G_active`` with ``mu_G_orig[i] < -tol`` had G pinned but the
    multiplier sign indicates H should have been pinned (the TNLP imposed the
    wrong equality).  Move them to ``I_H_active`` and vice-versa.

    Returns ``(I_G_new, I_H_new, n_flipped)``.
    """
    to_H = np.array([i for i in I_G_active if mu_G_orig[i] < -tol], dtype=np.intp)
    to_G = np.array([i for i in I_H_active if mu_H_orig[i] < -tol], dtype=np.intp)
    n_flip = len(to_H) + len(to_G)
    if n_flip == 0:
        return I_G_active, I_H_active, 0
    I_G_new = np.union1d(np.setdiff1d(I_G_active, to_H), to_G).astype(np.intp)
    I_H_new = np.union1d(np.setdiff1d(I_H_active, to_G), to_H).astype(np.intp)
    return I_G_new, I_H_new, n_flip


def _infer_active_set(
    G: np.ndarray,
    H: np.ndarray,
    mpcc_mult_G: Optional[np.ndarray],
    mpcc_mult_H: Optional[np.ndarray],
    *,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition comp pairs into G-pinned and H-pinned index arrays.

    Default rule: pin the smaller side (magnitude).  Biactive pairs
    (both within ``max(sqrt(comp_residual), tol)``) are resolved by the
    multiplier rule when approximate MPCC multipliers are available: pin
    the side whose multiplier magnitude is smaller (the solver considers
    it less binding).
    """
    n_c = len(G)
    if n_c == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty

    comp = float(np.max(np.abs(G * H)))
    bi_tol = max(np.sqrt(comp), tol)

    pin_G = G < H  # default: pin the smaller side

    if mpcc_mult_G is not None and mpcc_mult_H is not None:
        biactive = (G <= bi_tol) & (H <= bi_tol)
        if np.any(biactive):
            mu_G_bi = np.abs(mpcc_mult_G[biactive])
            mu_H_bi = np.abs(mpcc_mult_H[biactive])
            pin_G[biactive] = mu_G_bi <= mu_H_bi

    I_G = np.where(pin_G)[0].astype(np.intp)
    I_H = np.where(~pin_G)[0].astype(np.intp)
    return I_G, I_H


def run_tnlp_refinement(
    result: "MPCCResult",
    problem: "MPCCProblem",
    strategy,
    *,
    tol: Optional[float] = None,
    max_iter: int = 500,
) -> TNLPResult:
    """Re-solve MPCC with fixed active set for certified MPCC multipliers.

    Determines the complementarity active set from ``result.G`` and
    ``result.H``, then re-solves the active-set-fixed NLP to obtain
    proper MPCC-stationarity multipliers with correct signs.

    The TNLP result is stored as ``MPCCResult.tnlp_refined`` and does
    **not** replace the main relaxation result.  When the TNLP solve
    succeeds, ``MPCCResult.mult_comp_G_mpcc`` and
    ``MPCCResult.mult_comp_H_mpcc`` are also populated.

    Parameters
    ----------
    result : MPCCResult
        Converged relaxation result to refine.
    problem : MPCCProblem
        The MPCC problem in original (pre-presolve) variable space.
    strategy : BaseStrategy
        The instantiated strategy from the solver (for NLP construction).
    tol : float or None
        IPOPT tolerance for the TNLP solve.  Defaults to the strategy's
        ``ipopt_options['tol']`` (typically ``1e-8``).
    max_iter : int
        IPOPT maximum iterations for the TNLP solve.

    Returns
    -------
    TNLPResult
    """
    from types import SimpleNamespace

    from ._stationarity import compute_kkt_residual
    from .strategies._base import BaseStrategy

    p = problem

    # Extract approximate MPCC multipliers from the relaxed result for
    # biactive tie-breaking.  These are from the *relaxed* NLP so they
    # are only approximate — the TNLP will produce certified replacements.
    mpcc_mult_G: Optional[np.ndarray] = None
    mpcc_mult_H: Optional[np.ndarray] = None
    if result.mult_g is not None:
        offset = p.n_ineq + p.n_eq
        if len(result.mult_g) >= offset + 2 * p.n_comp:
            mpcc_mult_G = -np.asarray(result.mult_g[offset : offset + p.n_comp])
            mpcc_mult_H = -np.asarray(
                result.mult_g[offset + p.n_comp : offset + 2 * p.n_comp]
            )

    G_arr = np.asarray(result.G, dtype=float)
    H_arr = np.asarray(result.H, dtype=float)

    # Biactivity pre-screen: count pairs where both G_i and H_i are near zero.
    # The equality constraints in the TNLP (G_i=0 or H_i=0) are near-infeasible
    # from x_star for biactive pairs, making IPOPT restoration failure likely.
    # Skip the TNLP when more than _BIACTIVE_THRESHOLD of pairs are biactive.
    _comp = float(np.max(np.abs(G_arr * H_arr))) if G_arr.size else 0.0
    # Upward slack matches _attach_per_pair_status so classification is consistent.
    _bi_tol = max(np.sqrt(_comp) * (1.0 + BIACTIVE_TOL_FLOOR), BIACTIVE_TOL_FLOOR)
    _n_biactive = int(np.sum((G_arr <= _bi_tol) & (H_arr <= _bi_tol)))
    _biactive_frac = _n_biactive / max(p.n_comp, 1)
    _BIACTIVE_THRESHOLD = 0.10
    if _biactive_frac > _BIACTIVE_THRESHOLD:
        return TNLPResult(
            x=np.asarray(result.x, dtype=float),
            obj=result.obj,
            status=-999,
            message=(
                f"{_n_biactive}/{p.n_comp} biactive pairs "
                f"(frac={_biactive_frac:.2f}>{_BIACTIVE_THRESHOLD:.2f}); "
                "TNLP skipped to avoid restoration failure"
            ),
            success=False,
            mult_comp_G=np.zeros(p.n_comp),
            mult_comp_H=np.zeros(p.n_comp),
            mult_ineq=None,
            mult_eq=None,
            kkt_residual=None,
            stationarity="skipped",
            n_iter=0,
            solve_time=0.0,
            active_set=(np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)),
            n_violations=_n_biactive,
        )

    I_G_active, I_H_active = _infer_active_set(
        G_arr,
        H_arr,
        mpcc_mult_G,
        mpcc_mult_H,
    )

    # Inject cleanup attributes that _build_cleanup_nlp requires when
    # the strategy doesn't support cleanup (e.g. DirectStrategy).
    for attr, default in [
        ("cleanup_max_iter", max_iter),
        ("cleanup_tol", None),
    ]:
        if not hasattr(strategy, attr):
            setattr(strategy, attr, default)

    nlp, _cache = strategy._build_cleanup_nlp(I_G_active, I_H_active)

    # Override max_iter; preserve _build_cleanup_nlp's tol (which applies a
    # 1e-6 floor for L-BFGS).  Only override tol when the caller passes one
    # explicitly — forcing the raw strategy tol (1e-8) onto an L-BFGS solve
    # causes IPOPT to spin without converging and return with unreliable
    # equality multipliers.
    nlp.add_option("max_iter", max_iter)
    if tol is not None:
        nlp.add_option("tol", float(tol))

    # Cold solve from the relaxation iterate (no dual warm-start).
    t0 = time.perf_counter()
    x_tnlp, info_tnlp = nlp.solve(np.asarray(result.x, dtype=float))
    elapsed = time.perf_counter() - t0

    status = int(info_tnlp["status"])
    success = status in (0, 1, 3)
    n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp
    n_iter_total = 0

    def _extract_mults(info):
        """Parse IPOPT info dict → (mult_g_raw, mult_ineq, mult_eq, mu_G_orig, mu_H_orig)."""
        raw = info.get("mult_g")
        m_ineq: Optional[np.ndarray] = None
        m_eq: Optional[np.ndarray] = None
        mG = np.zeros(n_c)
        mH = np.zeros(n_c)
        if raw is not None:
            lam = np.asarray(raw, dtype=float)
            if n_g > 0:
                m_ineq = lam[:n_g]
            if n_h > 0:
                m_eq = lam[n_g : n_g + n_h]
            mG = lam[n_g + n_h : n_g + n_h + n_c].copy()
            mH = lam[n_g + n_h + n_c : n_g + n_h + 2 * n_c].copy()
        mG_orig = mG * np.asarray(p.comp_G_scale, dtype=float) if p.comp_G_scale is not None else mG
        mH_orig = mH * np.asarray(p.comp_H_scale, dtype=float) if p.comp_H_scale is not None else mH
        return raw, m_ineq, m_eq, mG_orig, mH_orig

    def _kkt(x_sol, info, mG_orig, mH_orig):
        if info.get("mult_g") is None:
            return None
        proxy = SimpleNamespace(
            x=x_sol,
            mult_g=info["mult_g"],
            G=np.asarray(p.comp_G(x_sol)),
            H=np.asarray(p.comp_H(x_sol)),
            success=True,
        )
        return compute_kkt_residual(
            proxy, p,
            mpcc_mult_G=mG_orig,
            mpcc_mult_H=mH_orig,
            mult_x_L=info.get("mult_x_L"),
            mult_x_U=info.get("mult_x_U"),
        )

    mult_g_raw, mult_ineq, mult_eq, mu_G_orig, mu_H_orig = _extract_mults(info_tnlp)
    n_iter_total += int(getattr(nlp, "n_ipopt_iter", 0))

    # KKT residual and stationarity for initial solve.
    kkt_res: Optional[float] = _kkt(x_tnlp, info_tnlp, mu_G_orig, mu_H_orig) if success else None
    stat, n_viol = ("unknown", 0)
    if success:
        stat, n_viol = _classify_tnlp_stationarity(mu_G_orig, mu_H_orig, I_G_active, I_H_active)

    # Flip-and-retry: if the initial TNLP found "not S-stationary", some pairs
    # had the wrong side pinned (typical when the relaxation's kkt_res is large).
    # Swap those pairs to the other active side and re-solve once from x_tnlp.
    # Guard: only attempt when violations are a small fraction of pairs — large
    # violation counts (> 20 % of n_comp) indicate the relaxation iterate is far
    # from any MPCC stationary point and flipping just oscillates, doubling cost.
    _flip_threshold = max(5, int(0.20 * n_c))
    if success and stat == "W-stationary":
        I_G2, I_H2, n_flip = _flip_wrong_pairs(mu_G_orig, mu_H_orig, I_G_active, I_H_active)
        if 0 < n_flip <= _flip_threshold:
            nlp2, _cache2 = strategy._build_cleanup_nlp(I_G2, I_H2)
            nlp2.add_option("max_iter", max_iter)
            if tol is not None:
                nlp2.add_option("tol", float(tol))
            t1 = time.perf_counter()
            x2, info2 = nlp2.solve(np.asarray(x_tnlp, dtype=float))
            elapsed += time.perf_counter() - t1
            n_iter_total += int(getattr(nlp2, "n_ipopt_iter", 0))
            status2 = int(info2["status"])
            if status2 in (0, 1, 3):
                # Accept the flipped result — it may be S-stationary.
                x_tnlp, info_tnlp, status = x2, info2, status2
                I_G_active, I_H_active = I_G2, I_H2
                mult_g_raw, mult_ineq, mult_eq, mu_G_orig, mu_H_orig = _extract_mults(info2)
                kkt_res = _kkt(x_tnlp, info_tnlp, mu_G_orig, mu_H_orig)
                stat, n_viol = _classify_tnlp_stationarity(mu_G_orig, mu_H_orig, I_G_active, I_H_active)

    return TNLPResult(
        x=np.asarray(x_tnlp, dtype=float),
        obj=float(info_tnlp.get("obj_val", np.inf)),
        status=status,
        message=BaseStrategy._decode_msg(info_tnlp.get("status_msg", "")),
        success=success,
        mult_comp_G=np.asarray(mu_G_orig, dtype=float),
        mult_comp_H=np.asarray(mu_H_orig, dtype=float),
        mult_ineq=mult_ineq,
        mult_eq=mult_eq,
        kkt_residual=kkt_res,
        stationarity=stat,
        n_iter=n_iter_total,
        solve_time=elapsed,
        active_set=(I_G_active, I_H_active),
        n_violations=n_viol,
    )
