"""Smooth NCP-function reformulation strategies for MPCC.

Strategy classes that share the ε-continuation harness from
:mod:`pympcc.strategies.smoothing` but differ in the NCP function used to
replace complementarity:

* :class:`SmoothMinStrategy` — smoothed min-NCP
  ``φ_ε(G,H) = ½(G + H − √((G−H)² + 4ε²))``
* :class:`ChenChenKanzowStrategy` — convex combination of Fischer-Burmeister
  and inner-product: ``φ_{λ,ε}(G,H) = λ·φ_FB,ε(G,H) + (1−λ)·G·H``
* :class:`KanzowSchwartzStrategy` — one-parameter family interpolating
  between FB (λ=0) and a modified FB:
  ``φ_{λ,ε}(G,H) = G + H − √(G² + H² + 2λGH + ε²)``,  λ ∈ [0, 1)
* :class:`ChenMangasarianStrategy` — α-asymmetric FB↔min interpolation:
  ``φ_{α,ε}(G,H) = (G + H) − √(G² + H² − 2αGH + ε²)``,  α ∈ [0, 1]
* :class:`BillupsStrategy` — Fischer-Burmeister minus a positive-part penalty:
  ``φ_{γ,ε}(G,H) = φ_FB,ε(G,H) − γ·G₊·H₊``
* :class:`VeelkenUlbrichPowStrategy` — smooth-min via piecewise-polynomial
  C² smoothing of ``|·|``.
* :class:`VeelkenUlbrichSinStrategy` — smooth-min via arctan-based C^∞
  smoothing of ``|·|``.

All follow the same constraint layout as :class:`SmoothingStrategy`::

    [g(x) ≤ 0,  h(x) = 0,  G(x) ≥ 0,  H(x) ≥ 0,  φ_ε(G,H) = 0]

MPCC multipliers are recovered via the chain-rule identity:
``μ_G = λ_G + α ⊙ λ_φ``,  ``μ_H = λ_H + β ⊙ λ_φ``
where ``α = ∂φ/∂G``, ``β = ∂φ/∂H``.
"""
from __future__ import annotations

import abc

import numpy as np

from .._kernels import eval_weighted_union as _eval_wu
from .._stationarity import classify_stationarity, compute_kkt_residual
from ..result import IterationInfo, MPCCResult
from ._base import CLEANUP_DEFAULTS, SAFEGUARD_DEFAULTS, BaseStrategy

_INF = 2e19

_DEFAULTS = dict(epsilon_0=1.0, reduction=0.1, max_iter=20, epsilon_min=1e-8,
                 dual_warmstart=True, comp_tol=None,
                 **SAFEGUARD_DEFAULTS, **CLEANUP_DEFAULTS)


class _SmoothNCPBase(BaseStrategy, abc.ABC):
    """Shared ε-continuation harness for smooth NCP-function strategies.

    Subclasses must implement :meth:`_phi` and :meth:`_phi_grad_coeffs`.
    """

    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        super().__init__(problem, ipopt_options,
                         backend=kwargs.pop("backend", "ipopt"),
                         solver_options=kwargs.pop("solver_options", None),
                         callback=kwargs.pop("callback", None),
                         inner_callback=kwargs.pop("inner_callback", None),
                         time_limit=kwargs.pop("time_limit", None))
        opts = {**_DEFAULTS, **kwargs}
        opts = self._maybe_resolve_auto_epsilon_0(opts)
        self._validate_continuation_options(
            epsilon_0=opts["epsilon_0"],
            reduction=opts["reduction"],
            max_iter=opts["max_iter"],
            epsilon_min=opts["epsilon_min"],
            comp_tol=opts["comp_tol"],
        )
        self.epsilon_0: float = opts["epsilon_0"]
        self.reduction: float = opts["reduction"]
        self.max_iter: int = opts["max_iter"]
        self.epsilon_min: float = opts["epsilon_min"]
        self.dual_warmstart: bool = bool(opts["dual_warmstart"])
        self.comp_tol: float | None = opts["comp_tol"]
        self._init_safeguards(opts)
        self._init_cleanup(opts, user_kwargs=kwargs)

    @abc.abstractmethod
    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        """NCP function values — shape ``(n_comp,)``."""

    @abc.abstractmethod
    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(α, β)`` — shape ``(n_comp,)`` each.

        The Jacobian of ``φ`` w.r.t. ``x`` is ``α[:,None]*JG + β[:,None]*JH``.
        MPCC multipliers: ``μ_G = λ_G + α⊙λ_φ``, ``μ_H = λ_H + β⊙λ_φ``.
        """

    def solve(self) -> MPCCResult:
        p = self.problem
        n_g, n_h, n_c = p.n_ineq, p.n_eq, p.n_comp

        cl = np.concatenate([
            np.full(n_g, -_INF),
            np.zeros(n_h),
            np.zeros(n_c),        # G ≥ 0
            np.zeros(n_c),        # H ≥ 0
            np.zeros(n_c),        # φ_ε = 0
        ])
        cu = np.concatenate([
            np.zeros(n_g),
            np.zeros(n_h),
            np.full(n_c, _INF),
            np.full(n_c, _INF),
            np.zeros(n_c),
        ])

        eps_ref = [self.epsilon_0]
        cache = self._new_callback_cache()

        union_maps = self._make_union_maps(
            p.comp_G_jacobian_sparsity, p.comp_H_jacobian_sparsity
        )
        gh_sp = union_maps[0] if union_maps is not None else None

        jac_structure = self._make_jac_structure([
            (n_g, p.ineq_jacobian_sparsity),
            (n_h, p.eq_jacobian_sparsity),
            (n_c, p.comp_G_jacobian_sparsity),
            (n_c, p.comp_H_jacobian_sparsity),
            (n_c, gh_sp),
        ]) if p.is_sparse else None

        _union_buf: np.ndarray | None
        if p.is_sparse and gh_sp is not None:
            assert jac_structure is not None
            assert (p.comp_G_jacobian_sparsity is not None
                    and p.comp_H_jacobian_sparsity is not None)
            _nnz_G   = len(p.comp_G_jacobian_sparsity[0])
            _nnz_H   = len(p.comp_H_jacobian_sparsity[0])
            _nnz_gh  = len(gh_sp[0])
            _nnz_tot = len(jac_structure[0])
            _off_G   = _nnz_tot - _nnz_G - _nnz_H - _nnz_gh
            _off_H   = _off_G + _nnz_G
            _off_gh  = _off_H + _nnz_H
            _jac_flat_buf = np.empty(_nnz_tot)
            _union_buf    = _jac_flat_buf[_off_gh:]
        else:
            _jac_flat_buf = None
            _union_buf    = np.empty(len(gh_sp[0])) if gh_sp is not None else None
        _gh_buf = np.empty((n_c, p.n)) if not p.is_sparse else None

        def constraints(x):
            std_parts = self._eval_standard_con_values(x, cache)
            G, H = self._eval_comp_values(x, cache)
            phi = self._phi(G, H, eps_ref[0])
            return np.concatenate([*std_parts, G, H, phi])

        if p.is_sparse:
            def jacobian(x):
                G, H = self._eval_comp_values(x, cache)
                vG_raw, vH_raw = self._eval_comp_jac_raw(x, cache)
                v_G = vG_raw.ravel() if vG_raw.ndim == 2 else vG_raw
                v_H = vH_raw.ravel() if vH_raw.ndim == 2 else vH_raw
                alpha, beta = self._phi_grad_coeffs(G, H, eps_ref[0])
                if union_maps is not None:
                    (r_u, _), map1, map2 = union_maps
                    _eval_wu(v_G, v_H, alpha, beta, r_u, map1, map2, _union_buf)
                    std_flat = self._build_std_jac_flat(x, cache)
                    _jac_flat_buf[:_off_G]        = std_flat
                    _jac_flat_buf[_off_G:_off_H]  = v_G
                    _jac_flat_buf[_off_H:_off_gh] = v_H
                    return _jac_flat_buf
                else:
                    std_flat = self._build_std_jac_flat(x, cache)
                    JG = (vG_raw if vG_raw.ndim == 2
                          else self._to_dense_block(vG_raw, p.comp_G_jacobian_sparsity, p.n_comp, p.n))
                    JH = (vH_raw if vH_raw.ndim == 2
                          else self._to_dense_block(vH_raw, p.comp_H_jacobian_sparsity, p.n_comp, p.n))
                    phi_vals = (alpha[:, None] * JG + beta[:, None] * JH).ravel()
                    return np.concatenate([std_flat, v_G, v_H, phi_vals])
        else:
            def jacobian(x):
                _, jac_rows = self._build_standard_constraints(x, cache)
                G, H, JG, JH = self._build_comp_jacobians(x, cache)
                alpha, beta = self._phi_grad_coeffs(G, H, eps_ref[0])
                self._weighted_row_sum(alpha, JG, beta, JH, _gh_buf)
                jac_rows.extend([JG, JH, _gh_buf])
                return np.vstack(jac_rows)

        nlp = self._build_nlp(cl, cu, constraints, jacobian, jac_structure)

        def make_iteration(eps, x, last_info, n_ipopt_iter, iter_time):
            G, H = self._eval_comp_values(x, cache)
            _mg = last_info.get("mult_g")
            if _mg is not None and len(_mg):
                _off = n_g + n_h
                _lam_G   = _mg[_off           : _off + n_c]
                _lam_H   = _mg[_off + n_c     : _off + 2 * n_c]
                _lam_phi = _mg[_off + 2 * n_c : _off + 3 * n_c]
                _alpha, _beta = self._phi_grad_coeffs(G, H, eps)
                _kkt = self._compute_kkt_iter(
                    x, _mg,
                    mpcc_mult_G=_lam_G + _alpha * _lam_phi,
                    mpcc_mult_H=_lam_H + _beta  * _lam_phi,
                    mult_x_L=last_info.get("mult_x_L"),
                    mult_x_U=last_info.get("mult_x_U"),
                )
            else:
                _kkt = None
            return IterationInfo(
                epsilon=eps,
                x=x.copy(),
                obj=float(last_info["obj_val"]),
                status=last_info["status"],
                message=self._decode_msg(last_info["status_msg"]),
                comp_residual=float(np.max(np.abs(G * H))),
                comp_residual_mean=float(np.mean(np.abs(G * H))),
                n_ipopt_iter=n_ipopt_iter,
                iter_time=iter_time,
                kkt_residual=_kkt,
            )

        x, last_info, total_time, history = self._run_epsilon_continuation(
            nlp, p.x0, eps_ref, make_iteration
        )

        G, H = self._eval_comp_values(x, cache)
        alpha, beta = self._phi_grad_coeffs(G, H, eps_ref[0])

        result = MPCCResult(
            x=x,
            obj=float(last_info["obj_val"]),
            status=last_info["status"],
            message=self._decode_msg(last_info["status_msg"]),
            G=G,
            H=H,
            comp_residual=float(np.max(np.abs(G * H))),
            comp_residual_mean=float(np.mean(np.abs(G * H))),
            success=last_info["status"] in (0, 1, 3),
            strategy=self.name,
            solve_time=total_time,
            history=history,
            mult_g=last_info.get("mult_g"),
        )
        result.stationarity = classify_stationarity(result, self.problem)

        mg = last_info.get("mult_g")
        if mg is not None and len(mg):
            _off = n_g + n_h
            lam_G   = mg[_off           : _off + n_c]
            lam_H   = mg[_off + n_c     : _off + 2 * n_c]
            lam_phi = mg[_off + 2 * n_c : _off + 3 * n_c]
            mpcc_mult_G = lam_G + alpha * lam_phi
            mpcc_mult_H = lam_H + beta  * lam_phi
            result.kkt_residual = compute_kkt_residual(
                result, self.problem,
                mpcc_mult_G=mpcc_mult_G,
                mpcc_mult_H=mpcc_mult_H,
                mult_x_L=last_info.get("mult_x_L"),
                mult_x_U=last_info.get("mult_x_U"),
            )
        else:
            mpcc_mult_G = None
            mpcc_mult_H = None
        result = self._maybe_run_cleanup(
            result, last_info, x, mpcc_mult_G, mpcc_mult_H,
        )
        return result


# --------------------------------------------------------------------------- #
# Smooth-min NCP                                                               #
# --------------------------------------------------------------------------- #

class SmoothMinStrategy(_SmoothNCPBase):
    """Smoothed min-NCP strategy.

    Replaces the complementarity conditions with::

        φ_ε(G, H) = ½(G + H − √((G−H)² + 4ε²)) = 0

    As ε→0, φ_0(a,b) = min(a,b) = 0, which is equivalent to
    a≥0, b≥0, a·b = 0 for non-negative a, b.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "smooth_min"

    @staticmethod
    def _phi(G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        sq = np.sqrt((G - H) ** 2 + 4.0 * eps ** 2)
        return 0.5 * (G + H - sq)

    @staticmethod
    def _phi_grad_coeffs(
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sq = np.sqrt((G - H) ** 2 + 4.0 * eps ** 2)
        diff = (G - H) / sq
        alpha = 0.5 * (1.0 - diff)
        beta  = 0.5 * (1.0 + diff)
        return alpha, beta


# --------------------------------------------------------------------------- #
# Chen-Chen-Kanzow (CCK)                                                       #
# --------------------------------------------------------------------------- #

class ChenChenKanzowStrategy(_SmoothNCPBase):
    r"""Chen-Chen-Kanzow NCP-function strategy.

    Replaces the complementarity conditions with a convex combination of the
    smoothed Fischer-Burmeister function and the inner-product penalty::

        φ_{λ,ε}(G, H) = λ·φ_FB,ε(G,H) + (1−λ)·G·H = 0

    where ``φ_FB,ε(G,H) = G + H − √(G² + H² + ε²)``.
    At ``λ=1`` this reduces to the standard smoothing strategy.
    At ``λ=0`` it reduces to ``G·H = 0`` (not smooth at zero; use ε>0).

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    lam : float
        Convex parameter λ ∈ (0, 1] (default 0.5).
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "chen_chen_kanzow"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "lam"})

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        lam = float(kwargs.pop("lam", 0.5))
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"ChenChenKanzow: lam must be in (0, 1], got {lam}")
        super().__init__(problem, ipopt_options, **kwargs)
        self._lam = lam

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        lam = self._lam
        r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
        fb = G + H - r
        return lam * fb + (1.0 - lam) * G * H

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lam = self._lam
        r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
        alpha = lam * (1.0 - G / r) + (1.0 - lam) * H
        beta  = lam * (1.0 - H / r) + (1.0 - lam) * G
        return alpha, beta


# --------------------------------------------------------------------------- #
# Kanzow-Schwartz (KS)                                                         #
# --------------------------------------------------------------------------- #

class KanzowSchwartzStrategy(_SmoothNCPBase):
    r"""Kanzow-Schwartz NCP-function strategy.

    Replaces the complementarity conditions with::

        φ_{λ,ε}(G, H) = G + H − √(G² + H² + 2λGH + ε²) = 0,  λ ∈ [0, 1)

    At ``λ=0`` this is the standard Fischer-Burmeister smoothing.
    As ``λ→1`` the function approaches the smoothed 1-norm ``|G+H|``.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    lam : float
        Parameter λ ∈ [0, 1) (default 0.5).
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "kanzow_schwartz"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "lam"})

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        lam = float(kwargs.pop("lam", 0.5))
        if not (0.0 <= lam < 1.0):
            raise ValueError(f"KanzowSchwartz: lam must be in [0, 1), got {lam}")
        super().__init__(problem, ipopt_options, **kwargs)
        self._lam = lam

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        lam = self._lam
        sq = np.sqrt(G ** 2 + H ** 2 + 2.0 * lam * G * H + eps ** 2)
        return G + H - sq

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        lam = self._lam
        sq = np.sqrt(G ** 2 + H ** 2 + 2.0 * lam * G * H + eps ** 2)
        alpha = 1.0 - (G + lam * H) / sq
        beta  = 1.0 - (H + lam * G) / sq
        return alpha, beta


# --------------------------------------------------------------------------- #
# Chen-Mangasarian (CM)                                                        #
# --------------------------------------------------------------------------- #

class ChenMangasarianStrategy(_SmoothNCPBase):
    r"""Chen-Mangasarian asymmetric NCP-function strategy.

    Replaces the complementarity conditions with the one-parameter family

    .. math::

        \varphi_{\alpha,\varepsilon}(G, H)
            = (G + H) - \sqrt{G^2 + H^2 - 2\alpha\,G\,H + \varepsilon^2} = 0,
            \qquad \alpha \in [0, 1].

    The radicand :math:`G^2 + H^2 - 2\alpha G H` is non-negative for any
    real ``G, H`` when ``α ∈ [0, 1]`` (it equals
    ``(1-α)(G²+H²) + α(G-H)²``).  Limits:

    * ``α = 0`` recovers smoothed Fischer-Burmeister.
    * ``α = 1`` recovers ``(G+H) − |G-H| = 2·min(G, H)`` (the min-NCP).

    For a general ``α`` the function is a smooth FB↔min interpolation,
    matching NLPEC's ``CMxf`` / ``CMfx`` rows.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    alpha : float
        Asymmetry parameter ``α ∈ [0, 1]`` (default 0.5).
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "chen_mangasarian"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "alpha"})

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        alpha = float(kwargs.pop("alpha", 0.5))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"ChenMangasarian: alpha must be in [0, 1], got {alpha}")
        super().__init__(problem, ipopt_options, **kwargs)
        self._alpha = alpha

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        a = self._alpha
        r = np.sqrt(G ** 2 + H ** 2 - 2.0 * a * G * H + eps ** 2)
        return G + H - r

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        a = self._alpha
        r = np.sqrt(G ** 2 + H ** 2 - 2.0 * a * G * H + eps ** 2)
        alpha_c = 1.0 - (G - a * H) / r
        beta_c  = 1.0 - (H - a * G) / r
        return alpha_c, beta_c


# --------------------------------------------------------------------------- #
# Billups composite                                                            #
# --------------------------------------------------------------------------- #

class BillupsStrategy(_SmoothNCPBase):
    r"""Billups composite NCP-function strategy.

    Composes smoothed Fischer-Burmeister with a positive-part penalty term::

        φ_{γ,ε}(G, H) = (G + H) − √(G² + H² + ε²) − γ · G₊_ε · H₊_ε

    where ``t₊_ε = ½(t + √(t² + ε²))`` is the standard CHKS smoothed
    positive-part.  At ``γ = 0`` this reduces to FB; for ``γ > 0`` the extra
    penalty drives feasible iterates harder onto the complementary cone.
    Matches NLPEC's ``Bill`` / ``fBill`` rows.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    gamma : float
        Penalty weight ``γ ≥ 0`` (default ``0.1``).  Larger values pull
        iterates more aggressively toward the complementary cone but can
        introduce spurious infeasibility for the smoothed φ at finite
        ``ε``; ``0.1`` is a safe default that keeps the FB term in
        control.
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "billups"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "gamma"})

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        gamma = float(kwargs.pop("gamma", 0.1))
        if gamma < 0.0:
            raise ValueError(f"Billups: gamma must be ≥ 0, got {gamma}")
        super().__init__(problem, ipopt_options, **kwargs)
        self._gamma = gamma

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        g = self._gamma
        rg = np.sqrt(G ** 2 + eps ** 2)
        rh = np.sqrt(H ** 2 + eps ** 2)
        Gp = 0.5 * (G + rg)
        Hp = 0.5 * (H + rh)
        r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
        return G + H - r - g * Gp * Hp

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        g = self._gamma
        rg = np.sqrt(G ** 2 + eps ** 2)
        rh = np.sqrt(H ** 2 + eps ** 2)
        Gp = 0.5 * (G + rg)
        Hp = 0.5 * (H + rh)
        dGp = 0.5 * (1.0 + G / rg)
        dHp = 0.5 * (1.0 + H / rh)
        r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
        alpha = 1.0 - G / r - g * dGp * Hp
        beta  = 1.0 - H / r - g * Gp * dHp
        return alpha, beta


# --------------------------------------------------------------------------- #
# Veelken-Ulbrich smoothings of the min-NCP                                    #
# --------------------------------------------------------------------------- #

def _smooth_min_phi_from_sigma(G, H, sigma_vals):
    """Common smooth-min body: ``φ = ½(G + H − σ_ε(G − H))``."""
    return 0.5 * (G + H - sigma_vals)


def _smooth_min_grad_from_sigma_prime(sigma_prime):
    """Common smooth-min gradient: ``α = ½(1 − σ'(G−H))``,
    ``β = ½(1 + σ'(G−H))``."""
    return 0.5 * (1.0 - sigma_prime), 0.5 * (1.0 + sigma_prime)


class VeelkenUlbrichPowStrategy(_SmoothNCPBase):
    r"""Smooth-min strategy with Veelken-Ulbrich piecewise-polynomial smoothing.

    Replaces complementarity with::

        φ_ε(G, H) = ½(G + H − σ_ε^pow(G − H)) = 0,

    where ``σ_ε^pow`` is the unique even polynomial of degree 4 that matches
    ``|t|`` and its first two derivatives at ``|t| = ε``::

        σ_ε^pow(t) = |t|                                          if |t| ≥ ε,
                   = 3ε/8 + (3/(4ε))·t² − (1/(8ε³))·t⁴            if |t| < ε.

    This is C² globally (matching σ'' = 0 on both sides at ``|t| = ε``)
    and approaches ``|·|`` from above as ``ε → 0``.  Retains MFCQ-friendly
    behaviour near degeneracy because the smoothing is strict at the
    transition.  Matches NLPEC's ``fVUpow``.

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "veelken_ulbrich_pow"

    @staticmethod
    def _sigma(t: np.ndarray, eps: float) -> np.ndarray:
        abst = np.abs(t)
        inner = 3.0 * eps / 8.0 + (3.0 / (4.0 * eps)) * t * t \
            - (1.0 / (8.0 * eps ** 3)) * t ** 4
        return np.where(abst >= eps, abst, inner)

    @staticmethod
    def _sigma_prime(t: np.ndarray, eps: float) -> np.ndarray:
        abst = np.abs(t)
        inner = (3.0 / (2.0 * eps)) * t - (1.0 / (2.0 * eps ** 3)) * t ** 3
        return np.where(abst >= eps, np.sign(t), inner)

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        return _smooth_min_phi_from_sigma(G, H, self._sigma(G - H, eps))

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _smooth_min_grad_from_sigma_prime(self._sigma_prime(G - H, eps))


class VeelkenUlbrichSinStrategy(_SmoothNCPBase):
    r"""Smooth-min strategy with Veelken-Ulbrich arctan-based smoothing.

    Replaces complementarity with::

        φ_ε(G, H) = ½(G + H − σ_ε^sin(G − H)) = 0,

    where ``σ_ε^sin`` is the C^∞ approximation of ``|·|``::

        σ_ε^sin(t) = (2t/π) · arctan(π t / (2ε)).

    Asymptotics: ``σ_ε^sin(t) → |t|`` as ``ε → 0`` (since
    ``(2/π)·arctan(πt/(2ε)) → sign(t)``), and ``σ_ε^sin(0) = 0`` exactly
    so the smoothed-min vanishes at ``G = H = 0``.  The function and all
    derivatives are smooth, which can help second-order solvers.  Matches
    NLPEC's ``fVUsin`` in spirit; the precise transcendental form is
    self-contained (no piecewise transition).

    Parameters (passed as **strategy_options** to :class:`MPCCSolver`)
    ------------------------------------------------------------------
    epsilon_0, reduction, max_iter, epsilon_min : same as SmoothingStrategy.
    """

    name = "veelken_ulbrich_sin"

    @staticmethod
    def _sigma(t: np.ndarray, eps: float) -> np.ndarray:
        return (2.0 * t / np.pi) * np.arctan(np.pi * t / (2.0 * eps))

    @staticmethod
    def _sigma_prime(t: np.ndarray, eps: float) -> np.ndarray:
        u = np.pi * t / (2.0 * eps)
        return (2.0 / np.pi) * np.arctan(u) + t / (eps * (1.0 + u * u))

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        return _smooth_min_phi_from_sigma(G, H, self._sigma(G - H, eps))

    def _phi_grad_coeffs(
        self,
        G: np.ndarray,
        H: np.ndarray,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _smooth_min_grad_from_sigma_prime(self._sigma_prime(G - H, eps))
