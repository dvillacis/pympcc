"""Smooth NCP-function reformulation strategies for MPCC.

Three strategy classes that share the ε-continuation harness from
:mod:`pympcc.strategies.smoothing` but differ in the NCP function used to
replace complementarity:

* :class:`SmoothMinStrategy` — smoothed min-NCP
  ``φ_ε(G,H) = ½(G + H − √((G−H)² + 4ε²))``
* :class:`ChenChenKanzowStrategy` — convex combination of Fischer-Burmeister
  and inner-product: ``φ_{λ,ε}(G,H) = λ·φ_FB,ε(G,H) + (1−λ)·G·H``
* :class:`KanzowSchwartzStrategy` — one-parameter family interpolating
  between FB (λ=0) and a modified FB:
  ``φ_{λ,ε}(G,H) = G + H − √(G² + H² + 2λGH + ε²)``,  λ ∈ [0, 1)

All three follow the same constraint layout as :class:`SmoothingStrategy`::

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
                         callback=kwargs.pop("callback", None))
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
            _off = n_g + n_h
            _lam_G   = last_info["mult_g"][_off           : _off + n_c]
            _lam_H   = last_info["mult_g"][_off + n_c     : _off + 2 * n_c]
            _lam_phi = last_info["mult_g"][_off + 2 * n_c : _off + 3 * n_c]
            _alpha, _beta = self._phi_grad_coeffs(G, H, eps)
            _kkt = self._compute_kkt_iter(
                x, last_info["mult_g"],
                mpcc_mult_G=_lam_G + _alpha * _lam_phi,
                mpcc_mult_H=_lam_H + _beta  * _lam_phi,
                mult_x_L=last_info.get("mult_x_L"),
                mult_x_U=last_info.get("mult_x_U"),
            )
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

        _off = n_g + n_h
        lam_G   = last_info["mult_g"][_off           : _off + n_c]
        lam_H   = last_info["mult_g"][_off + n_c     : _off + 2 * n_c]
        lam_phi = last_info["mult_g"][_off + 2 * n_c : _off + 3 * n_c]
        mpcc_mult_G = lam_G + alpha * lam_phi
        mpcc_mult_H = lam_H + beta  * lam_phi
        result.kkt_residual = compute_kkt_residual(
            result, self.problem,
            mpcc_mult_G=mpcc_mult_G,
            mpcc_mult_H=mpcc_mult_H,
            mult_x_L=last_info.get("mult_x_L"),
            mult_x_U=last_info.get("mult_x_U"),
        )
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
