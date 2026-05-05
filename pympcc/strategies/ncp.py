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
from functools import partial

import numpy as np

from .._kernels import eval_weighted_union as _eval_wu
from .._reformulation import (
    _vu_pow_sigma,
    _vu_pow_sigma_prime,
    lookup_ncp,
)
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
        self._init_continuation_options(problem, ipopt_options, _DEFAULTS, kwargs)

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
        # Dense Jacobian path pre-allocation: full (m, n) output buffer
        # plus offsets for the (G, H, φ_ε) block rows.  Per-call writes
        # the standard block rows in-place from `_build_standard_constraints`,
        # then the comp blocks, avoiding a fresh np.vstack on every IPOPT
        # iteration.
        if not p.is_sparse:
            _m_total       = n_g + n_h + 3 * n_c
            _dense_jac_buf = np.empty((_m_total, p.n))
            _off_jg_dense  = n_g + n_h
            _off_jh_dense  = _off_jg_dense + n_c
            _off_gh_dense  = _off_jh_dense + n_c
            _dense_gh_view = _dense_jac_buf[_off_gh_dense:]
        else:
            _dense_jac_buf = None
            _off_jg_dense = _off_jh_dense = _off_gh_dense = 0
            _dense_gh_view = None

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
                # Write each block in-place into the pre-allocated (m, n)
                # buffer.  The φ_ε block sits inside the buffer via the
                # view captured at construction; weighted_row_sum writes
                # through that view.
                row_off = 0
                for block in jac_rows:
                    nrows = block.shape[0]
                    _dense_jac_buf[row_off:row_off + nrows] = block
                    row_off += nrows
                _dense_jac_buf[_off_jg_dense:_off_jh_dense] = JG
                _dense_jac_buf[_off_jh_dense:_off_gh_dense] = JH
                self._weighted_row_sum(alpha, JG, beta, JH, _dense_gh_view)
                return _dense_jac_buf

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
# Registry-driven shims for the dedicated NCP-variant entry points             #
# --------------------------------------------------------------------------- #
#
# The seven classes below all delegate to the (phi, grad) callables in
# :data:`pympcc._reformulation.NCP_REGISTRY` — the same registry that
# powers ``strategy="ncp"``.  They exist as standalone strategy names so
# users can write ``solve(..., strategy="smooth_min", lam=0.5)`` directly
# without going through ``ncp_function`` plumbing.  Variant-specific
# kwargs (``lam``, ``alpha``, ``gamma``) are validated here for parity
# with the legacy entry points; everything else flows through
# :class:`_SmoothNCPBase` unchanged.


class _RegistryShim(_SmoothNCPBase):
    """Bind a registry-supplied ``(phi, grad)`` pair to ``_SmoothNCPBase``.

    Subclasses set ``_NCP_NAME`` and (optionally) override
    :meth:`_extract_ncp_params` to validate / transform the
    variant-specific kwargs into a registry-overrides dict.
    """

    _NCP_NAME: str = ""

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        ncp_params = self._extract_ncp_params(kwargs)
        phi_fn, grad_fn, merged = lookup_ncp(self._NCP_NAME, ncp_params)
        self._phi_fn = partial(phi_fn, **merged) if merged else phi_fn
        self._grad_fn = partial(grad_fn, **merged) if merged else grad_fn
        super().__init__(problem, ipopt_options, **kwargs)

    def _extract_ncp_params(self, kwargs: dict) -> dict | None:
        """Pop variant-specific kwargs out of *kwargs* and validate them.

        Default: no extras.  Subclasses with one parameter override and
        return a single-key dict.
        """
        return None

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        return self._phi_fn(G, H, eps)

    def _phi_grad_coeffs(
        self, G: np.ndarray, H: np.ndarray, eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._grad_fn(G, H, eps)


class SmoothMinStrategy(_RegistryShim):
    r"""Smoothed min-NCP strategy.

    Replaces the complementarity conditions with
    ``φ_ε(G, H) = ½(G + H − √((G−H)² + 4ε²)) = 0``.  As ``ε → 0`` this
    converges to ``min(G, H) = 0``, equivalent to ``G ≥ 0, H ≥ 0,
    G·H = 0`` for non-negative ``G, H``.
    """

    name = "smooth_min"
    _NCP_NAME = "smooth_min"
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)


class ChenChenKanzowStrategy(_RegistryShim):
    r"""Chen-Chen-Kanzow NCP-function strategy.

    Replaces complementarity with the convex combination
    ``φ_{λ,ε}(G, H) = λ·φ_FB,ε(G, H) + (1 − λ)·G·H``.  ``λ = 1``
    recovers the standard smoothing strategy; ``λ = 0`` recovers the
    pure inner-product penalty (use ``ε > 0`` to keep the smoothing
    well-defined at zero).

    Parameters
    ----------
    lam : float
        Convex parameter ``λ ∈ (0, 1]`` (default ``0.5``).
    """

    name = "chen_chen_kanzow"
    _NCP_NAME = "chen_chen_kanzow"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "lam"})

    def _extract_ncp_params(self, kwargs: dict) -> dict | None:
        if "lam" not in kwargs:
            return None
        lam = float(kwargs.pop("lam"))
        if not (0.0 < lam <= 1.0):
            raise ValueError(f"ChenChenKanzow: lam must be in (0, 1], got {lam}")
        return {"lam": lam}


class KanzowSchwartzStrategy(_RegistryShim):
    r"""Kanzow-Schwartz NCP-function strategy.

    Replaces complementarity with
    ``φ_{λ,ε}(G, H) = G + H − √(G² + H² + 2λGH + ε²) = 0``,
    ``λ ∈ [0, 1)``.  ``λ = 0`` recovers the standard FB smoothing;
    as ``λ → 1`` the function approaches the smoothed 1-norm
    ``|G + H|``.

    Parameters
    ----------
    lam : float
        Parameter ``λ ∈ [0, 1)`` (default ``0.5``).
    """

    name = "kanzow_schwartz"
    _NCP_NAME = "kanzow_schwartz"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "lam"})

    def _extract_ncp_params(self, kwargs: dict) -> dict | None:
        if "lam" not in kwargs:
            return None
        lam = float(kwargs.pop("lam"))
        if not (0.0 <= lam < 1.0):
            raise ValueError(f"KanzowSchwartz: lam must be in [0, 1), got {lam}")
        return {"lam": lam}


class ChenMangasarianStrategy(_RegistryShim):
    r"""Chen-Mangasarian asymmetric NCP-function strategy.

    Replaces complementarity with
    ``φ_{α,ε}(G, H) = (G + H) − √(G² + H² − 2αGH + ε²) = 0``,
    ``α ∈ [0, 1]``.  ``α = 0`` recovers smoothed FB; ``α = 1``
    recovers ``2·min(G, H)``.  Matches NLPEC's ``CMxf`` / ``CMfx``
    rows.

    Parameters
    ----------
    alpha : float
        Asymmetry parameter ``α ∈ [0, 1]`` (default ``0.5``).
    """

    name = "chen_mangasarian"
    _NCP_NAME = "chen_mangasarian"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "alpha"})

    def _extract_ncp_params(self, kwargs: dict) -> dict | None:
        if "alpha" not in kwargs:
            return None
        alpha = float(kwargs.pop("alpha"))
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"ChenMangasarian: alpha must be in [0, 1], got {alpha}")
        return {"alpha": alpha}


class BillupsStrategy(_RegistryShim):
    r"""Billups composite NCP-function strategy.

    Composes smoothed Fischer-Burmeister with a positive-part penalty:
    ``φ_{γ,ε}(G, H) = φ_FB,ε(G, H) − γ · G₊_ε · H₊_ε``.  At
    ``γ = 0`` this reduces to FB; for ``γ > 0`` the extra penalty
    pulls iterates harder onto the complementary cone.  Matches
    NLPEC's ``Bill`` / ``fBill`` rows.

    Parameters
    ----------
    gamma : float
        Penalty weight ``γ ≥ 0`` (default ``0.1``).
    """

    name = "billups"
    _NCP_NAME = "billups"
    _VALID_OPTIONS: frozenset = frozenset({*_DEFAULTS, "gamma"})

    def _extract_ncp_params(self, kwargs: dict) -> dict | None:
        if "gamma" not in kwargs:
            return None
        gamma = float(kwargs.pop("gamma"))
        if gamma < 0.0:
            raise ValueError(f"Billups: gamma must be ≥ 0, got {gamma}")
        return {"gamma": gamma}


class VeelkenUlbrichPowStrategy(_RegistryShim):
    r"""Smooth-min strategy with Veelken-Ulbrich piecewise-polynomial smoothing.

    ``φ_ε(G, H) = ½(G + H − σ_ε^pow(G − H)) = 0`` where ``σ_ε^pow`` is
    the unique even degree-4 polynomial matching ``|t|`` and its first
    two derivatives at ``|t| = ε``.  C² globally; approaches ``|·|``
    from above as ``ε → 0``.  Matches NLPEC's ``fVUpow``.
    """

    name = "veelken_ulbrich_pow"
    _NCP_NAME = "veelken_ulbrich_pow"
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    # Module-level helpers re-exposed as static methods so the existing
    # σ / σ' unit tests can probe the smoothing without instantiating a
    # strategy.  Source of truth lives in :mod:`pympcc._reformulation`.
    _sigma = staticmethod(_vu_pow_sigma)
    _sigma_prime = staticmethod(_vu_pow_sigma_prime)


class VeelkenUlbrichSinStrategy(_RegistryShim):
    r"""Smooth-min strategy with Veelken-Ulbrich arctan-based smoothing.

    ``φ_ε(G, H) = ½(G + H − σ_ε^sin(G − H)) = 0`` where
    ``σ_ε^sin(t) = (2t/π)·arctan(πt/(2ε))``.  C^∞ globally;
    ``σ_ε^sin(t) → |t|`` as ``ε → 0`` and ``σ_ε^sin(0) = 0``
    exactly.  Matches NLPEC's ``fVUsin`` in spirit.
    """

    name = "veelken_ulbrich_sin"
    _NCP_NAME = "veelken_ulbrich_sin"
    _VALID_OPTIONS: frozenset = frozenset(_DEFAULTS)

    # Test-helper static methods — the registry's
    # ``phi_veelken_ulbrich_sin`` / ``grad_veelken_ulbrich_sin`` inline
    # the same expressions.
    @staticmethod
    def _sigma(t: np.ndarray, eps: float) -> np.ndarray:
        return (2.0 * t / np.pi) * np.arctan(np.pi * t / (2.0 * eps))

    @staticmethod
    def _sigma_prime(t: np.ndarray, eps: float) -> np.ndarray:
        u = np.pi * t / (2.0 * eps)
        return (2.0 / np.pi) * np.arctan(u) + t / (eps * (1.0 + u * u))
