"""NCP-function registry for the unified reformulation strategy (§6.6).

Each entry is a self-contained ``(phi, grad)`` pair with default
parameters.  The classical strategy classes in
:mod:`pympcc.strategies.ncp` ship the same formulas as bound methods on
:class:`_SmoothNCPBase` subclasses; this module re-exposes them as plain
callables keyed by name so they can compose orthogonally with other
reformulation modifiers (constraint type, slack lifting, ...).

Conventions
-----------
``phi(G, H, eps, **params) -> ndarray of shape (n_comp,)``
    NCP function value.  At ``eps = 0`` and ``G, H ≥ 0`` it is zero iff
    ``G·H = 0`` (i.e. the complementary cone).

``grad(G, H, eps, **params) -> (alpha, beta)`` — each ``ndarray`` of shape
    ``(n_comp,)``.  The Jacobian of ``phi`` with respect to ``x`` is
    ``alpha[:, None] * JG + beta[:, None] * JH``.

The registry is the single source of truth used by
:class:`NCPReformulationStrategy`.  The legacy strategy classes remain
in place for backward compatibility and as the canonical reference
implementation; the formulas here mirror them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Inner-product NCP — the Scholtes / direct relaxation kernel                  #
# --------------------------------------------------------------------------- #

def phi_inner_product(G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
    return G * H


def grad_inner_product(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    return H.copy(), G.copy()


# --------------------------------------------------------------------------- #
# Fischer-Burmeister                                                           #
# --------------------------------------------------------------------------- #

def phi_fischer_burmeister(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> np.ndarray:
    return G + H - np.sqrt(G ** 2 + H ** 2 + eps ** 2)


def grad_fischer_burmeister(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
    return 1.0 - G / r, 1.0 - H / r


# --------------------------------------------------------------------------- #
# Smooth-min                                                                   #
# --------------------------------------------------------------------------- #

def phi_smooth_min(G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
    sq = np.sqrt((G - H) ** 2 + 4.0 * eps ** 2)
    return 0.5 * (G + H - sq)


def grad_smooth_min(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    sq = np.sqrt((G - H) ** 2 + 4.0 * eps ** 2)
    diff = (G - H) / sq
    return 0.5 * (1.0 - diff), 0.5 * (1.0 + diff)


# --------------------------------------------------------------------------- #
# Chen-Chen-Kanzow                                                             #
# --------------------------------------------------------------------------- #

def phi_chen_chen_kanzow(
    G: np.ndarray, H: np.ndarray, eps: float, *, lam: float = 0.5,
) -> np.ndarray:
    r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
    return lam * (G + H - r) + (1.0 - lam) * G * H


def grad_chen_chen_kanzow(
    G: np.ndarray, H: np.ndarray, eps: float, *, lam: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
    alpha = lam * (1.0 - G / r) + (1.0 - lam) * H
    beta  = lam * (1.0 - H / r) + (1.0 - lam) * G
    return alpha, beta


# --------------------------------------------------------------------------- #
# Kanzow-Schwartz                                                              #
# --------------------------------------------------------------------------- #

def phi_kanzow_schwartz(
    G: np.ndarray, H: np.ndarray, eps: float, *, lam: float = 0.5,
) -> np.ndarray:
    sq = np.sqrt(G ** 2 + H ** 2 + 2.0 * lam * G * H + eps ** 2)
    return G + H - sq


def grad_kanzow_schwartz(
    G: np.ndarray, H: np.ndarray, eps: float, *, lam: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    sq = np.sqrt(G ** 2 + H ** 2 + 2.0 * lam * G * H + eps ** 2)
    return 1.0 - (G + lam * H) / sq, 1.0 - (H + lam * G) / sq


# --------------------------------------------------------------------------- #
# Chen-Mangasarian                                                             #
# --------------------------------------------------------------------------- #

def phi_chen_mangasarian(
    G: np.ndarray, H: np.ndarray, eps: float, *, alpha: float = 0.5,
) -> np.ndarray:
    r = np.sqrt(G ** 2 + H ** 2 - 2.0 * alpha * G * H + eps ** 2)
    return G + H - r


def grad_chen_mangasarian(
    G: np.ndarray, H: np.ndarray, eps: float, *, alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(G ** 2 + H ** 2 - 2.0 * alpha * G * H + eps ** 2)
    return 1.0 - (G - alpha * H) / r, 1.0 - (H - alpha * G) / r


# --------------------------------------------------------------------------- #
# Billups composite                                                            #
# --------------------------------------------------------------------------- #

def phi_billups(
    G: np.ndarray, H: np.ndarray, eps: float, *, gamma: float = 0.1,
) -> np.ndarray:
    rg = np.sqrt(G ** 2 + eps ** 2)
    rh = np.sqrt(H ** 2 + eps ** 2)
    Gp = 0.5 * (G + rg)
    Hp = 0.5 * (H + rh)
    r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
    return G + H - r - gamma * Gp * Hp


def grad_billups(
    G: np.ndarray, H: np.ndarray, eps: float, *, gamma: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    rg = np.sqrt(G ** 2 + eps ** 2)
    rh = np.sqrt(H ** 2 + eps ** 2)
    Gp = 0.5 * (G + rg)
    Hp = 0.5 * (H + rh)
    dGp = 0.5 * (1.0 + G / rg)
    dHp = 0.5 * (1.0 + H / rh)
    r = np.sqrt(G ** 2 + H ** 2 + eps ** 2)
    return 1.0 - G / r - gamma * dGp * Hp, 1.0 - H / r - gamma * Gp * dHp


# --------------------------------------------------------------------------- #
# Veelken-Ulbrich pow                                                          #
# --------------------------------------------------------------------------- #

def _vu_pow_sigma(t: np.ndarray, eps: float) -> np.ndarray:
    abst = np.abs(t)
    inner = (3.0 * eps / 8.0
             + (3.0 / (4.0 * eps)) * t * t
             - (1.0 / (8.0 * eps ** 3)) * t ** 4)
    return np.where(abst >= eps, abst, inner)


def _vu_pow_sigma_prime(t: np.ndarray, eps: float) -> np.ndarray:
    abst = np.abs(t)
    inner = (3.0 / (2.0 * eps)) * t - (1.0 / (2.0 * eps ** 3)) * t ** 3
    return np.where(abst >= eps, np.sign(t), inner)


def phi_veelken_ulbrich_pow(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> np.ndarray:
    return 0.5 * (G + H - _vu_pow_sigma(G - H, eps))


def grad_veelken_ulbrich_pow(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    sp = _vu_pow_sigma_prime(G - H, eps)
    return 0.5 * (1.0 - sp), 0.5 * (1.0 + sp)


# --------------------------------------------------------------------------- #
# Veelken-Ulbrich sin                                                          #
# --------------------------------------------------------------------------- #

def phi_veelken_ulbrich_sin(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> np.ndarray:
    t = G - H
    return 0.5 * (G + H - (2.0 * t / np.pi) * np.arctan(np.pi * t / (2.0 * eps)))


def grad_veelken_ulbrich_sin(
    G: np.ndarray, H: np.ndarray, eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = G - H
    u = np.pi * t / (2.0 * eps)
    sp = (2.0 / np.pi) * np.arctan(u) + t / (eps * (1.0 + u * u))
    return 0.5 * (1.0 - sp), 0.5 * (1.0 + sp)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

# Each entry: (phi_fn, grad_fn, default_param_dict).
# ``default_param_dict`` lists the per-NCP options users may override via
# the ``ncp_params`` kwarg of :class:`NCPReformulationStrategy`.
NCP_REGISTRY: dict[str, tuple[Callable, Callable, dict]] = {
    "inner_product":       (phi_inner_product,      grad_inner_product,      {}),
    "fischer_burmeister":  (phi_fischer_burmeister, grad_fischer_burmeister, {}),
    "smooth_min":          (phi_smooth_min,         grad_smooth_min,         {}),
    "chen_chen_kanzow":    (phi_chen_chen_kanzow,   grad_chen_chen_kanzow,   {"lam": 0.5}),
    "kanzow_schwartz":     (phi_kanzow_schwartz,    grad_kanzow_schwartz,    {"lam": 0.5}),
    "chen_mangasarian":    (phi_chen_mangasarian,   grad_chen_mangasarian,   {"alpha": 0.5}),
    "billups":             (phi_billups,            grad_billups,            {"gamma": 0.1}),
    "veelken_ulbrich_pow": (phi_veelken_ulbrich_pow, grad_veelken_ulbrich_pow, {}),
    "veelken_ulbrich_sin": (phi_veelken_ulbrich_sin, grad_veelken_ulbrich_sin, {}),
}


def lookup_ncp(name: str, params: dict | None = None) -> tuple[Callable, Callable, dict]:
    """Resolve an NCP name to ``(phi_fn, grad_fn, merged_params)``.

    ``params`` overrides the registry's defaults; unknown keys raise
    ``ValueError`` (catches typos like ``"alpha"`` for an NCP that uses
    ``"lam"``).
    """
    if name not in NCP_REGISTRY:
        raise ValueError(
            f"unknown ncp_function {name!r}; "
            f"available: {sorted(NCP_REGISTRY)}"
        )
    phi_fn, grad_fn, defaults = NCP_REGISTRY[name]
    merged = dict(defaults)
    if params:
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(
                f"ncp_function {name!r} does not accept ncp_params "
                f"keys {sorted(unknown)}; valid: {sorted(defaults)}"
            )
        merged.update(params)
    return phi_fn, grad_fn, merged
