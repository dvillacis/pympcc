"""Unified NCP-reformulation strategy (§6.6 modifier matrix, Phase 1).

Single configurable strategy that dispatches the existing NCP-function
toolbox through a registry, exposing ``ncp_function`` and ``ncp_params``
as orthogonal modifier axes.  Phase 1 covers the ``ncp_function`` axis
only; ``constraint`` (eq / ineq / band), ``slack`` (none / positive /
free / one), ``aggregate``, and ``ncp_bounds`` ship in Phase 2.

Usage::

    pympcc.solve(
        problem,
        strategy="ncp",
        ncp_function="fischer_burmeister",
        ncp_params={},                    # NCP-specific overrides (lam, alpha, ...)
        epsilon_0=1.0,                    # standard ε-continuation options
        reduction=0.1,
        max_iter=20,
    )

The legacy strategy classes (`smoothing`, `chen_chen_kanzow`, ...) keep
working unchanged; this strategy is additive.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .._reformulation import lookup_ncp
from .ncp import _SmoothNCPBase


class NCPReformulationStrategy(_SmoothNCPBase):
    """Unified ε-smoothing strategy parameterized over the NCP-function registry.

    Parameters
    ----------
    ncp_function : str
        Registry key — one of ``inner_product``, ``fischer_burmeister``,
        ``smooth_min``, ``chen_chen_kanzow``, ``kanzow_schwartz``,
        ``chen_mangasarian``, ``billups``, ``veelken_ulbrich_pow``,
        ``veelken_ulbrich_sin``.  Each entry of
        :data:`pympcc._reformulation.NCP_REGISTRY` carries the matching
        ``(phi, grad)`` callables.
    ncp_params : dict, optional
        Per-NCP parameter overrides (e.g. ``{"lam": 0.7}`` for CCK /
        Kanzow-Schwartz, ``{"alpha": 0.3}`` for Chen-Mangasarian,
        ``{"gamma": 0.05}`` for Billups).  Unknown keys raise
        ``ValueError``.
    epsilon_0, reduction, max_iter, epsilon_min, dual_warmstart, comp_tol :
        Standard ε-continuation options forwarded to
        :class:`_SmoothNCPBase`.
    """

    name = "ncp"
    _VALID_OPTIONS = frozenset({
        *_SmoothNCPBase._VALID_OPTIONS,
        "ncp_function", "ncp_params",
    })

    def __init__(self, problem, ipopt_options: dict, **kwargs) -> None:
        ncp_function = kwargs.pop("ncp_function", "fischer_burmeister")
        ncp_params = kwargs.pop("ncp_params", None)
        if ncp_params is not None and not isinstance(ncp_params, dict):
            raise TypeError(
                f"ncp_params must be a dict or None, got {type(ncp_params).__name__}"
            )
        phi_fn, grad_fn, merged = lookup_ncp(ncp_function, ncp_params)

        super().__init__(problem, ipopt_options, **kwargs)

        self._ncp_function: str = ncp_function
        self._ncp_params: dict = merged
        self._phi_fn = partial(phi_fn, **merged) if merged else phi_fn
        self._grad_fn = partial(grad_fn, **merged) if merged else grad_fn

    def _phi(self, G: np.ndarray, H: np.ndarray, eps: float) -> np.ndarray:
        return self._phi_fn(G, H, eps)

    def _phi_grad_coeffs(
        self, G: np.ndarray, H: np.ndarray, eps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._grad_fn(G, H, eps)

    @property
    def ncp_function(self) -> str:
        """Resolved NCP-function name."""
        return self._ncp_function

    @property
    def ncp_params(self) -> dict:
        """Merged NCP parameter dict (registry defaults + user overrides)."""
        return dict(self._ncp_params)
