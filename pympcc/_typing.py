"""Centralised public ``Literal`` aliases for stringly-typed parameters.

Single source of truth so callers benefit from IDE auto-complete and
mypy catches typos at static-check time.  Defined here (not in
``solver.py``) to keep ``_base.py`` and ``_autodiff.py`` free of any
circular import on the solver entry point.
"""
from __future__ import annotations

from typing import Literal

#: Strategies dispatched by ``pympcc.solve(strategy=...)``.
StrategyName = Literal[
    "direct",
    "scholtes",
    "smoothing",
    "lin_fukushima",
    "augmented_lagrangian",
    "slack",
    "smooth_min",
    "chen_chen_kanzow",
    "kanzow_schwartz",
    "chen_mangasarian",
    "billups",
    "veelken_ulbrich_pow",
    "veelken_ulbrich_sin",
    "ncp",
]

#: NLP backends the strategies can dispatch to.
BackendName = Literal["ipopt", "filterSQP", "scipy"]

#: Finite-difference scheme used by the FD-Jacobian / FD-gradient helpers.
FDMode = Literal["forward", "central"]

#: Derivative backend selector for the bilevel KKT emitter.
Derivatives = Literal["jax", "fd"]

#: ε-continuation inner-tolerance schedule.
InnerTolMode = Literal["linear", "quadratic", "matched"]

__all__ = [
    "StrategyName",
    "BackendName",
    "FDMode",
    "Derivatives",
    "InnerTolMode",
]
