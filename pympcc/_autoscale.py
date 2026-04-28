"""Automatic per-pair scaling detector for the complementarity block.

A pair ``(G_i, H_i)`` is *imbalanced* when ``|G_i|`` and ``|H_i|`` typically
differ by orders of magnitude across the problem's working region — IPOPT
struggles to balance KKT residuals on such pairs and the relax-and-drive
strategies can stall on the under-scaled side.

:func:`autoscale_comp_pairs` probes each side at ``x0`` and a handful of
random perturbations, takes the per-pair median magnitude, and returns
diagonal scales ``(s_G, s_H)`` that bring imbalanced pairs to magnitude
~1.  Well-conditioned pairs are left at unit scale.

The returned scales plug straight into :class:`MPCCProblem.comp_G_scale`
and :class:`MPCCProblem.comp_H_scale`.  Multipliers reported by the
strategy live in scaled space; use :func:`pympcc.unscale_multipliers`
to recover original-space duals.
"""
from __future__ import annotations

import numpy as np

__all__ = ["autoscale_comp_pairs"]


def autoscale_comp_pairs(
    problem,
    *,
    threshold: float = 1e3,
    n_probes: int = 5,
    seed: int = 0,
    floor: float = 1e-12,
    perturb_scale: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pair diagonal scales that equilibrate the comp block.

    Parameters
    ----------
    problem : MPCCProblem
        Source problem.  Must expose ``comp_G``, ``comp_H``, ``x0``, and
        ``n_comp``.  ``xl`` / ``xu`` are honoured when probing perturbations.
    threshold : float, optional
        A pair is rescaled only when ``max(|G|, |H|) / min(|G|, |H|)`` —
        evaluated on per-pair medians across the probe set — exceeds this
        ratio.  Default ``1e3``.
    n_probes : int, optional
        Number of random perturbations of ``x0`` evaluated in addition to
        ``x0`` itself.  Default ``5`` (six samples total).
    seed : int, optional
        Seed for the perturbation RNG.  Default ``0``.
    floor : float, optional
        Lower clip on per-pair median magnitudes before division.  Default
        ``1e-12``.
    perturb_scale : float, optional
        Standard deviation of the random perturbation, expressed as a
        fraction of ``max(|x0|, 1)`` per coordinate.  Default ``0.1``.

    Returns
    -------
    s_G : ndarray, shape (n_comp,)
    s_H : ndarray, shape (n_comp,)
        Strictly positive scales.  ``1.0`` for well-conditioned pairs;
        ``1 / median(|·|)`` for imbalanced pairs (clipped at ``floor``).
    """
    if threshold <= 1.0:
        raise ValueError("threshold must be > 1")
    if n_probes < 0:
        raise ValueError("n_probes must be >= 0")
    if floor <= 0.0:
        raise ValueError("floor must be > 0")
    if perturb_scale <= 0.0:
        raise ValueError("perturb_scale must be > 0")

    rng = np.random.default_rng(seed)
    x0 = np.asarray(problem.x0, dtype=float)
    n = problem.n
    n_comp = problem.n_comp

    # Build probe set: x0 plus n_probes random perturbations clipped to bounds.
    span = np.maximum(np.abs(x0), 1.0) * float(perturb_scale)
    xl = np.asarray(problem.xl, dtype=float) if problem.xl is not None else np.full(n, -np.inf)
    xu = np.asarray(problem.xu, dtype=float) if problem.xu is not None else np.full(n, np.inf)

    pts: list[np.ndarray] = [x0]
    for _ in range(n_probes):
        x = x0 + rng.normal(scale=span)
        np.maximum(x, xl, out=x)
        np.minimum(x, xu, out=x)
        pts.append(x)

    G_samples: list[np.ndarray] = []
    H_samples: list[np.ndarray] = []
    for x in pts:
        try:
            G = np.asarray(problem.comp_G(x), dtype=float)
            H = np.asarray(problem.comp_H(x), dtype=float)
        except Exception:                                    # pragma: no cover
            # Skip points where the user's callable can't be evaluated
            # (NaN/Inf branches, domain errors).
            continue
        if np.all(np.isfinite(G)) and np.all(np.isfinite(H)):
            G_samples.append(np.abs(G))
            H_samples.append(np.abs(H))

    if not G_samples:
        return np.ones(n_comp), np.ones(n_comp)

    G_med = np.median(np.stack(G_samples), axis=0)
    H_med = np.median(np.stack(H_samples), axis=0)
    G_med = np.maximum(G_med, floor)
    H_med = np.maximum(H_med, floor)

    ratio = np.maximum(G_med / H_med, H_med / G_med)
    rescale_mask = ratio > threshold

    s_G = np.ones(n_comp)
    s_H = np.ones(n_comp)
    s_G[rescale_mask] = 1.0 / G_med[rescale_mask]
    s_H[rescale_mask] = 1.0 / H_med[rescale_mask]
    return s_G, s_H
