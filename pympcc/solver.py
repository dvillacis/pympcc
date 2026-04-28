"""Top-level solver interface."""
from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, Union

import numpy as np

from ._autoscale import autoscale_comp_pairs as _autoscale_comp_pairs
from ._kernels import HAS_NUMBA
from ._diagnostics import classify_cq as _classify_cq
from ._sosc import sosc_check as _sosc_check
from ._presolve import PresolveMap, presolve as _presolve
from ._stationarity import verify_b_stationarity as _verify_b_stat
from .models import StructuredMPCC
from .problem import MPCCProblem
from .result import IPOPTStatus, IterationInfo, MPCCResult
from .strategies.augmented_lagrangian import AugmentedLagrangianStrategy
from .strategies.direct import DirectStrategy
from .strategies.lin_fukushima import LinFukushimaStrategy
from .strategies.scholtes import ScholtesStrategy
from .strategies.slack import SlackStrategy
from .strategies.smoothing import SmoothingStrategy

__all__ = ["MPCCSolver", "solve"]

BackendName = Literal["ipopt", "filterSQP", "scipy"]
ProblemLike = Union[MPCCProblem, StructuredMPCC]


def _as_mpcc_problem(problem: ProblemLike) -> MPCCProblem:
    """Convert *problem* to :class:`MPCCProblem` if needed."""
    if isinstance(problem, StructuredMPCC):
        return problem.to_mpcc_problem()
    return problem

StrategyName = Literal[
    "direct", "scholtes", "smoothing", "lin_fukushima",
    "augmented_lagrangian", "slack",
]

_STRATEGIES = {
    "direct": DirectStrategy,
    "scholtes": ScholtesStrategy,
    "smoothing": SmoothingStrategy,
    "lin_fukushima": LinFukushimaStrategy,
    "augmented_lagrangian": AugmentedLagrangianStrategy,
    "slack": SlackStrategy,
}

_log = logging.getLogger("pympcc")
_log.addHandler(logging.NullHandler())


def _print_verbose_preamble(problem: MPCCProblem, strategy_name: str,
                            backend: str = "ipopt") -> None:
    """One-line banner printed to stdout before the iteration table when ``verbose=True``."""
    kernel_str = (
        "Numba JIT" if HAS_NUMBA
        else "NumPy (pip install numba for faster runs)"
    )
    parts = [f"strategy={strategy_name}", f"backend={backend}",
             f"n={problem.n}", f"n_comp={problem.n_comp}"]
    if problem.n_ineq:
        parts.append(f"n_ineq={problem.n_ineq}")
    if problem.n_eq:
        parts.append(f"n_eq={problem.n_eq}")
    parts.append(f"kernel={kernel_str}")
    print("pympcc  " + "  ".join(parts))


def _default_verbose_callback(k: int, info: IterationInfo) -> None:
    """Built-in per-iteration printer used when ``verbose=True``."""
    try:
        status_str = IPOPTStatus(info.status).name
    except ValueError:
        status_str = str(info.status)
    if k == 0:
        print(f"{'iter':>4}  {'epsilon':>12}  {'obj':>14}  {'comp_max':>12}  {'comp_mean':>12}  {'kkt_res':>10}  {'ipopt_it':>8}  {'status':<13}  {'time(s)':>9}")
        print("-" * 110)
    kkt_str = f"{info.kkt_residual:.3e}" if info.kkt_residual is not None else "n/a"
    print(f"{k+1:>4}  {info.epsilon:>12.4e}  {info.obj:>14.6g}  "
          f"{info.comp_residual:>12.3e}  {info.comp_residual_mean:>12.3e}  "
          f"{kkt_str:>10}  {info.n_ipopt_iter:>8d}  {status_str:<13}  {info.iter_time:>9.3f}")

# IPOPT options applied by default (users can override via ipopt_options)
_DEFAULT_IPOPT_OPTIONS: dict = {
    "print_level": 0,   # suppress per-iteration output
    "sb": "yes",        # suppress startup banner
}


class MPCCSolver:
    """
    Solver for Mathematical Programs with Complementarity Constraints (MPCC).

    Parameters
    ----------
    problem : MPCCProblem
        The MPCC problem instance.
    strategy : {'direct', 'scholtes', 'smoothing', 'lin_fukushima', 'augmented_lagrangian', 'slack'}
        Reformulation strategy (default ``'scholtes'``).

        - ``'direct'`` — single NLP solve with ``G·H ≤ 0``.
        - ``'scholtes'`` — outer loop relaxing ``G·H ≤ ε`` with ``ε → 0``.
        - ``'smoothing'`` — Fischer-Burmeister smoothing ``φ_ε(G,H) = 0``.
        - ``'lin_fukushima'`` — Scholtes + ``G+H ≥ ε`` to preserve MPCC-MFCQ.
        - ``'augmented_lagrangian'`` — PHR penalty; complementarity in objective only.
        - ``'slack'`` — lifting strategy with slack variables ``s_G = G(x)``,
          ``s_H = H(x)``; the complementarity Jacobian rows have zero x-block,
          which is efficient when ``n_comp ≪ n``.  Incompatible with
          ``backend='filterSQP'``.
    backend : {'ipopt', 'filterSQP', 'scipy'}, optional
        NLP backend solver (default ``'ipopt'``).

        - ``'ipopt'`` — uses IPOPT via cyipopt (default, fully supported).
        - ``'filterSQP'`` — uses pyfiltersqp (L-BFGS SQP).  Requires the
          ``pyfiltersqp`` package to be installed.  The ``'slack'`` strategy
          is incompatible with this backend.
        - ``'scipy'`` — uses ``scipy.optimize.minimize`` with
          ``method='trust-constr'``.  Requires ``scipy>=1.10``.  No IPOPT
          installation required; suitable for small problems and
          IPOPT-free environments.  Dual warm-starting is accepted but
          not forwarded (scipy does not support multiplier warm-starts);
          exact Hessians are not used.
    ipopt_options : dict, optional
        Options passed to the IPOPT backend (e.g. ``{"max_iter": 500,
        "tol": 1e-8}``).  When ``backend='filterSQP'`` the common keys
        ``"tol"`` and ``"max_iter"`` are translated to filterSQP equivalents;
        IPOPT-specific keys are silently ignored.  Merged with package
        defaults; user values take precedence.
    solver_options : dict, optional
        Options forwarded directly to :class:`~pyfiltersqp.SQPSolver`
        (e.g. ``{"tol_feas": 1e-7, "lbfgs_memory": 20}``).  Ignored when
        ``backend='ipopt'``.
    **strategy_options
        Extra keyword arguments forwarded to the strategy class
        (e.g. ``epsilon_0``, ``reduction`` for Scholtes / smoothing).
        Pass ``dual_warmstart=False`` to disable dual warm-starting between
        outer iterations (default ``True`` for all iterative strategies).
    callback : callable, optional
        ``f(k: int, info: IterationInfo) -> None`` called after each outer
        iteration, where ``k`` is the 0-based iteration index.  Not called
        by the ``'direct'`` strategy (single solve, no outer loop).
    verbose : bool, optional
        If ``True`` and no *callback* is provided, prints a formatted
        progress table to stdout after each outer iteration (default ``False``).

    Examples
    --------
    >>> solver = MPCCSolver(problem, strategy='scholtes',
    ...                     epsilon_0=0.5, reduction=0.1)
    >>> result = solver.solve()

    Using the filterSQP backend::

        result = pympcc.solve(problem, backend='filterSQP',
                              solver_options={'lbfgs_memory': 20})
    """

    def __init__(
        self,
        problem: ProblemLike,
        strategy: StrategyName = "scholtes",
        backend: BackendName = "ipopt",
        ipopt_options: dict | None = None,
        solver_options: dict | None = None,
        callback: Optional[Callable[[int, IterationInfo], None]] = None,
        verbose: bool = False,
        presolve: bool = False,
        diagnostics: bool = False,
        autoscale: bool = False,
        b_stat_max_biactive: int = 10,
        tnlp_refine: bool = False,
        tnlp_max_iter: int = 500,
        **strategy_options,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Available: {sorted(_STRATEGIES)}"
            )
        if backend not in ("ipopt", "filterSQP", "scipy"):
            raise ValueError(
                f"Unknown backend {backend!r}. "
                f"Choose 'ipopt', 'filterSQP', or 'scipy'."
            )
        # Pop linear_solver_fn before the _VALID_OPTIONS check: it's a solver-level
        # option, not a strategy option, and it bypasses IPOPT's built-in linear solver.
        linear_solver_fn = strategy_options.pop("linear_solver_fn", None)

        strategy_cls = _STRATEGIES[strategy]
        valid_opts: frozenset = getattr(strategy_cls, "_VALID_OPTIONS", frozenset())
        unknown = set(strategy_options) - valid_opts
        if unknown:
            raise TypeError(
                f"Unknown option(s) for strategy {strategy!r}: "
                f"{sorted(unknown)}. "
                f"Valid options: {sorted(valid_opts) if valid_opts else '(none)'}"
            )
        self.problem_orig = _as_mpcc_problem(problem)
        if presolve:
            self.problem, self._presolve_map = _presolve(self.problem_orig)
        else:
            self.problem, self._presolve_map = self.problem_orig, None
        if autoscale:
            self._apply_autoscale()
        self.strategy_name = strategy
        self.backend = backend
        self.ipopt_options = {**_DEFAULT_IPOPT_OPTIONS, **(ipopt_options or {})}
        self.solver_options = solver_options or {}
        self.strategy_options = strategy_options
        self._verbose = verbose and callback is None
        if self._verbose:
            callback = _default_verbose_callback
        self._strategy = _STRATEGIES[strategy](
            self.problem, self.ipopt_options,
            backend=backend,
            solver_options=self.solver_options,
            callback=callback,
            **strategy_options,
        )
        # linear_solver_fn bypasses the strategy's _VALID_OPTIONS and is injected
        # directly so individual strategies don't need to forward it.
        if linear_solver_fn is not None:
            self._strategy._linear_solver_fn = linear_solver_fn
        self._diagnostics = diagnostics
        self._b_stat_max_biactive = b_stat_max_biactive
        self._tnlp_refine = tnlp_refine
        self._tnlp_max_iter = tnlp_max_iter

    def solve(self) -> MPCCResult:
        """Run the solver and return an :class:`MPCCResult`."""
        if self._verbose:
            _print_verbose_preamble(self.problem, self.strategy_name, self.backend)
        result = self._strategy.solve()
        # Propagate complementarity-pair scaling (if any) to the result so
        # downstream callers can recover unscaled multipliers.  Done here in
        # one place rather than in every strategy.
        if self.problem.comp_G_scale is not None:
            result.comp_G_scale = self.problem.comp_G_scale
        if self.problem.comp_H_scale is not None:
            result.comp_H_scale = self.problem.comp_H_scale
        if self._presolve_map is not None and not self._presolve_map.is_identity:
            result = self._presolve_map.expand_result(result, self.problem_orig)
        if self._diagnostics:
            self._attach_diagnostics(result)
        # Always populate per_pair_status from G, H (O(n_comp), no cost).
        self._attach_per_pair_status(result)
        if self._tnlp_refine:
            self._attach_tnlp_refinement(result)
        return result

    def _apply_autoscale(self) -> None:
        """Populate ``problem.comp_G_scale`` / ``comp_H_scale`` from a probe.

        Skips silently when either scale is already user-supplied so manual
        scaling always wins.  When autoscale rescales at least one pair,
        emits a one-line ``UserWarning`` reporting the count.
        """
        import warnings as _warnings
        if (self.problem.comp_G_scale is not None
                or self.problem.comp_H_scale is not None):
            return
        s_G, s_H = _autoscale_comp_pairs(self.problem)
        n_rescaled = int(np.sum((s_G != 1.0) | (s_H != 1.0)))
        if n_rescaled == 0:
            return
        self.problem.comp_G_scale = s_G
        self.problem.comp_H_scale = s_H
        _warnings.warn(
            f"pympcc autoscale: rescaled {n_rescaled}/{self.problem.n_comp} "
            "complementarity pair(s) to balance |G_i| / |H_i|.",
            UserWarning,
            stacklevel=3,
        )

    def _attach_diagnostics(self, result: MPCCResult) -> None:
        """Run §2.1 / §2.2 / §2.3 diagnostics on the original-space result."""
        cq = _classify_cq(result, self.problem_orig)
        result.cq = cq["cq"]
        result.cq_active_set_sizes = cq["active_set_sizes"]
        result.cq_rank_deficit = cq["rank_deficit"]
        bs = _verify_b_stat(result, self.problem_orig,
                            max_biactive=self._b_stat_max_biactive)
        result.b_stationary = bs["status"]
        result.b_stationary_witness = bs["witness_branch"]
        result.b_stationary_min_descent = bs["min_descent"]
        sc = _sosc_check(result, self.problem_orig)
        result.sosc = sc["sosc"]
        result.sosc_min_eigenvalue = sc["min_eigenvalue"]
        result.sosc_skipped_reason = sc["skipped_reason"]

    @staticmethod
    def _attach_per_pair_status(result: MPCCResult) -> None:
        """Populate ``result.per_pair_status`` from G, H values (§4.6).

        Uses an adaptive threshold: ``max(sqrt(comp_residual), 1e-6)``
        matching the biactive-detection heuristic in ``_infer_active_set``.
        A flat 1e-6 cutoff misclassifies near-biactive points where both
        G and H sit at O(sqrt(ε)) due to the relaxation barrier.
        """
        G = np.asarray(result.G)
        H = np.asarray(result.H)
        comp = float(np.max(np.abs(G * H))) if G.size else 0.0
        # Slight upward slack (1 ppm) so that G≈H≈sqrt(comp) cases aren't
        # rejected by a ULP-level floating-point boundary.
        tol = max(np.sqrt(comp) * (1.0 + 1e-6), 1e-6)
        status = []
        for g, h in zip(G, H):
            if g <= tol and h <= tol:
                status.append("biactive")
            elif g <= tol:
                status.append("G_active")
            elif h <= tol:
                status.append("H_active")
            else:
                status.append("inactive")
        result.per_pair_status = status

    def _attach_tnlp_refinement(self, result: MPCCResult) -> None:
        """Run §2.6 TNLP active-set refinement and populate ``result.tnlp_refined``."""
        if not result.success:
            return
        from ._tnlp import run_tnlp_refinement as _tnlp
        tnlp = _tnlp(
            result,
            self.problem_orig,
            self._strategy,
            max_iter=self._tnlp_max_iter,
        )
        result.tnlp_refined = tnlp
        if tnlp.success:
            result.mult_comp_G_mpcc = tnlp.mult_comp_G
            result.mult_comp_H_mpcc = tnlp.mult_comp_H


def solve(
    problem: ProblemLike,
    strategy: StrategyName = "scholtes",
    backend: BackendName = "ipopt",
    ipopt_options: dict | None = None,
    solver_options: dict | None = None,
    callback: Optional[Callable[[int, IterationInfo], None]] = None,
    verbose: bool = False,
    presolve: bool = False,
    diagnostics: bool = False,
    autoscale: bool = False,
    b_stat_max_biactive: int = 10,
    tnlp_refine: bool = False,
    tnlp_max_iter: int = 500,
    n_starts: int = 1,
    perturb_scale: float = 0.1,
    multistart_seed: int = 0,
    **strategy_options,
) -> MPCCResult:
    """
    Solve an MPCC problem — convenience wrapper around :class:`MPCCSolver`.

    Parameters
    ----------
    problem : MPCCProblem
        The MPCC problem instance.
    strategy : {'direct', 'scholtes', 'smoothing', 'lin_fukushima', 'augmented_lagrangian', 'slack'}
        Reformulation strategy (default ``'scholtes'``).  See
        :class:`MPCCSolver` for a description of each option.
    backend : {'ipopt', 'filterSQP', 'scipy'}, optional
        NLP backend solver (default ``'ipopt'``).  ``'filterSQP'`` requires
        the ``pyfiltersqp`` package and is incompatible with ``strategy='slack'``.
        ``'scipy'`` uses ``scipy.optimize.minimize`` with ``method='trust-constr'``
        and requires no IPOPT installation.
    ipopt_options : dict, optional
        IPOPT solver options (merged with package defaults).  When
        ``backend='filterSQP'`` the common keys ``"tol"`` and ``"max_iter"``
        are translated; IPOPT-specific keys are silently ignored.
    solver_options : dict, optional
        Options forwarded directly to :class:`~pyfiltersqp.SQPSolver`.
        Ignored when ``backend='ipopt'``.
    callback : callable, optional
        ``f(k: int, info: IterationInfo) -> None`` called after each outer
        iteration.  Not called by the ``'direct'`` strategy.
    verbose : bool, optional
        If ``True`` and no *callback* is provided, prints a formatted
        progress table to stdout after each outer iteration (default ``False``).
    **strategy_options
        Additional options for the strategy
        (e.g. ``epsilon_0``, ``reduction``, ``max_iter``).

    Returns
    -------
    MPCCResult

    Examples
    --------
    Minimal call::

        result = pympcc.solve(problem)

    Use the filterSQP backend::

        result = pympcc.solve(problem, backend='filterSQP')

    Print progress after each outer iteration::

        result = pympcc.solve(problem, strategy='scholtes', verbose=True)

    Custom callback::

        def my_cb(k, info):
            print(f"[{k}] obj={info.obj:.6f}  comp={info.comp_residual:.2e}")

        result = pympcc.solve(problem, strategy='smoothing', callback=my_cb)

    Enable verbose IPOPT output::

        result = pympcc.solve(problem, ipopt_options={'print_level': 5})
    """
    if n_starts > 1:
        from .multistart import multistart as _multistart
        return _multistart(
            problem,
            n_starts=n_starts,
            perturb_scale=perturb_scale,
            seed=multistart_seed,
            strategy=strategy,
            backend=backend,
            ipopt_options=ipopt_options,
            solver_options=solver_options,
            callback=callback,
            verbose=verbose,
            presolve=presolve,
            diagnostics=diagnostics,
            autoscale=autoscale,
            b_stat_max_biactive=b_stat_max_biactive,
            tnlp_refine=tnlp_refine,
            tnlp_max_iter=tnlp_max_iter,
            **strategy_options,
        )
    return MPCCSolver(
        _as_mpcc_problem(problem), strategy,
        backend=backend,
        ipopt_options=ipopt_options,
        solver_options=solver_options,
        callback=callback, verbose=verbose,
        presolve=presolve,
        diagnostics=diagnostics,
        autoscale=autoscale,
        b_stat_max_biactive=b_stat_max_biactive,
        tnlp_refine=tnlp_refine,
        tnlp_max_iter=tnlp_max_iter,
        **strategy_options,
    ).solve()
