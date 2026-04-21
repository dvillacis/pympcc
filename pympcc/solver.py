"""Top-level solver interface."""
from __future__ import annotations

from typing import Callable, Literal, Optional, Union

from .problem import MPCCProblem
from .models import StructuredMPCC
from .result import IPOPTStatus, IterationInfo, MPCCResult
from .strategies.direct import DirectStrategy
from .strategies.scholtes import ScholtesStrategy
from .strategies.smoothing import SmoothingStrategy
from .strategies.lin_fukushima import LinFukushimaStrategy
from .strategies.augmented_lagrangian import AugmentedLagrangianStrategy
from .strategies.slack import SlackStrategy

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

def _default_verbose_callback(k: int, info: IterationInfo) -> None:
    """Built-in per-iteration printer used when ``verbose=True``."""
    try:
        status_str = IPOPTStatus(info.status).name
    except ValueError:
        status_str = str(info.status)
    if k == 0:
        print(f"{'iter':>4}  {'epsilon':>12}  {'obj':>14}  {'comp_max':>12}  {'comp_mean':>12}  {'ipopt_it':>8}  {'status':<13}")
        print("-" * 82)
    print(f"{k+1:>4}  {info.epsilon:>12.4e}  {info.obj:>14.6g}  "
          f"{info.comp_residual:>12.3e}  {info.comp_residual_mean:>12.3e}  "
          f"{info.n_ipopt_iter:>8d}  {status_str:<13}")


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

        - ``'direct'`` — single IPOPT solve with ``G·H ≤ 0``.
        - ``'scholtes'`` — outer loop relaxing ``G·H ≤ ε`` with ``ε → 0``.
        - ``'smoothing'`` — Fischer-Burmeister smoothing ``φ_ε(G,H) = 0``.
        - ``'lin_fukushima'`` — Scholtes + ``G+H ≥ ε`` to preserve MPCC-MFCQ.
        - ``'augmented_lagrangian'`` — PHR penalty; complementarity in objective only.
        - ``'slack'`` — lifting strategy with slack variables ``s_G = G(x)``,
          ``s_H = H(x)``; the complementarity Jacobian rows have zero x-block,
          which is efficient when ``n_comp ≪ n``.
    ipopt_options : dict, optional
        IPOPT options passed directly to cyipopt (e.g.
        ``{"max_iter": 500, "tol": 1e-8}``).  These are merged with
        the package defaults; user values take precedence.
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
    """

    def __init__(
        self,
        problem: ProblemLike,
        strategy: StrategyName = "scholtes",
        ipopt_options: dict | None = None,
        callback: Optional[Callable[[int, IterationInfo], None]] = None,
        verbose: bool = False,
        **strategy_options,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {sorted(_STRATEGIES)}"
            )
        self.problem = _as_mpcc_problem(problem)
        self.strategy_name = strategy
        self.ipopt_options = {**_DEFAULT_IPOPT_OPTIONS, **(ipopt_options or {})}
        self.strategy_options = strategy_options
        if verbose and callback is None:
            callback = _default_verbose_callback
        self._strategy = _STRATEGIES[strategy](
            self.problem, self.ipopt_options, callback=callback, **strategy_options
        )

    def solve(self) -> MPCCResult:
        """Run the solver and return an :class:`MPCCResult`."""
        return self._strategy.solve()


def solve(
    problem: ProblemLike,
    strategy: StrategyName = "scholtes",
    ipopt_options: dict | None = None,
    callback: Optional[Callable[[int, IterationInfo], None]] = None,
    verbose: bool = False,
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
    ipopt_options : dict, optional
        IPOPT solver options (merged with package defaults).
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

    Print progress after each outer iteration::

        result = pympcc.solve(problem, strategy='scholtes', verbose=True)

    Custom callback::

        def my_cb(k, info):
            print(f"[{k}] obj={info.obj:.6f}  comp={info.comp_residual:.2e}")

        result = pympcc.solve(problem, strategy='smoothing', callback=my_cb)

    Enable verbose IPOPT output::

        result = pympcc.solve(problem, ipopt_options={'print_level': 5})
    """
    return MPCCSolver(
        _as_mpcc_problem(problem), strategy, ipopt_options,
        callback=callback, verbose=verbose, **strategy_options
    ).solve()
