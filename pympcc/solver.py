"""Top-level solver interface."""
from __future__ import annotations

import logging
from typing import Callable, Literal, Optional, Union

from ._kernels import HAS_NUMBA
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
        strategy_cls = _STRATEGIES[strategy]
        valid_opts: frozenset = getattr(strategy_cls, "_VALID_OPTIONS", frozenset())
        unknown = set(strategy_options) - valid_opts
        if unknown:
            raise TypeError(
                f"Unknown option(s) for strategy {strategy!r}: "
                f"{sorted(unknown)}. "
                f"Valid options: {sorted(valid_opts) if valid_opts else '(none)'}"
            )
        self.problem = _as_mpcc_problem(problem)
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

    def solve(self) -> MPCCResult:
        """Run the solver and return an :class:`MPCCResult`."""
        if self._verbose:
            _print_verbose_preamble(self.problem, self.strategy_name, self.backend)
        return self._strategy.solve()


def solve(
    problem: ProblemLike,
    strategy: StrategyName = "scholtes",
    backend: BackendName = "ipopt",
    ipopt_options: dict | None = None,
    solver_options: dict | None = None,
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
    return MPCCSolver(
        _as_mpcc_problem(problem), strategy,
        backend=backend,
        ipopt_options=ipopt_options,
        solver_options=solver_options,
        callback=callback, verbose=verbose,
        **strategy_options,
    ).solve()
