"""MacMPEC benchmark runner.

CLI usage
---------
# Run all problems with scholtes (default):
    python -m pympcc.benchmarks.macmpec

# Run specific strategies and problems:
    python -m pympcc.benchmarks.macmpec --strategy scholtes,smoothing
    python -m pympcc.benchmarks.macmpec --problems kth1,bard1,gauvin

# Save results to CSV:
    python -m pympcc.benchmarks.macmpec --out results.csv

Programmatic usage
------------------
    from pympcc.benchmarks import run_benchmark
    results = run_benchmark(strategies=["scholtes", "smoothing"])
    for r in results:
        print(r.problem, r.strategy, r.obj, r.stationarity)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import pympcc

from ._problems import ALL_PROBLEMS, PROBLEM_NAMES, ProblemSpec

__all__ = ["BenchmarkResult", "run_benchmark"]

_ALL_STRATEGIES = [
    "scholtes", "smoothing", "lin_fukushima", "augmented_lagrangian", "direct",
]


@dataclass
class BenchmarkResult:
    """Single-solve record: one problem × one strategy."""

    problem: str
    strategy: str
    n: int
    n_comp: int
    obj: float
    f_opt: Optional[float]
    f_gap: Optional[float]
    comp_residual: float
    stationarity: str
    success: bool
    ipopt_status: int
    solve_time: float
    error: Optional[str] = None

    def passed(self, comp_tol: float = 1e-4, f_rtol: float = 1e-2) -> bool:
        """True when the solve converged and the solution is close to the known optimum."""
        if self.error or not self.success:
            return False
        if self.comp_residual > comp_tol:
            return False
        if self.f_gap is not None and self.f_gap > f_rtol:
            return False
        return True


def run_benchmark(
    problems: Optional[Sequence[ProblemSpec]] = None,
    strategies: Sequence[str] = ("scholtes",),
    ipopt_options: Optional[dict] = None,
    max_iter: int = 3000,
    tol: float = 1e-8,
    verbose: bool = False,
) -> list[BenchmarkResult]:
    """Solve every problem with every strategy and return structured results.

    Parameters
    ----------
    problems :
        List of :class:`ProblemSpec` to solve.  Defaults to ``ALL_PROBLEMS``.
    strategies :
        Strategy names to run (one pass per strategy per problem).
    ipopt_options :
        Extra IPOPT options merged on top of the runner defaults
        (``{"tol": tol, "max_iter": max_iter, "print_level": 0}``).
    max_iter :
        IPOPT maximum iterations (default 3000).
    tol :
        IPOPT convergence tolerance (default 1e-8).
    verbose :
        Print a one-line progress message for each solve.

    Returns
    -------
    list[BenchmarkResult]
    """
    if problems is None:
        problems = ALL_PROBLEMS

    base_opts: dict = {"tol": tol, "max_iter": max_iter, "print_level": 0, "sb": "yes"}
    if ipopt_options:
        base_opts.update(ipopt_options)

    results: list[BenchmarkResult] = []

    for spec in problems:
        for strategy in strategies:
            if verbose:
                print(f"  {spec.name:<15} {strategy:<22} … ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                # augmented_lagrangian uses comp_tol to drive the outer loop,
                # which plays the role of epsilon for the other strategies.
                extra: dict = {}
                if strategy == "augmented_lagrangian":
                    extra["comp_tol"] = spec.comp_tol

                result = pympcc.solve(
                    spec.problem,
                    strategy=strategy,  # type: ignore[arg-type]
                    ipopt_options=base_opts,
                    **extra,
                )
                obj = result.obj
                comp_res = result.comp_residual
                stat = result.stationarity
                success = result.success
                ipopt_status = result.status
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                err_msg = f"{type(exc).__name__}: {exc}"
                if verbose:
                    print(f"CRASH  {err_msg}")
                results.append(BenchmarkResult(
                    problem=spec.name,
                    strategy=strategy,
                    n=spec.problem.n,
                    n_comp=spec.problem.n_comp,
                    obj=float("nan"),
                    f_opt=spec.f_opt,
                    f_gap=None,
                    comp_residual=float("nan"),
                    stationarity="error",
                    success=False,
                    ipopt_status=-999,
                    solve_time=elapsed,
                    error=err_msg,
                ))
                continue

            elapsed = time.perf_counter() - t0
            f_gap: Optional[float] = None
            if spec.f_opt is not None:
                denom = max(1.0, abs(spec.f_opt))
                f_gap = abs(obj - spec.f_opt) / denom

            br = BenchmarkResult(
                problem=spec.name,
                strategy=strategy,
                n=spec.problem.n,
                n_comp=spec.problem.n_comp,
                obj=obj,
                f_opt=spec.f_opt,
                f_gap=f_gap,
                comp_residual=comp_res,
                stationarity=stat,
                success=success,
                ipopt_status=ipopt_status,
                solve_time=elapsed,
            )
            if verbose:
                mark = "ok" if br.passed(spec.comp_tol) else "FAIL"
                print(f"{mark}  obj={obj:.4e}  comp={comp_res:.1e}  {stat:<15}  {elapsed:.2f}s")
            results.append(br)

    return results


def print_table(results: list[BenchmarkResult], *, file=None) -> None:
    """Print a Leyffer-style results table to *file* (default stdout)."""
    if file is None:
        file = sys.stdout

    col_w = dict(idx=3, problem=13, strategy=22, n=5, n_comp=6,
                 obj=11, f_gap=9, comp=9, stat=15, ok=4, time=7)

    hdr = (
        f"{'#':>{col_w['idx']}}  "
        f"{'problem':<{col_w['problem']}}  "
        f"{'strategy':<{col_w['strategy']}}  "
        f"{'n':>{col_w['n']}}  "
        f"{'n_comp':>{col_w['n_comp']}}  "
        f"{'obj':>{col_w['obj']}}  "
        f"{'f_gap':>{col_w['f_gap']}}  "
        f"{'comp_res':>{col_w['comp']}}  "
        f"{'stationarity':<{col_w['stat']}}  "
        f"{'ok':<{col_w['ok']}}  "
        f"{'time(s)':>{col_w['time']}}"
    )
    sep = "─" * len(hdr)
    print(sep, file=file)
    print(hdr, file=file)
    print(sep, file=file)

    for idx, r in enumerate(results, 1):
        obj_s = f"{r.obj:.4e}" if r.obj == r.obj else "NaN      "
        gap_s = f"{r.f_gap:.2e}" if r.f_gap is not None else "    n/a"
        comp_s = f"{r.comp_residual:.2e}" if r.comp_residual == r.comp_residual else "    NaN"

        # Truncate stationarity label to fit column
        stat_s = r.stationarity[:col_w['stat']]

        ok_s = "ok" if r.success else ("err" if r.error else "FAIL")
        print(
            f"{idx:>{col_w['idx']}}  "
            f"{r.problem:<{col_w['problem']}}  "
            f"{r.strategy:<{col_w['strategy']}}  "
            f"{r.n:>{col_w['n']}}  "
            f"{r.n_comp:>{col_w['n_comp']}}  "
            f"{obj_s:>{col_w['obj']}}  "
            f"{gap_s:>{col_w['f_gap']}}  "
            f"{comp_s:>{col_w['comp']}}  "
            f"{stat_s:<{col_w['stat']}}  "
            f"{ok_s:<{col_w['ok']}}  "
            f"{r.solve_time:>{col_w['time']}.2f}",
            file=file,
        )

    print(sep, file=file)

    n_ok = sum(1 for r in results if r.success and not r.error)
    n_total = len(results)
    total_time = sum(r.solve_time for r in results)
    comp_vals = [r.comp_residual for r in results if r.comp_residual == r.comp_residual]
    comp_max = max(comp_vals) if comp_vals else float("nan")
    print(
        f"Summary: {n_ok}/{n_total} converged  "
        f"comp_res_max={comp_max:.2e}  "
        f"total_time={total_time:.1f}s",
        file=file,
    )


def save_csv(results: list[BenchmarkResult], path: str) -> None:
    """Write results to a CSV file."""
    fields = [
        "problem", "strategy", "n", "n_comp",
        "obj", "f_opt", "f_gap", "comp_residual",
        "stationarity", "success", "ipopt_status", "solve_time", "error",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({f: getattr(r, f) for f in fields})


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the pympcc MacMPEC benchmark suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategy", default="scholtes",
        help=(
            "Comma-separated list of strategies to run.  "
            f"Choices: {', '.join(_ALL_STRATEGIES)}."
        ),
    )
    p.add_argument(
        "--problems", default=None,
        help=(
            "Comma-separated list of problem names to run (default: all).  "
            f"Available: {', '.join(PROBLEM_NAMES)}."
        ),
    )
    p.add_argument(
        "--skip", default=None,
        help="Comma-separated list of problem names to exclude.",
    )
    p.add_argument("--max-iter", type=int, default=3000, dest="max_iter",
                   help="IPOPT max iterations per NLP solve.")
    p.add_argument("--tol", type=float, default=1e-8,
                   help="IPOPT convergence tolerance.")
    p.add_argument("--out", default=None, metavar="FILE",
                   help="Save results to a CSV file.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-solve progress lines.")
    p.add_argument(
        "--from-nl", default=None, metavar="DIR", dest="from_nl",
        help=(
            "Load problems from a directory of AMPL .nl files instead of the "
            "built-in registry.  Each <name>.nl is matched against the "
            "registry by stem to recover f_opt and tolerances; unknown names "
            "are skipped."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse()

    strategies = [s.strip() for s in args.strategy.split(",")]
    for s in strategies:
        if s not in _ALL_STRATEGIES:
            print(f"Unknown strategy {s!r}. Choose from: {', '.join(_ALL_STRATEGIES)}")
            sys.exit(1)

    skip_set = (
        {n.strip() for n in args.skip.split(",")} if args.skip else set()
    )

    if args.from_nl:
        from ._nl_loader import load_nl_directory
        only = (
            {n.strip() for n in args.problems.split(",")}
            if args.problems else None
        )
        problems = load_nl_directory(args.from_nl, only=only)
        if skip_set:
            problems = [p for p in problems if p.name not in skip_set]
        if not problems:
            print(f"No matching .nl fixtures found in {args.from_nl}")
            sys.exit(1)
    elif args.problems:
        names = [n.strip() for n in args.problems.split(",")]
        unknown = [n for n in names if n not in PROBLEM_NAMES]
        if unknown:
            print(f"Unknown problem(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(PROBLEM_NAMES)}")
            sys.exit(1)
        problems = [PROBLEM_NAMES[n] for n in names]
    else:
        problems = list(ALL_PROBLEMS)

    if skip_set and not args.from_nl:
        problems = [p for p in problems if p.name not in skip_set]

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    strat_str = ", ".join(strategies)
    print(
        f"\nMacMPEC Benchmark — strategy={strat_str}  "
        f"{len(problems)} problem(s)  {ts}\n"
    )

    results = run_benchmark(
        problems=problems,
        strategies=strategies,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=not args.quiet,
    )

    print()
    print_table(results)

    if args.out:
        save_csv(results, args.out)
        print(f"\nSaved CSV → {args.out}")


if __name__ == "__main__":
    main()
