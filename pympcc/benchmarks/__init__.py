"""pympcc.benchmarks — benchmark runners for the MacMPEC and other problem suites."""
from ._problems import ALL_PROBLEMS, PROBLEM_NAMES, ProblemSpec

__all__ = [
    "ALL_PROBLEMS",
    "PROBLEM_NAMES",
    "ProblemSpec",
    "BenchmarkResult",
    "run_benchmark",
]


def __getattr__(name: str):
    if name in ("BenchmarkResult", "run_benchmark", "print_table", "save_csv"):
        from . import macmpec as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
