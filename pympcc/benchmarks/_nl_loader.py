"""Load MacMPEC ``.nl`` fixtures as :class:`ProblemSpec` records.

Each ``<name>.nl`` file in the supplied directory is parsed via
:func:`pympcc.frontend.ampl.from_nl`.  The known optimal value, objective
tolerance, and complementarity tolerance are pulled from the matching entry
in :data:`pympcc.benchmarks._problems.PROBLEM_NAMES`; problems without a
registered analytical counterpart are skipped (so a directory may contain
extra ``.nl`` files without breaking the loader).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from ._problems import PROBLEM_NAMES, ProblemSpec
from ..frontend.ampl import from_nl


def load_nl_directory(
    directory: Union[str, Path],
    *,
    only: Iterable[str] | None = None,
    matched_only: bool = False,
) -> list[ProblemSpec]:
    """Scan ``directory`` for ``*.nl`` files and return one ``ProblemSpec`` each.

    Parameters
    ----------
    directory
        Path containing ``<name>.nl`` files.  ``<name>`` is matched against
        the names registered in :data:`PROBLEM_NAMES` to recover the known
        optimum and tolerances.
    only
        Optional iterable of names to keep; everything else is skipped.
    matched_only
        When ``True``, skip ``.nl`` files whose stem doesn't appear in the
        registry.  When ``False`` (default) those problems are still
        returned with ``f_opt`` set to ``NaN`` and slack tolerances, so
        the benchmark runner can solve them — only the optimum-gap check
        is dropped.

    Returns
    -------
    specs
        List of :class:`ProblemSpec` records, one per matching fixture.
    """
    d = Path(directory)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a directory")
    keep = set(only) if only is not None else None

    specs: list[ProblemSpec] = []
    for path in sorted(d.glob("*.nl")):
        name = path.stem
        if keep is not None and name not in keep:
            continue
        ref = PROBLEM_NAMES.get(name)
        if ref is None and matched_only:
            continue
        try:
            problem = from_nl(path)
        except Exception as exc:
            # Skip fixtures that the reader can't parse (unsupported ops,
            # exotic AMPL features); the benchmark runner reports a clean
            # summary at the end so individual failures don't abort.
            print(f"  skip  {name}: {type(exc).__name__}: {exc}")
            continue
        if ref is not None:
            specs.append(
                ProblemSpec(
                    name=name, problem=problem,
                    f_opt=ref.f_opt, f_atol=ref.f_atol, comp_tol=ref.comp_tol,
                )
            )
        else:
            specs.append(
                ProblemSpec(
                    name=name, problem=problem,
                    f_opt=float("nan"), f_atol=float("inf"), comp_tol=1e-4,
                )
            )
    return specs


__all__ = ["load_nl_directory"]
