"""One-off MacMPEC `.mod` → `.nl` converter via amplpy.

Scans a directory of MacMPEC AMPL sources and writes text-format ``.nl``
files (with the ``cvar`` complementarity suffix embedded) into an output
directory.  The resulting files load directly via
:func:`pympcc.frontend.ampl.from_nl` and the
``python -m pympcc.benchmarks.macmpec --from-nl <dir>`` runner.

Usage
-----
    pip install amplpy           # bundles AMPL CE; no license server needed
    python scripts/convert_macmpec.py path/to/MacMPEC tests/fixtures/nl

Pairing rules
-------------
For each ``<name>.mod`` in the source directory:

* If ``<name>.dat`` exists alongside it, both are loaded.
* Otherwise only the ``.mod`` is loaded.

``option auxfiles cr;`` is set before write so AMPL embeds the ``cvar``
suffix that the reader uses to recover ``(var, con)`` complementarity
pairs.  Without this, the resulting ``.nl`` would have no comp structure.

The script writes text-format ``.nl`` (``g`` prefix); binary ``.nl`` is
rejected by our reader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _convert_one(ampl_obj, mod_path: Path, dat_path: Path | None, out_path: Path) -> None:
    ampl_obj.reset()
    ampl_obj.read(str(mod_path))
    if dat_path is not None and dat_path.exists():
        ampl_obj.read_data(str(dat_path))
    # 'cr' = constraint + variable suffixes; 'cvar' is a var-side integer
    # suffix linking each complementarity var to its constraint index.
    ampl_obj.option["auxfiles"] = "cr"
    # exportModel writes text-format .nl by default with amplpy >=0.10.
    ampl_obj.export_model(str(out_path))


def convert_directory(
    src: Path,
    dst: Path,
    *,
    only: set[str] | None = None,
    overwrite: bool = False,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Convert every ``.mod`` under ``src`` into a ``.nl`` under ``dst``.

    Returns
    -------
    n_ok, n_skip, failures
        Counts and a list of ``(name, error_message)`` pairs.
    """
    try:
        from amplpy import AMPL
    except ImportError as exc:
        raise SystemExit(
            "amplpy is required for this script.  Install via "
            "`pip install amplpy` (bundles AMPL CE)."
        ) from exc

    dst.mkdir(parents=True, exist_ok=True)
    ampl_obj = AMPL()

    n_ok = 0
    n_skip = 0
    failures: list[tuple[str, str]] = []

    for mod_path in sorted(src.glob("*.mod")):
        name = mod_path.stem
        if only is not None and name not in only:
            continue
        out_path = dst / f"{name}.nl"
        if out_path.exists() and not overwrite:
            n_skip += 1
            continue
        dat_path = mod_path.with_suffix(".dat")
        try:
            _convert_one(
                ampl_obj, mod_path,
                dat_path if dat_path.exists() else None,
                out_path,
            )
            n_ok += 1
            print(f"  ok    {name}")
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}: {exc}")

    return n_ok, n_skip, failures


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("src", type=Path, help="Source dir with MacMPEC .mod/.dat files")
    p.add_argument("dst", type=Path, help="Output dir for .nl files")
    p.add_argument(
        "--only", default=None,
        help="Comma-separated list of problem stems to convert (default: all).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-convert problems whose .nl already exists in dst.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    if not args.src.is_dir():
        print(f"src is not a directory: {args.src}", file=sys.stderr)
        return 2
    only = (
        {n.strip() for n in args.only.split(",")} if args.only else None
    )
    print(f"Converting {args.src} → {args.dst}")
    n_ok, n_skip, failures = convert_directory(
        args.src, args.dst, only=only, overwrite=args.overwrite,
    )
    print()
    print(f"Summary: {n_ok} converted, {n_skip} skipped, {len(failures)} failed")
    if failures:
        print("\nFailures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
