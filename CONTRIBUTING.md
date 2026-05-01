# Contributing to pympcc

Thanks for your interest in pympcc. This document covers how to set up a
development environment, the conventions the codebase follows, and what
to expect when you open a pull request.

## Reporting bugs

Please open an issue at <https://github.com/dvillacis/pympcc/issues> with:

- A minimal reproducer (an `MPCCProblem` instance is ideal).
- The pympcc, cyipopt, IPOPT, and Python versions (`pympcc.__version__`,
  `pip show cyipopt`, `ipopt --version`).
- The full traceback or unexpected output.

If the bug is numerical (wrong objective, multiplier, or stationarity
classification), please include the strategy and any non-default options
you passed to `pympcc.solve(...)`.

## Development setup

pympcc requires a working IPOPT installation for the default backend.
On macOS use `brew install ipopt`; on Debian/Ubuntu use
`sudo apt-get install coinor-libipopt-dev`.

We use [`uv`](https://github.com/astral-sh/uv) for environment management.
With IPOPT already installed:

```bash
git clone https://github.com/dvillacis/pympcc
cd pympcc
uv pip install -e ".[dev]"
```

## Running the tests

```bash
uv run pytest tests/ -v                   # full suite
uv run pytest tests/test_strategies.py    # one file
uv run pytest tests/ -k "scholtes"        # filter by name
```

The `direct` strategy tests are marked `xfail(strict=False)` because LICQ
generically fails at MPCC feasible points; that is expected.

## Project layout

See `CLAUDE.md` for a high-level architecture sketch (data flow,
strategy contract, where to add a new reformulation). The roadmap is
in `ROADMAP.md`.

## Coding conventions

- Format and lint with `ruff` (`uv run ruff check . && uv run ruff format .`).
- Type-check with `mypy` (`uv run mypy pympcc/`). New code should be
  type-annotated end-to-end; we are tightening typing module by module.
- Tests for new behaviour go under `tests/`. Strategy-level features
  should be exercised on at least one MacMPEC problem in
  `tests/test_macmpec.py`.
- Keep callbacks (`objective`, `gradient`, `constraints`, `jacobian`,
  `hessian`) allocation-free where reasonable — they run inside the
  IPOPT iteration loop.

Naming conventions for private helpers (used throughout the package):

- `_detect_*` — discovery returning indices or booleans.
- `_build_*` — construction returning new objects.
- `_apply_*` — in-place modifications.
- `_resolve_*` — conversions / lookups.
- `_compute_*` — pure mathematical evaluation.

## Pull requests

- Branch from `master` with a short, descriptive name
  (`feature/ncp-billups`, `fix/slack-jacobian-offset`).
- Keep PRs focused. Refactors and feature changes belong in separate
  PRs from bug fixes.
- Run the full test suite locally before pushing.
- Update `CHANGELOG.md` under the `Unreleased` heading.
- If your change is research-grade (a new strategy, a new diagnostic),
  cite the relevant paper in the docstring and add a short entry under
  the matching `ROADMAP.md` section.

## License

By contributing, you agree that your contributions will be licensed
under the MIT License (see `LICENSE`).
