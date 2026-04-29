"""Parity tests: each ``tests/fixtures/nl/<name>.nl`` must solve to the same
optimum as the matching analytical :class:`ProblemSpec` in
:mod:`pympcc.benchmarks._problems`.

This is the round-trip check for the AMPL ``.nl`` reader: it confirms the
hand-authored fixtures reproduce the reference problems' objective values,
and exercises the full pipeline (parse + op-tree eval + comp_var_pairs_bulk
build + IPOPT solve) on the simplest MacMPEC suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import pympcc
from pympcc.benchmarks._nl_loader import load_nl_directory


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "nl"


@pytest.fixture(scope="module")
def nl_specs() -> list:
    return load_nl_directory(FIXTURE_DIR)


def test_fixtures_present(nl_specs):
    names = {s.name for s in nl_specs}
    assert {"simple", "kth1", "ralph1"}.issubset(names), (
        f"missing fixtures in {FIXTURE_DIR}: got {sorted(names)}"
    )


@pytest.mark.parametrize(
    "name,expected_n_eq,expected_n_ineq,expected_n_comp",
    [
        ("simple",   0, 0, 1),
        ("kth1",     0, 0, 1),
        ("bard1",    1, 0, 3),
        ("ex9.1.1",  7, 0, 5),
    ],
)
def test_nl_constraint_counts_wired(
    nl_specs, name, expected_n_eq, expected_n_ineq, expected_n_comp,
):
    """Regression: ``from_nl`` must populate ``n_eq`` / ``n_ineq`` so the
    NLP includes the constraints (previously dropped silently)."""
    spec = next(s for s in nl_specs if s.name == name)
    p = spec.problem
    assert p.n_eq == expected_n_eq, f"{name}: n_eq={p.n_eq}"
    assert p.n_ineq == expected_n_ineq, f"{name}: n_ineq={p.n_ineq}"
    assert p.n_comp == expected_n_comp, f"{name}: n_comp={p.n_comp}"


@pytest.mark.parametrize("name", ["simple", "kth1", "ralph1"])
def test_nl_solves_to_known_optimum(nl_specs, name):
    spec = next(s for s in nl_specs if s.name == name)
    result = pympcc.solve(
        spec.problem, strategy="scholtes", max_iter=30, epsilon_min=1e-10
    )
    assert result.success, f"{name}: IPOPT did not converge"
    assert result.comp_residual < spec.comp_tol, (
        f"{name}: comp_residual={result.comp_residual:.2e} "
        f"> tol {spec.comp_tol:.2e}"
    )
    assert abs(result.obj - spec.f_opt) < spec.f_atol, (
        f"{name}: obj={result.obj:.6g} vs f*={spec.f_opt} "
        f"(tol={spec.f_atol})"
    )
