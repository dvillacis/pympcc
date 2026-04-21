"""
Pytest suite for the MacMPEC benchmark problems.

Each problem is solved with all three strategies (direct, scholtes, smoothing).
Tests verify:
  1. Complementarity feasibility  (comp_residual < tol)
  2. Objective value              (|obj - f*| < atol)
  3. IPOPT convergence            (result.success is True)
"""

from __future__ import annotations

import pytest

import pympcc
from .macmpec_problems import ALL_PROBLEMS, ProblemSpec

# The direct strategy enforces G*H <= 0 as a single NLP.  At any MPCC
# feasible point LICQ fails (active constraint gradients are linearly
# dependent), so IPOPT often cannot certify second-order conditions and may
# report an "acceptable level" solve or fail outright.  All direct-strategy
# parametrized cases are therefore marked xfail(strict=False): the tests
# document the known limitation without breaking CI, and any incidental
# passes are still reported as successes.
_DIRECT_XFAIL = pytest.mark.xfail(
    reason=(
        "Direct NLP violates LICQ at MPCC feasible points; "
        "IPOPT convergence is unreliable for this strategy."
    ),
    strict=False,
)

STRATEGIES = [
    pytest.param("direct", marks=_DIRECT_XFAIL),
    "scholtes",
    "smoothing",
    "lin_fukushima",
]

# ======================================================================= #
# Helpers                                                                   #
# ======================================================================= #

def _solve(spec: ProblemSpec, strategy: str) -> pympcc.MPCCResult:
    """Solve a problem spec with the given strategy."""
    return pympcc.solve(
        spec.problem,
        strategy=strategy,
        # Tighter outer loop for the test suite
        max_iter=30,
        epsilon_min=1e-10,
    )


# ======================================================================= #
# Main parametrized test                                                    #
# ======================================================================= #

@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("spec", ALL_PROBLEMS, ids=lambda s: s.name)
def test_complementarity_residual(spec: ProblemSpec, strategy: str):
    """Complementarity infeasibility must be below spec tolerance."""
    result = _solve(spec, strategy)
    assert result.comp_residual < spec.comp_tol, (
        f"[{spec.name}/{strategy}] comp_residual={result.comp_residual:.2e} "
        f"exceeds tolerance {spec.comp_tol:.2e}"
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("spec", ALL_PROBLEMS, ids=lambda s: s.name)
def test_objective_value(spec: ProblemSpec, strategy: str):
    """Objective value must be within absolute tolerance of the known optimum."""
    result = _solve(spec, strategy)
    err = abs(result.obj - spec.f_opt)
    assert err < spec.f_atol, (
        f"[{spec.name}/{strategy}] obj={result.obj:.6f}, "
        f"f*={spec.f_opt:.6f}, |err|={err:.2e} exceeds {spec.f_atol:.2e}"
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("spec", ALL_PROBLEMS, ids=lambda s: s.name)
def test_solver_converged(spec: ProblemSpec, strategy: str):
    """IPOPT must report success (status 0 or 1) for all problems."""
    result = _solve(spec, strategy)
    assert result.success, (
        f"[{spec.name}/{strategy}] solver did not converge: "
        f"status={result.status}, message={result.message!r}"
    )
