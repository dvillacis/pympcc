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

# (strategy, problem_name) pairs where the strategy is known to converge to a
# local minimum instead of the global optimum.  Only test_objective_value is
# marked xfail; comp_residual and solver_converged tests still apply (IPOPT
# converges, and the local minimum has complementarity residual ≈ 0).
_KNOWN_OBJECTIVE_FAILURES: frozenset[tuple[str, str]] = frozenset({
    # bilevel1: at f*=0 one comp pair has G=H=0.
    # smoothing: phi_eps(0,0,eps)=-eps can never be zero → solver forced to f≈2.
    # lin_fukushima: G+H >= eps forces G+H > 0 → cannot reach G=H=0.
    # scholtes: G*H = 0 <= eps ✓, so scholtes does reach the global optimum.
    ("smoothing",     "bilevel1"),
    ("lin_fukushima", "bilevel1"),
})

STRATEGIES = [
    pytest.param("direct", marks=_DIRECT_XFAIL),
    "scholtes",
    "smoothing",
    "lin_fukushima",
    "augmented_lagrangian",
]

# ======================================================================= #
# Helpers                                                                   #
# ======================================================================= #

def _solve(spec: ProblemSpec, strategy: str) -> pympcc.MPCCResult:
    """Solve a problem spec with the given strategy."""
    # augmented_lagrangian uses comp_tol instead of epsilon_min.
    if strategy == "augmented_lagrangian":
        return pympcc.solve(
            spec.problem,
            strategy=strategy,
            max_iter=30,
            comp_tol=1e-10,
        )
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
    if (strategy, spec.name) in _KNOWN_OBJECTIVE_FAILURES:
        pytest.xfail(
            f"{strategy!r} cannot reach the global optimum of {spec.name!r}: "
            "one complementarity pair has G=H=0 at f*, which is incompatible "
            "with this strategy's regularization."
        )
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


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("spec", ALL_PROBLEMS, ids=lambda s: s.name)
def test_kkt_residual(spec: ProblemSpec, strategy: str):
    """kkt_residual must be populated and below 1e-4 for converged solves."""
    result = _solve(spec, strategy)
    assert result.kkt_residual is not None, (
        f"[{spec.name}/{strategy}] kkt_residual was not populated"
    )
    assert result.kkt_residual < 1e-4, (
        f"[{spec.name}/{strategy}] kkt_residual={result.kkt_residual:.2e} exceeds 1e-4"
    )
