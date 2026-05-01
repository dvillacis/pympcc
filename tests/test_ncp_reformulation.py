"""Tests for the unified NCP-reformulation strategy (§6.6 Phase 1).

Covers:

* All 9 registry NCPs converge through ``strategy="ncp"`` on the SIMPLE
  benchmark.
* Each ``ncp_function`` value produces results equivalent to its dedicated
  strategy class (``smoothing``, ``smooth_min``, ``chen_chen_kanzow``, ...).
* ``ncp_params`` overrides registry defaults; unknown keys raise.
* Unknown ``ncp_function`` raises ``ValueError``.
* The ``ncp_function`` / ``ncp_params`` properties expose the resolved
  configuration.
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._reformulation import NCP_REGISTRY, lookup_ncp
from pympcc.strategies.ncp_reformulation import NCPReformulationStrategy

from .macmpec_problems import PROBLEM_NAMES

SIMPLE = PROBLEM_NAMES["simple"]

_ALL_NCPS = sorted(NCP_REGISTRY)

# Mapping registry name → equivalent dedicated strategy name.
# ``inner_product`` has no exact dedicated counterpart (Scholtes uses
# ``G·H ≤ ε`` rather than ``= 0``), so it is omitted from equivalence checks.
_DEDICATED = {
    "fischer_burmeister":  "smoothing",
    "smooth_min":          "smooth_min",
    "chen_chen_kanzow":    "chen_chen_kanzow",
    "kanzow_schwartz":     "kanzow_schwartz",
    "chen_mangasarian":    "chen_mangasarian",
    "billups":             "billups",
    "veelken_ulbrich_pow": "veelken_ulbrich_pow",
    "veelken_ulbrich_sin": "veelken_ulbrich_sin",
}


def _solve_ncp(ncp_function: str, **kwargs) -> pympcc.MPCCResult:
    return pympcc.solve(
        SIMPLE.problem, strategy="ncp",
        ncp_function=ncp_function, **kwargs,
    )


# --------------------------------------------------------------------------- #
# Convergence across the registry                                              #
# --------------------------------------------------------------------------- #

class TestUnifiedConvergence:
    @pytest.mark.parametrize("ncp_function", _ALL_NCPS)
    def test_converges(self, ncp_function):
        result = _solve_ncp(ncp_function)
        assert result.success, f"{ncp_function}: did not converge"

    @pytest.mark.parametrize("ncp_function", _ALL_NCPS)
    def test_comp_residual_small(self, ncp_function):
        result = _solve_ncp(ncp_function)
        assert result.comp_residual < 1e-4

    @pytest.mark.parametrize("ncp_function", _ALL_NCPS)
    def test_strategy_name_in_result(self, ncp_function):
        result = _solve_ncp(ncp_function)
        assert result.strategy == "ncp"


# --------------------------------------------------------------------------- #
# Equivalence with dedicated strategy classes                                  #
# --------------------------------------------------------------------------- #

class TestEquivalenceWithDedicated:
    """Each unified ncp_function must reproduce its dedicated strategy.

    The ε-continuation harness, defaults, and chain-rule are shared, so the
    final objective and complementarity residual should match to within
    IPOPT's tolerance.
    """

    @pytest.mark.parametrize("ncp_function", sorted(_DEDICATED))
    def test_objective_matches(self, ncp_function):
        dedicated = _DEDICATED[ncp_function]
        r_uni = _solve_ncp(ncp_function)
        r_ded = pympcc.solve(SIMPLE.problem, strategy=dedicated)
        assert r_uni.success and r_ded.success
        assert abs(r_uni.obj - r_ded.obj) < 1e-3, (
            f"{ncp_function}: unified obj={r_uni.obj:.6f} vs "
            f"{dedicated} obj={r_ded.obj:.6f}"
        )

    @pytest.mark.parametrize("ncp_function", sorted(_DEDICATED))
    def test_x_matches(self, ncp_function):
        dedicated = _DEDICATED[ncp_function]
        r_uni = _solve_ncp(ncp_function)
        r_ded = pympcc.solve(SIMPLE.problem, strategy=dedicated)
        np.testing.assert_allclose(r_uni.x, r_ded.x, atol=1e-3)


# --------------------------------------------------------------------------- #
# ncp_params overrides                                                         #
# --------------------------------------------------------------------------- #

class TestNcpParams:
    def test_default_is_fischer_burmeister(self):
        # Omitting ncp_function falls back to FB.
        r_default = pympcc.solve(SIMPLE.problem, strategy="ncp")
        r_fb = _solve_ncp("fischer_burmeister")
        assert abs(r_default.obj - r_fb.obj) < 1e-6

    def test_lam_override_cck(self):
        r_uni = _solve_ncp("chen_chen_kanzow", ncp_params={"lam": 1.0})
        r_ded = pympcc.solve(SIMPLE.problem, strategy="smoothing")
        # CCK with λ=1 == FB.
        assert abs(r_uni.obj - r_ded.obj) < 1e-3

    def test_alpha_override_chen_mangasarian(self):
        r_uni = _solve_ncp("chen_mangasarian", ncp_params={"alpha": 0.0})
        r_ded = pympcc.solve(SIMPLE.problem, strategy="smoothing")
        # CM with α=0 == FB.
        assert abs(r_uni.obj - r_ded.obj) < 1e-3

    def test_gamma_override_billups(self):
        r_uni = _solve_ncp("billups", ncp_params={"gamma": 0.0})
        r_ded = pympcc.solve(SIMPLE.problem, strategy="smoothing")
        # Billups with γ=0 == FB.
        assert abs(r_uni.obj - r_ded.obj) < 1e-3

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="does not accept"):
            _solve_ncp("chen_chen_kanzow", ncp_params={"alpha": 0.5})

    def test_unknown_key_for_paramless_ncp_raises(self):
        with pytest.raises(ValueError, match="does not accept"):
            _solve_ncp("fischer_burmeister", ncp_params={"lam": 0.5})

    def test_ncp_params_type_check(self):
        with pytest.raises(TypeError, match="ncp_params must be a dict"):
            _solve_ncp("chen_chen_kanzow", ncp_params=0.5)


# --------------------------------------------------------------------------- #
# Error handling                                                               #
# --------------------------------------------------------------------------- #

class TestErrors:
    def test_unknown_ncp_function_raises(self):
        with pytest.raises(ValueError, match="unknown ncp_function"):
            _solve_ncp("not_a_real_ncp")

    def test_unknown_kwarg_rejected_by_solver(self):
        # ncp_params is the channel for NCP-specific options; passing them
        # as bare kwargs (e.g. lam=...) must be rejected.
        with pytest.raises(TypeError, match="Unknown option"):
            pympcc.solve(SIMPLE.problem, strategy="ncp",
                         ncp_function="chen_chen_kanzow", lam=0.7)


# --------------------------------------------------------------------------- #
# Resolved configuration is exposed via properties                             #
# --------------------------------------------------------------------------- #

class TestPropertyAccess:
    def test_resolved_function_and_params(self):
        strat = NCPReformulationStrategy(
            SIMPLE.problem, ipopt_options={},
            ncp_function="chen_chen_kanzow", ncp_params={"lam": 0.7},
        )
        assert strat.ncp_function == "chen_chen_kanzow"
        assert strat.ncp_params == {"lam": 0.7}

    def test_default_params_filled_from_registry(self):
        strat = NCPReformulationStrategy(
            SIMPLE.problem, ipopt_options={},
            ncp_function="chen_mangasarian",
        )
        assert strat.ncp_params == {"alpha": 0.5}

    def test_paramless_ncp_has_empty_params(self):
        strat = NCPReformulationStrategy(
            SIMPLE.problem, ipopt_options={},
            ncp_function="fischer_burmeister",
        )
        assert strat.ncp_params == {}

    def test_params_dict_is_a_copy(self):
        strat = NCPReformulationStrategy(
            SIMPLE.problem, ipopt_options={},
            ncp_function="chen_chen_kanzow",
        )
        out = strat.ncp_params
        out["lam"] = 999.0
        assert strat.ncp_params == {"lam": 0.5}


# --------------------------------------------------------------------------- #
# Registry helper directly                                                     #
# --------------------------------------------------------------------------- #

class TestLookupHelper:
    def test_registry_keys_complete(self):
        expected = {
            "inner_product", "fischer_burmeister", "smooth_min",
            "chen_chen_kanzow", "kanzow_schwartz", "chen_mangasarian",
            "billups", "veelken_ulbrich_pow", "veelken_ulbrich_sin",
        }
        assert set(NCP_REGISTRY) == expected

    def test_lookup_merges_defaults(self):
        _, _, params = lookup_ncp("chen_chen_kanzow", {"lam": 0.7})
        assert params == {"lam": 0.7}

    def test_lookup_uses_defaults_when_no_override(self):
        _, _, params = lookup_ncp("chen_chen_kanzow")
        assert params == {"lam": 0.5}

    def test_lookup_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown ncp_function"):
            lookup_ncp("does_not_exist")

    def test_lookup_unknown_param_raises(self):
        with pytest.raises(ValueError, match="does not accept"):
            lookup_ncp("billups", {"lam": 0.5})


# --------------------------------------------------------------------------- #
# Sparse problem path                                                          #
# --------------------------------------------------------------------------- #

SIMPLE_SPARSE = pympcc.MPCCProblem(
    n=2, n_comp=1,
    x0=np.array([0.5, 0.5]),
    objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
    gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
    comp_G=lambda x: np.array([x[0]]),
    comp_G_jacobian=lambda x: np.array([1.0]),
    comp_G_jacobian_sparsity=(np.array([0]), np.array([0])),
    comp_H=lambda x: np.array([x[1]]),
    comp_H_jacobian=lambda x: np.array([1.0]),
    comp_H_jacobian_sparsity=(np.array([0]), np.array([1])),
)


class TestUnifiedSparse:
    @pytest.mark.parametrize("ncp_function", _ALL_NCPS)
    def test_sparse_converges(self, ncp_function):
        result = pympcc.solve(
            SIMPLE_SPARSE, strategy="ncp", ncp_function=ncp_function,
        )
        assert result.success
        assert result.comp_residual < 1e-4
