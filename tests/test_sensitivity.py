"""Tests for §5.1 parametric sensitivity analysis (``pympcc.sensitivity``)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pympcc
from pympcc import MPCCProblem


# ---------------------------------------------------------------------------
# Problem builders parametrised by ``p``
# ---------------------------------------------------------------------------

def _branch_select(a: float, b: float, c: float, d: float):
    """Build the 2-pair quadratic MPCC:

        min ½(x0-a)² + ½(x1-b)² + ½(x2-c)² + ½(x3-d)²
        s.t. x0 ≥ 0 ⊥ x2 ≥ 0,  x1 ≥ 0 ⊥ x3 ≥ 0

    Returned with x0 = (a, b, c, d) clipped at zero — already at the
    likely optimum so IPOPT lands cleanly with no biactive pairs.
    """
    a, b, c, d = float(a), float(b), float(c), float(d)
    n = 4
    n_comp = 2

    def obj(x):
        return 0.5 * ((x[0] - a) ** 2 + (x[1] - b) ** 2
                      + (x[2] - c) ** 2 + (x[3] - d) ** 2)

    def grad(x):
        return np.array([x[0] - a, x[1] - b, x[2] - c, x[3] - d])

    def G(x):
        return np.array([x[0], x[1]])

    def H(x):
        return np.array([x[2], x[3]])

    JG = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])
    JH = np.array([[0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

    x0 = np.array([max(a, 0.0), max(b, 0.0),
                   max(c, 0.0), max(d, 0.0)])

    return MPCCProblem(
        n=n, n_comp=n_comp, x0=x0,
        objective=obj, gradient=grad,
        comp_G=G, comp_H=H,
        comp_G_jacobian=lambda _x: JG,
        comp_H_jacobian=lambda _x: JH,
    )


def _eq_box_problem(p: float):
    """Equality-constrained quadratic with one trivial comp pair.

        min ½(x-1)² + ½(y-1)² + ½(s² + t²)
        s.t. x + y = p,  s ≥ 0 ⊥ t ≥ 0

    The comp pair (s, t) is decoupled from (x, y) and pins to s=t=0
    cleanly under any strategy.  IFT closed form on (x, y):
        x* = y* = p/2,  λ_h* = 1 - p/2
        dx*/dp = dy*/dp = 0.5
    """
    n, n_comp, n_eq = 4, 1, 1

    def obj(z):
        x, y, s, t = z
        return 0.5 * ((x - 1) ** 2 + (y - 1) ** 2 + s ** 2 + t ** 2)

    def grad(z):
        x, y, s, t = z
        return np.array([x - 1, y - 1, s, t])

    def heq(z):
        return np.array([z[0] + z[1] - p])

    def Jheq(_z):
        return np.array([[1.0, 1.0, 0.0, 0.0]])

    def G(z):
        return np.array([z[2]])

    def H(z):
        return np.array([z[3]])

    JG = np.array([[0.0, 0.0, 1.0, 0.0]])
    JH = np.array([[0.0, 0.0, 0.0, 1.0]])

    return MPCCProblem(
        n=n, n_comp=n_comp, n_eq=n_eq,
        x0=np.array([p / 2, p / 2, 0.5, 0.0]),
        objective=obj, gradient=grad,
        eq_constraints=heq, eq_jacobian=Jheq,
        comp_G=G, comp_H=H,
        comp_G_jacobian=lambda _z: JG,
        comp_H_jacobian=lambda _z: JH,
    )


def _biactive_problem():
    """A problem whose unique stationary point is biactive (s = t = 0).

        min s + t  s.t. s ≥ 0 ⊥ t ≥ 0
    """
    return MPCCProblem(
        n=2, n_comp=1, x0=np.array([0.5, 0.5]),
        objective=lambda x: x[0] + x[1],
        gradient=lambda _x: np.array([1.0, 1.0]),
        comp_G=lambda x: np.array([x[0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_G_jacobian=lambda _x: np.array([[1.0, 0.0]]),
        comp_H_jacobian=lambda _x: np.array([[0.0, 1.0]]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActiveRowLabels:
    def test_eq_only(self):
        p = _eq_box_problem(p=1.0)
        result = pympcc.solve(p, strategy="scholtes")
        labels = pympcc.active_row_labels(result, p)
        # Exactly one equality row, and exactly one of G_0 / H_0 active
        # (s=t=0 is biactive in fact, but the eq pair is trivial: pympcc
        # often labels both G and H active here; just check eq is present).
        assert ("h", 0) in labels

    def test_branch_select_two_pairs(self):
        # Pick (a,b,c,d) so optimum is x0=2, x1=0, x2=0, x3=3.
        # → I_G = {1}, I_H = {0}
        p = _branch_select(a=2.0, b=0.0, c=0.0, d=3.0)
        # Force the chosen branch by aligning x0.
        p.x0 = np.array([2.0, 0.0, 0.0, 3.0])
        result = pympcc.solve(p, strategy="scholtes")
        labels = pympcc.active_row_labels(result, p)
        kinds = [k for k, _ in labels]
        # Should see exactly one G label and one H label
        assert kinds.count("G") == 1
        assert kinds.count("H") == 1


class TestSkippedPaths:
    def test_not_converged_skipped(self):
        p = _eq_box_problem(p=1.0)
        # Build a faux non-converged result by running a real solve and
        # flipping success.
        result = pympcc.solve(p, strategy="scholtes")
        result.success = False

        sens = pympcc.sensitivity(
            result, p,
            dgrad_L_dp=np.zeros((p.n, 1)),
            dc_dp=np.zeros((1, 1)),  # any shape; validation happens later
        )
        assert sens.skipped_reason == "not_converged"
        assert sens.dx_dp.shape == (p.n, 1)
        assert sens.dlam_dp.shape == (0, 1)

    def test_biactive_skipped(self):
        prob = _biactive_problem()
        result = pympcc.solve(prob, strategy="scholtes")
        # The optimum is (0, 0) — biactive.
        sens = pympcc.sensitivity(
            result, prob,
            dgrad_L_dp=np.zeros((prob.n, 1)),
            dc_dp=np.zeros((0, 1)),
        )
        assert sens.skipped_reason == "biactive_pairs"


class TestEqualityNLPClosedForm:
    """Verify dx/dp matches analytical IFT on a parametric equality NLP."""

    @pytest.mark.parametrize("p_val", [0.5, 1.0, 1.5])
    def test_eq_box(self, p_val):
        prob = _eq_box_problem(p=p_val)
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        assert result.success

        labels = pympcc.active_row_labels(result, prob)
        # Find the equality-row index in `labels`.
        h_idx = labels.index(("h", 0))

        # ∂(∇_xL)/∂p = 0 (p only enters c).
        # ∂c/∂p = -1 for the equality x + y - p = 0.
        n_active = len(labels)
        dgrad_L_dp = np.zeros((prob.n, 1))
        dc_dp = np.zeros((n_active, 1))
        dc_dp[h_idx, 0] = -1.0

        # The trivial comp pair is biactive at the optimum, so TNLP
        # refinement skips and we hit the zero-multiplier fallback.
        # All constraints here are linear in x, so the fallback is exact.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=dgrad_L_dp, dc_dp=dc_dp,
            )
        assert sens.skipped_reason is None
        # dx*/dp = dy*/dp = 0.5; ds*/dp = dt*/dp = 0
        assert sens.dx_dp.shape == (prob.n, 1)
        assert abs(sens.dx_dp[0, 0] - 0.5) < 1e-6
        assert abs(sens.dx_dp[1, 0] - 0.5) < 1e-6


class TestBranchSelectFiniteDifference:
    """Verify dx/dp matches a finite-difference re-solve at the active branch."""

    def test_param_in_objective_only(self):
        # ½(x0-a)² + … with parameter a.  Branch x0=a, x2=0 active for a>1.
        p_val = 2.0
        prob = _branch_select(a=p_val, b=0.0, c=0.0, d=3.0)
        prob.x0 = np.array([p_val, 0.0, 0.0, 3.0])
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        assert result.success
        # Active set: I_G = {1} (x1 pinned), I_H = {0} (x2 pinned).
        labels = pympcc.active_row_labels(result, prob)
        n_active = len(labels)

        # ∂(∇_xL)/∂a where L = ½(x0-a)² + … + λ_G x1 + λ_H x2:
        # ∂(∇_x f)/∂a = (-1, 0, 0, 0).  Constraint terms (λ·∇c) are
        # constant in a so contribute nothing.
        dgrad_L_dp = np.array([[-1.0], [0.0], [0.0], [0.0]])
        # Active rows are (x1)=0 and (x2)=0; neither depends on a.
        dc_dp = np.zeros((n_active, 1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=dgrad_L_dp, dc_dp=dc_dp,
            )
        assert sens.skipped_reason is None

        # FD reference: re-solve at a + eps and a - eps.
        eps = 1e-3
        prob_p = _branch_select(a=p_val + eps, b=0.0, c=0.0, d=3.0)
        prob_p.x0 = np.array([p_val + eps, 0.0, 0.0, 3.0])
        prob_m = _branch_select(a=p_val - eps, b=0.0, c=0.0, d=3.0)
        prob_m.x0 = np.array([p_val - eps, 0.0, 0.0, 3.0])
        x_p = pympcc.solve(prob_p, strategy="scholtes").x
        x_m = pympcc.solve(prob_m, strategy="scholtes").x
        dx_fd = (x_p - x_m) / (2 * eps)

        # Sensitivity should match FD to ~1e-4
        np.testing.assert_allclose(sens.dx_dp[:, 0], dx_fd, atol=1e-4, rtol=1e-3)

    def test_two_parameters(self):
        # Two parameters: a (objective shift on x0) and d (shift on x3).
        a0, d0 = 2.0, 3.0
        prob = _branch_select(a=a0, b=0.0, c=0.0, d=d0)
        prob.x0 = np.array([a0, 0.0, 0.0, d0])
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        assert result.success

        labels = pympcc.active_row_labels(result, prob)
        n_active = len(labels)
        # Two parameters: column 0 = a, column 1 = d.
        dgrad_L_dp = np.zeros((prob.n, 2))
        dgrad_L_dp[0, 0] = -1.0   # ∂/∂a of (x0-a)
        dgrad_L_dp[3, 1] = -1.0   # ∂/∂d of (x3-d)
        dc_dp = np.zeros((n_active, 2))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=dgrad_L_dp, dc_dp=dc_dp,
            )
        assert sens.skipped_reason is None
        assert sens.dx_dp.shape == (4, 2)

        # FD verification
        eps = 1e-3
        for k, (da, dd) in enumerate([(eps, 0.0), (0.0, eps)]):
            prob_p = _branch_select(a=a0 + da, b=0.0, c=0.0, d=d0 + dd)
            prob_p.x0 = np.array([a0 + da, 0.0, 0.0, d0 + dd])
            prob_m = _branch_select(a=a0 - da, b=0.0, c=0.0, d=d0 - dd)
            prob_m.x0 = np.array([a0 - da, 0.0, 0.0, d0 - dd])
            x_p = pympcc.solve(prob_p, strategy="scholtes").x
            x_m = pympcc.solve(prob_m, strategy="scholtes").x
            dx_fd = (x_p - x_m) / (2 * eps)
            np.testing.assert_allclose(
                sens.dx_dp[:, k], dx_fd, atol=1e-4, rtol=1e-3,
            )


class TestShapeValidation:
    def test_dgrad_L_dp_wrong_n_rows(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes")
        with pytest.raises(ValueError, match="dgrad_L_dp"):
            pympcc.sensitivity(
                result, prob,
                dgrad_L_dp=np.zeros((prob.n + 1, 1)),  # wrong row count
                dc_dp=np.zeros((1, 1)),
            )

    def test_dc_dp_wrong_n_active(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes")
        labels = pympcc.active_row_labels(result, prob)
        with pytest.raises(ValueError, match="dc_dp"):
            pympcc.sensitivity(
                result, prob,
                dgrad_L_dp=np.zeros((prob.n, 1)),
                dc_dp=np.zeros((len(labels) + 5, 1)),  # wrong row count
            )

    def test_n_p_mismatch_between_blocks(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes")
        labels = pympcc.active_row_labels(result, prob)
        with pytest.raises(ValueError, match="parameter columns"):
            pympcc.sensitivity(
                result, prob,
                dgrad_L_dp=np.zeros((prob.n, 2)),     # n_p = 2
                dc_dp=np.zeros((len(labels), 3)),     # n_p = 3
            )

    def test_1d_inputs_promoted_to_single_column(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        labels = pympcc.active_row_labels(result, prob)
        h_idx = labels.index(("h", 0))

        dgrad = np.zeros(prob.n)                    # 1-D
        dc = np.zeros(len(labels))                  # 1-D
        dc[h_idx] = -1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=dgrad, dc_dp=dc,
            )
        assert sens.skipped_reason is None
        assert sens.dx_dp.shape == (prob.n, 1)


class TestZeroMultiplierWarning:
    def test_warns_when_no_tnlp_and_no_analytic_hessian(self):
        prob = _branch_select(a=2.0, b=0.0, c=0.0, d=3.0)
        prob.x0 = np.array([2.0, 0.0, 0.0, 3.0])
        # Solve WITHOUT tnlp_refine.  Constraints here are linear so the
        # answer would still be exact; the warning is informational.
        result = pympcc.solve(prob, strategy="scholtes")
        assert result.tnlp_refined is None

        labels = pympcc.active_row_labels(result, prob)
        with pytest.warns(UserWarning, match="zero multipliers"):
            pympcc.sensitivity(
                result, prob,
                dgrad_L_dp=np.array([[-1.0], [0.0], [0.0], [0.0]]),
                dc_dp=np.zeros((len(labels), 1)),
            )

    def test_no_warning_when_tnlp_refined(self):
        prob = _branch_select(a=2.0, b=0.0, c=0.0, d=3.0)
        prob.x0 = np.array([2.0, 0.0, 0.0, 3.0])
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        assert result.tnlp_refined is not None and result.tnlp_refined.success

        labels = pympcc.active_row_labels(result, prob)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # turn any warning into an error
            pympcc.sensitivity(
                result, prob,
                dgrad_L_dp=np.array([[-1.0], [0.0], [0.0], [0.0]]),
                dc_dp=np.zeros((len(labels), 1)),
            )


class TestSensitivityResultMetadata:
    def test_kkt_condition_populated(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        labels = pympcc.active_row_labels(result, prob)
        h_idx = labels.index(("h", 0))
        dc = np.zeros(len(labels))
        dc[h_idx] = -1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=np.zeros(prob.n), dc_dp=dc,
            )
        assert sens.kkt_condition is not None and sens.kkt_condition > 0.0

    def test_default_flags(self):
        prob = _eq_box_problem(p=1.0)
        result = pympcc.solve(prob, strategy="scholtes", tnlp_refine=True)
        labels = pympcc.active_row_labels(result, prob)
        h_idx = labels.index(("h", 0))
        dc = np.zeros(len(labels))
        dc[h_idx] = -1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sens = pympcc.sensitivity(
                result, prob, dgrad_L_dp=np.zeros(prob.n), dc_dp=dc,
            )
        assert sens.used_pseudoinverse is False
        assert sens.rank_deficit == 0
        assert sens.skipped_reason is None
