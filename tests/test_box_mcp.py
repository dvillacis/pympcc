"""Tests for §4.9 box-MCP / doubly-bounded canonical form.

``comp_box_pairs`` declares ``xl[var_idx] <= x[var_idx] <= xu[var_idx]
⊥ F_fn(x)`` and is dispatched by bound finiteness:

* lower-only finite → comp pair ``(x - ell) ≥ 0 ⊥ F(x) ≥ 0``
* upper-only finite → comp pair ``(u - x) ≥ 0 ⊥ -F(x) ≥ 0``
* both infinite (free) → equality ``F(x) = 0`` appended to eq block
* both finite (doubly-bounded) → ``NotImplementedError`` (deferred)
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc import MPCCProblem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quad_obj(target):
    """Return (f, grad) for f(x) = sum (x_i - target_i)^2."""
    target = np.asarray(target, dtype=float)

    def f(x):
        return float(np.sum((x - target) ** 2))

    def g(x):
        return 2.0 * (np.asarray(x, dtype=float) - target)

    return f, g


# ---------------------------------------------------------------------------
# Construction — lower-only
# ---------------------------------------------------------------------------

class TestLowerOnly:
    def test_pure_lower_only_xl_zero_builds(self):
        """Lower-only at xl=0 builds comp_G/H with shifted G = x - 0 = x."""
        f, g = _quad_obj([1.0, 1.0])
        p = MPCCProblem(
            n=2, n_comp=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([x[1] - 1.0]),
                 lambda x: np.array([0.0, 1.0])),
                (1, lambda x: np.array([x[0] - 1.0]),
                 lambda x: np.array([1.0, 0.0])),
            ],
        )
        assert p.n_comp == 2
        np.testing.assert_allclose(p.comp_G(p.x0), [0.5, 0.5])
        np.testing.assert_allclose(p.comp_H(p.x0), [-0.5, -0.5])

    def test_lower_only_with_shift(self):
        """Lower-only at xl=ell != 0 emits G = x - ell."""
        f, g = _quad_obj([3.0])
        p = MPCCProblem(
            n=1, n_comp=0,
            x0=np.array([2.5]),
            xl=np.array([2.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([x[0] - 4.0]),
                 lambda x: np.array([1.0])),
            ],
        )
        np.testing.assert_allclose(p.comp_G(p.x0), [0.5])  # 2.5 - 2.0
        np.testing.assert_allclose(p.comp_H(p.x0), [-1.5])  # 2.5 - 4.0

    def test_lower_only_g_jacobian_is_identity_row(self):
        """G Jacobian for lower-only is e_{var_idx} (single 1)."""
        f, g = _quad_obj([0.0, 0.0])
        p = MPCCProblem(
            n=2, n_comp=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (1, lambda x: np.array([x[0]]),
                 lambda x: np.array([1.0, 0.0])),
            ],
        )
        J = p.comp_G_jacobian(p.x0)
        assert J.shape == (1, 2)
        np.testing.assert_array_equal(J, [[0.0, 1.0]])

    def test_lower_only_solves_to_optimum(self):
        """Quadratic with two lower-only pairs converges to interior (1, 1)."""
        f, g = _quad_obj([1.0, 1.0])
        p = MPCCProblem(
            n=2, n_comp=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([x[1] - 1.0]),
                 lambda x: np.array([0.0, 1.0])),
                (1, lambda x: np.array([x[0] - 1.0]),
                 lambda x: np.array([1.0, 0.0])),
            ],
        )
        r = pympcc.solve(p, strategy="scholtes")
        assert r.status == 0
        np.testing.assert_allclose(r.x, [1.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Construction — upper-only
# ---------------------------------------------------------------------------

class TestUpperOnly:
    def test_upper_only_g_is_u_minus_x(self):
        """Upper-only emits G = u - x[var_idx]."""
        f, g = _quad_obj([4.0])
        p = MPCCProblem(
            n=1, n_comp=0,
            x0=np.array([3.0]),
            xu=np.array([5.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([2.0 - x[0]]),
                 lambda x: np.array([-1.0])),
            ],
        )
        np.testing.assert_allclose(p.comp_G(p.x0), [2.0])  # 5 - 3
        # H = -F  → -(2 - 3) = 1
        np.testing.assert_allclose(p.comp_H(p.x0), [1.0])

    def test_upper_only_g_jacobian_is_negative_identity(self):
        """G Jacobian for upper-only has -1 at var_idx."""
        f, g = _quad_obj([0.0])
        p = MPCCProblem(
            n=1, n_comp=0,
            x0=np.array([3.0]),
            xu=np.array([5.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([2.0 - x[0]]),
                 lambda x: np.array([-1.0])),
            ],
        )
        J = p.comp_G_jacobian(p.x0)
        np.testing.assert_array_equal(J, [[-1.0]])

    def test_upper_only_h_negates_F(self):
        """H Jacobian for upper-only is -∇F."""
        f, g = _quad_obj([0.0])
        p = MPCCProblem(
            n=1, n_comp=0,
            x0=np.array([3.0]),
            xu=np.array([5.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([2.0 - x[0]]),
                 lambda x: np.array([-1.0])),
            ],
        )
        JH = p.comp_H_jacobian(p.x0)
        # ∇F = -1 → ∇H = +1
        np.testing.assert_array_equal(JH, [[1.0]])

    def test_upper_only_solves_to_optimum(self):
        """min (x - 4)^2 s.t. x ≤ 5 ⊥ (2 - x); converges to x = 2."""
        f, g = _quad_obj([4.0])
        p = MPCCProblem(
            n=1, n_comp=0,
            x0=np.array([3.0]),
            xu=np.array([5.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([2.0 - x[0]]),
                 lambda x: np.array([-1.0])),
            ],
        )
        r = pympcc.solve(p, strategy="scholtes")
        assert r.status == 0
        # F = 0 → x = 2 (interior of (-inf, 5])
        np.testing.assert_allclose(r.x, [2.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Construction — free (CNS equality row)
# ---------------------------------------------------------------------------

class TestFree:
    def test_free_appends_to_eq(self):
        """Free entry appends F(x) = 0 to eq block when base eq is None."""
        f, g = _quad_obj([2.0, 0.0])
        # Base comp pair to satisfy n_comp >= 1
        p = MPCCProblem(
            n=2, n_comp=1, n_eq=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([-np.inf, 0.0]),
            objective=f, gradient=g,
            comp_G=lambda z: np.array([z[1]]),
            comp_H=lambda z: np.array([10.0 - z[1]]),
            comp_G_jacobian=lambda z: np.array([[0.0, 1.0]]),
            comp_H_jacobian=lambda z: np.array([[0.0, -1.0]]),
            comp_box_pairs=[
                (0, lambda z: np.array([z[0] - z[1]]),
                 lambda z: np.array([1.0, -1.0])),
            ],
        )
        assert p.n_eq == 1
        # eq(x0) = x0[0] - x0[1] = 0
        np.testing.assert_allclose(p.eq_constraints(p.x0), [0.0])
        np.testing.assert_array_equal(p.eq_jacobian(p.x0), [[1.0, -1.0]])

    def test_free_extends_existing_eq(self):
        """Free entry appends to a non-empty base eq block."""
        f, g = _quad_obj([0.0, 0.0])
        p = MPCCProblem(
            n=2, n_comp=1, n_eq=1,
            x0=np.array([0.0, 0.0]),
            xl=np.array([-np.inf, 0.0]),
            objective=f, gradient=g,
            comp_G=lambda z: np.array([z[1]]),
            comp_H=lambda z: np.array([1.0 - z[1]]),
            comp_G_jacobian=lambda z: np.array([[0.0, 1.0]]),
            comp_H_jacobian=lambda z: np.array([[0.0, -1.0]]),
            eq_constraints=lambda z: np.array([z[0] + z[1]]),
            eq_jacobian=lambda z: np.array([[1.0, 1.0]]),
            comp_box_pairs=[
                (0, lambda z: np.array([z[0] - 2.0]),
                 lambda z: np.array([1.0, 0.0])),
            ],
        )
        assert p.n_eq == 2
        np.testing.assert_allclose(p.eq_constraints(p.x0), [0.0, -2.0])
        np.testing.assert_array_equal(p.eq_jacobian(p.x0),
                                      [[1.0, 1.0], [1.0, 0.0]])

    def test_free_solves_with_base_comp(self):
        """Mixed free + base comp solves to the right optimum."""
        f, g = _quad_obj([2.0, 0.0])
        p = MPCCProblem(
            n=2, n_comp=1, n_eq=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([-np.inf, 0.0]),
            objective=f, gradient=g,
            comp_G=lambda z: np.array([z[1]]),
            comp_H=lambda z: np.array([10.0 - z[1]]),
            comp_G_jacobian=lambda z: np.array([[0.0, 1.0]]),
            comp_H_jacobian=lambda z: np.array([[0.0, -1.0]]),
            comp_box_pairs=[
                (0, lambda z: np.array([z[0] - z[1]]),
                 lambda z: np.array([1.0, -1.0])),
            ],
        )
        r = pympcc.solve(p, strategy="scholtes")
        assert r.status == 0
        # y in {0, 10} (comp); x = y (free); min over y in {0, 10} of (y-2)^2 + y^2
        # y=0 → 4; y=10 → 164. So y=0, x=0.
        np.testing.assert_allclose(r.x, [0.0, 0.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Mixed lower + upper + free
# ---------------------------------------------------------------------------

class TestMixed:
    def test_lower_upper_free_combine(self):
        """All three categories in one problem auto-bump n_comp / n_eq."""
        f, g = _quad_obj([1.0, 3.2, 0.0])
        p = MPCCProblem(
            n=3, n_comp=0, n_eq=0,
            x0=np.array([1.0, 4.0, 2.5]),
            xl=np.array([0.0, -np.inf, -np.inf]),
            xu=np.array([np.inf, 5.0, np.inf]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda z: np.array([z[2] - z[0]]),       # lower-only
                 lambda z: np.array([-1.0, 0.0, 1.0])),
                (1, lambda z: np.array([z[1] - 3.0]),        # upper-only
                 lambda z: np.array([0.0, 1.0, 0.0])),
                (2, lambda z: np.array([z[2] - 2.0]),        # free
                 lambda z: np.array([0.0, 0.0, 1.0])),
            ],
        )
        assert p.n_comp == 2  # 1 lower + 1 upper
        assert p.n_eq == 1    # 1 free
        r = pympcc.solve(p, strategy="scholtes")
        assert r.status == 0
        # Free: w = 2.  Comp: x ∈ {0, 2}, y ∈ {3, 5}.  Min objective:
        # (x-1)^2 + (y-3.2)^2: x=0 → 1, x=2 → 1.  y=3 → 0.04, y=5 → 3.24.
        # x=0 or 2 is degenerate; Scholtes typically picks the lower-bound branch.
        assert r.x[1] == pytest.approx(3.0, abs=1e-5)
        assert r.x[2] == pytest.approx(2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# fd fallback for F_jac
# ---------------------------------------------------------------------------

class TestFiniteDifferenceJacobian:
    def test_two_tuple_uses_fd(self):
        """2-tuple (var_idx, F_fn) form falls back to forward fd for F_jac."""
        f, g = _quad_obj([1.0, 1.0])
        p = MPCCProblem(
            n=2, n_comp=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            objective=f, gradient=g,
            comp_box_pairs=[
                (0, lambda x: np.array([x[1] - 1.0])),  # 2-tuple, fd row
                (1, lambda x: np.array([x[0] - 1.0]),
                 lambda x: np.array([1.0, 0.0])),
            ],
        )
        JH = p.comp_H_jacobian(p.x0)
        # Row 0: ∇(y - 1) = [0, 1] (fd should match exactly to ~1e-7)
        np.testing.assert_allclose(JH[0], [0.0, 1.0], atol=1e-6)
        np.testing.assert_array_equal(JH[1], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrors:
    def test_doubly_bounded_raises(self):
        """Both bounds finite → NotImplementedError pointing to §4.9 Phase 2."""
        f, g = _quad_obj([0.0])
        with pytest.raises(NotImplementedError, match="doubly-bounded"):
            MPCCProblem(
                n=1, n_comp=0,
                x0=np.array([0.5]),
                xl=np.array([0.0]),
                xu=np.array([1.0]),
                objective=f, gradient=g,
                comp_box_pairs=[(0, lambda x: np.array([x[0] - 0.5]))],
            )

    def test_pure_free_no_base_raises(self):
        """All-free + no base comp_G is a pure CNS → ValueError."""
        f, g = _quad_obj([0.0])
        with pytest.raises(ValueError, match="pure square nonlinear system"):
            MPCCProblem(
                n=1, n_comp=0,
                x0=np.array([0.5]),
                objective=f, gradient=g,
                comp_box_pairs=[
                    (0, lambda x: np.array([x[0] - 1.0]),
                     lambda x: np.array([1.0])),
                ],
            )

    def test_mutual_exclusion_with_var_pairs(self):
        """comp_box_pairs and comp_var_pairs cannot coexist."""
        f, g = _quad_obj([0.0, 0.0])
        with pytest.raises(ValueError, match="cannot be combined"):
            MPCCProblem(
                n=2, n_comp=2,
                x0=np.array([0.5, 0.5]),
                xl=np.array([0.0, 0.0]),
                objective=f, gradient=g,
                comp_var_pairs=[
                    (0, lambda x: np.array([x[1]]),
                     lambda x: np.array([0.0, 1.0])),
                ],
                comp_box_pairs=[
                    (1, lambda x: np.array([x[0]]),
                     lambda x: np.array([1.0, 0.0])),
                ],
            )

    def test_var_idx_out_of_range(self):
        f, g = _quad_obj([0.0])
        with pytest.raises(ValueError, match="out of range"):
            MPCCProblem(
                n=1, n_comp=0,
                x0=np.array([0.5]),
                xl=np.array([0.0]),
                objective=f, gradient=g,
                comp_box_pairs=[
                    (5, lambda x: np.array([x[0]]),
                     lambda x: np.array([1.0])),
                ],
            )

    def test_duplicate_var_idx(self):
        f, g = _quad_obj([0.0, 0.0])
        with pytest.raises(ValueError, match="appears in more than one entry"):
            MPCCProblem(
                n=2, n_comp=0,
                x0=np.array([0.5, 0.5]),
                xl=np.array([0.0, 0.0]),
                objective=f, gradient=g,
                comp_box_pairs=[
                    (0, lambda x: np.array([x[0]]),
                     lambda x: np.array([1.0, 0.0])),
                    (0, lambda x: np.array([x[1]]),
                     lambda x: np.array([0.0, 1.0])),
                ],
            )

    def test_bad_tuple_length(self):
        f, g = _quad_obj([0.0])
        with pytest.raises(ValueError, match=r"must be \(var_idx, F_fn\)"):
            MPCCProblem(
                n=1, n_comp=0,
                x0=np.array([0.5]),
                xl=np.array([0.0]),
                objective=f, gradient=g,
                comp_box_pairs=[(0,)],
            )

    def test_mismatched_n_comp_when_box_only(self):
        """User passing n_comp > 0 with no base comp_G + box-pairs is rejected."""
        f, g = _quad_obj([0.0])
        with pytest.raises(ValueError, match="set n_comp=0"):
            MPCCProblem(
                n=1, n_comp=2,  # wrong: should be 0
                x0=np.array([0.5]),
                xl=np.array([0.0]),
                objective=f, gradient=g,
                comp_box_pairs=[
                    (0, lambda x: np.array([x[0]]),
                     lambda x: np.array([1.0])),
                ],
            )


# ---------------------------------------------------------------------------
# Composition with derivatives='fd' / 'jax'
# ---------------------------------------------------------------------------

class TestDerivativesDefault:
    def test_derivatives_fd_with_box_pairs(self):
        """derivatives='fd' fills gradient + base eq Jac when base eq is set."""
        f, _ = _quad_obj([1.0, 1.0])
        p = MPCCProblem(
            n=2, n_comp=0,
            x0=np.array([0.5, 0.5]),
            xl=np.array([0.0, 0.0]),
            derivatives="fd",
            objective=f,
            comp_box_pairs=[
                (0, lambda x: np.array([x[1] - 1.0]),
                 lambda x: np.array([0.0, 1.0])),
                (1, lambda x: np.array([x[0] - 1.0]),
                 lambda x: np.array([1.0, 0.0])),
            ],
        )
        # gradient resolved to fd
        np.testing.assert_allclose(p.gradient(p.x0), [-1.0, -1.0], atol=1e-5)
