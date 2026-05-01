"""Tests for the presolve layer (pinned-variable elimination + dead-pair pruning)."""

from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc._presolve import PresolveMap, presolve


# ======================================================================= #
# Builders                                                                  #
# ======================================================================= #

def _basic(n=2, n_comp=1, *, xl=None, xu=None,
           x0=None, comp_G=None, comp_G_jac=None, comp_G_sp=None,
           comp_H=None, comp_H_jac=None, comp_H_sp=None):
    if x0 is None:
        x0 = np.full(n, 0.5)
    if comp_G is None:
        comp_G     = lambda x: np.array([x[0]])
        comp_G_jac = lambda x: np.array([[1.0] + [0.0] * (n - 1)])
    if comp_H is None:
        comp_H     = lambda x: np.array([x[1]])
        comp_H_jac = lambda x: np.array([[0.0, 1.0] + [0.0] * (n - 2)])
    kw = {}
    if xl is not None: kw["xl"] = xl
    if xu is not None: kw["xu"] = xu
    if comp_G_sp is not None: kw["comp_G_jacobian_sparsity"] = comp_G_sp
    if comp_H_sp is not None: kw["comp_H_jacobian_sparsity"] = comp_H_sp
    return pympcc.MPCCProblem(
        n=n, n_comp=n_comp, x0=np.asarray(x0, dtype=float),
        objective=lambda x: float(np.sum((x - 1.0) ** 2)),
        gradient=lambda x: 2.0 * (x - 1.0),
        comp_G=comp_G, comp_G_jacobian=comp_G_jac,
        comp_H=comp_H, comp_H_jacobian=comp_H_jac,
        **kw,
    )


# ======================================================================= #
# Identity / no-op                                                          #
# ======================================================================= #

class TestNoOp:
    def test_no_pinned_no_dead_returns_identity(self):
        p = _basic()
        reduced, pmap = presolve(p)
        assert reduced is p                # exact same instance
        assert isinstance(pmap, PresolveMap)
        assert pmap.is_identity
        assert pmap.fixed_vars.size == 0
        assert pmap.keep_comp.size == p.n_comp

    def test_solver_no_op_matches_no_presolve(self):
        p = _basic()
        r1 = pympcc.solve(p, strategy="scholtes",
                          ipopt_options={"max_iter": 50, "tol": 1e-7})
        r2 = pympcc.solve(p, strategy="scholtes",
                          ipopt_options={"max_iter": 50, "tol": 1e-7},
                          presolve=True)
        assert r1.success and r2.success
        np.testing.assert_allclose(r1.x, r2.x, atol=1e-6)


# ======================================================================= #
# Pinned variables                                                          #
# ======================================================================= #

class TestPinnedVariables:
    def test_pinned_single_variable(self):
        # n=3 with x[2] pinned to 0.5
        n = 3
        xl = np.array([-np.inf, -np.inf, 0.5])
        xu = np.array([+np.inf, +np.inf, 0.5])
        p = _basic(n=n,
                   xl=xl, xu=xu,
                   x0=np.array([0.5, 0.5, 0.5]),
                   comp_G=lambda x: np.array([x[0]]),
                   comp_G_jac=lambda x: np.array([[1.0, 0.0, 0.0]]),
                   comp_H=lambda x: np.array([x[1] + 0.1 * x[2]]),
                   comp_H_jac=lambda x: np.array([[0.0, 1.0, 0.1]]))
        reduced, pmap = presolve(p)
        assert reduced.n == 2
        assert reduced.n_comp == 1
        assert pmap.fixed_vars.tolist() == [2]
        assert pmap.fixed_vals.tolist() == [0.5]

        # Reduced callables must reinject x[2]=0.5.
        x_red = np.array([0.0, 0.0])
        H_red = reduced.comp_H(x_red)
        assert H_red[0] == pytest.approx(0.05)  # 0 + 0.1*0.5

    def test_pinned_solve_round_trip(self):
        n = 3
        xl = np.array([-np.inf, -np.inf, 0.5])
        xu = np.array([+np.inf, +np.inf, 0.5])
        p = _basic(n=n,
                   xl=xl, xu=xu,
                   x0=np.array([0.5, 0.5, 0.5]),
                   comp_G=lambda x: np.array([x[0]]),
                   comp_G_jac=lambda x: np.array([[1.0, 0.0, 0.0]]),
                   comp_H=lambda x: np.array([x[1]]),
                   comp_H_jac=lambda x: np.array([[0.0, 1.0, 0.0]]))
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 50, "tol": 1e-7},
                              presolve=True)
        assert result.success
        assert len(result.x) == n
        assert result.x[2] == pytest.approx(0.5, abs=1e-9)


# ======================================================================= #
# Dead-pair pruning                                                         #
# ======================================================================= #

class TestDeadPair:
    def test_dead_comp_pair_pruned(self):
        # n=2, two comp pairs.  Pair 0: G=x[0], H=x[1].
        # Pair 1: G=5.0 (constant, structurally empty Jacobian), H=x[0]+x[1].
        # Sparsity is supplied so the dead-pair detector activates.
        n, n_comp = 2, 2
        sG = (np.array([0]), np.array([0]))                     # only row 0
        sH = (np.array([0, 1, 1]), np.array([1, 0, 1]))         # rows 0, 1
        p = pympcc.MPCCProblem(
            n=n, n_comp=n_comp,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: float(np.sum((x - 1.0) ** 2)),
            gradient=lambda x: 2.0 * (x - 1.0),
            comp_G=lambda x: np.array([x[0], 5.0]),
            comp_G_jacobian=lambda x: np.array([1.0]),  # COO values
            comp_G_jacobian_sparsity=sG,
            comp_H=lambda x: np.array([x[1], x[0] + x[1]]),
            comp_H_jacobian=lambda x: np.array([1.0, 1.0, 1.0]),
            comp_H_jacobian_sparsity=sH,
        )
        reduced, pmap = presolve(p)
        assert reduced.n_comp == 1
        assert pmap.keep_comp.tolist() == [0]
        # Reduced G/H operate on remaining pair only.
        np.testing.assert_allclose(reduced.comp_G(p.x0), [0.5])
        np.testing.assert_allclose(reduced.comp_H(p.x0), [0.5])

    def test_expand_result_fills_pruned_pair(self):
        n, n_comp = 2, 2
        sG = (np.array([0]), np.array([0]))
        sH = (np.array([0, 1, 1]), np.array([1, 0, 1]))
        p = pympcc.MPCCProblem(
            n=n, n_comp=n_comp,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            comp_G=lambda x: np.array([x[0], 5.0]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=sG,
            comp_H=lambda x: np.array([x[1], x[0] + x[1]]),
            comp_H_jacobian=lambda x: np.array([1.0, 1.0, 1.0]),
            comp_H_jacobian_sparsity=sH,
        )
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 50, "tol": 1e-7},
                              presolve=True)
        assert result.success
        assert len(result.G) == n_comp
        assert len(result.H) == n_comp
        # Pruned pair index 1 must have G=5.0 in the expanded result.
        assert result.G[1] == pytest.approx(5.0)


# ======================================================================= #
# Combined                                                                  #
# ======================================================================= #

class TestCombined:
    def test_pinned_plus_pruned_round_trip(self):
        # n=3 with x[2] pinned, two comp pairs with pair 1 dead.
        n, n_comp = 3, 2
        xl = np.array([-np.inf, -np.inf, 0.7])
        xu = np.array([+np.inf, +np.inf, 0.7])
        sG = (np.array([0]), np.array([0]))
        sH = (np.array([0, 1, 1]), np.array([1, 0, 1]))
        p = pympcc.MPCCProblem(
            n=n, n_comp=n_comp,
            x0=np.array([0.5, 0.5, 0.7]),
            xl=xl, xu=xu,
            objective=lambda x: float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + x[2] ** 2),
            gradient=lambda x: np.array([2.0 * (x[0] - 2.0),
                                         2.0 * (x[1] - 1.0),
                                         2.0 * x[2]]),
            comp_G=lambda x: np.array([x[0], 5.0]),
            comp_G_jacobian=lambda x: np.array([1.0]),
            comp_G_jacobian_sparsity=sG,
            comp_H=lambda x: np.array([x[1], x[0] + x[1]]),
            comp_H_jacobian=lambda x: np.array([1.0, 1.0, 1.0]),
            comp_H_jacobian_sparsity=sH,
        )
        reduced, pmap = presolve(p)
        assert reduced.n == 2
        assert reduced.n_comp == 1
        assert pmap.fixed_vars.tolist() == [2]
        assert pmap.keep_comp.tolist() == [0]

        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 50, "tol": 1e-7},
                              presolve=True)
        assert result.success
        assert len(result.x) == n
        assert result.x[2] == pytest.approx(0.7, abs=1e-9)
        assert len(result.G) == n_comp
        assert result.G[1] == pytest.approx(5.0)


# ======================================================================= #
# End-to-end composition (§7.3.5)                                           #
# ======================================================================= #

class TestComposition:
    """Verify multiple presolve passes fire together and the round-trip
    through ``solve(presolve=True)`` matches ``solve(presolve=False)`` in
    both ``x`` and ``mult_g`` blocks (modulo documented info loss on
    promoted/prefix-eq rows)."""

    def _build_multi_pass_problem(self):
        # Synthetic MPCC engineered so several presolve passes all fire:
        #
        #   x[0] : pinned via xl[0] == xu[0] == 0.5     (pinned-var, A1)
        #   x[2] : FBBT tightens xu[2] from 1.0 to 0.8  (FBBT, A2)
        #          via the linear ineq x[2] - 0.8 ≤ 0
        #   ineq[1] : structurally empty row, feasible  (empty-row, A4)
        #
        #   pair 0 : G_0 = 5.0 (const, empty G-row)     (dead, B1)
        #   pair 1 : H_1 = 0.0 (const, empty H-row)     (forced, B2)
        #            G_1 = x[1]+2 ≥ 1 over [-1,1] gets promoted to ineq.
        #   pair 2 : G_2 = x[3] + 1 linear ≥ 1 over [0,2] (prefix-eq, B3)
        #            so H_2 = x[1]-x[3] = 0 is enforced as eq.
        #   pair 3 : G_3 = x[1]+3, H_3 = x[2]-0.5       (survives)
        n, n_comp, n_ineq = 4, 4, 2
        xl = np.array([0.5, -1.0, 0.5, 0.0])
        xu = np.array([0.5,  1.0, 1.0, 2.0])
        x0 = np.array([0.5, 0.0, 0.6, 1.0])

        # Linear ineq: row 0 nonempty (x[2] - 0.8 ≤ 0); row 1 empty (0.0).
        ineq_sp = (np.array([0], dtype=np.intp), np.array([2], dtype=np.intp))

        def ineq(x):
            return np.array([x[2] - 0.8, 0.0])

        def ineq_jac(x):
            return np.array([1.0])  # COO values for row-0 only

        # G: rows 1, 2, 3 nonempty (pair 0 dead).
        sG = (np.array([1, 2, 3], dtype=np.intp),
              np.array([1, 3, 1], dtype=np.intp))

        def comp_G(x):
            return np.array([5.0,           # dead
                             x[1] + 2.0,    # forced (gets promoted)
                             x[3] + 1.0,    # prefix-eq
                             x[1]])         # normal — spans 0 over [-1,1]

        def comp_G_jac(x):
            return np.array([1.0, 1.0, 1.0])

        # H: rows 0, 2, 3 nonempty (pair 1 forced via empty H-row).
        # Pair 2's H is x[1] - x[3] which is structurally two-column.
        sH = (np.array([0, 2, 2, 3], dtype=np.intp),
              np.array([3, 1, 3, 2], dtype=np.intp))

        def comp_H(x):
            return np.array([x[3],          # dead pair (H = 0 at x[3]=0)
                             0.0,            # forced: structurally zero
                             x[1] - x[3],    # prefix-eq target (drives to 0)
                             x[2] - 0.5])    # normal

        def comp_H_jac(x):
            # Same order as sH: (0,3), (2,1), (2,3), (3,2)
            return np.array([1.0, 1.0, -1.0, 1.0])

        return pympcc.MPCCProblem(
            n=n, n_comp=n_comp, n_ineq=n_ineq,
            x0=x0, xl=xl, xu=xu,
            objective=lambda x: float(
                (x[0] - 0.5) ** 2 + (x[1] - 0.4) ** 2
                + (x[2] - 0.5) ** 2 + (x[3]) ** 2
            ),
            gradient=lambda x: np.array([
                2.0 * (x[0] - 0.5),
                2.0 * (x[1] - 0.4),
                2.0 * (x[2] - 0.5),
                2.0 * x[3],
            ]),
            ineq_constraints=ineq,
            ineq_jacobian=ineq_jac,
            ineq_jacobian_sparsity=ineq_sp,
            comp_G=comp_G,
            comp_G_jacobian=comp_G_jac,
            comp_G_jacobian_sparsity=sG,
            comp_H=comp_H,
            comp_H_jacobian=comp_H_jac,
            comp_H_jacobian_sparsity=sH,
        ), n, n_comp

    def test_multiple_passes_fire(self):
        """The constructed problem triggers multiple passes; presolve is
        not the identity map."""
        p, n, n_comp = self._build_multi_pass_problem()
        reduced, pmap = presolve(p)
        assert not pmap.is_identity, (
            "expected at least one presolve pass to fire on this problem"
        )
        # Pinned-var pass (A1): x[0] pinned by bounds.
        assert 0 in pmap.fixed_vars.tolist(), (
            f"x[0] should be pinned; fixed_vars={pmap.fixed_vars.tolist()}"
        )
        # Dead-pair pass (B1): pair 0 dropped (G_0 = 5.0 const).
        assert 0 not in pmap.keep_comp.tolist(), (
            f"pair 0 should be dead; keep_comp={pmap.keep_comp.tolist()}"
        )
        # Forced pass (B2): pair 1 promoted via empty H-row (G is the
        # surviving side, hence ``promote_G``).
        assert 1 in pmap.promote_G.tolist(), (
            f"pair 1 should be in promote_G (H_1 ≡ 0); "
            f"promote_G={pmap.promote_G.tolist()}"
        )
        # Prefix-eq pass (B3): pair 2 dropped, H_2 forced to equality.
        assert 2 in pmap.prefix_H_eq.tolist(), (
            f"pair 2 should be in prefix_H_eq; "
            f"prefix_H_eq={pmap.prefix_H_eq.tolist()}"
        )
        # Pair 3 is the only one still in keep_comp.
        assert pmap.keep_comp.tolist() == [3]

    def test_round_trip_feasibility_for_surviving_constraints(self):
        """The expanded ``result`` is feasible w.r.t. bounds, surviving
        linear inequalities, and surviving comp pairs.  Dropped pairs are
        intentionally not enforced (that is the presolve contract — see
        ROADMAP §1 B1/B2/B3 docs on info loss); they are listed but only
        the surviving pairs are checked here."""
        p, n, n_comp = self._build_multi_pass_problem()
        _, pmap = presolve(p)
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 200, "tol": 1e-8},
                              presolve=True)
        assert result.success
        x = result.x
        # Bounds.
        assert np.all(x >= np.asarray(p.xl) - 1e-7)
        assert np.all(x <= np.asarray(p.xu) + 1e-7)
        # Surviving inequality rows only (empty-row pass dropped the rest).
        g_full = np.asarray(p.ineq_constraints(x))
        kept_ineq = pmap.keep_ineq if pmap.keep_ineq is not None else slice(None)
        assert np.all(g_full[kept_ineq] <= 1e-6), (
            f"surviving ineq violated: {g_full[kept_ineq]}"
        )
        # Complementarity holds for surviving pairs only.
        G_full = np.asarray(p.comp_G(x))
        H_full = np.asarray(p.comp_H(x))
        keep = pmap.keep_comp
        assert np.all(G_full[keep] >= -1e-6), f"G violated: {G_full[keep]}"
        assert np.all(H_full[keep] >= -1e-6), f"H violated: {H_full[keep]}"
        comp_resid = float(np.max(np.abs(G_full[keep] * H_full[keep])))
        assert comp_resid < 1e-4, (
            f"surviving complementarity violated: max |G*H| = {comp_resid}"
        )

    def test_round_trip_shapes_match_original(self):
        """Result is reported in original problem space regardless of
        which passes fired during presolve."""
        p, n, n_comp = self._build_multi_pass_problem()
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 200, "tol": 1e-8},
                              presolve=True)
        assert result.success
        assert result.x.shape == (n,)
        assert result.G.shape == (n_comp,)
        assert result.H.shape == (n_comp,)
        # Pinned x[0] reported at its fixed value.
        assert result.x[0] == pytest.approx(0.5, abs=1e-9)
        # Dead pair G_0 reported at its constant value.
        assert result.G[0] == pytest.approx(5.0, abs=1e-9)


# ======================================================================= #
# History expansion                                                         #
# ======================================================================= #

class TestHistory:
    def test_history_x_expanded(self):
        n = 3
        xl = np.array([-np.inf, -np.inf, 0.25])
        xu = np.array([+np.inf, +np.inf, 0.25])
        p = _basic(n=n,
                   xl=xl, xu=xu,
                   x0=np.array([0.5, 0.5, 0.25]),
                   comp_G=lambda x: np.array([x[0]]),
                   comp_G_jac=lambda x: np.array([[1.0, 0.0, 0.0]]),
                   comp_H=lambda x: np.array([x[1]]),
                   comp_H_jac=lambda x: np.array([[0.0, 1.0, 0.0]]))
        result = pympcc.solve(p, strategy="scholtes",
                              ipopt_options={"max_iter": 50, "tol": 1e-7},
                              presolve=True)
        assert result.history, "expected at least one outer iteration"
        for info in result.history:
            assert len(info.x) == n
            assert info.x[2] == pytest.approx(0.25, abs=1e-9)


# ======================================================================= #
# Strategy parametrisation                                                  #
# ======================================================================= #

class TestStrategies:
    @pytest.mark.parametrize("strategy", [
        "scholtes", "smoothing", "lin_fukushima", "slack", "direct",
    ])
    def test_pinned_var_across_strategies(self, strategy):
        n = 3
        xl = np.array([-np.inf, -np.inf, 0.4])
        xu = np.array([+np.inf, +np.inf, 0.4])
        p = _basic(n=n,
                   xl=xl, xu=xu,
                   x0=np.array([0.5, 0.5, 0.4]),
                   comp_G=lambda x: np.array([x[0]]),
                   comp_G_jac=lambda x: np.array([[1.0, 0.0, 0.0]]),
                   comp_H=lambda x: np.array([x[1]]),
                   comp_H_jac=lambda x: np.array([[0.0, 1.0, 0.0]]))
        result = pympcc.solve(p, strategy=strategy,
                              ipopt_options={"max_iter": 100, "tol": 1e-7},
                              presolve=True)
        assert result.success, f"strategy={strategy} failed: {result.message}"
        assert len(result.x) == n
        assert result.x[2] == pytest.approx(0.4, abs=1e-9)
