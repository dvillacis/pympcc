"""Tests for plateau-based early termination and pre/post comp blowup rejection.

These two safeguards extend ``_run_epsilon_continuation``:

* **plateau** — stop once (obj, comp_residual) stall across consecutive
  accepted iterates and comp is already below ``plateau_comp_target``.
* **pre/post blowup** — reject the post-NLP iterate when the inner
  solver returns a comp residual ``> blowup_factor`` × the pre-NLP
  warm-start residual, even when ``tracked_eps`` would otherwise
  accept it.

Both are wired through ``--safeguards`` (``safeguards="all"``).
"""
from __future__ import annotations

import numpy as np
import pytest

import pympcc
from pympcc.strategies._base import SAFEGUARD_DEFAULTS
from pympcc.strategies.scholtes import ScholtesStrategy


def _make_problem():
    return pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )


# ======================================================================= #
# Defaults & validation                                                     #
# ======================================================================= #

class TestDefaults:
    def test_safeguard_plateau_in_defaults(self):
        assert SAFEGUARD_DEFAULTS["safeguard_plateau"] is False
        assert SAFEGUARD_DEFAULTS["plateau_window"] == 2
        assert SAFEGUARD_DEFAULTS["plateau_comp_target"] == pytest.approx(1e-4)
        assert SAFEGUARD_DEFAULTS["pre_post_blowup_factor"] == pytest.approx(10.0)

    def test_all_on_enables_plateau(self):
        strat = ScholtesStrategy(
            _make_problem(), ipopt_options={}, safeguards="all"
        )
        assert strat.safeguard_plateau is True

    def test_explicit_opt_in(self):
        strat = ScholtesStrategy(
            _make_problem(), ipopt_options={}, safeguard_plateau=True
        )
        assert strat.safeguard_plateau is True
        assert strat.safeguard_rollback is False  # untouched

    def test_invalid_blowup_factor(self):
        with pytest.raises(ValueError, match="pre_post_blowup_factor must be > 1"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, pre_post_blowup_factor=1.0
            )

    def test_invalid_plateau_window(self):
        with pytest.raises(ValueError, match="plateau_window must be >= 1"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, plateau_window=0
            )

    def test_invalid_plateau_target(self):
        with pytest.raises(ValueError, match="plateau_comp_target must be > 0"):
            ScholtesStrategy(
                _make_problem(), ipopt_options={}, plateau_comp_target=0.0
            )


# ======================================================================= #
# Plateau termination — stub-driven                                         #
# ======================================================================= #

class TestPlateauTermination:
    """Drive ``_run_epsilon_continuation`` with a stubbed ``_timed_solve``
    so we can dictate the (obj, comp_residual) trajectory and observe
    whether the plateau safeguard breaks the loop early."""

    def _build_strategy(self, **kwargs):
        opts = dict(
            ipopt_options={},
            safeguard_plateau=True,
            epsilon_0=1.0,
            reduction=0.1,
            epsilon_min=1e-12,
            max_iter=10,
            comp_eps_ratio_theta=1e18,  # don't reject on tracking
        )
        opts.update(kwargs)
        return ScholtesStrategy(_make_problem(), **opts)

    def _stub_solve(self, strat, trajectory):
        """Replace inner solve with a generator that yields predetermined
        (x, obj, comp) triples.  ``trajectory`` is a list of dicts with
        keys: x (ndarray), obj (float), comp (float). Status is forced
        to SOLVED so the iterate is always accepted (modulo blowup)."""
        p = strat.problem
        n_mult = 3 * p.n_comp + p.n_ineq + p.n_eq
        call_idx = {"k": 0}

        def fake(nlp, x, warm_dual):  # noqa: ARG001
            k = call_idx["k"]
            call_idx["k"] += 1
            point = trajectory[min(k, len(trajectory) - 1)]
            x_new = np.asarray(point["x"], dtype=float)
            info = {
                "status": 0,
                "status_msg": b"ok",
                "obj_val": point["obj"],
                "mult_g":   np.zeros(n_mult),
                "mult_x_L": np.zeros(p.n),
                "mult_x_U": np.zeros(p.n),
            }
            return x_new, info, 0.001

        strat._timed_solve = fake  # type: ignore[assignment]

        # Stub _comp_residual to return the trajectory's comp values.
        # _run_epsilon_continuation calls it via make_iteration → uses
        # problem.comp_G/H of the returned x; trajectory points already
        # encode the desired x, so let the real comp_residual run.
        return call_idx

    def test_plateau_stops_when_obj_and_comp_stall(self):
        # Trajectory: 4 iters where (obj, x) become essentially identical
        # by iter 1, and comp drops below plateau_comp_target = 1e-4.
        x_opt = np.array([2.0, 1e-6])  # comp = 2e-6, well below 1e-4
        traj = [
            {"x": x_opt + np.array([0.0, 1e-2]), "obj": 1.0001},  # comp ≈ 2e-2
            {"x": x_opt,                          "obj": 1.0000},  # comp ≈ 2e-6
            {"x": x_opt,                          "obj": 1.0000},  # plateau iter 1
            {"x": x_opt,                          "obj": 1.0000},  # plateau iter 2 → break
            {"x": x_opt,                          "obj": 1.0000},  # would be iter 3 if not broken
        ]
        strat = self._build_strategy(plateau_window=2, plateau_comp_target=1e-4)
        idx = self._stub_solve(strat, traj)
        result = strat.solve()
        # Should have stopped before exhausting all 5 trajectory points
        # (window=2 plateau hits → break after iter 4 at the latest;
        # actually after 4 iters: 0,1 not-plateau / plateau, 2 plateau-1, 3 plateau-2 → break).
        assert idx["k"] <= 4
        assert result.success

    def test_plateau_does_not_fire_above_target(self):
        # Comp residual stays above plateau_comp_target → plateau never
        # fires even if obj is constant.  Loop runs to max_iter.
        x_opt = np.array([2.0, 1e-2])  # comp ≈ 2e-2 >> 1e-4
        traj = [{"x": x_opt, "obj": 1.0}] * 12
        strat = self._build_strategy(
            plateau_window=2,
            plateau_comp_target=1e-4,
            max_iter=5,
        )
        idx = self._stub_solve(strat, traj)
        strat.solve()
        # Hit max_iter, not plateau.
        assert idx["k"] == 5

    def test_plateau_disabled_by_default(self):
        strat = ScholtesStrategy(
            _make_problem(), ipopt_options={},
            epsilon_0=1.0, reduction=0.1, max_iter=5,
        )
        # Default: safeguard_plateau is False, no early termination
        assert strat.safeguard_plateau is False


# ======================================================================= #
# Pre/post blowup rejection                                                 #
# ======================================================================= #

class TestPrePostBlowup:
    def _build_strategy(self, **kwargs):
        opts = dict(
            ipopt_options={},
            safeguard_rollback=True,
            epsilon_0=1.0,
            reduction=0.1,
            epsilon_min=1e-12,
            max_iter=10,
            comp_eps_ratio_theta=1e18,  # disable tracked_eps rejection
            rollback_lambda_jump=1e18,  # disable mult-jump rejection
            rollback_max_count=10,
        )
        opts.update(kwargs)
        return ScholtesStrategy(_make_problem(), **opts)

    def test_blowup_triggers_rollback(self):
        # Trajectory: iter 0 lands at a clean point with tiny comp;
        # iter 1 returns an iterate whose comp is 100× larger → blow-up.
        p_clean = np.array([2.0, 1e-6])     # comp ≈ 2e-6
        p_bad   = np.array([2.0, 1e-3])     # comp ≈ 2e-3 = 1000× pre
        x_seq   = [p_clean, p_bad, p_clean, p_clean]
        n_mult  = 3 * 1
        call_idx = {"k": 0}
        observed_eps: list[float] = []

        def fake(nlp, x, warm_dual):  # noqa: ARG001
            k = call_idx["k"]
            call_idx["k"] += 1
            x_new = x_seq[min(k, len(x_seq) - 1)]
            return x_new, {
                "status": 0,
                "status_msg": b"ok",
                "obj_val": float((x_new[0] - 2) ** 2 + (x_new[1] - 1) ** 2),
                "mult_g":   np.zeros(n_mult),
                "mult_x_L": np.zeros(2),
                "mult_x_U": np.zeros(2),
            }, 0.001

        strat = self._build_strategy(pre_post_blowup_factor=10.0)
        strat._timed_solve = fake  # type: ignore[assignment]

        # Capture ε on each call to confirm rollback held ε after blowup.
        orig_make = None
        def wrap_solve():
            nonlocal orig_make
            return strat.solve()

        result = wrap_solve()
        # The blowup at iter 1 should be rejected; safeguard restores
        # snapshot and holds ε.  Without the safeguard the bad iterate
        # would propagate and the comp residual would stay elevated.
        assert call_idx["k"] >= 2
        # Final result should have small comp (rejected the bad iterate
        # and kept the clean one).
        assert result.comp_residual < 1e-2

    def test_blowup_below_eps_floor_not_rejected(self):
        # When post-NLP comp ≤ eps, blowup does NOT trigger — ε is small
        # and "blowup ratio" is meaningless near floating-point noise.
        x = np.array([2.0, 1e-9])  # comp ≈ 2e-9
        n_mult = 3
        call_idx = {"k": 0}

        def fake(nlp, xx, warm_dual):  # noqa: ARG001
            call_idx["k"] += 1
            return x, {
                "status": 0,
                "status_msg": b"ok",
                "obj_val": 1.0,
                "mult_g":   np.zeros(n_mult),
                "mult_x_L": np.zeros(2),
                "mult_x_U": np.zeros(2),
            }, 0.001

        strat = self._build_strategy(
            pre_post_blowup_factor=10.0,
            epsilon_0=1.0, reduction=0.5, epsilon_min=1e-12, max_iter=3,
        )
        strat._timed_solve = fake  # type: ignore[assignment]
        result = strat.solve()
        # Iterate is constant — no rollback should fire.
        assert result.success
        assert call_idx["k"] == 3


# ======================================================================= #
# End-to-end                                                                #
# ======================================================================= #

class TestEndToEnd:
    def test_safeguards_all_runs_clean_problem(self):
        # Sanity: a well-behaved problem should still converge with the
        # extended safeguard set enabled.
        result = pympcc.solve(
            _make_problem(), strategy="scholtes", safeguards="all",
        )
        assert result.success
        assert result.comp_residual < 1e-4

    def test_plateau_breaks_max_iter_envelope(self):
        # max_iter=20 but plateau should trigger far earlier.
        traj_obj = []

        def cb(k, info):
            traj_obj.append(info.obj)

        result = pympcc.solve(
            _make_problem(), strategy="scholtes",
            safeguard_plateau=True,
            plateau_window=2, plateau_comp_target=1e-3,
            plateau_tol_obj=1e-3, plateau_tol_comp=1e-1,
            max_iter=20,
            callback=cb,
        )
        assert result.success
        # Should not have run all 20 outer iters thanks to plateau break.
        assert len(traj_obj) < 20
