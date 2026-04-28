"""
Tests for §4.6: per-pair complementarity status, to_json(), to_dataframe().

Uses the canonical 'simple' problem (n=2, n_comp=1, f*=1, x*=(2,0)):
  - Optimal solution: G(x*)=x0*=2>0 (inactive), H(x*)=x1*=0 (H_active)
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import pympcc


# ----------------------------------------------------------------------- #
# Shared fixture                                                            #
# ----------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def simple_result():
    problem = pympcc.MPCCProblem(
        n=2, n_comp=1,
        x0=np.array([0.5, 0.5]),
        objective=lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        gradient=lambda x: np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        comp_G=lambda x: np.array([x[0]]),
        comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
        comp_H=lambda x: np.array([x[1]]),
        comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
    )
    return pympcc.solve(problem, strategy="scholtes")


# ----------------------------------------------------------------------- #
# per_pair_status                                                           #
# ----------------------------------------------------------------------- #

class TestPerPairStatus:
    def test_always_populated(self, simple_result):
        assert simple_result.per_pair_status is not None

    def test_length_matches_n_comp(self, simple_result):
        assert len(simple_result.per_pair_status) == 1

    def test_known_solution_status(self, simple_result):
        # At x*=(2,0): G=2>0 (inactive on G side), H=0 (active on H side)
        assert simple_result.per_pair_status[0] == "H_active"

    def test_valid_status_values(self, simple_result):
        valid = {"G_active", "H_active", "biactive", "inactive"}
        for s in simple_result.per_pair_status:
            assert s in valid

    def test_biactive_detected(self):
        # Build a problem whose solution forces both G and H to zero.
        # min x0^2 + x1^2  →  x* = (0, 0), both at zero.
        # The adaptive threshold (sqrt(comp_residual)) classifies the point
        # as "biactive" even when both G and H sit at O(sqrt(ε)) due to the
        # Scholtes relaxation barrier.
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.1, 0.1]),
            objective=lambda x: x[0] ** 2 + x[1] ** 2,
            gradient=lambda x: np.array([2.0 * x[0], 2.0 * x[1]]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        result = pympcc.solve(p, strategy="scholtes")
        # Both G and H near 0 → biactive (with adaptive threshold)
        status = result.per_pair_status[0]
        assert status == "biactive", (
            f"Expected biactive at (0,0), got {status!r}. "
            f"G={result.G[0]:.3e}, H={result.H[0]:.3e}, "
            f"comp={result.comp_residual:.3e}"
        )

    def test_g_active_detected(self):
        # min (x0 - 0)^2 + (x1 - 2)^2  →  x* = (0, 2), G=0 (G_active).
        p = pympcc.MPCCProblem(
            n=2, n_comp=1,
            x0=np.array([0.5, 0.5]),
            objective=lambda x: x[0] ** 2 + (x[1] - 2.0) ** 2,
            gradient=lambda x: np.array([2.0 * x[0], 2.0 * (x[1] - 2.0)]),
            comp_G=lambda x: np.array([x[0]]),
            comp_G_jacobian=lambda x: np.array([[1.0, 0.0]]),
            comp_H=lambda x: np.array([x[1]]),
            comp_H_jacobian=lambda x: np.array([[0.0, 1.0]]),
        )
        result = pympcc.solve(p, strategy="scholtes")
        assert result.per_pair_status[0] == "G_active"

    def test_multiple_pairs(self):
        # 2 comp pairs: x*=(2,0,0,3) → pair0=H_active, pair1=G_active
        p = pympcc.MPCCProblem(
            n=4, n_comp=2,
            x0=np.array([0.5, 0.5, 0.5, 0.5]),
            objective=lambda x: (
                (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
                + x[2] ** 2 + (x[3] - 3.0) ** 2
            ),
            gradient=lambda x: np.array([
                2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0),
                2.0 * x[2], 2.0 * (x[3] - 3.0),
            ]),
            comp_G=lambda x: np.array([x[0], x[2]]),
            comp_G_jacobian=lambda x: np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]),
            comp_H=lambda x: np.array([x[1], x[3]]),
            comp_H_jacobian=lambda x: np.array([
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
        )
        result = pympcc.solve(p, strategy="scholtes")
        assert len(result.per_pair_status) == 2
        # pair0: optimal at x0=2>0 (G>0) and x1=0 (H≈0)  → H_active
        # pair1: optimal at x2=0 (G≈0) and x3=3>0 (H>0)  → G_active
        assert result.per_pair_status[0] == "H_active"
        assert result.per_pair_status[1] == "G_active"


# ----------------------------------------------------------------------- #
# to_json                                                                   #
# ----------------------------------------------------------------------- #

class TestToJson:
    def test_returns_string(self, simple_result):
        s = simple_result.to_json()
        assert isinstance(s, str)

    def test_valid_json(self, simple_result):
        d = json.loads(simple_result.to_json())
        assert isinstance(d, dict)

    def test_required_fields_present(self, simple_result):
        d = json.loads(simple_result.to_json())
        for key in ("obj", "success", "status", "strategy", "x", "G", "H",
                    "comp_residual", "stationarity", "per_pair_status"):
            assert key in d, f"missing key: {key!r}"

    def test_x_is_list(self, simple_result):
        d = json.loads(simple_result.to_json())
        assert isinstance(d["x"], list)
        assert len(d["x"]) == 2

    def test_per_pair_status_in_json(self, simple_result):
        d = json.loads(simple_result.to_json())
        assert d["per_pair_status"] == ["H_active"]

    def test_roundtrip_obj(self, simple_result):
        d = json.loads(simple_result.to_json())
        assert abs(d["obj"] - simple_result.obj) < 1e-12

    def test_tnlp_not_in_json_when_absent(self, simple_result):
        d = json.loads(simple_result.to_json())
        assert "tnlp_refined" not in d or d.get("tnlp_refined") is None


# ----------------------------------------------------------------------- #
# to_dataframe                                                              #
# ----------------------------------------------------------------------- #

try:
    import pandas as pd  # noqa: F401
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

_skip_no_pandas = pytest.mark.skipif(not _HAS_PANDAS, reason="pandas not installed")


class TestToDataframe:
    @_skip_no_pandas
    def test_returns_dataframe(self, simple_result):
        import pandas as pd
        df = simple_result.to_dataframe()
        assert isinstance(df, pd.DataFrame)

    @_skip_no_pandas
    def test_one_row_per_pair(self, simple_result):
        df = simple_result.to_dataframe()
        assert len(df) == 1

    @_skip_no_pandas
    def test_required_columns(self, simple_result):
        df = simple_result.to_dataframe()
        for col in ("pair", "G", "H", "GH", "status"):
            assert col in df.columns, f"missing column: {col!r}"

    @_skip_no_pandas
    def test_status_column_value(self, simple_result):
        df = simple_result.to_dataframe()
        assert df["status"].iloc[0] == "H_active"

    @_skip_no_pandas
    def test_gh_product(self, simple_result):
        df = simple_result.to_dataframe()
        assert abs(df["GH"].iloc[0] - df["G"].iloc[0] * df["H"].iloc[0]) < 1e-12

    def test_raises_without_pandas(self, simple_result, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("pandas not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pandas"):
            simple_result.to_dataframe()
