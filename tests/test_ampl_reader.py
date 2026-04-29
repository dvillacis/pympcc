"""Tests for :mod:`pympcc.frontend.ampl`.

The ``.nl`` reader is being shipped incrementally; tests in this file cover
each piece as it lands:

* :class:`TestHeaderParser` — fixed-width numeric header
* :class:`TestOpTreeReader` — recursive prefix-notation op-tree
* :class:`TestTokenStream` — line/token bookkeeping
"""

from __future__ import annotations

import pytest

import numpy as np

import pympcc
from pympcc.frontend.ampl import (
    from_nl,
    NLBody,
    NLHeader,
    NLParseError,
    OpNode,
    Suffix,
    _read_optree,
    _TokenStream,
    eval_grad,
    eval_value,
    parse_body,
    parse_header,
)


def _parse_expr(text: str) -> OpNode:
    return _read_optree(_TokenStream(text.splitlines(), 0))


def _split(text: str) -> list[str]:
    """Split a multi-line string into lines without the trailing empty line."""
    return text.splitlines()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


class TestHeaderParser:

    def _minimal_header_lines(self) -> list[str]:
        # 2 vars, 1 constraint (the complementarity row), no objectives.
        # Line 4 declares 1 linear-only complementarity constraint.
        return [
            "g3 1 1 0",            # 1: magic
            "2 1 0 0 0 0",         # 2: nvar ncon nobj nranges neqns nlcon
            "0 0",                 # 3: nlc nlo
            "0 1 0",               # 4: nlcc lnc nccc  (1 linear comp)
            "0 0",                 # 5: nlnc nlno (network)
            "0 0 0",               # 6: nlvc nlvo nlvb
            "0 0 0 0",             # 7: lin_arith nfunc narith flags
            "0 0 0 0 0",           # 8: nbv niv nlvbi nlvci nlvoi
            "1 0",                 # 9: nzc nzo
            "0 0",                 # 10: maxconname maxvarname
        ]

    def test_basic_dims(self):
        lines = self._minimal_header_lines()
        header, n = parse_header(lines)
        assert isinstance(header, NLHeader)
        assert header.arith == 3
        assert header.flags1 == (1, 1, 0)
        assert header.n_var == 2
        assert header.n_con == 1
        assert header.n_obj == 0
        assert header.n_compl_lin == 1
        assert header.n_compl == 1
        assert header.nzc == 1
        assert n == 10

    def test_rejects_binary(self):
        lines = ["b3 1 1 0"] + self._minimal_header_lines()[1:]
        with pytest.raises(NLParseError, match="binary"):
            parse_header(lines)

    def test_rejects_missing_magic(self):
        with pytest.raises(NLParseError, match="magic"):
            parse_header(["x3 1 1 0"])

    def test_rejects_empty_file(self):
        with pytest.raises(NLParseError, match="empty"):
            parse_header([])

    def test_truncated_header(self):
        # Header line 7 absent; line 5 already says "0 0".
        lines = self._minimal_header_lines()[:6] + ["C0", "n0"]
        with pytest.raises(NLParseError, match="body marker"):
            parse_header(lines)

    def test_optional_suffix_line(self):
        lines = self._minimal_header_lines() + ["1 1 0 0 0", "C0", "n0"]
        header, n = parse_header(lines)
        assert header.suffix_counts == (1, 1, 0, 0, 0)
        assert n == 11
        assert lines[n][0] == "C"

    def test_no_suffix_line_when_body_follows(self):
        lines = self._minimal_header_lines() + ["C0", "n0"]
        header, n = parse_header(lines)
        assert header.suffix_counts == ()
        assert n == 10
        assert lines[n][0] == "C"


# ---------------------------------------------------------------------------
# Op-tree
# ---------------------------------------------------------------------------


class TestOpTreeReader:

    def _read(self, text: str) -> OpNode:
        stream = _TokenStream(text.splitlines(), 0)
        return _read_optree(stream)

    def test_single_variable(self):
        node = self._read("v0")
        assert node == OpNode(kind="var", index=0)

    def test_single_number(self):
        node = self._read("n3.5")
        assert node.kind == "num"
        assert node.value == 3.5

    def test_binary_plus(self):
        # plus(v0, n2.0)
        node = self._read("o0\nv0\nn2.0\n")
        assert node.kind == "op"
        assert node.op == 0
        assert len(node.children) == 2
        assert node.children[0].kind == "var"
        assert node.children[1].kind == "num"

    def test_unary_neg(self):
        # neg(v1)
        node = self._read("o16\nv1\n")
        assert node.kind == "op"
        assert node.op == 16
        assert len(node.children) == 1

    def test_pow(self):
        # pow(v0, n2)
        node = self._read("o5\nv0\nn2\n")
        assert node.kind == "op"
        assert node.op == 5

    def test_nested_tree(self):
        # mult(plus(v0, v1), n3)
        node = self._read("o2\no0\nv0\nv1\nn3\n")
        assert node.op == 2
        assert node.children[0].op == 0
        assert node.children[0].children[0].index == 0
        assert node.children[0].children[1].index == 1
        assert node.children[1].value == 3.0

    def test_sumlist(self):
        # sumlist(3 children: v0, v1, v2)
        node = self._read("o54\n3\nv0\nv1\nv2\n")
        assert node.op == 54
        assert len(node.children) == 3
        assert [c.index for c in node.children] == [0, 1, 2]

    def test_unsupported_op_raises(self):
        # op 79 (function call) is intentionally unsupported.
        with pytest.raises(NLParseError, match="op code 79"):
            self._read("o79\nv0\n")

    def test_eof_raises(self):
        # Binary op missing second arg.
        with pytest.raises(NLParseError, match="EOF"):
            self._read("o0\nv0\n")


# ---------------------------------------------------------------------------
# Token stream
# ---------------------------------------------------------------------------


class TestTokenStream:

    def test_token_iteration(self):
        s = _TokenStream(["a b c", "d e", "", "f"], 0)
        toks = []
        while True:
            t = s.next_token()
            if t is None:
                break
            toks.append(t)
        assert toks == ["a", "b", "c", "d", "e", "f"]

    def test_peek_line_marker(self):
        s = _TokenStream(["o0", "v0", "C0", "n3"], 0)
        # consume "o0", "v0"
        s.next_token()
        s.next_token()
        assert s.peek_line_marker() == "C"

    def test_read_int_and_float(self):
        s = _TokenStream(["3 4.5"], 0)
        assert s.read_int() == 3
        assert s.read_float() == 4.5

    def test_advance_to_next_line(self):
        s = _TokenStream(["a b c", "d"], 0)
        s.next_token()  # "a"
        s.advance_to_next_line()
        assert s.next_token() == "d"


# ---------------------------------------------------------------------------
# Body parser
# ---------------------------------------------------------------------------


def _stub_header(n_var: int = 2, n_con: int = 1, n_obj: int = 0) -> NLHeader:
    return NLHeader(
        arith=3, flags1=(1, 1, 0),
        n_var=n_var, n_con=n_con, n_obj=n_obj,
        n_ranges=0, n_eqns=0, n_lcon=0,
        n_nl_con=0, n_nl_obj=0,
        n_compl_nl=0, n_compl_lin=0,
        n_compl_net=0, n_compl_extra=0,
        n_nl_net_con=0, n_nl_net_obj=0,
        nlvc=0, nlvo=0, nlvb=0,
        n_lin_arith_only=0, n_func=0, n_arith=0, flags7=0,
        nbv=0, niv=0, nlvbi=0, nlvci=0, nlvoi=0,
        nzc=0, nzo=0,
        max_con_name=0, max_var_name=0,
    )


class TestBodyParser:

    def test_C_segment_nonlinear_constraint(self):
        # Constraint 0 is plus(v0, n3.5)
        lines = ["C0", "o0", "v0", "n3.5"]
        body = parse_body(lines, 0, _stub_header(n_var=1, n_con=1))
        assert 0 in body.nl_cons
        node = body.nl_cons[0]
        assert node.op == 0  # plus
        assert node.children[0].kind == "var"
        assert node.children[1].value == 3.5

    def test_O_segment_with_sense(self):
        # Objective 0: max v0
        lines = ["O0 1", "v0"]
        body = parse_body(lines, 0, _stub_header(n_var=1, n_con=0, n_obj=1))
        assert body.obj_senses[0] == 1
        assert body.nl_objs[0].kind == "var"

    def test_b_segment_bounds(self):
        # 3 variables: range [0,1], lower bound 0, free
        lines = ["b", "0 0 1", "2 0", "3"]
        body = parse_body(lines, 0, _stub_header(n_var=3, n_con=0))
        assert body.var_bounds == [(0, 0.0, 1.0), (2.0, 0.0), (3.0,)]

    def test_b_segment_complementarity_bound(self):
        # Variable 1 is complementarity-bounded against constraint 0,
        # complementarity variable kind 1.
        lines = ["b", "0 0 1", "5 1 0"]
        body = parse_body(lines, 0, _stub_header(n_var=2, n_con=1))
        assert body.var_bounds[1] == (5.0, 1.0, 0.0)

    def test_r_segment_constraint_ranges(self):
        # 2 constraints: equality at 0, upper bound at 5
        lines = ["r", "4 0", "1 5"]
        body = parse_body(lines, 0, _stub_header(n_var=0, n_con=2))
        assert body.con_bounds == [(4.0, 0.0), (1.0, 5.0)]

    def test_k_segment_col_counts(self):
        # n_var=3 → k segment has 2 entries
        lines = ["k2", "1", "3"]
        body = parse_body(lines, 0, _stub_header(n_var=3, n_con=0))
        assert body.jac_col_counts == [1, 3]

    def test_J_segment_linear_constraint(self):
        # Constraint 0 has 2 linear terms: 2*v0 + 3*v1
        lines = ["J0 2", "0 2.0", "1 3.0"]
        body = parse_body(lines, 0, _stub_header(n_var=2, n_con=1))
        assert body.jac_lin[0] == [(0, 2.0), (1, 3.0)]

    def test_G_segment_linear_objective(self):
        lines = ["G0 1", "0 5.0"]
        body = parse_body(lines, 0, _stub_header(n_var=1, n_con=0, n_obj=1))
        assert body.obj_lin[0] == [(0, 5.0)]

    def test_x_segment_primal_init(self):
        lines = ["x2", "0 0.5", "1 1.5"]
        body = parse_body(lines, 0, _stub_header(n_var=2, n_con=0))
        assert body.primal_init == {0: 0.5, 1: 1.5}

    def test_d_segment_dual_init(self):
        lines = ["d1", "0 -2.0"]
        body = parse_body(lines, 0, _stub_header(n_var=0, n_con=1))
        assert body.dual_init == {0: -2.0}

    def test_S_segment_suffix(self):
        # Constraint-side integer suffix "cvar" mapping con 0 -> var 1.
        # flag = 1 (con) | 4 (int) = 5
        lines = ["S5 1 cvar", "0 1"]
        body = parse_body(lines, 0, _stub_header(n_var=2, n_con=1))
        assert len(body.suffixes) == 1
        suf = body.suffixes[0]
        assert suf.name == "cvar"
        assert suf.target == "con"
        assert suf.is_int is True
        assert suf.values == {0: 1.0}

    def test_find_suffix(self):
        body = NLBody()
        body.suffixes.append(
            Suffix(flag=5, name="cvar", values={0: 3.0, 1: 4.0})
        )
        body.suffixes.append(
            Suffix(flag=4, name="cvar", values={0: 7.0})  # var-side, also "cvar"
        )
        assert body.find_suffix("cvar", target="con").values == {0: 3.0, 1: 4.0}
        assert body.find_suffix("cvar", target="var").values == {0: 7.0}
        assert body.find_suffix("does_not_exist") is None

    def test_unsupported_V_segment_raises(self):
        with pytest.raises(NLParseError, match="V"):
            parse_body(["V0 0 0"], 0, _stub_header())

    def test_unknown_marker_raises(self):
        with pytest.raises(NLParseError, match="unknown body marker"):
            parse_body(["Z0"], 0, _stub_header())

    def test_full_round_trip(self):
        """End-to-end: header + body for a 2-var, 1-comp problem."""
        # min  v0
        # s.t. 0 <= v0 ⊥ v1 - 1 >= 0
        # The complementarity constraint is "v1 - 1" (linear).
        # Bound type 5 says v0 is complementarity-paired with constraint 0
        # (kind 1 = G(x) = x_var-form).
        lines = [
            # Header
            "g3 1 1 0",
            "2 1 1 0 0 0",     # 2 vars, 1 con, 1 obj
            "0 0",
            "0 1 0",            # 1 linear-only complementarity
            "0 0",
            "0 0 0",
            "0 0 0 0",
            "0 0 0 0 0",
            "1 1",              # nzc=1, nzo=1
            "0 0",
            # Body
            "C0",               # nonlinear part of con 0 (none)
            "n0",
            "O0 0",             # objective 0, sense=min
            "v0",
            "x2",
            "0 0.5",
            "1 0.5",
            "r",
            "1 0",              # con 0 upper-bounded at 0 (complementarity row)
            "b",
            "5 1 1",            # var 0: complementarity with con 0 (1-indexed), kind 1
            "2 0",              # var 1: lower-bounded at 0
            "k1",
            "0",                # col-count: col 0 ends at 0
            "J0 1",
            "1 1.0",            # linear part of con 0: 1.0 * v1
            "G0 1",
            "0 1.0",            # linear part of obj: 1.0 * v0
        ]
        header, n = parse_header(lines)
        assert header.n_var == 2
        assert header.n_con == 1
        assert header.n_compl_lin == 1
        body = parse_body(lines, n, header)
        assert body.var_bounds[0][0] == 5.0   # complementarity bound type
        assert body.var_bounds[0][1] == 1.0   # complementarity kind
        assert body.var_bounds[0][2] == 1.0   # paired with con 0 (1-indexed in raw .nl)
        assert body.jac_lin[0] == [(1, 1.0)]
        assert body.obj_lin[0] == [(0, 1.0)]
        assert body.obj_senses[0] == 0
        assert body.primal_init == {0: 0.5, 1: 0.5}


# ---------------------------------------------------------------------------
# Op-tree evaluation
# ---------------------------------------------------------------------------


class TestEvalValue:

    def test_constant(self):
        assert eval_value(_parse_expr("n3.5"), np.array([])) == 3.5

    def test_variable(self):
        assert eval_value(_parse_expr("v1"), np.array([10.0, 20.0])) == 20.0

    def test_arithmetic(self):
        # plus(v0, mult(v1, n2))
        x = np.array([3.0, 4.0])
        assert eval_value(_parse_expr("o0\nv0\no2\nv1\nn2"), x) == 11.0

    def test_pow_and_neg(self):
        # neg(pow(v0, n3))
        x = np.array([2.0])
        assert eval_value(_parse_expr("o16\no5\nv0\nn3"), x) == -8.0

    def test_transcendentals(self):
        # log(exp(v0)) ≈ v0
        x = np.array([1.5])
        result = eval_value(_parse_expr("o43\no44\nv0"), x)
        assert abs(result - 1.5) < 1e-12

    def test_sumlist(self):
        # sumlist(v0, v1, v2, n5)
        x = np.array([1.0, 2.0, 3.0])
        result = eval_value(_parse_expr("o54\n4\nv0\nv1\nv2\nn5"), x)
        assert result == 11.0

    def test_square(self):
        # op 76 is x^2
        x = np.array([7.0])
        assert eval_value(_parse_expr("o76\nv0"), x) == 49.0


class TestEvalGrad:

    def _grad_array(self, node: OpNode, x: np.ndarray) -> np.ndarray:
        _, partials = eval_grad(node, x)
        out = np.zeros_like(x, dtype=float)
        for k, v in partials.items():
            out[k] = v
        return out

    def test_constant_has_no_partials(self):
        val, partials = eval_grad(_parse_expr("n2.5"), np.array([]))
        assert val == 2.5
        assert partials == {}

    def test_variable_partial_one(self):
        val, partials = eval_grad(_parse_expr("v0"), np.array([3.0]))
        assert val == 3.0
        assert partials == {0: 1.0}

    def test_polynomial(self):
        # f(x, y) = 3 x^2 + 2 x y + y^2
        # plus(plus(mult(n3, square(v0)), mult(n2, mult(v0, v1))), square(v1))
        text = "o0\no0\no2\nn3\no76\nv0\no2\nn2\no2\nv0\nv1\no76\nv1"
        node = _parse_expr(text)
        x = np.array([2.0, 1.0])
        val, partials = eval_grad(node, x)
        # f(2,1) = 12 + 4 + 1 = 17; df/dx = 6x + 2y = 14; df/dy = 2x + 2y = 6
        assert val == 17.0
        assert partials[0] == 14.0
        assert partials[1] == 6.0

    def test_sin_cos_chain(self):
        # f(x) = sin(x^2)
        node = _parse_expr("o41\no76\nv0")
        x = np.array([1.3])
        val, partials = eval_grad(node, x)
        assert abs(val - np.sin(1.69)) < 1e-12
        # df/dx = cos(x^2) * 2x
        expected = np.cos(1.69) * 2.6
        assert abs(partials[0] - expected) < 1e-12

    def test_sumlist_gradient(self):
        # sum(v0, v1, v2)
        node = _parse_expr("o54\n3\nv0\nv1\nv2")
        x = np.array([1.0, 1.0, 1.0])
        val, partials = eval_grad(node, x)
        assert val == 3.0
        assert partials == {0: 1.0, 1: 1.0, 2: 1.0}

    def test_repeated_variable_accumulates(self):
        # f(x) = x + x*x  (v0 appears in two places)
        node = _parse_expr("o0\nv0\no2\nv0\nv0")
        x = np.array([3.0])
        val, partials = eval_grad(node, x)
        # f(3) = 3 + 9 = 12; df/dx = 1 + 2x = 7
        assert val == 12.0
        assert partials[0] == 7.0

    def test_against_finite_differences(self):
        """Random expression: gradient must match central-FD within 1e-6."""
        # f(x, y) = exp(x) * log(y) + sqrt(x*y)
        text = "o0\no2\no44\nv0\no43\nv1\no39\no2\nv0\nv1"
        node = _parse_expr(text)
        x = np.array([1.5, 2.0])
        val, partials = eval_grad(node, x)
        h = 1e-6
        for i in range(2):
            xp = x.copy(); xp[i] += h
            xm = x.copy(); xm[i] -= h
            fd = (eval_value(node, xp) - eval_value(node, xm)) / (2 * h)
            assert abs(partials.get(i, 0.0) - fd) < 1e-6

    def test_division_gradient(self):
        # f(x, y) = x / y
        node = _parse_expr("o3\nv0\nv1")
        x = np.array([6.0, 3.0])
        val, partials = eval_grad(node, x)
        assert val == 2.0
        # df/dx = 1/y = 1/3; df/dy = -x/y^2 = -2/3
        assert abs(partials[0] - 1.0 / 3.0) < 1e-12
        assert abs(partials[1] - (-6.0 / 9.0)) < 1e-12


# ---------------------------------------------------------------------------
# from_nl end-to-end
# ---------------------------------------------------------------------------


class TestFromNL:

    def _write(self, tmp_path, lines: list[str]):
        p = tmp_path / "p.nl"
        p.write_text("\n".join(lines) + "\n")
        return p

    def _simple_mpcc(self, tmp_path):
        """min (x0-2)² + (x1+1)² s.t. x0 ≥ 0 ⊥ x1 ≥ 0.

        Unique global optimum at (2, 0) with obj = 1.
        """
        # Op-tree for (x0-2)² + (x1+1)²:
        #   plus(square(minus(v0, n2)), square(plus(v1, n1)))
        # = "o0 o76 o1 v0 n2 o76 o0 v1 n1"
        lines = [
            # Header (10 lines)
            "g3 1 1 0",
            "2 1 1 0 0 0",     # nvar=2 ncon=1 nobj=1
            "0 1",              # nlc=0 nlo=1
            "0 1 0 0",          # 1 linear-only complementarity
            "0 0",
            "0 2 0",            # nlvc=0 nlvo=2 nlvb=0
            "0 0 0 0",
            "0 0 0 0 0",
            "1 0",              # nzc=1 nzo=0
            "0 0",
            # Body
            "C0",               # con 0 has no nonlinear part
            "n0",
            "O0 0",             # obj 0, sense=min
            "o0",
            "o76",
            "o1",
            "v0",
            "n2",
            "o76",
            "o0",
            "v1",
            "n1",
            "x2",
            "0 1.0",
            "1 1.0",
            "r",
            "2 0",              # con 0: c0(x) ≥ 0
            "b",
            "5 1 1",            # var 0 complements con 0 (1-indexed cvar), kind 1
            "2 0",              # var 1 lower-bounded at 0
            "k1",
            "0",                # col 0 ends at offset 0 (no linear J in col 0)
            "J0 1",
            "1 1.0",            # con 0 = 1.0 * v1
        ]
        return self._write(tmp_path, lines)

    def test_returns_mpcc_problem(self, tmp_path):
        path = self._simple_mpcc(tmp_path)
        problem = from_nl(path)
        assert problem.n == 2
        assert problem.n_comp == 1

    def test_objective_value_at_x0(self, tmp_path):
        path = self._simple_mpcc(tmp_path)
        problem = from_nl(path)
        # objective(x0=[1, 1]) = (1-2)² + (1+1)² = 1 + 4 = 5
        assert abs(problem.objective(problem.x0) - 5.0) < 1e-12

    def test_gradient_matches_analytic(self, tmp_path):
        path = self._simple_mpcc(tmp_path)
        problem = from_nl(path)
        # ∇f = [2(x0-2), 2(x1+1)] at x0=[1,1] is [-2, 4]
        g = problem.gradient(problem.x0)
        assert np.allclose(g, [-2.0, 4.0])

    def test_complementarity_extracted(self, tmp_path):
        path = self._simple_mpcc(tmp_path)
        problem = from_nl(path)
        # G(x0=[1,1]) should be x[0] = 1
        assert problem.comp_G(problem.x0)[0] == 1.0
        # H(x0=[1,1]) should be x[1] = 1
        assert problem.comp_H(problem.x0)[0] == 1.0

    def test_solves_to_known_optimum(self, tmp_path):
        path = self._simple_mpcc(tmp_path)
        problem = from_nl(path)
        result = pympcc.solve(problem, strategy="scholtes")
        assert result.success
        # Unique optimum at (2, 0) with obj = 1
        assert abs(result.obj - 1.0) < 1e-4
        assert abs(result.x[0] - 2.0) < 1e-3
        assert abs(result.x[1] - 0.0) < 1e-3
