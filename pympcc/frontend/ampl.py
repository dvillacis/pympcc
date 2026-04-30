"""AMPL ``.nl`` reader.

Parses AMPL's text-format ``.nl`` files (Gay 2005, "Writing .nl files") and
produces an :class:`pympcc.MPCCProblem`.  Complementarity structure is read
from the ``cvar`` variable suffix and the complementarity counts in line 4
of the header.

Public API
----------

:func:`from_nl(path) <from_nl>` — read a ``.nl`` file and return an
:class:`pympcc.MPCCProblem`.  Function/objective values, gradients, and
Jacobians are evaluated by walking the parsed op-tree directly; no AMPL
runtime or external solver SDK is required.

Status
------

This is an in-progress implementation.  The header parser, op-tree reader,
forward evaluator, reverse-mode AD walker, and complementarity-suffix
resolver all live in this module.  Operator coverage is the ~30 ops used by
MacMPEC; unsupported ops raise :class:`NLParseError` with the op code in
the message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


class NLParseError(ValueError):
    """Raised when an ``.nl`` file cannot be parsed or uses unsupported features."""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLHeader:
    """Parsed ``.nl`` text-format header.

    Field naming follows Gay's "Writing .nl files" (most recent revision).
    Line numbers below refer to that document.
    """

    # Line 1: magic prefix "g<arith> a b c"
    arith: int                 # arithmetic kind (3 = any-endian)
    flags1: tuple[int, ...]    # remaining tokens on line 1

    # Line 2: problem dimensions
    n_var: int                 # number of variables
    n_con: int                 # number of constraints (incl. comp + logic)
    n_obj: int                 # number of objectives
    n_ranges: int              # number of range constraints
    n_eqns: int                # number of equality constraints
    n_lcon: int                # number of logical constraints (rare)

    # Line 3: nonlinear counts
    n_nl_con: int              # constraints with nonlinear part
    n_nl_obj: int              # objectives with nonlinear part

    # Line 4: complementarity counts
    n_compl_nl: int            # complementarity constraints w/ nonlinear part
    n_compl_lin: int           # linear-only complementarity constraints
    n_compl_net: int           # network complementarity (usually 0)
    n_compl_extra: int         # spare slot (some AMPL versions emit 4)

    # Line 5: network counts
    n_nl_net_con: int
    n_nl_net_obj: int

    # Line 6: nonlinear-variable partitions
    nlvc: int                  # nonlinear in constraints only
    nlvo: int                  # nonlinear in objectives only
    nlvb: int                  # nonlinear in both

    # Line 7: extension counts
    n_lin_arith_only: int
    n_func: int
    n_arith: int
    flags7: int

    # Line 8: discrete / integer-variable counts
    nbv: int                   # binary
    niv: int                   # other integer
    nlvbi: int                 # nonlinear-in-both integer
    nlvci: int                 # nonlinear-in-cons integer
    nlvoi: int                 # nonlinear-in-objs integer

    # Line 9: Jacobian + objective-gradient nnz
    nzc: int
    nzo: int

    # Line 10: name-length caps (informational)
    max_con_name: int
    max_var_name: int

    # Line 11 (optional): suffix counts
    suffix_counts: tuple[int, ...] = field(default_factory=tuple)

    @property
    def n_compl(self) -> int:
        """Total number of complementarity constraints in this problem."""
        return self.n_compl_nl + self.n_compl_lin + self.n_compl_net


# Body-segment markers (first non-whitespace character of a body line).
# When the header parser sees one of these as the start of a line, it stops
# reading header lines.
_BODY_MARKERS = frozenset("COdxrbkJGSVLF")


def _ints(line: str) -> list[int]:
    """Parse a whitespace-separated list of integers; tolerate trailing junk."""
    out: list[int] = []
    for tok in line.split():
        try:
            out.append(int(tok))
        except ValueError:
            # Some AMPL writers append a trailing comment like "# nvar".
            break
    return out


def _floats(tokens: list[str]) -> list[float]:
    return [float(t) for t in tokens]


def _pad(values: list[int], width: int, *, fill: int = 0) -> list[int]:
    """Right-pad ``values`` with ``fill`` to length ``width``."""
    if len(values) >= width:
        return values[:width]
    return values + [fill] * (width - len(values))


def parse_header(text_lines: list[str]) -> tuple[NLHeader, int]:
    """Parse the header of a ``.nl`` text-format file.

    Parameters
    ----------
    text_lines
        The full file split on ``\\n``.  Trailing newlines and the final
        empty line (if any) must be preserved by the caller.

    Returns
    -------
    header
        Parsed :class:`NLHeader`.
    n_consumed
        Number of input lines consumed by the header.  The body parser
        starts at ``text_lines[n_consumed]``.
    """
    if not text_lines:
        raise NLParseError("empty file")

    line1 = text_lines[0].strip()
    if not line1 or line1[0] not in ("g", "b"):
        raise NLParseError(
            f"invalid magic line {line1!r}: must start with 'g' (text) or 'b' (binary)"
        )
    if line1[0] == "b":
        raise NLParseError("binary .nl format is not supported yet")

    # "g3 1 1 0" → arith = 3, flags1 = [1, 1, 0]
    head_tok = line1.split()
    if not head_tok or not head_tok[0].startswith("g"):
        raise NLParseError(f"malformed magic line: {line1!r}")
    try:
        arith = int(head_tok[0][1:])
    except ValueError:
        raise NLParseError(f"could not parse arith from {head_tok[0]!r}") from None
    flags1 = tuple(_ints(" ".join(head_tok[1:])))

    # Lines 2-10 are fixed-position numeric.  Line 11 is a suffix-counts line
    # only when comb/obo > 0; it's not always emitted.
    cursor = 1

    def _read_line(min_fields: int = 0) -> list[int]:
        nonlocal cursor
        if cursor >= len(text_lines):
            raise NLParseError("unexpected EOF in header")
        line = text_lines[cursor]
        if line and line[0] in _BODY_MARKERS:
            raise NLParseError(
                f"header truncated at line {cursor}: body marker '{line[0]}' encountered"
            )
        cursor += 1
        vals = _ints(line)
        if min_fields and len(vals) < min_fields:
            raise NLParseError(
                f"header line {cursor} has {len(vals)} fields, expected ≥ {min_fields}"
            )
        return vals

    l2 = _pad(_read_line(min_fields=3), 6)
    l3 = _pad(_read_line(min_fields=2), 2)
    l4 = _pad(_read_line(), 4)
    l5 = _pad(_read_line(), 2)
    l6 = _pad(_read_line(), 3)
    l7 = _pad(_read_line(), 4)
    l8 = _pad(_read_line(), 5)
    l9 = _pad(_read_line(min_fields=2), 2)
    l10 = _pad(_read_line(), 2)

    # Line 11 is optional: only if it's still numeric.
    suffix_counts: tuple[int, ...] = ()
    if cursor < len(text_lines):
        nxt = text_lines[cursor]
        if nxt and nxt[0] not in _BODY_MARKERS:
            stripped = nxt.strip()
            if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                vals11 = _ints(nxt)
                if vals11:
                    suffix_counts = tuple(vals11)
                    cursor += 1

    header = NLHeader(
        arith=arith,
        flags1=flags1,
        n_var=l2[0], n_con=l2[1], n_obj=l2[2],
        n_ranges=l2[3], n_eqns=l2[4], n_lcon=l2[5],
        n_nl_con=l3[0], n_nl_obj=l3[1],
        n_compl_nl=l4[0], n_compl_lin=l4[1],
        n_compl_net=l4[2], n_compl_extra=l4[3],
        n_nl_net_con=l5[0], n_nl_net_obj=l5[1],
        nlvc=l6[0], nlvo=l6[1], nlvb=l6[2],
        n_lin_arith_only=l7[0], n_func=l7[1],
        n_arith=l7[2], flags7=l7[3],
        nbv=l8[0], niv=l8[1],
        nlvbi=l8[2], nlvci=l8[3], nlvoi=l8[4],
        nzc=l9[0], nzo=l9[1],
        max_con_name=l10[0], max_var_name=l10[1],
        suffix_counts=suffix_counts,
    )
    return header, cursor


# ---------------------------------------------------------------------------
# Op-tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpNode:
    """Node of an AMPL op-tree.

    ``kind`` is one of:
    - ``"op"`` — operator; ``op`` holds the AMPL op code, ``children`` the args.
    - ``"var"`` — variable reference; ``index`` holds the var index.
    - ``"num"`` — numeric literal; ``value`` holds the constant.
    """

    kind: str
    op: int = -1
    index: int = -1
    value: float = 0.0
    children: tuple["OpNode", ...] = ()


# AMPL op codes used by MacMPEC.  Anything outside this set raises an error
# with the op number in the message.
_BINARY_OPS = {
    0: "plus",        # a + b
    1: "minus",       # a - b
    2: "mult",        # a * b
    3: "div",         # a / b
    5: "pow",         # a ** b
    35: "atan2",      # atan2(a, b)  -- AMPL ASL OPATAN2
    55: "intdiv",     # a // b       -- AMPL ASL OPINTDIV
}
_UNARY_OPS = {
    13: "floor",
    14: "ceil",
    15: "abs",
    16: "neg",        # -a
    37: "tanh",
    38: "tan",
    39: "sqrt",
    40: "sinh",
    41: "sin",
    42: "log10",
    43: "log",
    44: "exp",
    45: "cosh",
    46: "cos",
    47: "atanh",
    48: "atan",
    49: "asinh",
    50: "asin",
    51: "acosh",
    52: "acos",
}
# n-ary ops carry an explicit count line after the op header.
_NARY_OPS = {
    54: "sumlist",    # sum of N children -- AMPL ASL OPSUMLIST
}
# Special: x^2 (op 76) carries no second operand; it's encoded as a unary square.
_SPECIAL_OPS = {
    76: "square",     # a^2 (alias for pow(a, 2))
}


def _read_optree(stream: "_TokenStream") -> OpNode:
    """Recursively read one op-tree node from ``stream``.

    Each non-blank token starts with one of:
    - ``o<N>`` — operator with code N
    - ``v<N>`` — variable index N
    - ``n<float>`` — numeric literal
    - ``f<N>`` — defined function (not yet supported)
    """
    tok = stream.next_token()
    if not tok:
        raise NLParseError("unexpected EOF inside op-tree")
    head = tok[0]
    rest = tok[1:]
    if head == "v":
        return OpNode(kind="var", index=int(rest))
    if head == "n":
        return OpNode(kind="num", value=float(rest))
    if head == "o":
        op = int(rest)
        if op in _BINARY_OPS:
            a = _read_optree(stream)
            b = _read_optree(stream)
            return OpNode(kind="op", op=op, children=(a, b))
        if op in _UNARY_OPS or op in _SPECIAL_OPS:
            a = _read_optree(stream)
            return OpNode(kind="op", op=op, children=(a,))
        if op in _NARY_OPS:
            count_tok = stream.next_token()
            if not count_tok:
                raise NLParseError(f"missing count for n-ary op {op}")
            n_children = int(count_tok)
            kids = tuple(_read_optree(stream) for _ in range(n_children))
            return OpNode(kind="op", op=op, children=kids)
        raise NLParseError(
            f"unsupported AMPL op code {op}; extend pympcc.frontend.ampl "
            "if MacMPEC requires it"
        )
    raise NLParseError(f"unexpected op-tree token {tok!r}")


# ---------------------------------------------------------------------------
# Token stream over the body
# ---------------------------------------------------------------------------


class _TokenStream:
    """Cursor over whitespace-separated tokens drawn from a list of lines.

    Op-tree segments in a ``.nl`` file are written one token per line, but
    that's a convention, not a requirement.  This stream tolerates either.
    """

    __slots__ = ("_lines", "_idx", "_buf", "_buf_idx")

    def __init__(self, lines: list[str], start_line: int) -> None:
        self._lines = lines
        self._idx = start_line
        self._buf: list[str] = []
        self._buf_idx = 0

    def next_token(self) -> Optional[str]:
        """Return the next non-empty token, or ``None`` at EOF."""
        while self._buf_idx >= len(self._buf):
            if self._idx >= len(self._lines):
                return None
            self._buf = self._lines[self._idx].split()
            self._idx += 1
            self._buf_idx = 0
        tok = self._buf[self._buf_idx]
        self._buf_idx += 1
        return tok

    def peek_line_marker(self) -> Optional[str]:
        """Peek at the first character of the next non-empty line."""
        # Skip any unread tokens on the current buffered line first.
        if self._buf_idx < len(self._buf):
            return self._buf[self._buf_idx][:1] or None
        i = self._idx
        while i < len(self._lines):
            line = self._lines[i]
            stripped = line.strip()
            if stripped:
                return stripped[0]
            i += 1
        return None

    def advance_to_next_line(self) -> None:
        """Discard any tokens left in the current line buffer."""
        self._buf_idx = len(self._buf)

    def current_line_idx(self) -> int:
        return self._idx

    def read_int(self) -> int:
        tok = self.next_token()
        if tok is None:
            raise NLParseError("unexpected EOF reading int")
        return int(tok)

    def read_float(self) -> float:
        tok = self.next_token()
        if tok is None:
            raise NLParseError("unexpected EOF reading float")
        return float(tok)


# ---------------------------------------------------------------------------
# Body segments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Suffix:
    """A ``.nl`` suffix table entry.

    AMPL encodes per-(var|con|obj|problem) metadata via suffixes.  Bit-flag
    layout (Gay 2005):

    * Bits 0-1 (``flag & 3``): target — 0=var, 1=con, 2=obj, 3=problem.
    * Bit 2 (``flag & 4``): 0=real-valued, 4=integer-valued.

    The ``cvar`` suffix used by MPCC problems is a constraint-side integer
    suffix (``flag = 1 | 4 = 5``) mapping each complementarity constraint
    to its complementary variable index.
    """

    flag: int
    name: str
    values: dict[int, float]   # idx -> value (int values stored as float)

    @property
    def target(self) -> str:
        return ("var", "con", "obj", "prob")[self.flag & 3]

    @property
    def is_int(self) -> bool:
        return bool(self.flag & 4)


@dataclass
class NLBody:
    """All segments parsed from the body of a ``.nl`` text-format file.

    Attributes
    ----------
    nl_cons
        Map ``con_idx -> OpNode`` for the nonlinear part of constraint
        ``con_idx``.  Constraints absent from this map are linear-only.
    nl_objs
        Map ``obj_idx -> OpNode`` for the nonlinear part of objective
        ``obj_idx``.
    obj_senses
        Map ``obj_idx -> int`` (0=min, 1=max) from the ``O`` segment header.
    primal_init
        Sparse map ``var_idx -> float`` from the ``x`` segment.
    dual_init
        Sparse map ``con_idx -> float`` from the ``d`` segment.
    var_bounds
        Length-``n_var`` list of ``(type, *floats)`` tuples; ``type``
        follows AMPL's bound-type code (0=range, 1=upper, 2=lower,
        3=free, 4=fixed, 5=complementarity).
    con_bounds
        Length-``n_con`` list of ``(type, *floats)`` tuples (same scheme).
    jac_col_counts
        Cumulative column-pointer for the linear Jacobian sparsity from
        the ``k`` segment.  Length ``n_var - 1`` per Gay's spec.
    jac_lin
        Map ``con_idx -> [(var_idx, coeff), ...]`` from ``J`` segments.
    obj_lin
        Map ``obj_idx -> [(var_idx, coeff), ...]`` from ``G`` segments.
    suffixes
        List of :class:`Suffix` records from the ``S`` segments.
    """

    nl_cons: dict[int, OpNode] = field(default_factory=dict)
    nl_objs: dict[int, OpNode] = field(default_factory=dict)
    obj_senses: dict[int, int] = field(default_factory=dict)
    primal_init: dict[int, float] = field(default_factory=dict)
    dual_init: dict[int, float] = field(default_factory=dict)
    var_bounds: list[tuple[float, ...]] = field(default_factory=list)
    con_bounds: list[tuple[float, ...]] = field(default_factory=list)
    jac_col_counts: list[int] = field(default_factory=list)
    jac_lin: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    obj_lin: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    suffixes: list[Suffix] = field(default_factory=list)

    def find_suffix(self, name: str, target: Optional[str] = None) -> Optional[Suffix]:
        """Look up a suffix by ``name`` (and optional ``target``)."""
        for s in self.suffixes:
            if s.name == name and (target is None or s.target == target):
                return s
        return None


# Bound-type 5 in `b` segments encodes complementarity:
#   "5 <comp_var_kind> <con_idx>"
# where comp_var_kind is 1..4 per AMPL's complementarity convention.
# (See Gay's "Hooking Solvers" doc for the exact table.)


def _parse_bound_entry(line: str) -> tuple[float, ...]:
    """Parse one entry of a ``b`` or ``r`` segment.

    AMPL bound types:

    * ``0 a b`` — range ``[a, b]``
    * ``1 b``  — upper bound ``b`` (lower = -inf)
    * ``2 a``  — lower bound ``a`` (upper = +inf)
    * ``3``    — free
    * ``4 v``  — equality / fixed at ``v``
    * ``5 k i`` — complementarity (``b`` segment only): variable
      complements constraint ``i`` with kind ``k``
    """
    toks = line.split()
    if not toks:
        raise NLParseError("empty bound entry")
    btype = int(toks[0])
    if btype == 0:
        if len(toks) < 3:
            raise NLParseError(f"range bound needs 2 floats: {line!r}")
        return (0, float(toks[1]), float(toks[2]))
    if btype in (1, 2, 4):
        if len(toks) < 2:
            raise NLParseError(f"bound type {btype} needs 1 float: {line!r}")
        return (float(btype), float(toks[1]))
    if btype == 3:
        return (3.0,)
    if btype == 5:
        if len(toks) < 3:
            raise NLParseError(f"complementarity bound needs 2 ints: {line!r}")
        return (5.0, float(toks[1]), float(toks[2]))
    raise NLParseError(f"unknown bound type {btype} in entry {line!r}")


def _parse_suffix_marker(marker: str) -> tuple[int, int]:
    """Parse an ``S<flag> <count> <name>`` header line.

    Returns ``(flag, count)``; the name is on the same line and is read
    by the caller via the returned token offset.
    """
    # Marker is the full line.  After the leading 'S', the next int is the
    # flag, the second int is the count of entries.  The remainder is the
    # suffix name.
    if not marker.startswith("S"):
        raise NLParseError(f"not a suffix marker: {marker!r}")
    rest = marker[1:].split()
    if len(rest) < 3:
        raise NLParseError(f"suffix header missing fields: {marker!r}")
    flag = int(rest[0])
    count = int(rest[1])
    return flag, count


def parse_body(
    text_lines: list[str],
    start_line: int,
    header: NLHeader,
) -> NLBody:
    """Parse all body segments of a ``.nl`` text-format file.

    Parameters
    ----------
    text_lines
        Full file split on newlines.
    start_line
        Index of the first body line (returned by :func:`parse_header`).
    header
        Already-parsed header; used to pre-size ``var_bounds`` and
        ``con_bounds`` and to validate segment lengths.
    """
    body = NLBody()
    n_var = header.n_var
    n_con = header.n_con

    i = start_line
    while i < len(text_lines):
        line = text_lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        marker = stripped[0]

        if marker == "C":
            con_idx = int(stripped[1:].strip() or "0")
            stream = _TokenStream(text_lines, i + 1)
            body.nl_cons[con_idx] = _read_optree(stream)
            i = stream.current_line_idx()
            continue

        if marker == "O":
            tail = stripped[1:].split()
            if len(tail) < 2:
                raise NLParseError(f"malformed O segment: {stripped!r}")
            obj_idx = int(tail[0])
            sense = int(tail[1])
            body.obj_senses[obj_idx] = sense
            stream = _TokenStream(text_lines, i + 1)
            body.nl_objs[obj_idx] = _read_optree(stream)
            i = stream.current_line_idx()
            continue

        if marker == "x":
            count = int(stripped[1:].strip() or "0")
            for j in range(count):
                tok = text_lines[i + 1 + j].split()
                body.primal_init[int(tok[0])] = float(tok[1])
            i += 1 + count
            continue

        if marker == "d":
            count = int(stripped[1:].strip() or "0")
            for j in range(count):
                tok = text_lines[i + 1 + j].split()
                body.dual_init[int(tok[0])] = float(tok[1])
            i += 1 + count
            continue

        if marker == "b":
            for j in range(n_var):
                body.var_bounds.append(_parse_bound_entry(text_lines[i + 1 + j]))
            i += 1 + n_var
            continue

        if marker == "r":
            for j in range(n_con):
                body.con_bounds.append(_parse_bound_entry(text_lines[i + 1 + j]))
            i += 1 + n_con
            continue

        if marker == "k":
            count = int(stripped[1:].strip() or str(n_var - 1))
            body.jac_col_counts = [
                int(text_lines[i + 1 + j].strip()) for j in range(count)
            ]
            i += 1 + count
            continue

        if marker == "J":
            tail = stripped[1:].split()
            if len(tail) < 2:
                raise NLParseError(f"malformed J segment: {stripped!r}")
            con_idx = int(tail[0])
            count = int(tail[1])
            body.jac_lin[con_idx] = [
                (int(t.split()[0]), float(t.split()[1]))
                for t in text_lines[i + 1:i + 1 + count]
            ]
            i += 1 + count
            continue

        if marker == "G":
            tail = stripped[1:].split()
            if len(tail) < 2:
                raise NLParseError(f"malformed G segment: {stripped!r}")
            obj_idx = int(tail[0])
            count = int(tail[1])
            body.obj_lin[obj_idx] = [
                (int(t.split()[0]), float(t.split()[1]))
                for t in text_lines[i + 1:i + 1 + count]
            ]
            i += 1 + count
            continue

        if marker == "S":
            flag, count = _parse_suffix_marker(stripped)
            name = stripped.split()[2] if len(stripped.split()) >= 3 else ""
            values: dict[int, float] = {}
            for j in range(count):
                tok = text_lines[i + 1 + j].split()
                values[int(tok[0])] = float(tok[1])
            body.suffixes.append(Suffix(flag=flag, name=name, values=values))
            i += 1 + count
            continue

        if marker in ("V", "L", "F"):
            # Defined variables (V), logical constraints (L), and imported
            # functions (F) are valid AMPL constructs but not used by
            # MacMPEC.  Skip them with a clear error so we don't silently
            # produce a wrong problem.
            raise NLParseError(
                f"AMPL .nl segment {marker!r} (defined-var / logical / "
                "imported-function) is not yet supported"
            )

        raise NLParseError(f"unknown body marker {marker!r} at line {i}")

    return body


# ---------------------------------------------------------------------------
# Op-tree evaluation and gradient
# ---------------------------------------------------------------------------


def eval_value(node: OpNode, x: np.ndarray) -> float:
    """Evaluate the op-tree rooted at ``node`` at point ``x``.

    Raises :class:`NLParseError` if the tree contains an op the evaluator
    doesn't implement (mirroring the parser's coverage).
    """
    if node.kind == "num":
        return float(node.value)
    if node.kind == "var":
        return float(x[node.index])
    op = node.op
    kids = node.children
    if op == 0:   # plus
        return eval_value(kids[0], x) + eval_value(kids[1], x)
    if op == 1:   # minus
        return eval_value(kids[0], x) - eval_value(kids[1], x)
    if op == 2:   # mult
        return eval_value(kids[0], x) * eval_value(kids[1], x)
    if op == 3:   # div
        return eval_value(kids[0], x) / eval_value(kids[1], x)
    if op == 5:   # pow
        return eval_value(kids[0], x) ** eval_value(kids[1], x)
    if op == 15:  # abs
        return abs(eval_value(kids[0], x))
    if op == 16:  # neg
        return -eval_value(kids[0], x)
    if op == 35:  # atan2
        return float(np.arctan2(eval_value(kids[0], x), eval_value(kids[1], x)))
    if op == 37:  # tanh
        return float(np.tanh(eval_value(kids[0], x)))
    if op == 38:  # tan
        return float(np.tan(eval_value(kids[0], x)))
    if op == 39:  # sqrt
        return float(np.sqrt(eval_value(kids[0], x)))
    if op == 40:  # sinh
        return float(np.sinh(eval_value(kids[0], x)))
    if op == 41:  # sin
        return float(np.sin(eval_value(kids[0], x)))
    if op == 42:  # log10
        return float(np.log10(eval_value(kids[0], x)))
    if op == 43:  # log
        return float(np.log(eval_value(kids[0], x)))
    if op == 44:  # exp
        return float(np.exp(eval_value(kids[0], x)))
    if op == 45:  # cosh
        return float(np.cosh(eval_value(kids[0], x)))
    if op == 46:  # cos
        return float(np.cos(eval_value(kids[0], x)))
    if op == 47:  # atanh
        return float(np.arctanh(eval_value(kids[0], x)))
    if op == 48:  # atan
        return float(np.arctan(eval_value(kids[0], x)))
    if op == 49:  # asinh
        return float(np.arcsinh(eval_value(kids[0], x)))
    if op == 50:  # asin
        return float(np.arcsin(eval_value(kids[0], x)))
    if op == 51:  # acosh
        return float(np.arccosh(eval_value(kids[0], x)))
    if op == 52:  # acos
        return float(np.arccos(eval_value(kids[0], x)))
    if op == 54:  # sumlist
        return sum(eval_value(c, x) for c in kids)
    if op == 76:  # square (x^2)
        v = eval_value(kids[0], x)
        return v * v
    raise NLParseError(f"eval_value: op {op} not implemented")


def _add_partials(
    out: dict[int, float], partials: dict[int, float], scale: float
) -> None:
    """Accumulate ``scale * partials`` into ``out``."""
    for k, v in partials.items():
        out[k] = out.get(k, 0.0) + scale * v


def eval_grad(
    node: OpNode, x: np.ndarray
) -> tuple[float, dict[int, float]]:
    """Forward-mode AD over an op-tree.

    Returns ``(value, partials)`` where ``partials`` maps variable index
    to ``∂node/∂x[idx]`` evaluated at ``x``.  Variables that don't appear
    in the tree are absent from the dict (the caller should treat them as
    zero gradient).

    Forward-mode is used because op-trees in ``.nl`` are typically narrow
    (few variables, shallow); reverse-mode would add bookkeeping without
    measurable savings on this scale.
    """
    if node.kind == "num":
        return float(node.value), {}
    if node.kind == "var":
        return float(x[node.index]), {node.index: 1.0}

    op = node.op
    kids = node.children

    # Binary arithmetic
    if op == 0:   # plus
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        out: dict[int, float] = {}
        _add_partials(out, da, 1.0)
        _add_partials(out, db, 1.0)
        return a + b, out
    if op == 1:   # minus
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        out = {}
        _add_partials(out, da, 1.0)
        _add_partials(out, db, -1.0)
        return a - b, out
    if op == 2:   # mult
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        out = {}
        _add_partials(out, da, b)
        _add_partials(out, db, a)
        return a * b, out
    if op == 3:   # div
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        out = {}
        _add_partials(out, da, 1.0 / b)
        _add_partials(out, db, -a / (b * b))
        return a / b, out
    if op == 5:   # pow
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        # d/da (a^b) = b * a^(b-1); d/db (a^b) = a^b * ln(a)
        val = a ** b
        out = {}
        if a != 0.0 or b >= 1.0:
            _add_partials(out, da, b * (a ** (b - 1.0)))
        if db and a > 0.0:
            _add_partials(out, db, val * np.log(a))
        return float(val), out
    if op == 35:  # atan2(a, b)
        a, da = eval_grad(kids[0], x)
        b, db = eval_grad(kids[1], x)
        denom = a * a + b * b
        out = {}
        _add_partials(out, da, b / denom)
        _add_partials(out, db, -a / denom)
        return float(np.arctan2(a, b)), out

    # Unary functions: pattern is `value, derivative-of-outer * inner_grad`.
    if op in _UNARY_OPS or op == 76:
        a, da = eval_grad(kids[0], x)
        if op == 15:                       # abs
            val = abs(a)
            local = float(np.sign(a))
        elif op == 16:                     # neg
            val = -a
            local = -1.0
        elif op == 37:                     # tanh
            val = float(np.tanh(a))
            local = 1.0 - val * val
        elif op == 38:                     # tan
            val = float(np.tan(a))
            local = 1.0 + val * val
        elif op == 39:                     # sqrt
            val = float(np.sqrt(a))
            local = 0.5 / val
        elif op == 40:                     # sinh
            val = float(np.sinh(a))
            local = float(np.cosh(a))
        elif op == 41:                     # sin
            val = float(np.sin(a))
            local = float(np.cos(a))
        elif op == 42:                     # log10
            val = float(np.log10(a))
            local = 1.0 / (a * np.log(10.0))
        elif op == 43:                     # log
            val = float(np.log(a))
            local = 1.0 / a
        elif op == 44:                     # exp
            val = float(np.exp(a))
            local = val
        elif op == 45:                     # cosh
            val = float(np.cosh(a))
            local = float(np.sinh(a))
        elif op == 46:                     # cos
            val = float(np.cos(a))
            local = -float(np.sin(a))
        elif op == 47:                     # atanh
            val = float(np.arctanh(a))
            local = 1.0 / (1.0 - a * a)
        elif op == 48:                     # atan
            val = float(np.arctan(a))
            local = 1.0 / (1.0 + a * a)
        elif op == 49:                     # asinh
            val = float(np.arcsinh(a))
            local = 1.0 / float(np.sqrt(1.0 + a * a))
        elif op == 50:                     # asin
            val = float(np.arcsin(a))
            local = 1.0 / float(np.sqrt(1.0 - a * a))
        elif op == 51:                     # acosh
            val = float(np.arccosh(a))
            local = 1.0 / float(np.sqrt(a * a - 1.0))
        elif op == 52:                     # acos
            val = float(np.arccos(a))
            local = -1.0 / float(np.sqrt(1.0 - a * a))
        elif op == 76:                     # square
            val = a * a
            local = 2.0 * a
        else:
            raise NLParseError(f"eval_grad: unary op {op} unhandled")
        out = {}
        _add_partials(out, da, local)
        return val, out

    if op == 54:                           # sumlist
        out = {}
        total = 0.0
        for c in kids:
            v, g = eval_grad(c, x)
            total += v
            _add_partials(out, g, 1.0)
        return total, out

    raise NLParseError(f"eval_grad: op {op} not implemented")


# ---------------------------------------------------------------------------
# MPCCProblem builder
# ---------------------------------------------------------------------------


_INF = float("inf")


def _var_bounds_to_xl_xu(
    var_bounds: list[tuple[float, ...]], n_var: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Convert ``b``-segment entries into ``(xl, xu, comp_pairs)``.

    Bound type 5 on a variable encodes ``var v complements constraint c``.
    AMPL writes the constraint index as **1-indexed** (matching the
    ``cvar`` suffix convention from Gay's "Hooking Solvers"), so we
    subtract one before recording.
    """
    xl = np.full(n_var, -_INF)
    xu = np.full(n_var, _INF)
    comp_pairs: list[tuple[int, int]] = []
    for v, entry in enumerate(var_bounds):
        btype = int(entry[0])
        if btype == 0:                 # range [a, b]
            xl[v] = entry[1]
            xu[v] = entry[2]
        elif btype == 1:               # upper only
            xu[v] = entry[1]
        elif btype == 2:               # lower only
            xl[v] = entry[1]
        elif btype == 3:               # free
            pass
        elif btype == 4:               # fixed
            xl[v] = entry[1]
            xu[v] = entry[1]
        elif btype == 5:               # complementarity (var-side)
            xl[v] = max(xl[v], 0.0)
            comp_pairs.append((v, int(entry[2]) - 1))
        else:
            raise NLParseError(f"unknown bound type {btype} on var {v}")
    return xl, xu, comp_pairs


def _con_bounds_extract_comp(
    con_bounds: list[tuple[float, ...]],
    n_var: int,
    xl: np.ndarray,
) -> tuple[list[tuple[int, int]], set[int]]:
    """Extract complementarity pairs from ``r``-segment bound-type-5 entries.

    AMPL emits ``5 kind cvar_1indexed`` on the constraint side: this
    constraint complements the variable at the given (1-indexed) index.
    We mutate ``xl`` to lower-bound the comp variable at 0 (matching the
    MCP convention) and return the list of ``(var_idx, con_idx)`` pairs
    plus the set of con indices that should be excluded from the regular
    eq/ineq blocks.
    """
    comp_pairs: list[tuple[int, int]] = []
    comp_con_idx: set[int] = set()
    for ci, entry in enumerate(con_bounds):
        if int(entry[0]) != 5:
            continue
        # entry = (5, kind, cvar_1indexed)
        v = int(entry[2]) - 1
        if not 0 <= v < n_var:
            raise NLParseError(
                f"r-segment comp on con {ci} references var index "
                f"{v + 1} (1-based), but problem has {n_var} vars"
            )
        xl[v] = max(xl[v], 0.0)
        comp_pairs.append((v, ci))
        comp_con_idx.add(ci)
    return comp_pairs, comp_con_idx


def _make_constraint_callable(
    body: NLBody, con_idx: int
) -> tuple[Any, Any, np.ndarray]:
    """Build (value_fn, grad_fn, sparsity_cols) for constraint ``con_idx``.

    ``value_fn(x) -> float`` evaluates ``c_i(x)`` (nonlinear + linear).
    ``grad_fn(x) -> (val, dict[var_idx, partial])`` returns both pieces.
    ``sparsity_cols`` is the sorted array of variable indices that appear
    nonlinearly *or* linearly in this constraint — this is the constraint's
    Jacobian-row column set.
    """
    nl_node = body.nl_cons.get(con_idx)
    lin_terms = body.jac_lin.get(con_idx, [])
    lin_idx = np.array([v for v, _ in lin_terms], dtype=np.intp)
    lin_coef = np.array([c for _, c in lin_terms], dtype=float)

    nl_vars = sorted(_collect_vars(nl_node)) if nl_node is not None else []
    all_vars = sorted(set(int(v) for v in lin_idx) | set(nl_vars))
    sparsity_cols = np.array(all_vars, dtype=np.intp)

    def _value(x: np.ndarray) -> float:
        v_nl = eval_value(nl_node, x) if nl_node is not None else 0.0
        v_lin = float(np.dot(lin_coef, x[lin_idx])) if lin_idx.size else 0.0
        return v_nl + v_lin

    def _grad(x: np.ndarray) -> tuple[float, dict[int, float]]:
        if nl_node is not None:
            v_nl, partials = eval_grad(nl_node, x)
        else:
            v_nl, partials = 0.0, {}
        v_lin = float(np.dot(lin_coef, x[lin_idx])) if lin_idx.size else 0.0
        for v, c in lin_terms:
            partials[v] = partials.get(v, 0.0) + c
        return v_nl + v_lin, partials

    return _value, _grad, sparsity_cols


def _make_jac_row_writer(
    body: NLBody, con_idx: int
) -> tuple[Any, Any, np.ndarray]:
    """Like :func:`_make_constraint_callable` but the gradient pathway
    writes directly into a caller-provided ``out`` slice aligned with
    ``sparsity_cols``.

    Avoids the per-call ``dict.items()`` iteration and ``int(col)`` casts
    in :func:`_h_bulk_jac` etc.; for problems with thousands of comp /
    eq rows this is the difference between minutes and seconds per
    Jacobian callback.
    """
    nl_node = body.nl_cons.get(con_idx)
    lin_terms = body.jac_lin.get(con_idx, [])
    lin_idx = np.array([v for v, _ in lin_terms], dtype=np.intp)
    lin_coef = np.array([c for _, c in lin_terms], dtype=float)

    nl_vars = sorted(_collect_vars(nl_node)) if nl_node is not None else []
    all_vars = sorted(set(int(v) for v in lin_idx) | set(nl_vars))
    sparsity_cols = np.array(all_vars, dtype=np.intp)
    col_to_local = {int(c): i for i, c in enumerate(sparsity_cols)}
    lin_local = np.array(
        [col_to_local[int(v)] for v, _ in lin_terms], dtype=np.intp
    )

    def _value(x: np.ndarray) -> float:
        v_nl = eval_value(nl_node, x) if nl_node is not None else 0.0
        v_lin = float(np.dot(lin_coef, x[lin_idx])) if lin_idx.size else 0.0
        return v_nl + v_lin

    def _row(x: np.ndarray, out: np.ndarray) -> None:
        out[:] = 0.0
        if nl_node is not None:
            _, partials = eval_grad(nl_node, x)
            for v, p in partials.items():
                out[col_to_local[v]] = p
        if lin_local.size:
            out[lin_local] += lin_coef

    return _value, _row, sparsity_cols


def _collect_vars(node: OpNode) -> set[int]:
    """Recursively collect every variable index referenced in ``node``."""
    if node.kind == "var":
        return {node.index}
    if node.kind == "num":
        return set()
    out: set[int] = set()
    for c in node.children:
        out |= _collect_vars(c)
    return out


def _build_objective(
    body: NLBody, n_var: int
) -> tuple[Any, Any, int]:
    """Construct ``(objective, gradient, sense)`` for objective 0.

    AMPL emits the objective sense as ``0`` (min) or ``1`` (max).  When
    the sense is max, the returned callables negate so the downstream
    NLP path always sees a minimisation.
    """
    if 0 not in body.nl_objs and 0 not in body.obj_lin:
        # No objective declared in the .nl file.
        def _zero_obj(x: np.ndarray) -> float:
            return 0.0
        def _zero_grad(x: np.ndarray) -> np.ndarray:
            return np.zeros(n_var)
        return _zero_obj, _zero_grad, 0

    nl_node = body.nl_objs.get(0)
    lin_terms = body.obj_lin.get(0, [])
    lin_idx = np.array([v for v, _ in lin_terms], dtype=np.intp)
    lin_coef = np.array([c for _, c in lin_terms], dtype=float)
    sense = body.obj_senses.get(0, 0)
    sign = -1.0 if sense == 1 else 1.0

    def _obj(x: np.ndarray) -> float:
        v_nl = eval_value(nl_node, x) if nl_node is not None else 0.0
        v_lin = float(np.dot(lin_coef, x[lin_idx])) if lin_idx.size else 0.0
        return sign * (v_nl + v_lin)

    def _grad(x: np.ndarray) -> np.ndarray:
        out = np.zeros(n_var)
        if nl_node is not None:
            _, partials = eval_grad(nl_node, x)
            for k, v in partials.items():
                out[k] += v
        for v, c in lin_terms:
            out[v] += c
        return sign * out

    return _obj, _grad, sense


def from_nl(path: Union[str, Path]) -> Any:
    """Read an AMPL text-format ``.nl`` file and return an
    :class:`pympcc.MPCCProblem`.

    Complementarity constraints (encoded via bound-type-5 entries in the
    ``b`` segment) are routed into the MCP bulk form
    (``comp_var_pairs_bulk``).  Remaining constraints are split between
    equality and inequality blocks based on the AMPL bound type codes
    in the ``r`` segment.

    Not supported:

    * Binary ``.nl`` files (rejected with :class:`NLParseError`).
    * Defined variables (``V`` segments) and logical constraints (``L``).
    * Imported function calls (``f<N>``).
    * Multiple objectives — only objective 0 is honoured; the rest are
      ignored with a :class:`UserWarning`.
    """
    from ..problem import MPCCProblem  # avoid circular import at module load

    text = Path(path).read_text()
    lines = text.splitlines()
    header, body_start = parse_header(lines)
    body = parse_body(lines, body_start, header)

    n_var = header.n_var

    # Variable bounds + complementarity pairs from the b segment.
    xl, xu, comp_pairs = _var_bounds_to_xl_xu(body.var_bounds, n_var)

    # AMPL also emits comp links on the r segment (constraint side).
    r_pairs, r_comp_cons = _con_bounds_extract_comp(body.con_bounds, n_var, xl)
    comp_pairs.extend(r_pairs)

    # x0 — fall back to clipping zero into [xl, xu] for any var not in `x`.
    x0 = np.zeros(n_var)
    for v, val in body.primal_init.items():
        x0[v] = val
    # Clip to bounds so MPCCProblem doesn't warn.
    x0 = np.minimum(np.maximum(x0, xl), xu)

    # Objective.
    objective, gradient, _sense = _build_objective(body, n_var)

    # Split constraints.  Cons referenced by either b- or r-side comp
    # entries are routed through comp_var_pairs_bulk; everything else
    # falls into eq or ineq based on its r-segment bound type.
    comp_con_idx = {ci for _, ci in comp_pairs} | r_comp_cons
    n_compl_pairs = len(comp_pairs)

    # Per-row writers: (value_fn, row_writer, sparsity_cols).
    # rhs_or_lo, hi: equality bound is a single value (cb[1]); inequalities
    # split into 1-2 rows below.
    eq_writers: list[tuple[Any, Any, np.ndarray, float]] = []
    ineq_writers: list[tuple[Any, Any, np.ndarray, float, float]] = []

    for ci in range(header.n_con):
        if ci in comp_con_idx:
            continue   # handled via comp_var_pairs_bulk below
        cb = body.con_bounds[ci] if ci < len(body.con_bounds) else (3.0,)
        btype = int(cb[0])
        val_fn, row_fn, cols = _make_jac_row_writer(body, ci)
        if btype == 4:                        # equality
            eq_writers.append((val_fn, row_fn, cols, cb[1]))
        elif btype == 0:                      # range
            ineq_writers.append((val_fn, row_fn, cols, cb[1], cb[2]))
        elif btype == 1:                      # upper only
            ineq_writers.append((val_fn, row_fn, cols, -_INF, cb[1]))
        elif btype == 2:                      # lower only
            ineq_writers.append((val_fn, row_fn, cols, cb[1], _INF))
        elif btype == 3:                      # free → no constraint
            continue
        else:
            raise NLParseError(
                f"unsupported constraint bound type {btype} on con {ci}"
            )

    # Build complementarity bulk callables (sparse, single-pass writes).
    comp_kwargs: dict[str, Any] = {}
    if n_compl_pairs > 0:
        var_idxs = np.array([v for v, _ in comp_pairs], dtype=np.intp)
        comp_con_list = [ci for _, ci in comp_pairs]
        comp_writers = [_make_jac_row_writer(body, ci) for ci in comp_con_list]

        comp_row_sizes = np.array(
            [w[2].size for w in comp_writers], dtype=np.intp
        )
        comp_row_offsets = np.concatenate(
            ([0], np.cumsum(comp_row_sizes))
        ).astype(np.intp)
        h_jac_rows = np.concatenate(
            [np.full(w[2].size, i, dtype=np.intp)
             for i, w in enumerate(comp_writers)]
        ) if comp_writers else np.empty(0, dtype=np.intp)
        h_jac_cols = np.concatenate(
            [w[2] for w in comp_writers]
        ) if comp_writers else np.empty(0, dtype=np.intp)
        comp_total_nnz = int(comp_row_offsets[-1])
        comp_value_fns = [w[0] for w in comp_writers]
        comp_row_fns = [w[1] for w in comp_writers]

        def _h_bulk(
            x: np.ndarray, _vfns=comp_value_fns
        ) -> np.ndarray:
            out = np.empty(len(_vfns))
            for i in range(len(_vfns)):
                out[i] = _vfns[i](x)
            return out

        def _h_bulk_jac(
            x: np.ndarray,
            _rfns=comp_row_fns,
            _off=comp_row_offsets,
            _nnz=comp_total_nnz,
        ) -> np.ndarray:
            out = np.empty(_nnz)
            for i in range(len(_rfns)):
                _rfns[i](x, out[_off[i]:_off[i + 1]])
            return out

        comp_kwargs.update(
            n_comp=n_compl_pairs,
            comp_var_pairs_bulk=(
                var_idxs,
                _h_bulk,
                _h_bulk_jac,
                (h_jac_rows, h_jac_cols),
            ),
        )
    else:
        comp_kwargs["n_comp"] = 0

    # Equality block: sparse, flat-values Jacobian.
    eq_kwargs: dict[str, Any] = {}
    if eq_writers:
        eq_row_sizes = np.array(
            [w[2].size for w in eq_writers], dtype=np.intp
        )
        eq_row_offsets = np.concatenate(
            ([0], np.cumsum(eq_row_sizes))
        ).astype(np.intp)
        eq_jac_rows = np.concatenate(
            [np.full(w[2].size, i, dtype=np.intp)
             for i, w in enumerate(eq_writers)]
        )
        eq_jac_cols = np.concatenate([w[2] for w in eq_writers])
        eq_total_nnz = int(eq_row_offsets[-1])
        eq_value_fns = [w[0] for w in eq_writers]
        eq_row_fns = [w[1] for w in eq_writers]
        eq_rhs = np.array([w[3] for w in eq_writers], dtype=float)

        def _eq_constraints(
            x: np.ndarray, _vfns=eq_value_fns, _r=eq_rhs
        ) -> np.ndarray:
            out = np.empty(len(_vfns))
            for i in range(len(_vfns)):
                out[i] = _vfns[i](x)
            return out - _r

        def _eq_jac(
            x: np.ndarray,
            _rfns=eq_row_fns,
            _off=eq_row_offsets,
            _nnz=eq_total_nnz,
        ) -> np.ndarray:
            out = np.empty(_nnz)
            for i in range(len(_rfns)):
                _rfns[i](x, out[_off[i]:_off[i + 1]])
            return out

        eq_kwargs["n_eq"] = len(eq_writers)
        eq_kwargs["eq_constraints"] = _eq_constraints
        eq_kwargs["eq_jacobian"] = _eq_jac
        eq_kwargs["eq_jacobian_sparsity"] = (eq_jac_rows, eq_jac_cols)

    # Inequality block: each writer contributes 1-2 rows (lo and/or hi).
    # Sparse, flat-values Jacobian.
    ineq_kwargs: dict[str, Any] = {}
    if ineq_writers:
        # Materialise a flat list of physical rows: (writer_idx, sign, bias).
        # Final residual = sign * c(x) + bias; row values = sign * jac_row.
        ineq_rows_meta: list[tuple[int, float, float]] = []
        for w_idx, (_vfn, _rfn, _cols, lo, hi) in enumerate(ineq_writers):
            if hi != _INF:
                ineq_rows_meta.append((w_idx, +1.0, -hi))   # c - hi ≤ 0
            if lo != -_INF:
                ineq_rows_meta.append((w_idx, -1.0, lo))    # lo - c ≤ 0

        n_ineq_rows = len(ineq_rows_meta)
        ineq_row_sizes = np.array(
            [ineq_writers[m[0]][2].size for m in ineq_rows_meta],
            dtype=np.intp,
        )
        ineq_row_offsets = np.concatenate(
            ([0], np.cumsum(ineq_row_sizes))
        ).astype(np.intp)
        ineq_jac_rows = np.concatenate(
            [np.full(s, i, dtype=np.intp)
             for i, s in enumerate(ineq_row_sizes)]
        )
        ineq_jac_cols = np.concatenate(
            [ineq_writers[m[0]][2] for m in ineq_rows_meta]
        )
        ineq_total_nnz = int(ineq_row_offsets[-1])
        ineq_value_fns = [ineq_writers[m[0]][0] for m in ineq_rows_meta]
        ineq_row_fns = [ineq_writers[m[0]][1] for m in ineq_rows_meta]
        ineq_signs = np.array([m[1] for m in ineq_rows_meta], dtype=float)
        ineq_biases = np.array([m[2] for m in ineq_rows_meta], dtype=float)

        def _ineq_rows_fn(
            x: np.ndarray,
            _vfns=ineq_value_fns,
            _signs=ineq_signs,
            _biases=ineq_biases,
        ) -> np.ndarray:
            out = np.empty(len(_vfns))
            for i in range(len(_vfns)):
                out[i] = _vfns[i](x)
            return _signs * out + _biases

        def _ineq_jac(
            x: np.ndarray,
            _rfns=ineq_row_fns,
            _off=ineq_row_offsets,
            _signs=ineq_signs,
            _nnz=ineq_total_nnz,
        ) -> np.ndarray:
            out = np.empty(_nnz)
            for i in range(len(_rfns)):
                sl = out[_off[i]:_off[i + 1]]
                _rfns[i](x, sl)
                if _signs[i] != 1.0:
                    sl *= _signs[i]
            return out

        ineq_kwargs["n_ineq"] = n_ineq_rows
        ineq_kwargs["ineq_constraints"] = _ineq_rows_fn
        ineq_kwargs["ineq_jacobian"] = _ineq_jac
        ineq_kwargs["ineq_jacobian_sparsity"] = (ineq_jac_rows, ineq_jac_cols)

    return MPCCProblem(
        n=n_var,
        x0=x0,
        xl=xl,
        xu=xu,
        objective=objective,
        gradient=gradient,
        **comp_kwargs,
        **eq_kwargs,
        **ineq_kwargs,
    )


__all__ = [
    "NLBody",
    "NLHeader",
    "NLParseError",
    "OpNode",
    "Suffix",
    "eval_grad",
    "eval_value",
    "from_nl",
    "parse_body",
    "parse_header",
]
