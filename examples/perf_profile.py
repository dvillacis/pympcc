"""
Performance profiling script for pympcc — discover bottlenecks in large problems.

Three analysis modes run in sequence by default:

  1. Size scaling sweep   — wall-clock, IPOPT iters, and Jacobian-callback cost
                            across several k values (n = 2k, n_comp = k)
  2. Dense vs Sparse      — same problem at a fixed size with and without COO
  3. cProfile breakdown   — function-level attribution on the largest case

Synthetic problem (parallel complementarity, nonlinear G):

    min  Σ_i [ (x_{2i} − 2)² + (x_{2i+1} − 1)² ]
    s.t. G_i(x) = x_{2i}²  ≥ 0
         H_i(x) = x_{2i+1} ≥ 0
         G_i · H_i = 0

Optimal: x_{2i} = √2, x_{2i+1} = 0  →  f* = k·(√2 − 2)² = k·(2 − 2√2 + 2)
         Wait: at x_{2i}=2 (for x_{2i}^2=4, but G_i·H_i=0 requires x_{2i+1}=0)
         Actually: H_i=0 is complementary. At H_i=0, x_{2i} free → minimise (x_{2i}-2)²
         which gives x_{2i}=2. So f* = k · [(2-2)² + (0-1)²] = k.

Usage:
    uv run python examples/perf_profile.py
    uv run python examples/perf_profile.py --kmax 800 --strategy smoothing
    uv run python examples/perf_profile.py --dump          # save .prof for snakeviz
    uv run python examples/perf_profile.py --skip-sweep --skip-compare
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

import pympcc

# ─────────────────────────────────────────────────────────────────────────────
# Problem builder
# ─────────────────────────────────────────────────────────────────────────────

class _TimedCallback:
    """Wraps a callable, counting invocations and accumulating wall time."""
    __slots__ = ("_fn", "calls", "total_s")

    def __init__(self, fn):
        self._fn   = fn
        self.calls  = 0
        self.total_s = 0.0

    def __call__(self, x):
        t0 = time.perf_counter()
        out = self._fn(x)
        self.total_s += time.perf_counter() - t0
        self.calls   += 1
        return out

    def reset(self):
        self.calls   = 0
        self.total_s = 0.0


def build_problem(
    k: int,
    *,
    sparse: bool,
    time_jacs: bool = False,
) -> tuple[pympcc.MPCCProblem, _TimedCallback | None, _TimedCallback | None]:
    """
    Build the k-pair parallel complementarity problem.

    Returns (problem, jac_G_counter, jac_H_counter).
    Counters are None when time_jacs=False.

    Jacobian sparsity (sparse=True):
        JG[i, 2i]   = 2·x_{2i}     (k nnz, diagonal)
        JH[i, 2i+1] = 1             (k nnz, diagonal)

    Jacobian dense (sparse=False):
        JG : (k, 2k)  full matrix — k² zeros wasted
        JH : (k, 2k)  full matrix
    """
    n      = 2 * k
    x0     = np.full(n, 0.5)
    idx_G  = np.arange(0, n, 2)   # even indices  → G variables
    idx_H  = np.arange(1, n, 2)   # odd  indices  → H variables

    def objective(x):
        return float(np.sum((x[idx_G] - 2.0) ** 2 + (x[idx_H] - 1.0) ** 2))

    def gradient(x):
        g = np.zeros(n)
        g[idx_G] = 2.0 * (x[idx_G] - 2.0)
        g[idx_H] = 2.0 * (x[idx_H] - 1.0)
        return g

    def comp_G(x):
        return x[idx_G] ** 2

    def comp_H(x):
        return x[idx_H].copy()

    if sparse:
        G_rows, G_cols = np.arange(k, dtype=np.intp), idx_G.astype(np.intp)
        H_rows, H_cols = np.arange(k, dtype=np.intp), idx_H.astype(np.intp)

        def _jac_G(x):
            return 2.0 * x[idx_G]   # k nnz values

        def _jac_H(x):
            return np.ones(k)        # k nnz values (constant)

        jac_G_fn = _TimedCallback(_jac_G) if time_jacs else _jac_G
        jac_H_fn = _TimedCallback(_jac_H) if time_jacs else _jac_H

        p = pympcc.MPCCProblem(
            n=n, n_comp=k, x0=x0,
            objective=objective, gradient=gradient,
            comp_G=comp_G,
            comp_G_jacobian=jac_G_fn,
            comp_G_jacobian_sparsity=(G_rows, G_cols),
            comp_H=comp_H,
            comp_H_jacobian=jac_H_fn,
            comp_H_jacobian_sparsity=(H_rows, H_cols),
        )
    else:
        def _jac_G_dense(x):
            J = np.zeros((k, n))
            J[np.arange(k), idx_G] = 2.0 * x[idx_G]
            return J

        def _jac_H_dense(x):
            J = np.zeros((k, n))
            J[np.arange(k), idx_H] = 1.0
            return J

        jac_G_fn = _TimedCallback(_jac_G_dense) if time_jacs else _jac_G_dense
        jac_H_fn = _TimedCallback(_jac_H_dense) if time_jacs else _jac_H_dense

        p = pympcc.MPCCProblem(
            n=n, n_comp=k, x0=x0,
            objective=objective, gradient=gradient,
            comp_G=comp_G,
            comp_G_jacobian=jac_G_fn,
            comp_H=comp_H,
            comp_H_jacobian=jac_H_fn,
        )

    g_ctr = jac_G_fn if time_jacs else None
    h_ctr = jac_H_fn if time_jacs else None
    return p, g_ctr, h_ctr


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1: size scaling sweep
# ─────────────────────────────────────────────────────────────────────────────

def size_sweep(sizes: list[int], strategy: str, sparse: bool, max_iter: int) -> None:
    jac_label = "sparse" if sparse else "dense"
    print(f"Size scaling sweep  —  strategy={strategy}  jacobians={jac_label}")
    print()

    # column widths chosen so the table stays under 100 chars
    hdr = (
        f"  {'k':>5}  {'n':>6}  {'outer':>5}  {'ipopt_it':>8}  "
        f"{'solve_s':>7}  {'jac_calls':>9}  {'jac_s':>6}  {'jac%':>5}  "
        f"{'comp_res':>9}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for k in sizes:
        p, ctr_G, ctr_H = build_problem(k, sparse=sparse, time_jacs=True)

        t0     = time.perf_counter()
        result = pympcc.solve(p, strategy=strategy, max_iter=max_iter)
        wall   = time.perf_counter() - t0

        ipopt_iters = sum(it.n_ipopt_iter for it in result.history)
        jac_calls   = ctr_G.calls + ctr_H.calls
        jac_s       = ctr_G.total_s + ctr_H.total_s
        jac_pct     = 100.0 * jac_s / wall if wall > 0 else 0.0

        print(
            f"  {k:>5}  {p.n:>6}  {len(result.history):>5}  {ipopt_iters:>8}  "
            f"{result.solve_time:>7.3f}  {jac_calls:>9}  {jac_s:>6.3f}  "
            f"{jac_pct:>4.1f}%  {result.comp_residual:>9.2e}"
        )

    print()
    print("  solve_s   = total wall time inside nlp.solve() calls (IPOPT + callbacks)")
    print("  jac_calls = comp_G_jacobian + comp_H_jacobian invocations")
    print("  jac_s     = wall time spent in those two callbacks")
    print("  jac%      = jac_s / wall_time — fraction attributable to Jacobian callbacks")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2: dense vs sparse comparison
# ─────────────────────────────────────────────────────────────────────────────

def dense_vs_sparse(k: int, strategy: str, max_iter: int) -> None:
    print(f"Dense vs Sparse Jacobians  —  k={k}  (n={2*k}, n_comp={k})")
    print()

    nnz_sparse = 2 * k
    nnz_dense  = 2 * k * (2 * k)

    results = {}
    for label, sp in [("dense", False), ("sparse", True)]:
        p, ctr_G, ctr_H = build_problem(k, sparse=sp, time_jacs=True)
        t0     = time.perf_counter()
        r      = pympcc.solve(p, strategy=strategy, max_iter=max_iter)
        wall   = time.perf_counter() - t0
        jac_s  = ctr_G.total_s + ctr_H.total_s
        results[label] = dict(r=r, wall=wall, jac_s=jac_s)

    dense_wall  = results["dense"]["wall"]
    sparse_wall = results["sparse"]["wall"]
    speedup     = dense_wall / sparse_wall if sparse_wall > 0 else float("inf")

    print(f"  {'':6}  {'nnz/call':>9}  {'wall_s':>7}  {'solve_s':>7}  "
          f"{'jac_s':>6}  {'comp_res':>9}")
    print("  " + "─" * 60)
    for label, sp in [("dense", False), ("sparse", True)]:
        d   = results[label]
        nnz = nnz_dense if not sp else nnz_sparse
        print(
            f"  {label:>6}  {nnz:>9}  {d['wall']:>7.3f}  "
            f"{d['r'].solve_time:>7.3f}  {d['jac_s']:>6.3f}  "
            f"{d['r'].comp_residual:>9.2e}"
        )

    print()
    ratio = nnz_dense / nnz_sparse
    print(f"  Jacobian nnz ratio   dense/sparse: {ratio:.0f}×")
    print(f"  Wall-clock speedup   sparse/dense: {speedup:.1f}×")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: cProfile breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _strip_pstats_header(text: str, keep_from: str = "ncalls") -> str:
    """Drop the verbose pstats preamble; keep from the column header onward."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if keep_from in line:
            return "\n".join(lines[i:])
    return text


def cprofile_breakdown(k: int, strategy: str, sparse: bool,
                       max_iter: int, dump: bool) -> None:
    p, _, _ = build_problem(k, sparse=sparse)

    pr = cProfile.Profile()
    pr.enable()
    result = pympcc.solve(p, strategy=strategy, max_iter=max_iter)
    pr.disable()

    sp_label = "sparse" if sparse else "dense"
    print(f"cProfile  —  k={k}  strategy={strategy}  jacobians={sp_label}")
    print()

    # ── cumulative time: shows call chain and total cost per entry point ──
    print("Top 25 by cumulative time  (where total cost accumulates)")
    print("─" * 80)
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(25)
    print(_strip_pstats_header(buf.getvalue()))

    # ── tottime: shows where CPU is actually burned (no subcall inflation) ──
    print("Top 15 by own time  (actual CPU hotspots, no subcall inflation)")
    print("─" * 80)
    buf2 = io.StringIO()
    pstats.Stats(pr, stream=buf2).sort_stats("tottime").print_stats(15)
    print(_strip_pstats_header(buf2.getvalue()))

    print(f"  outer iterations : {len(result.history)}")
    print(f"  solve_time       : {result.solve_time:.3f}s")
    print(f"  comp_residual    : {result.comp_residual:.2e}")
    if result.kkt_residual is not None:
        print(f"  kkt_residual     : {result.kkt_residual:.2e}")
    print()

    if dump:
        fname = f"pympcc_k{k}_{strategy}_{sp_label}.prof"
        pr.dump_stats(fname)
        print(f"  Profile saved → {fname}")
        print(f"  Visualise with:  python -m snakeviz {fname}")
        print(f"  or:              uv run python -m snakeviz {fname}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _sweep_sizes(kmax: int) -> list[int]:
    """Five geometrically spaced sizes from kmax/20 to kmax."""
    pts = [max(10, kmax // 20), max(20, kmax // 8),
           max(50, kmax // 4),  max(100, kmax // 2), kmax]
    return sorted(set(pts))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="pympcc performance profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--strategy", default="scholtes",
                    choices=["scholtes", "smoothing", "lin_fukushima",
                             "augmented_lagrangian", "slack"],
                    help="MPCC strategy to profile")
    ap.add_argument("--kmax", type=int, default=300,
                    help="Largest k (n=2k, n_comp=k) in the size sweep")
    ap.add_argument("--compare-k", type=int, default=None,
                    help="k for dense vs sparse comparison (default: kmax//2)")
    ap.add_argument("--profile-k", type=int, default=None,
                    help="k for cProfile analysis (default: kmax)")
    ap.add_argument("--dense", dest="sparse", action="store_false",
                    help="Use dense Jacobians in the sweep (default: sparse)")
    ap.add_argument("--max-iter", type=int, default=20,
                    help="Maximum outer iterations per solve")
    ap.add_argument("--dump", action="store_true",
                    help="Save .prof file (open with snakeviz or py-spy)")
    ap.add_argument("--skip-sweep",   action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()

    compare_k = args.compare_k or max(50, args.kmax // 2)
    profile_k = args.profile_k or args.kmax
    sizes     = _sweep_sizes(args.kmax)

    print("=" * 70)
    print("  pympcc — performance profile")
    print(f"  strategy={args.strategy}  kmax={args.kmax}  "
          f"jacobians={'sparse' if args.sparse else 'dense'}  "
          f"max_iter={args.max_iter}")
    print("  problem: G_i=x[2i]²  H_i=x[2i+1]  (nonlinear G, linear H)")
    print("=" * 70)
    print()

    if not args.skip_sweep:
        size_sweep(sizes, args.strategy, args.sparse, args.max_iter)

    if not args.skip_compare:
        dense_vs_sparse(compare_k, args.strategy, args.max_iter)

    if not args.skip_profile:
        cprofile_breakdown(profile_k, args.strategy, args.sparse,
                           args.max_iter, args.dump)


if __name__ == "__main__":
    main()
