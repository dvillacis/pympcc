"""
Example 9 — Slack (lifting) strategy for large-n problems
==========================================================

Demonstrates the ``strategy="slack"`` lifting formulation and the Jacobian
sparsity benefit it delivers when ``n_comp ≪ n``.

Motivation
----------
Every standard strategy (Scholtes, smoothing, Lin-Fukushima) includes a
constraint block for ``G_i(x) · H_i(x) ≤ ε``.  Its Jacobian row is

    ∂(G_i · H_i)/∂x = H_i · ∇G_i + G_i · ∇H_i

which has as many nonzeros as the union of the sparsity patterns of JG and JH.
When G or H are dense functions of x (e.g. inner products with a basis vector)
this row has *n* entries — even if only a handful of complementarity pairs exist.

The slack strategy avoids this by introducing slack variables s_G = G(x) and
s_H = H(x) and replacing the complementarity block with ``s_G · s_H ≤ ε``.
The Jacobian of the new block has **zero entries in x** and exactly
``2 · n_comp`` nonzeros regardless of ``n``::

        x (n)           s_G (n_comp)    s_H (n_comp)
    [ JG(x)         |      −I         |       0      ]   ← G − s_G = 0
    [ JH(x)         |       0         |      −I      ]   ← H − s_H = 0
    [    0          |  diag(s_H)      |  diag(s_G)   ]   ← s_G · s_H ≤ ε

Problem
-------
"Spectral complementarity" — a proxy for imaging / signal recovery problems
where a few spectral projections of a high-dimensional signal must be
complementary.

    min  ‖x − x_ref‖²
    s.t. G_i(x) = Φ_i · x ≥ 0,   i = 0 … k−1
         H_i(x) = Ψ_i · x ≥ 0,   i = 0 … k−1
         G_i(x) · H_i(x) = 0,     i = 0 … k−1

where Φ_i, Ψ_i ∈ ℝ^n are dense unit vectors (spectral basis rows).

Because each G_i and H_i depends on *all* n components of x, both JG and JH
are fully dense (k × n matrices).  The G·H Jacobian block in standard
strategies therefore has k · n entries; the slack s_G·s_H block has just 2k.

Nnz comparison (both strategies, per Jacobian callback):
────────────────────────────────────────────────────────
Strategy block          Scholtes             Slack
────────────────────────────────────────────────────────
G / H rows              k·n + k·n            same (pinning rows)
G·H / s_G·s_H rows      k·n (dense)          2·k  (diagonal)
────────────────────────────────────────────────────────
Comp-block entries      k·n                  2·k
Reduction factor        —                    n/2
────────────────────────────────────────────────────────

For n = 500, k = 5 this is a 250× reduction for the complementarity rows.
"""
from __future__ import annotations

import time

import numpy as np
import pympcc
from pympcc.strategies.slack import SlackStrategy


# ── Problem builder ──────────────────────────────────────────────────────── #

def build_spectral_mpcc(n: int, k: int, seed: int = 0) -> pympcc.MPCCProblem:
    """
    Build the spectral complementarity problem.

    Parameters
    ----------
    n : int   Number of signal components (the "large" dimension).
    k : int   Number of complementarity pairs (spectral channels).
    seed : int  Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Spectral basis rows: each Phi_i and Psi_i is a random unit vector in R^n
    Phi = rng.standard_normal((k, n))          # JG matrix (constant)
    Phi /= np.linalg.norm(Phi, axis=1, keepdims=True)

    Psi = rng.standard_normal((k, n))          # JH matrix (constant)
    Psi /= np.linalg.norm(Psi, axis=1, keepdims=True)

    # Reference signal: IPOPT should pull x toward x_ref, then complementarity
    # forces projections onto Phi and Psi to respect G·H = 0.
    x_ref = rng.standard_normal(n)
    x0    = np.zeros(n)                        # cold start

    def objective(x: np.ndarray) -> float:
        return float(np.sum((x - x_ref) ** 2))

    def gradient(x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - x_ref)

    def comp_G(x: np.ndarray) -> np.ndarray:
        return Phi @ x                         # k projections

    def comp_G_jacobian(_x: np.ndarray) -> np.ndarray:
        return Phi                             # constant k×n matrix (dense)

    def comp_H(x: np.ndarray) -> np.ndarray:
        return Psi @ x                         # k projections

    def comp_H_jacobian(_x: np.ndarray) -> np.ndarray:
        return Psi                             # constant k×n matrix (dense)

    return pympcc.MPCCProblem(
        n=n, n_comp=k, x0=x0,
        objective=objective, gradient=gradient,
        comp_G=comp_G, comp_G_jacobian=comp_G_jacobian,
        comp_H=comp_H, comp_H_jacobian=comp_H_jacobian,
    )


# ── Nnz accounting ───────────────────────────────────────────────────────── #

def scholtes_comp_nnz(n: int, k: int) -> int:
    """
    Nonzeros in the complementarity block of a Scholtes Jacobian.

    Constraint layout: [G, H, G·H] — three k×n dense blocks.
    """
    return 3 * k * n          # G block + H block + G·H block


def slack_comp_nnz(n: int, k: int) -> int:
    """
    Nonzeros in the complementarity block of a Slack Jacobian.

    Lifted layout [G−s_G, H−s_H, s_G·s_H]:
      G−s_G block : k·n (JG x-cols) + k (−I diagonal)
      H−s_H block : k·n (JH x-cols) + k (−I diagonal)
      s_G·s_H block: 2·k (only slack diagonals — zero in x)
    """
    return k * n + k + k * n + k + 2 * k      # pinning rows + comp rows


def slack_comp_rows_nnz(k: int) -> int:
    """Nonzeros in just the s_G·s_H rows — the x-free part."""
    return 2 * k


# ── Verification helper ──────────────────────────────────────────────────── #

def _verify_x_block_zero(problem: pympcc.MPCCProblem) -> bool:
    """
    Check that the complementarity rows in the lifted Jacobian have no
    x-column entries.
    """
    strategy = SlackStrategy(problem, {})
    rows, cols = strategy._build_lifted_jac_structure()
    row_comp = problem.n_ineq + problem.n_eq + 2 * problem.n_comp
    comp_mask = rows >= row_comp
    return not np.any(cols[comp_mask] < problem.n)


# ── Main ─────────────────────────────────────────────────────────────────── #

def main() -> None:
    # ── Section 1: small problem — correctness ─────────────────────────── #

    n_small, k = 20, 3
    problem_small = build_spectral_mpcc(n=n_small, k=k)

    print("Slack strategy — spectral complementarity MPCC")
    print("=" * 60)
    print(f"  n = {n_small}, n_comp = {k}  (small demo)")
    print()

    r_scholtes = pympcc.solve(problem_small, strategy="scholtes")
    r_slack    = pympcc.solve(problem_small, strategy="slack")

    delta_obj = abs(r_scholtes.obj - r_slack.obj)
    delta_x   = float(np.max(np.abs(r_scholtes.x - r_slack.x)))

    print(f"  {'strategy':<12}  {'f*':>10}  {'comp_res':>10}  {'ok?':>4}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*4}")
    for name, r in [("scholtes", r_scholtes), ("slack", r_slack)]:
        ok = r.success and r.comp_residual < 1e-4
        print(f"  {name:<12}  {r.obj:>10.6f}  {r.comp_residual:>10.2e}  "
              f"{'✓' if ok else '✗'}")

    print()
    print(f"  |Δ obj| = {delta_obj:.2e}   (scholtes vs slack)")
    print(f"  |Δ x|∞  = {delta_x:.2e}")
    print()

    # ── Section 2: structural check ────────────────────────────────────── #

    x_zero = _verify_x_block_zero(problem_small)
    print(f"  x-block of comp rows is zero: {x_zero}")
    print()

    # ── Section 3: nnz accounting ──────────────────────────────────────── #

    print("Jacobian nonzero comparison  (dense JG/JH, per callback)")
    print("=" * 60)
    sizes = [10, 50, 100, 200, 500, 1000, 10000]
    k_fixed = 5
    print(f"  k = {k_fixed} complementarity pairs")
    print()
    print(f"  {'n':>6}  {'Scholtes comp':>14}  {'Slack comp':>12}  "
          f"{'Slack comp rows':>16}  {'reduction':>10}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*12}  {'─'*16}  {'─'*10}")
    for n_ in sizes:
        s_nnz  = scholtes_comp_nnz(n_, k_fixed)
        sl_nnz = slack_comp_nnz(n_, k_fixed)
        sl_cr  = slack_comp_rows_nnz(k_fixed)
        # reduction ratio for just the complementarity rows
        ratio = scholtes_comp_nnz(n_, k_fixed) // 3 // sl_cr  # G*H rows vs s_G*s_H rows
        print(f"  {n_:>6}  {s_nnz:>14}  {sl_nnz:>12}  {sl_cr:>16}  {ratio:>9}×")

    print()
    print("  'Slack comp rows' = entries in s_G·s_H rows only (always 2·k).")
    print("  'reduction' = (Scholtes G·H block nnz) / (Slack s_G·s_H nnz).")
    print()

    # ── Section 4: timing on a moderate problem ────────────────────────── #

    n_med, k_med = 10000, 5
    problem_med = build_spectral_mpcc(n=n_med, k=k_med)

    print(f"Timing comparison  (n = {n_med}, k = {k_med})")
    print("=" * 60)

    t0 = time.perf_counter()
    r_sch = pympcc.solve(problem_med, strategy="scholtes")
    t_sch = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_slk = pympcc.solve(problem_med, strategy="slack")
    t_slk = time.perf_counter() - t0

    print(f"  scholtes : {t_sch:.3f} s   f* = {r_sch.obj:.6f}  "
          f"comp_res = {r_sch.comp_residual:.2e}")
    print(f"  slack    : {t_slk:.3f} s   f* = {r_slk.obj:.6f}  "
          f"comp_res = {r_slk.comp_residual:.2e}")
    print()
    print(f"  |Δ f*| = {abs(r_sch.obj - r_slk.obj):.2e}")
    print()

    # ── Section 5: stationarity ────────────────────────────────────────── #

    print("Stationarity classification")
    print("=" * 60)
    print("  (slack uses geometric biactive check — no multiplier signs)")
    print()
    print(f"  scholtes stationarity : {r_scholtes.stationarity}")
    print(f"  slack    stationarity : {r_slack.stationarity}")
    print()
    print("  'S-stationary' from slack means no biactive pairs found;")
    print("  'unknown' would mean biactive pairs exist (multiplier signs")
    print("  are ambiguous in the lifted formulation).")


if __name__ == "__main__":
    main()
