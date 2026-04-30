# AMPL `.nl` reader

```python
from pympcc.frontend.ampl import from_nl

problem = from_nl("path/to/problem.nl")
result = pympcc.solve(problem, strategy="scholtes")
```

`from_nl` parses the binary `.nl` format produced by AMPL (and `ampl_nlwrite`) and returns an `MPCCProblem` with sparse equality / inequality Jacobians emitted directly — no per-row dictionary lookups during evaluation.

## MacMPEC fixtures

A subset of the MacMPEC test set (Leyffer 2000) ships as `.nl` fixtures and is used by the regression suite (`tests/test_macmpec_nl.py`). To convert your own AMPL `.mod` files:

```bash
ampl -ogproblem problem.mod          # writes problem.nl
```

then load with `from_nl`. Path to the full 150-problem Leyffer collection: see `ROADMAP.md §6.4`.

## Benchmark runner

```python
from pympcc.benchmarks import run_benchmark, print_table, save_csv

results = run_benchmark(strategies=["scholtes", "smoothing"], verbose=True)
print_table(results)
save_csv(results, "results.csv")
```

CLI:

```bash
python -m pympcc.benchmarks.macmpec --strategy scholtes,smoothing --out results.csv
```
