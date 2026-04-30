# API reference

```{eval-rst}
.. currentmodule:: pympcc
```

## Problem types

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   MPCCProblem
   StructuredMPCC
   ParametricMPCC
```

## Solving

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   solve
   MPCCSolver
   solve_jax
   multistart
```

## Result types

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   MPCCResult
   IterationInfo
   IPOPTStatus
   MultiStartResult
   TNLPResult
   SensitivityResult
```

## Diagnostics

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   active_sets
   classify_cq
   degeneracy_report
   initial_point_statistics
   jac_norms
   merit_cross_check
   sosc_check
   classify_stationarity
   compute_kkt_residual
   verify_b_stationarity
```

## Sensitivity

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   sensitivity
   active_row_labels
```

## Presolve and scaling

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   presolve
   PresolveMap
   autoscale_comp_pairs
   unscale_multipliers
```

## Bilevel

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   bilevel.from_lower_level
```

## I/O

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   frontend.ampl.from_nl
```
