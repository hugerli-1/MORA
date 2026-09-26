# Reproduction Scope

## What Is Included

- The frozen numerical routines for segment means, historical features, ridge
  fitting, periodic templates, factorized estimation, stagewise residual maps,
  coordinate reliability, and validation calibration.
- Portable orchestration to fit the coarse estimators and adapter from explicit
  training/OOF/validation arrays, then run target-free inference.
- Numerical data exported for the paper's main and horizon-level tables.
- Executable smoke tests and structural/information-boundary tests.

## What Is Not Included

- Dataset files or private filesystem links.
- Backbone weights and full prediction caches.
- Research-directory launchers, local GPU schedulers, or all upstream model
  implementations.
- A claim that the complete benchmark was retrained during release preparation.

Exact table reproduction requires the corresponding backbone recipe, checkpoint,
normalization, indices, estimator realization, and source-seed adapter. The four
JSON examples illustrate configurations; they do not certify every dataset/model
choice. Porting the fitting interface was checked separately from reproducing
every original benchmark fit. Archived-state prediction replay checks are
listed in `VALIDATION.md`.

## Evaluation and Aggregation

Compute MSE and MAE over query, horizon, and channel axes. At a fixed setting,
aggregate available seed runs first; average the four horizon errors for a
dataset/backbone table entry. Its improvement is

```text
Imp. = 100 * (1 - mean_error_MORA / mean_error_baseline)
```

This is not generally equal to the mean of the four horizon-wise percentages.
Across dataset/backbone table entries, reported overall improvement is the
arithmetic mean of their percentages. `scripts/summarize_results.py` recomputes
these aggregates from the exported records without rerunning models.

Most reference values are local paired baselines. Six PEMS cells use
paper-reported references; their `quoted` flags and `local_base_*` columns are
retained. These reference comparisons are not same-checkpoint paired gains.
Never replace the local baseline column with a published number and call the
result paired. Per-seed files preserve the runs that actually exist.

The benchmarks were inspected during method development; they are not a new
blind-test evaluation. Fitted correction parameters use training/validation,
and the inference API has no test-target argument.

## Cost and Precision

The CPU reference implementation uses float32 trajectory arrays and float64
linear-system accumulation. It preserves fitted ridge precision when saving.
The `torch_cuda` option accelerates residual-map fitting, not all preprocessing
or ridge fitting. This portable package does not imply that the complete
backbone-plus-adapter pipeline has the same cost as the adapter alone.

Use the same numerical backend and environment when testing bit-level replay.
Different BLAS libraries or devices can introduce small floating-point changes.
