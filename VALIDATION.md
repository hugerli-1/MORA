# Release Validation

The release was checked without changing the research implementation or
retraining the benchmark backbones.

## Code and Execution

- All Python sources parse successfully.
- 66 extracted definitions across 14 numerical modules were compared using
  Python syntax trees. Numerical operations were preserved; differences are
  identifier naming and two documentation/error-message phrases.
- 25 automated tests pass, covering correction geometry, query shapes and
  indices, held-interval exclusion, both coarse-estimator implementations,
  target-free inference, query independence, calibration placement,
  serialization, paired metrics, and anonymous archive construction.
- Both synthetic examples run through fitting, prediction, and save/load.
  Predictions before and after serialization match exactly in the tested CPU
  environment. Example improvements are not benchmark claims.
- The `fit`, `predict`, and `evaluate` command sequence was exercised separately.
- A Python wheel was built without downloading dependencies.

Reference environment: Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1. The optional
CUDA residual-fitting path is retained but was not benchmarked in this release
validation; these checks used CPU.

## Archived-State Replay

The following checks used existing fitted states, not newly fitted replacements.

| Scope | Settings | Queries per setting | Maximum absolute difference |
|---|---:|---:|---:|
| ECL/Traffic, L512, H96/H720, channel-wise/factorized, complete corrected forecast | 8 | 4 | 0 |
| ETTh1/ETTh2/ETTm1/ETTm2/Weather, TimeFilter, L96, H96/H720, coarse estimates | 10 | 17 | 4.77e-7 |

The second check compares extrapolated and learned coarse states, not a full
forecast or a new benchmark score. The archived-state checks validate the
portable inference equations. They do not establish that every fresh adapter
fit reproduces every table cell; the complete benchmark was not rerun for this
release. Replay inputs and large fitted states are not included in the archive.

## Result Aggregation

The release preserves 736 individual run records spanning 480 settings,
including the foundation-model settings. Aggregation from run records to
horizon entries and then dataset/backbone averages passes consistency checks.

| Table scope | Run records | Settings | Mean MSE reduction |
|---|---:|---:|---:|
| Six backbones, long-term | 384 | 192 | 7.188192% |
| Foundation models, L96 | 128 | 96 | 23.973167% |
| Foundation models, L512 | 128 | 96 | 6.931660% |
| Six backbones, PEMS | 96 | 96 | 28.095082% |

Task-specific MSE wins are 277/288 = 96.180556%. The PEMS reference table
retains its six quoted-reference flags and local-baseline columns. No missing
seed records were synthesized.

## Anonymous Packaging

`scripts/build_release.py` exports only the documented code, configurations,
tests, and numerical records. It rejects unexpected binary artifacts and
symlinks, checks for personal paths, email addresses, private-network addresses,
and credential-like content, and omits Git history and build caches. ZIP member
timestamps are fixed and file permissions normalized. A clean extracted copy
is used for a separate smoke/test run.

Automated text scanning does not guarantee anonymity against every possible
external inference. Review any future additions before rebuilding the archive.
