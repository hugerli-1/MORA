# Forecast Arrays and Backbone Integration

## Required Files

All arrays must use the **same normalization and channel order as the backbone**.
Do not denormalize only one of the history, forecast, or target arrays.

`training_series.npy` is a finite float array `[T_train,C]` containing the
training split only, in normalized backbone coordinates. Its first row is
absolute sampling index zero. Validation and test rows must not be appended.

`validation.npz` contains:

| Key | Shape | Meaning |
|---|---|---|
| `history` | N x L x C | Exactly the history supplied to the frozen backbone |
| `backbone_prediction` | N x H x C | Forecast from the matched frozen checkpoint |
| `sample_index` | N | Absolute **history-start** index, integer |
| `y_true` | N x H x C | Validation targets for final calibration |

`oof.npz` contains the same four keys for held training queries, plus:

| Key | Shape | Meaning |
|---|---|---|
| `fold_id` | N | Integer fold label, starting at zero |
| `held_raw_intervals` | B x 2 | Half-open raw index ranges `[start,end)` excluded from each fold fit |

`test.npz` contains the same query keys as validation. Its `y_true` is optional
for prediction and required only for evaluation. The prediction command does
not read `y_true`, even if the key exists in the archive.

All saved arrays use `allow_pickle=False`. Sample indices must be unique and
chronological in fitting/evaluation archives; no positional truncation is used
to conceal misaligned windows. Direct `MORA.predict` also supports isolated
queries and arbitrary query order.

## Building Training OOF Forecasts

1. Train the deployment backbone with its corresponding benchmark recipe.
2. Use `blocked_oof_geometry(T_train,L,H)` to generate four held blocks from
   the latter half of the training-window origins, sampling at stride six.
3. For each held raw interval, exclude every training window whose history or
   future intersects the interval. `legal_training_origins` implements this
   exclusion with span `L+H`.
4. Fit that backbone replica on the legal training windows, using the original
   replica checkpoint-selection recipe. Do not replace this with an additional
   outer-nested procedure. Never select a checkpoint on test targets.
5. Predict the held queries. Convert replica-normalized outputs back to the
   deployment backbone's official normalized coordinates before storing them.
6. Store the exact held indices, fold labels, and excluded raw intervals in
   `oof.npz`. The adapter refits each training-dependent coarse estimator using
   these same exclusions. It also checks that histories and targets match the
   declared training-row indices.

The archive records the contract; it cannot prove that an external backbone
trainer really excluded the held intervals. That responsibility remains with
the backbone exporter. A sample's fitted in-sample prediction must not be
renamed as an OOF prediction.

For foundation models, pretrained weights stay fixed. The training-dependent
coarse estimators still follow the held-interval exclusion. Context lengths
96 and 512 require separately fitted adapters and matched prediction archives.

## Converting an Existing Prediction Archive

If an upstream export has `y_pred` and `sample_index`, use:

```bash
python scripts/prepare_queries.py \
    --series data/normalized_complete_series.npy \
    --predictions data/backbone_validation.npz \
    --history-length 96 --index-convention history_start \
    --include-targets --output data/validation.npz
```

For an export indexed by the start of the future rather than the history, pass
`--index-convention future_start`. This is an explicit conversion, not an
automatic guess. If the source archive includes targets, they are checked
against the supplied series before writing. Export the training-only prefix
separately as `training_series.npy`.

## Dataset Configuration

| Dataset | Base period P | Main coarse estimator |
|---|---:|---|
| ETTh1 / ETTh2 | 24 | Channel-wise |
| ETTm1 / ETTm2 | 96 | Channel-wise |
| Weather / Solar | 144 | Channel-wise |
| ECL / Traffic | 24 | Factorized, phase period 168 |
| PEMS03 / 04 / 07 / 08 | 288 | Factorized, phase period 2016 |

ETT uses the standard 12/4/4-month split; other long-term datasets use 7:1:2;
PEMS uses 6:2:2. The adapter itself does not redefine these splits. Preserve the
native backbone's exact boundary-window policy, missing-value handling, scale,
and metric definition. In particular, do not silently reuse another backbone's
PEMS window indices when their loaders differ.

## Backbone Boundary

MORA consumes predictions, not internal features or architecture-specific
hidden states. The same API therefore covers the six task-specific backbones
and three foundation models in the paper. Their original trainers/checkpoints
are not included in this repository.

For TimeFilter, iTransformer, and CGTFra settings with published PEMS values,
compare the local unadapted baseline with the matching reported setting before
interpreting the adapter gain. TimeFilter H96 is an extension. The PEMS recipes
for PatchTST, DLinear, and VPNet are transferred settings, not reproductions of
PEMS scores absent from their papers.

Preserve individual seeds. Repeated backbone inference runs must not be
represented as independent foundation-model pretraining runs. The records in
`results/` retain the existing seed and reference-baseline distinctions.
