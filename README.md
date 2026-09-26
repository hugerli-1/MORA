# MORA: Correcting Time Series Forecasts with Coarse Future Signals

Anonymous review code for **Model-Output Residual Adaptation (MORA)**.
MORA refines a frozen forecaster using the same historical window and known
temporal coordinates. Its two components follow the terminology of the paper:

1. **Coarse Future Estimation** predicts segment-wise future means through
   temporal extrapolation and learned estimation.
2. **Stagewise Residual Adaptation** converts discrepancies with the backbone
   into two ordered residual updates, calibrates their sum, and reconstructs
   a boundary-safe correction.

The high-channel implementation is a **factorized coarse estimator**, not a
different adaptation method. Both implementations use the same residual updates.

## Install

The reference implementation needs Python 3.8 or newer, NumPy, and SciPy.
The tested CPU environment is recorded in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

PyTorch is optional, needed only for the `torch_cuda` residual-fitting backend
or for running a separate backbone. The default adapter works on CPU. Backbone
environments can remain separate: the interface between them is NumPy arrays.

## Run a Small Example

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.quickstart
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.quickstart \
    --estimator factorized --output runs/factorized_example
python -m unittest discover -s tests -v
```

The examples generate synthetic data and a deterministic history-only baseline.
They exercise fitting, calibration, inference, serialization, and evaluation.
**Their numerical results are smoke tests, not the paper's benchmark results.**

## Use a Frozen Backbone

```python
from mora import MORA

adapter = MORA.load("runs/my_experiment/mora.npz")
corrected = adapter.predict(
    history=X,                       # [N, L, C]
    backbone_prediction=Y_backbone,   # [N, H, C]
    sample_index=history_start,       # [N], absolute integer sampling indices
)
```

`predict` accepts no future targets. It does not update parameters, reuse earlier
test outcomes, or read history outside the supplied `X`. The first predicted
value is left unchanged. All inputs must use the backbone's normalization and
channel ordering.

## Fit on Your Forecasts

```bash
python -m mora fit \
    --config configs/channel_wise.json \
    --training-series data/training_series.npy \
    --oof data/oof.npz \
    --validation data/validation.npz \
    --output runs/my_experiment/mora.npz

python -m mora predict \
    --model runs/my_experiment/mora.npz \
    --queries data/test.npz \
    --output runs/my_experiment/predictions.npz

python -m mora evaluate \
    --queries data/test.npz \
    --predictions runs/my_experiment/predictions.npz
```

Fitting requires training-only out-of-fold (OOF) backbone forecasts. The coarse
estimators are refitted with the matching held raw intervals excluded; the final
calibration uses validation targets. Test targets are read only by `evaluate`.
See [the array contract and OOF protocol](docs/DATA_AND_BACKBONES.md) before
preparing a real experiment. The synthetic example writes files in this format.

The JSON files in `configs/` are starting configurations, not a claim that one
feature choice reproduces every table entry. The configuration must match the
dataset, horizon, context length, and corresponding frozen backbone experiment.

## Repository Layout

| File | Role |
|---|---|
| `mora/model.py` | Public `MORA.fit`, `predict`, `save`, and `load` |
| `mora/coarse_future.py` | Coarse Future Estimation |
| `mora/factorized_estimator.py` | Factorized coarse estimator |
| `mora/stagewise.py` | Stagewise Residual Adaptation and final calibration |
| `mora/config.py` | Explicit configuration and dimension checks |
| `mora/data.py` | Query-index, window, and blocked-OOF contracts |
| `mora/_numerics/` | Extracted, numerically preserved computation routines |
| `scripts/prepare_queries.py` | Convert aligned backbone prediction archives |
| `examples/quickstart.py` | Executable synthetic example |
| `tests/` | Geometry, fitting, inference, isolation, and serialization tests |
| `results/` | Anonymized numerical exports used for the paper tables |

## Paper Scope

- Six task-specific backbones: TimeFilter, iTransformer, CGTFra, PatchTST,
  DLinear, and VPNet.
- Eight long-term datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL, Traffic,
  and Solar, at `L=96`, `H=96/192/336/720`.
- Four PEMS datasets: PEMS03/04/07/08, at `L=96`, `H=12/24/48/96`.
- Three foundation models: Chronos-2, Sundial, and TimesFM 2.5, at `L=96/512`.
- The paper uses the factorized coarse estimator on ECL, Traffic, and PEMS,
  and the channel-wise estimator on the other datasets.

This release packages the adapter and its array-based reproduction interface.
It does **not** bundle datasets, backbone checkpoints, precomputed prediction
caches, or each author's backbone trainer. Consequently, downloading this code
alone does not regenerate the full benchmark matrix. Use the original backbone
implementations and the matched prediction/OOF assets described in the protocol.

See [method-to-code mapping](docs/METHOD.md), [reproduction scope](docs/REPRODUCIBILITY.md),
and [third-party notices](NOTICE.md).
