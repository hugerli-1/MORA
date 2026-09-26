# Method-to-Code Mapping

## Notation

For one channel, the frozen forecast is `y_hat_0`, and the future target is `y`.
The following symbols match the paper.

| Paper symbol | Public code | Shape |
|---|---|---|
| A | `model.coarse_future.A` | Q x H |
| U | `model.residual_adapter.U` | H x k |
| m^e | First array returned by `coarse_future.predict` | N x C x J x Q |
| m^ell | Second array returned by `coarse_future.predict` | N x C x Q |
| K_c^e | `residual_adapter.K_e` | C x k x JQ |
| K_c^ell | `residual_adapter.K_l` | C x k x Q |
| G^e, G^ell | `residual_adapter.G_e`, `G_l` | k diagonal values each |
| S_c | `residual_adapter.S` | C x k diagonal values |

Only the diagonals of G and S are stored. Each channel's S values repeat within
four consecutive coordinate groups. Internal numerical routines store `U.T`
and transposed mapping weights for efficient batched multiplication; the public
properties above use the paper's orientation.

## Coarse Future Estimation

`segment_mean_operator` partitions the horizon into contiguous intervals with
`Q = min(64, floor(H/4))`. Each row of A averages one interval.

`CoarseFutureEstimation` keeps its extrapolated estimates separate:

- **Current level:** repeat the last historical value.
- **Recent pattern:** repeat the last `min(P,L)` observations. No samples outside
  the input window are read even when `P > L`.
- **Periodic template, when enabled:** select a compact period dictionary on
  training rows, fit harmonic coefficients there, and register the query's
  historical level relative to that template.

`ChannelWiseCoarseEstimator` fits ridge regressions to `A y`, not to the backbone
residual. It averages the absolute and restored last-centered predictions.
The compact feature set has 78 features at L=96. The implementation also retains
the expanded feature realizations used by the recorded experiments; `feature_set`
specifies the choice explicitly.

`FactorizedCoarseEstimator` first fits a channel-wise periodic baseline on
training rows, subtracts it, standardizes the deviations, and projects them onto
leading training-covariance eigenvectors. A shared ridge model predicts coarse
factor states. The means, scales, and periodic baseline are restored when the
predictions are mapped back to channels. Training-internal rolling validation
selects the ridge strength and the absolute, last-relative, or averaged output
rule. The selected rule is fixed for every inference query.

## Stagewise Residual Adaptation

`StagewiseResidualAdaptation` constructs a tapered, orthonormal IDCT basis U.
Its first row is zero. The two updates are:

```text
d^e       = m^e - A^e y_hat_0
z_hat^e   = G^e K_c^e d^e
d^ell     = m^ell - A (y_hat_0 + U z_hat^e)
z_hat^ell = G^ell K_c^ell d^ell
y_hat    = y_hat_0 + U S_c (z_hat^e + z_hat^ell)
```

The mapping routine estimates regularized uncentered second-order moments of
the discrepancy and residual coordinates. Reliability scaling uses held-block
coordinate predictions, with a pooled rule for the extrapolation stage and a
blockwise rule for the learned stage. The second-stage fitting target removes
the cross-fitted first update; its coarse predictor still targets the original
future quantity `A y`.

Validation fits S with four scales per channel using the frozen mixed objective:
10% normalized MSE plus 90% smoothed normalized MAE, averaged equally across four
contiguous validation blocks. S is applied only to the final sum, never to the
intermediate forecast used by the learned stage.

For long-term settings `k=64`. The short-horizon configuration is:

| H | Q | k | Boundary taper |
|---:|---:|---:|---:|
| 12 | 3 | 8 | 2 |
| 24 | 6 | 16 | 3 |
| 48 | 12 | 32 | 6 |
| 96 | 24 | 64 | 13 |

These quantities are resolved by `MORAConfig`; they are not silently inherited
from a long-horizon rank that would exceed the short horizon.

## Numerical Preservation

The numerical kernels were extracted from the frozen experiment implementation.
Module/function identifiers were reorganized to follow the paper; the matrix
calculations, precision choices, regularization, and calibration solver were
retained. Public orchestration handles explicit arrays instead of local research
paths. CPU inference is the reference behavior. See `VALIDATION.md` for the
checks actually performed, including their scope.
