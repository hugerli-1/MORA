"""Paired forecast metrics; no fitting or selection."""

import numpy as np


def forecast_errors(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 3 or not prediction.size:
        raise ValueError("Forecast and target must share nonempty [N,H,C] shape")
    error = prediction - target
    if not np.isfinite(error).all():
        raise ValueError("Metrics require finite forecasts and targets")
    return {"mse": float(np.mean(error ** 2)), "mae": float(np.mean(np.abs(error)))}


def paired_metrics(backbone, corrected, target):
    base, adapted = forecast_errors(backbone, target), forecast_errors(corrected, target)
    output = {}
    for metric in ("mse", "mae"):
        output["backbone_" + metric] = base[metric]
        output["mora_" + metric] = adapted[metric]
        output[metric + "_reduction_percent"] = (
            100 * (1 - adapted[metric] / base[metric]) if base[metric] > 0 else None)
    return output
