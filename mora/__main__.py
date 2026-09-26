"""Portable fit, predict, and evaluate commands."""

import argparse
import json
from pathlib import Path
import numpy as np
from . import MORA, MORAConfig, ForecastBatch, OOFForecasts
from .metrics import paired_metrics


def main():
    parser = argparse.ArgumentParser(description="Model-Output Residual Adaptation")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit from normalized training rows, OOF forecasts, and validation")
    fit.add_argument("--config", required=True)
    fit.add_argument("--training-series", required=True)
    fit.add_argument("--oof", required=True)
    fit.add_argument("--validation", required=True)
    fit.add_argument("--output", required=True)
    predict = commands.add_parser("predict", help="X-only inference, without reading any future targets")
    predict.add_argument("--model", required=True)
    predict.add_argument("--queries", required=True)
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate frozen predictions; never fit")
    evaluate.add_argument("--queries", required=True)
    evaluate.add_argument("--predictions", required=True)
    args = parser.parse_args()
    if args.command == "fit":
        config = MORAConfig.from_json(args.config)
        training = np.load(args.training_series, allow_pickle=False, mmap_mode="r")
        model = MORA.fit(config, training, OOFForecasts.load(args.oof),
                         ForecastBatch.load(args.validation), progress=lambda s: print(s, flush=True))
        model.save(args.output)
        print(json.dumps(model.fit_summary, indent=2, allow_nan=False))
    elif args.command == "predict":
        model = MORA.load(args.model)
        batch = ForecastBatch.load(args.queries, targets=False).validate(model.config)
        result = model.predict(batch.history, batch.backbone_prediction, batch.sample_index)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.savez_compressed(handle, prediction=result, sample_index=batch.sample_index)
        print(f"Saved {len(result)} corrected forecasts")
    else:
        batch = ForecastBatch.load(args.queries, targets=True)
        if batch.y_true is None:
            raise ValueError("Evaluation requires y_true")
        with np.load(args.predictions, allow_pickle=False) as data:
            if not np.array_equal(data["sample_index"], batch.sample_index):
                raise ValueError("Prediction and target query indices do not align")
            print(json.dumps(paired_metrics(batch.backbone_prediction, data["prediction"], batch.y_true),
                             indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
