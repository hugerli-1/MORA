"""Convert backbone archives into the common MORA query format.

The series must already use the exact backbone normalization and channel order.
No model, normalization, split, or query index is silently changed here.
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mora.data import integer_indices, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", required=True, help="Normalized [T,C] NPY")
    parser.add_argument("--predictions", required=True, help="NPZ with y_pred and sample_index")
    parser.add_argument("--history-length", type=int, default=96)
    parser.add_argument("--index-convention", choices=("history_start", "future_start"), required=True)
    parser.add_argument("--include-targets", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    series = np.load(args.series, mmap_mode="r", allow_pickle=False)
    with np.load(args.predictions, allow_pickle=False) as source:
        prediction = np.asarray(source["y_pred"], dtype=np.float32)
        index = integer_indices(source["sample_index"])
        if args.index_convention == "future_start":
            index = index - args.history_length
        history = windows(series, index, args.history_length)
        if prediction.ndim != 3 or prediction.shape[0] != len(index) or prediction.shape[2] != history.shape[2]:
            raise ValueError("Backbone archive shape does not match the series")
        output = {"history": history, "backbone_prediction": prediction, "sample_index": index}
        if args.include_targets:
            truth = windows(series, index, prediction.shape[1], args.history_length)
            if "y_true" in source.files and not np.allclose(source["y_true"], truth, rtol=2e-5, atol=2e-5):
                raise ValueError("Archive truth differs: check normalization, split, and channel order")
            output["y_true"] = truth
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **output)
    print(f"Exported {len(index)} aligned queries")


if __name__ == "__main__":
    main()
