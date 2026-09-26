"""Check paper-table aggregation against retained per-seed records."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def check_close(value, expected, context):
    if abs(value - expected) > 1e-10 * max(1.0, abs(expected)):
        raise ValueError(f"Aggregation mismatch at {context}: {value} != {expected}")


def summarize(root):
    result = {}
    errors = ("base_mse", "base_mae", "mora_mse", "mora_mae", "local_base_mse", "local_base_mae")
    for prefix in ("table1", "table2_L96", "table2_L512", "table3"):
        runs = read(root / (prefix + "_runs.csv"))
        horizons = read(root / (prefix + "_horizons.csv"))
        averages = read(root / (prefix + "_averages.csv"))
        by_setting, by_group = defaultdict(list), defaultdict(list)
        for row in runs:
            by_setting[tuple(row[k] for k in ("dataset", "backbone", "L", "H"))].append(row)
        for row in horizons:
            key = tuple(row[k] for k in ("dataset", "backbone", "L", "H"))
            records = by_setting[key]
            if len(records) != int(row["runs"]):
                raise ValueError(f"Run count mismatch at {key}")
            for metric in errors:
                check_close(mean(float(x[metric]) for x in records), float(row[metric]), (key, metric))
            by_group[key[:3]].append(row)
        for row in averages:
            key = tuple(row[k] for k in ("dataset", "backbone", "L"))
            records = by_group[key]
            for metric in errors:
                check_close(mean(float(x[metric]) for x in records), float(row[metric]), (key, metric))
            for metric in ("mse", "mae"):
                improvement = 100 * (1 - float(row["mora_" + metric]) / float(row["base_" + metric]))
                check_close(improvement, float(row["imp_" + metric]), (key, metric))
        result[prefix] = {
            "seed_records": len(runs), "settings": len(horizons),
            "mse_wins": sum(float(x["mora_mse"]) < float(x["base_mse"]) for x in horizons),
            "mean_mse_reduction_percent": mean(float(x["imp_mse"]) for x in averages),
            "mean_mae_reduction_percent": mean(float(x["imp_mae"]) for x in averages),
            "quoted_reference_settings": sum(x["quoted"].lower() == "true" for x in horizons),
        }
    result["task_specific_mse_win_percent"] = 100 * (
        result["table1"]["mse_wins"] + result["table3"]["mse_wins"]) / (
        result["table1"]["settings"] + result["table3"]["settings"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    print(json.dumps(summarize(parser.parse_args().results), indent=2))
