# Paper Table Exports

These files preserve the numerical records used by the manuscript. They are
not newly rerun experiments from the synthetic example.

| Prefix | Scope |
|---|---|
| `table1` | Six task-specific backbones, eight long-term datasets, L96 |
| `table2_L96` | Three foundation models, eight datasets, L96 |
| `table2_L512` | Three foundation models, eight datasets, L512 |
| `table3` | Six task-specific backbones, four PEMS datasets, L96 |

For each prefix, `runs.csv` retains each available seed record, `horizons.csv`
aggregates runs at fixed dataset/backbone/context/horizon, and `averages.csv`
averages the four horizons. The `coarse_estimator` column uses the paper-aligned
names `channel_wise` and `factorized`.

Private source paths were removed. Numerical values, run counts, seed values,
quoted-reference flags, and local-baseline columns were not changed. A `quoted`
reference is distinct from the local paired baseline; see
`docs/REPRODUCIBILITY.md` for the aggregation and comparison definitions.

```bash
python scripts/summarize_results.py
```
