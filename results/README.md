# Results

[Project overview](../README.md) · [Evaluation](../Scripts/Evaluation/README.md)

The supplied dissertation results are stored in `thesis_results/`.
New training and measurements write to `new_runs/`.

## Select the source

Set the source in [Constants.py](../Constants.py):

```python
EVALUATION_SOURCE = 'thesis_results'  # Or 'new_runs'.
```

This setting applies to the research-question evaluations and appendix.
Tables and figures are written to `output/pdf/Evaluation/<source>/`.
Training and measurement outputs remain under `new_runs/`, regardless of
the evaluation source. Missing new results are not replaced with supplied ones.

## Result folders

Results are grouped by source and then by experiment:

```text
results/
  thesis_results/
    full_data_hpo/       # HPO on all training data, then final seeds
    data_fraction/       # Retraining with reduced training data
    fractional/          # NFDE and reference-model benchmark
    physics_components/  # RLC component comparison
    resources/           # Inference-time measurements
    solver/              # Numerical solver comparisons
  new_runs/              # The same experiment folders for new runs
```

| Folder within each source | Contents |
| --- | --- |
| `full_data_hpo/` | Hyperparameter search on all training data, followed by final training with multiple seeds. Contains found hyperparameters, configurations and per-seed results. |
| `data_fraction/` | One raw result file per dataset, model, seed and training fraction. |
| `fractional/nfde/` | NFDE hyperparameters and results for the three Fractional tasks. |
| `fractional/references/` | Full-data reference training. |
| `fractional/reference_curves/` | Separately trained reference data-fraction curves. |
| `physics_components/` | RLC physical and neural component evaluation. |
| `resources/` | Inference-time measurements. |
| `solver/` | Unforced and forced solver comparisons. |

Folders and filenames use the current model names. For example:

```text
full_data_hpo/CascadedTanks/LaplaceNet/Seed_42/evaluation_results.json
data_fraction/CascadedTanks/raw/GRU_seed42_005pct.json
```

The supplied results contain raw metrics, selected hyperparameters and the
configuration and measurement metadata needed for evaluation. Console logs,
tracebacks, local machine paths, loss histories, duplicate copies and generated
tables are excluded. Original model identifiers may remain inside recorded metadata.
Each seed folder contains a `training_info.yaml` file with the epoch and
parameter summaries used by the appendix.

## Required inputs

Saved metrics can be evaluated without model weights. New predictions,
component comparisons and inference-time measurements require trained
checkpoints. Model weights, Optuna databases and logs are excluded from Git.
Appendix D.1 also requires the prepared datasets.

The supplied files retain their original numerical values and configurations.
The fractional benchmark uses seeds 42, 1, 2 and 3 for NFDE and NFDE-t at all
training fractions. Their full-data statistics differ from the ten-seed
aggregates printed in the dissertation (Tables 8.11 and D.5, and the 100% curve
points). The other dataset, model and seed selections remain as used in the
dissertation, including the separate RLC comparison.

For the supplied inference-time measurements,
[source_selection.json](thesis_results/resources/source_selection.json) records
the selected runs and [forecast_durations.json](thesis_results/resources/forecast_durations.json)
records the physical forecast durations. New measurements use
`new_runs/resources/runtime_01/` by default. For another run, adjust `runtime_csv` in
[Evaluate_RQ2_3.py](../Scripts/Evaluation/RQs/Evaluate_RQ2_3.py).
