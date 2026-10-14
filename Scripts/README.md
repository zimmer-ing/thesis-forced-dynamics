# Workflow

[Project overview](../README.md) · [Environment](../environments/README.md) ·
[Models](../src/models/README.md) · [Results](../results/README.md)

Run commands from the repository root after activating the Python environment.
The workflow follows the experimental part of the dissertation:

| Stage | Guide | Main command |
| --- | --- | --- |
| Prepare the datasets | [Data preparation](DataPreparation/README.md) | `bash Scripts/Prepare_Data.sh all` |
| Search hyperparameters and train multiple seeds | [Training](Training/README.md) | `bash Scripts/Train.sh main` |
| Train with reduced data | [Training](Training/README.md#data-fractions) | `bash Scripts/Train.sh data-fraction` |
| Compare fractional systems | [Training](Training/README.md#fractional-dynamics) | `bash Scripts/Train.sh fractional` |
| Measure runtime and model components | [Evaluation](Evaluation/README.md#additional-measurements) | See the individual commands |
| Generate tables and figures | [Evaluation](Evaluation/README.md) | `bash Scripts/Evaluate.sh all` |

Datasets, models and seeds are selected at the top of each Python script.
Training scripts call `PredictionPerformance`. Runtime measurements call
`MeasureResources`.
New experiments write to `results/new_runs`. Evaluation can read either these
results or the supplied `results/thesis_results`.
