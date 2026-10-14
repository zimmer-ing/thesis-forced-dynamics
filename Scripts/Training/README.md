# Training

[Workflow](../README.md) · [Data preparation](../DataPreparation/README.md) ·
[Models](../../src/models/README.md) · [Evaluation](../Evaluation/README.md)

Prepare the datasets first. Run commands from the repository root in the
[Python environment](../../environments/README.md). Dataset, model, seed and
path selections are defined at the top of each script.

## Hyperparameters and multiple seeds

Dissertation: Sections 8.4 and 8.5.1. Appendix D.2 lists hyperparameters and
training epochs.

```sh
bash Scripts/Train.sh main
```

[Train_Models.py](Train_Models.py) runs hyperparameter optimization (HPO) and
final training with seed 42 for each selected dataset–model pair. It then trains
seeds 1–9 with the selected hyperparameters.
These runs are stored under `results/new_runs/full_data_hpo/`.

The main comparison includes FC, LSTM, GRU, TCN, NO-Obs, NO-Aug0, NO-AugH,
NO-Lat and LaplaceNet on seven datasets. NO-Prior is included for RLC.
Model and dataset settings are in `configs/models` and `configs/datasets`.
HPO and training settings are in
[PredictionPerformance.yaml](../../configs/experiments/PredictionPerformance.yaml).

HPO and seed 42 finish before the additional seeds start. These seeds are
assigned to visible CUDA GPUs, with one process per GPU. Set `final_gpus = [0, 1]`
to select two visible GPUs, or `final_gpus = []` for sequential training.
Without CUDA, training is sequential.

The script skips runs that have already completed successfully. For a fresh
experiment, choose a new `results_path`. This keeps the new run separate from
an incomplete experiment.

## Train with existing hyperparameters

Use `--hparams-from` to train with saved hyperparameters without starting HPO:

```sh
bash Scripts/Train.sh main \
  --hparams-from results/thesis_results/full_data_hpo \
  --results-path results/new_runs/repeat_full_data
```

The source directory contains one folder per dataset and model, with
`best_hparams_seed_42.json`. Selected seeds are trained from scratch with these
parameters. The parameters are also saved in the new result folder for later
data-fraction runs. Missing parameter files are reported. The script does not
start a new search in this case.

`--dataset`, `--model` and `--seeds` restrict the selection. If omitted, the
lists at the top of [Train_Models.py](Train_Models.py) are used. In this mode,
`--seeds` selects exactly the listed seeds. The other dataset, model and training
settings still come from `configs/`. Each run saves its resolved configuration.
Keep the source directory separate from the output directory.

## Example: LaplaceNet on SMD

Prepare SMD, then train LaplaceNet with the supplied hyperparameters and seed 42:

```sh
bash Scripts/Prepare_Data.sh smd
bash Scripts/Train.sh main \
  --dataset SMDSystem --model LaplaceNet --seeds 42 \
  --hparams-from results/thesis_results/full_data_hpo \
  --results-path results/new_runs/laplace_smd
```

Append `--dry-run` to the training command first to check the parameter source
and see the selected run. The training command writes the configuration, model
checkpoints and metrics under
`results/new_runs/laplace_smd/SMDSystem/LaplaceNet/Seed_42/`.
The training budget is defined in `PredictionPerformance.yaml`. This command
performs a full training run.

To include more seeds, use `--seeds 42 1 2 3`. To search hyperparameters again,
omit `--hparams-from` and select a fresh output directory. Seed 42 performs HPO
and final training before the additional seeds.

See [code structure](../../docs/Code_structure.md) to reuse this workflow with
your own dataset, model or model variant.

## Data fractions

Dissertation: Section 8.5.2, *Data efficiency (RQ 1.1, RQ 2.2)*.

```sh
bash Scripts/Train.sh data-fraction
```

[Train_DataFraction.py](Train_DataFraction.py) requires the hyperparameters from
the main training. It retrains the selected models with seeds 42, 1, 2 and 3
at 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90 and 100% of the training data.
Validation and test sets remain unchanged. The 100% point is also trained again.
Each run writes one result file under
`results/new_runs/data_fraction/<dataset>/raw/<model>_seed<seed>_<fraction>pct.json`.

If the main training used another output directory, set `training_path` at the
top of `Train_DataFraction.py` to that directory. Its dataset/model selections
must have the required seed-42 run and hyperparameter files there.

## Fractional dynamics

Dissertation: Section 8.5.3, *Fractional Dynamics (RQ 1.3)*.

```sh
bash Scripts/Train.sh fractional
```

[Train_NFDE.py](Train_NFDE.py) trains NFDE and NFDE-t on the reconstruction,
extrapolation and forcing tasks. Both use a fixed fractional order of 0.7.
NFDE-t includes time as an input. Full-data and reduced-data training use seeds
42, 1, 2 and 3. Reduced-data training also includes the 25% fraction.

| Setting | Selected stage |
| --- | --- |
| `run_hpo` | Hyperparameter search |
| `run_full` | Full-data NFDE training |
| `run_reduced` | Reduced-data NFDE training |
| `run_references` | Full-data NO-Obs, NO-Aug0 and I-LSTM references |
| `run_reference_curves` | Reference data-fraction curves |

The default runs the NFDE stages. Enable both reference switches and uncomment
FC-dt and TCN in `reference_variants` to generate the complete fractional tables.
The listed variants define the time input and reference-specific fraction grids.

## Inspect the selection

```sh
bash Scripts/Train.sh main --dry-run
bash Scripts/Train.sh data-fraction --dry-run
bash Scripts/Train.sh fractional --dry-run
```

`--dry-run` displays the selections without training or writing results.
Data-fraction training also checks for the required HPO files.
The scripts write to separate activity folders under
[`results/new_runs`](../../results/README.md). Keep the trained checkpoints
when running the additional runtime and component measurements.
