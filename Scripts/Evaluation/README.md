# Evaluation

[Workflow](../README.md) · [Training](../Training/README.md) ·
[Results](../../results/README.md)

The research-question scripts read saved experimental results and generate the
tables and figures. Run commands from the repository root in the
[Python environment](../../environments/README.md).

## Select the results

Set the source in [Constants.py](../../Constants.py):

```python
EVALUATION_SOURCE = 'thesis_results'  # or 'new_runs'
```

`thesis_results` contains the supplied dissertation results. `new_runs` reads
only newly generated results. Missing results are not filled from another source.
The supplied results include configurations and metrics. Generating new
predictions requires the checkpoints from a new training run.

## Tables and figures

```sh
bash Scripts/Evaluate.sh all
```

Replace `all` with one task for an individual evaluation:

| Task | Script and output | Dissertation |
| --- | --- | --- |
| `rq1_1` | [Evaluate_RQ1_1_RQ2_2.py](RQs/Evaluate_RQ1_1_RQ2_2.py): model accuracy and data fractions | Sections 8.5.1–8.5.2 |
| `rq1_2` | [Evaluate_RQ1_2.py](RQs/Evaluate_RQ1_2.py): RLC physics prior and component ablation | Section 8.5.4 |
| `rq1_3` | [Evaluate_RQ1_3.py](RQs/Evaluate_RQ1_3.py): fractional-system comparison and data fractions | Section 8.5.3 |
| `rq2_3` | [Evaluate_RQ2_3.py](RQs/Evaluate_RQ2_3.py): runtime comparison | Section 8.5.5 |
| `appendix` | [Evaluate_Appendix.py](Supplementary/Evaluate_Appendix.py): dataset windows, hyperparameters, epochs and full comparisons | Appendix D.1–D.4 |

The scripts save tables as LaTeX and CSV files and figures as PDF and PNG files.
Output is separated by source and research question under
`output/pdf/Evaluation/<source>/`.
These scripts do not start training or runtime measurements. The appendix also
reads the prepared datasets.

```sh
bash Scripts/Evaluate.sh all --dry-run
```

`--dry-run` displays the selected inputs and outputs without generating tables
or figures.

## Additional measurements

For new results, run the relevant measurements before generating their tables.
All commands below write to `results/new_runs`.

| Measurement | Command | Required input |
| --- | --- | --- |
| Prediction runtime | `bash Scripts/Train.sh runtime` | Trained checkpoints and CUDA |
| RLC component ablation | `python -m Scripts.Evaluation.Supplementary.Evaluate_Physics` | Trained RLC NO-Prior checkpoint |
| Persistence baseline | `python -m Scripts.Evaluation.Supplementary.Evaluate_Persistence` | Prepared Cascaded Tanks data |

[Evaluate_Resources.py](Supplementary/Evaluate_Resources.py) measures scaling with
forecast length and batch size. Each point uses five warm-up runs and ten timed
predictions. Choose a new `run_name` for each measurement. If the name differs
from `runtime_01`, adjust `runtime_csv` in the RQ2.3 script.

[Evaluate_Physics.py](Supplementary/Evaluate_Physics.py) compares the full model,
its physics component and its neural component. Use `--run-dir <directory>`
to select another checkpoint.
[Evaluate_Persistence.py](Supplementary/Evaluate_Persistence.py) repeats the last
observed output. Use `--dataset` to select another dataset.

Generate the persistence reference for all seven main datasets:

```sh
for dataset in SMDSystem RLCDataset IndPenSim CascadedTanks \
               IndustrialRobotForward NanoDroneOOD NanoDroneRunSplit; do
    python -m Scripts.Evaluation.Supplementary.Evaluate_Persistence --dataset "$dataset"
done
```

## Numerical examples and solver validation

These examples are separate from the trained-model comparisons and are excluded
from `Evaluate.sh all`.

```sh
bash Scripts/Evaluate.sh demos
python -m Scripts.Evaluation.Supplementary.Evaluate_FDE_solver_unforced_zero_forcing_GPU
python -m Scripts.Evaluation.Supplementary.Plot_FDE_solver_validation
```

| Script | Purpose | Dissertation |
| --- | --- | --- |
| [Evaluate_RQ2_1.py](../Demonstrations/Evaluate_RQ2_1.py) | ILT examples and DLT/FFLT reconstruction of sine and ramp signals | Sections 6.3.4 and 7.1 |
| [Evaluate_FDE_solver_unforced_zero_forcing_GPU.py](Supplementary/Evaluate_FDE_solver_unforced_zero_forcing_GPU.py) | GPU solver comparison with Coelho and an analytical reference | Section 5.2.4, Table 5.1 |
| [Plot_FDE_solver_validation.py](Supplementary/Plot_FDE_solver_validation.py) | Forced FDE comparison with the supplied FOMCON trajectory | Section 5.2.4, Table 5.2 |

The GPU solver comparison requires CUDA. The forced-FDE plot requires LaTeX.

Recreate Table 5.1 from the supplied per-run measurements without measuring again:

```sh
python -m Scripts.Evaluation.Supplementary.Evaluate_FDE_solver_unforced_zero_forcing_GPU \
  --raw-csv results/thesis_results/solver/zero_forcing/solver_unforced_zero_forcing_gpu_raw.csv
```

The table is written under `output/pdf/Evaluation/<source>/Supplementary/`.
