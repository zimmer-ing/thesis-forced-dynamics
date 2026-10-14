# Data preparation

[Project overview](../../README.md) · [Environment](../../environments/README.md) ·
[Training](../Training/README.md)

Activate the Python environment and run from the repository root:

```sh
bash Scripts/Prepare_Data.sh all
```

[Prepare_Datasets.py](Prepare_Datasets.py) downloads missing external data,
generates the simulated datasets and calls the preparation scripts. It stores raw
downloads in `Datasets/`. The dataset loaders read the prepared files in `data/`.
Complete prepared datasets are retained. If a dataset is incomplete, the script
reports this and leaves the existing files unchanged.

Dissertation references: Section 8.2 describes the datasets, Appendix B.1 the
fractional trajectories, and Appendix D.1 the forecasting windows.

## Select or check datasets

```sh
bash Scripts/Prepare_Data.sh smd
bash Scripts/Prepare_Data.sh all --dry-run
bash Scripts/Prepare_Data.sh all --check
```

`--dry-run` shows the available and missing datasets without downloading or
processing data. `--check` checks the required prepared files.

| Task | Datasets |
| --- | --- |
| `all` | All datasets below |
| `external` | IndPenSim, Cascaded Tanks, Industrial Robot and both NanoDrone splits |
| `indpensim`, `tanks`, `robot` | The corresponding external dataset |
| `nanodrone` | NanoDrone 3 and NanoDrone 4 |
| `smd`, `rlc` | The corresponding simulated dataset |
| `fractional` | Reconstruction, extrapolation and forcing tasks |

## Data sources

External datasets are downloaded from their original sources for local use.
They are not included in this repository. Cite the original datasets and papers
when using them.

| Dataset | Source and preparation |
| --- | --- |
| IndPenSim | [Mendeley Data, version 1](https://data.mendeley.com/datasets/pdnjz7zz5x/1), CC BY 4.0. The download is approximately 0.50 GB and expands to approximately 2.5 GB. Only process columns preceding the Raman channels are used. |
| Cascaded Tanks | [4TU.ResearchData](https://doi.org/10.4121/12960104.v1), CC BY-SA 4.0, and [benchmark description](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks). The script tries the official download first, then the mirror referenced by the benchmark loader. A local original MAT file can also be used. |
| Industrial Robot | [Original dataset](https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/). The source lists MIT and CC BY-SA terms. The script downloads a RAR archive and extracts only the forward-dynamics MAT file. |
| NanoDrone | [Original repository](https://github.com/idsia-robotics/nanodrone-sysid-benchmark/tree/2d921b57d166fe2debe08a5d39bd07297c5abc39) and [benchmark paper](https://doi.org/10.1016/j.conengprac.2026.106871). Fifteen CSV files, approximately 29 MB, downloaded directly from the pinned revision. No repository clone or Git LFS is needed. Refer to the original source for data licensing. |

Industrial Robot extraction requires `bsdtar` (available on macOS), `unar`,
`unrar` or `7z`. GNU `tar` alone cannot extract the RAR archive.

### IndPenSim download

The script first checks for complete prepared data and retains it. Otherwise, it
uses an existing `Datasets/IndPenSim/raw/data.zip` or downloads the original ZIP.
It checks the archive size (503,792,426 bytes), SHA-256 checksum, required CSV
files and ZIP integrity before preprocessing.

If the automatic download or ZIP check fails, the script prints the download
link, destination path and command to run again:

1. Open the [original ZIP download](https://data.mendeley.com/public-files/datasets/pdnjz7zz5x/files/e7007dff-4061-494f-a9f5-16c20f457d1d/file_downloaded).
2. Save it as `Datasets/IndPenSim/raw/data.zip` (replace an invalid file).
3. Run:

```sh
bash Scripts/Prepare_Data.sh indpensim
```

Expected SHA-256: `bc434c6cf96444fee3c915d434ac84e2a9c6492be4daebcfa4aed9cdff181f00`.
On restart, an existing ZIP is checked and used without downloading it again.
The script reports that it found the ZIP and is starting preprocessing. Failed
downloads are discarded. Complete prepared data skips download and preprocessing.

## Generated and included datasets

The SMD and RLC reference data used for the dissertation are included in
`data/SMD_Dataset` and `data/4th_order_RLC` (approximately 33 MB and 1.1 MB).
Preparation checks and retains these files. The generators can recreate missing
datasets. SMD generation uses seed 42. Numerical simulation can produce small
floating-point differences between environments.
SMD loads its trajectories in the fixed order recorded in
[SMD_file_order.json](../../src/datasets/SMD_file_order.json). This preserves the
reference file order across machines and repository clones. Keep this list
unchanged when regenerating the reference dataset.
The RLC model also appears in the earlier joint work
[Knowledge-Enhanced NODEs](https://github.com/zimmer-ing/KnowledgeEnhancedNODEs).
The repository includes the fractional benchmark trajectories and forced-FDE
reference under [data/Fractional_Data](../../data/Fractional_Data). They can be
used without MATLAB. Regeneration with
[SimpleFDE_DSgen.m](SimpleFDE_DSgen.m) requires MATLAB and
[FOMCON](https://www.mathworks.com/matlabcentral/fileexchange/66323-fomcon-toolbox-for-matlab).
The benchmark uses a fixed fractional order of 0.7.

## Splits and scaling

Paths below are relative to `data/`. Scaling reproduces the historical
preprocessing for each dataset. **IndPenSim uses all 100 original batches to fit
the scaler, including validation, test and batches unused by the loader.**
Cascaded Tanks, Industrial Robot and both NanoDrone splits fit their scalers on
their selected training data. SMD, RLC and the fractional benchmark have no
additional scaling.

| Dataset | Training / validation / test | Prepared directory |
| --- | --- | --- |
| SMD | 300 / 150 / 150 simulated trajectories with no additional scaling | `SMD_Dataset/{train,val,test}/` |
| RLC | Three separately simulated time series with no additional scaling | `4th_order_RLC/{train,val,test}/` |
| IndPenSim | 31 / 6 / 6 batches used by the loader. Min-max scaling is fitted to all 100 original batches. | `IndPenSim/minmax_scaled/` |
| Cascaded Tanks | First 80% of the estimation series for training and last 20% for validation. The official validation series is used for testing. | `NonlinearBenchmarks/CascadedTanks/minmax/` |
| Industrial Robot | Last 10% of the official training series for validation. A separate official series is used for testing. | `IndustrialRobot/forward_minmax/` |
| NanoDrone 3 / OOD | Random, Square and Chirp runs 1–3 for training and run 4 for validation. Melon runs 1–3 are used for testing. | `NonlinearBenchmarks/NanoDroneOOD/minmax/` |
| NanoDrone 4 / RunSplit | Random, Square and Chirp runs 1–2 plus Melon 1 for training, run 3 plus Melon 2 for validation, and run 4 plus Melon 3 for testing | `NonlinearBenchmarks/NanoDroneRunSplit/minmax/` |
| Fractional benchmark | 40 / 20 / 20 trajectories per task with no additional scaling | `Fractional_Data/fractional_benchmark_data/{reconstruction,extrapolation,forcing}/` |

IndPenSim preparation writes all 100 scaled batches. Its scaler is fitted to all
batches to retain the preprocessing used for the dissertation results. The loader
selects the 43 batches listed above. Process columns retain the original
floating-point representation so regenerated IndPenSim CSVs match the historical
files. Each NanoDrone split has its own scaler.
NanoDrone orientations are converted from quaternions to rotation vectors before
scaling.

## Where the split is defined

For Robot, Tanks and both NanoDrone splits, the dataset classes define
`TRAIN_FILES`, `VAL_FILES` and `TEST_FILES`. Preparation and training use the same
lists. IndPenSim preparation writes all 100 batches. Its loader selects
31 training, 6 validation and 6 test batches.

| Dataset class | Preparation script |
| --- | --- |
| [IndustrialRobotForward](../../src/datasets/IndustrialRobot.py), `VAL_FRACTION = 0.10` | [build_forward_dataset.py](build_forward_dataset.py) |
| [CascadedTanks](../../src/datasets/NonlinearBenchmarks.py), `TRAIN_FRACTION = 0.8` | [prepare_nonlinearbenchmark_core_datasets.py](prepare_nonlinearbenchmark_core_datasets.py) |
| [NanoDroneOOD](../../src/datasets/NonlinearBenchmarks.py) | [prepare_nanodrone_dataset.py](prepare_nanodrone_dataset.py) |
| [NanoDroneRunSplit](../../src/datasets/NonlinearBenchmarks.py) | [prepare_nanodrone_run_split_dataset.py](prepare_nanodrone_run_split_dataset.py) |

Robot and Tanks use temporal cuts within the official training series. NanoDrone
uses complete flights. `split_files.json` records the prepared split. The loaders
select files from the class lists.

Keep the scaled CSVs, `split_files.json`, `scaler_params.json` and `metadata.json`
together for Robot, Tanks and NanoDrone. If a split changes, move the prepared
directory aside and prepare it again so that the scaler matches the new training
data. Do not scale prepared files a second time. Use the common launcher for
normal preparation because some individual scripts replace existing outputs.
