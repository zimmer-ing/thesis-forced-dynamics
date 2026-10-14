# Project overview

This repository contains the software accompanying the dissertation
*Data-Driven Modelling of Forced Dynamical Systems with Neural Differential
Models and Laplace-Net*.
Section and appendix references below refer to the dissertation.

The dissertation investigates open-loop forecasting from observed system
history and known future forcing (Section 4.2). It compares model structure,
physical prior knowledge, fractional dynamics and Laplace-domain response modelling.
The evaluation considers forecasting accuracy, reduced training data and
inference time.

## Models

- **NODE:** Models with observed, augmented or latent states. The RLC case study
  also includes a model with physical prior knowledge (Sections 5.1.2–5.1.3 and Appendix C).
- **NFDE:** Fractional differential models for reconstruction, extrapolation
  and modified forcing (Section 5.2).
- **Laplace-Net:** A model that uses the observed history to describe local
  dynamics and separates these from future forcing in the Laplace domain
  (Sections 6.2–6.4 and Appendix A).
- **Discrete baselines:** FC, LSTM, GRU, TCN and I-LSTM
  (Sections 8.3 and 8.5.3).

See the [model guide](src/models/README.md) for names and configurations.

## Workflow

Run commands from the repository root. Settings such as datasets, models and
seeds are defined at the top of the corresponding Python scripts.

The workflow for new experiments follows these stages:

```text
Prepare data -> HPO and seed 42 -> further seeds -> data fractions and measurements -> tables and figures
```

The data-fraction runs reuse the selected hyperparameters. Runtime and component
measurements use the trained checkpoints. The Fractional benchmark has its own
HPO and training stages. To start with one model and existing hyperparameters,
see the [LaplaceNet example](Scripts/Training/README.md#example-laplacenet-on-smd).

### 1. Set up the environment

Use Mamba from [Miniforge3](https://github.com/conda-forge/miniforge), with
packages from conda-forge. See [installation](environments/README.md#install-mamba)
if Mamba is not installed yet.

Linux x86-64 with an NVIDIA GPU:

```sh
mamba create --name thesis-gpu --file environments/locks/thesis-linux-64-cuda.explicit.txt \
  --override-channels --channel conda-forge
mamba activate thesis-gpu
```

Apple Silicon Mac with CPU:

```sh
mamba create --name thesis-cpu --file environments/locks/thesis-macos-arm64.explicit.txt \
  --override-channels --channel conda-forge
mamba activate thesis-cpu
```

See [environment setup](environments/README.md) for platform requirements.

### 2. Prepare the datasets

```sh
bash Scripts/Prepare_Data.sh all
```

External data are downloaded and preprocessed. SMD, RLC and Fractional
reference data are supplied with the repository
(Section 8.2 and Appendix B.1).
See [data preparation](Scripts/DataPreparation/README.md) for sources,
splits and scaling.

### 3. Search hyperparameters and train multiple seeds

```sh
bash Scripts/Train.sh main
```

For each dataset/model pair, the script first performs hyperparameter
optimisation (HPO) and final training for Seed 42. It then trains Seeds 1–9
with the selected hyperparameters. These additional seeds run across the
visible GPUs when available. The training setup is described in Section 8.4,
with the full-data comparison in Section 8.5.1 and hyperparameters in Appendix D.2.
See [training](Scripts/Training/README.md).

Training can also use existing hyperparameters without a new search. See
[training with existing hyperparameters](Scripts/Training/README.md#train-with-existing-hyperparameters).

### 4. Train with reduced data

```sh
bash Scripts/Train.sh data-fraction
```

The selected models are retrained with different training-data fractions,
using the hyperparameters from step 3 (Section 8.5.2).

### 5. Run the further experiments

| Command | Purpose | Dissertation |
| --- | --- | --- |
| `bash Scripts/Train.sh fractional` | Fractional benchmark. Configure the reference models as described in the [training guide](Scripts/Training/README.md#fractional-dynamics). | Section 8.5.3 |
| `bash Scripts/Train.sh runtime` | Inference-time measurements from trained checkpoints. These measurements require CUDA. | Section 8.5.5 |
| `python -m Scripts.Evaluation.Supplementary.Evaluate_Physics` | RLC physical and neural component comparison. | Section 8.5.4 |

The [evaluation guide](Scripts/Evaluation/README.md) also describes the
persistence reference, numerical examples and solver comparisons.

### 6. Generate tables and figures

Choose `EVALUATION_SOURCE = 'thesis_results'` or `'new_runs'` in
[Constants.py](Constants.py), then run:

```sh
bash Scripts/Evaluate.sh all
```

The evaluation reads saved results and writes tables and figures to
`output/pdf/Evaluation/<source>/`. Individual research questions and the
appendix can also be evaluated separately (Sections 8.5.1–8.5.5 and Appendix D).

New experiments write to `results/new_runs/`. The supplied dissertation
results are stored under `results/thesis_results/`. See [results](results/README.md)
and [evaluation](Scripts/Evaluation/README.md) for details.

## Code structure

Experiments combine a dataset, a model and their configurations. The base
classes provide shared data handling and training methods. See
[code structure and extensions](docs/Code_structure.md) for the class diagram,
configuration flow, and how to add datasets, models or model variants.

Small [interface tests](tests/README.md) help check new models and datasets.

## Citation

If you use this software, please cite this repository using
[CITATION.cff](CITATION.cff). Also cite the publications relevant to the models
you use:

| Method or component | Publication |
| --- | --- |
| Laplace-Net framework | Zimmering et al. (2026), [Breaking Free: Decoupling Forced Systems with Laplace Neural Networks](https://doi.org/10.1007/978-3-032-06109-6_15). |
| RLC model with physical prior knowledge, derivative scaling and pretraining | Zimmering, Roche and Niggemann (2024), [Enhancing Nonlinear Electrical Circuit Modeling with Prior Knowledge-Infused Neural ODEs](https://doi.org/10.1109/ETFA61755.2024.10711112). |
| Optimised FDEint solver and NFDE comparison | Zimmering, Coelho and Niggemann (2024), [Optimising Neural Fractional Differential Equations for Performance and Efficiency](https://proceedings.mlr.press/v255/zimmering24a.html). |

The dissertation builds on these methods and evaluates them on the forecasting
tasks described above. In particular, it extends FDEint to known forcing.
The [model guide](src/models/README.md) lists the underlying work for individual
model variants.

The dissertation has been submitted and is not yet published. Its bibliographic
reference will be added after publication.

## License

Original code and documentation are licensed under the [MIT License](LICENSE).
Third-party components retain the terms listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with complete texts in
[licenses/](licenses/). Dataset permissions are separate from the software license.
See the [data preparation guide](Scripts/DataPreparation/README.md) for sources
and terms.
