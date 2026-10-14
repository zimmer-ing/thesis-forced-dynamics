# Models

[Project overview](../../README.md) · [Training](../../Scripts/Training/README.md)

The models forecast the system response over a finite window using observed
history and known future forcing (Section 4.2). Each model configuration is linked
below. Section and appendix references refer to the dissertation.

## Laplace-Net

[LaplaceNet](../../configs/models/LaplaceNet.yaml) separates the prediction into
a history-dependent initial-condition term and a dynamic response operator
applied to future forcing. The operator represents locally linear dynamics
within a forecast window and is conditioned on the observed history. Numerical
Laplace transformations reconstruct the response without an ODE solver.
The formulation is described in Sections 6.2–6.4 and Appendix A. The individual
components are described in Sections 6.3.1–6.3.6.

## Neural ordinary differential equations

| Model | State and initialisation | Dissertation |
| --- | --- | --- |
| [NO-Obs](../../configs/models/NO-Obs.yaml) | Observed response variables, initialised from the last observation. | Section 5.1.2, Eq. (5.5) |
| [NO-Aug0](../../configs/models/NO-Aug0.yaml) | Observed variables and additional coordinates initialised with zeros. | Section 5.1.2, Eq. (5.9) |
| [NO-AugH](../../configs/models/NO-AugH.yaml) | Observed variables and additional coordinates initialised from the input/output history. | Section 5.1.2, Eq. (5.10) |
| [NO-Lat](../../configs/models/NO-Lat.yaml) | History encoded into a latent state. A decoder maps the evolving state to the response. | Section 5.1.2, Eq. (5.11) |
| [NO-Prior](../../configs/models/NO-Prior.yaml) | RLC physical equations combined with a learned correction to the dynamics. | Section 5.1.3, Eq. (5.13) |

NO-Prior is used for the RLC physical-prior comparison. The Fractional benchmark
uses [NO-Aug](../../configs/models/NO-Aug.yaml) as the configuration for its
zero-augmented NODE reference variants.
Implementation details are given in Appendix C. The RLC comparison is presented
in Section 8.5.4.

## Neural fractional differential equations

[NFDE](../../configs/models/NFDE.yaml) uses a fractional differential equation
with fixed order `alpha = 0.7`. `NFDE-t` additionally supplies time to the learned
right-hand side. Both variants use the same implementation and configuration.
The variant is selected in [Train_NFDE.py](../../Scripts/Training/Train_NFDE.py).
The Fractional NODE references also have variants with a time input.
The forced NFDE formulation and solver are described in Sections 5.2.2–5.2.3.
Section 8.5.3 describes the fixed-order and time-input variants used for evaluation.

## Discrete baselines

| Model | Prediction structure | Dissertation |
| --- | --- | --- |
| [FC](../../configs/models/FC.yaml), [FC-dt](../../configs/models/FC-dt.yaml) | Feed-forward prediction of the response window from history and future forcing. FC uses relative time. FC-dt uses time-step differences. | Sections 8.3 and 8.5.3 |
| [LSTM](../../configs/models/LSTM.yaml), [GRU](../../configs/models/GRU.yaml) | A recurrent encoder represents the history. The decoder uses this representation and future forcing to predict the response. | Section 8.3 |
| [TCN](../../configs/models/TCN.yaml) | Temporal convolutional encoder–decoder for history and future forcing. | Section 8.3 |
| [I-LSTM](../../configs/models/I-LSTM.yaml) | LSTM initialised from the last historical input, response and time. | Section 8.5.3 |

FC, LSTM, GRU and TCN are used in the main comparison. I-LSTM and FC-dt are
additional references for the Fractional benchmark. Model and dataset lists
are selected in the [training scripts](../../Scripts/Training/README.md).
