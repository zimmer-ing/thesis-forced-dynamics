"""Model names, result-directory compatibility and plot labels."""

from pathlib import Path

from src.utils.model_names import (
    MODEL_ALIASES, VARIANT_ALIASES, model_key, model_names, variant_key, variant_names,
)

DATASET_LABELS = {'IndPenSim': 'IndPenSim',
 'RLCDataset': 'RLC',
 'SMDSystem': 'SMD',
 'IndustrialRobotForward': 'Industrial Robot',
 'CascadedTanks': 'Cascaded Tanks',
 'NanoDroneOOD': 'NanoDrone 3',
 'NanoDroneRunSplit': 'NanoDrone 4',
 'FractionalBenchmarkExtrapolation': 'Extrapolation',
 'FractionalBenchmarkForcing': 'Forcing',
 'FractionalBenchmarkReconstruction': 'Reconstruction'}

DATASET_SLUGS = {'SMDSystem': 'smd',
 'RLCDataset': 'rlc',
 'IndPenSim': 'indpensim',
 'IndustrialRobotForward': 'industrial_robot',
 'CascadedTanks': 'cascaded_tanks',
 'NanoDroneOOD': 'nanodrone_ood',
 'NanoDroneRunSplit': 'nanodrone_runsplit'}

MODEL_LABELS = {'PersistenceBaseline': 'Persistence',
 'FC': 'FC',
 'FC-dt': 'FC',
 'LSTM': 'LSTM',
 'GRU': 'GRU',
 'TCN': 'TCN',
 'I-LSTM': 'I-LSTM',
 'LaplaceNet': 'Laplace-Net',
 'NO-Obs': 'NO-Obs',
 'NO-Prior': 'NO-Prior',
 'NO-Aug0': 'NO-Aug0',
 'NO-AugH': 'NO-AugH',
 'NO-Aug': 'NO-Aug',
 'NO-Lat': 'NO-Lat',
 'NFDE': 'NFDE',
 'NFDE-t': 'NFDE-t',
 'NO-Obs-t': 'NO-Obs-t',
 'NO-Aug0-t': 'NO-Aug0-t',
 'NFDE_no_time': 'NFDE ($\\alpha$ learned)',
 'NFDE_with_time': 'NFDE-t ($\\alpha$ learned)',
 'NFDE_learned_no_time': 'NFDE ($\\alpha$ learned)',
 'NFDE_learned_with_time': 'NFDE-t ($\\alpha$ learned)'}

MODEL_TABLE_LABELS = dict(MODEL_LABELS)
MODEL_SHORT_LABELS = dict(MODEL_LABELS)

HYPERPARAMETER_LABELS = {'activation': 'Activation',
 'activation_decoder': '$\\varrho_{\\mathrm{dec}}$',
 'activation_ODE': '$\\varrho_f$',
 'augmented_dim_ODE': '$D_{\\mathrm{aug}}$',
 'decoder_hidden_size': '$n_{\\mathrm{dec}}$',
 'dropout': 'Dropout',
 'encoder_hidden_size': '$n_{\\mathrm{enc}}$',
 'hidden_layers': '$L_h$',
 'hidden_size': '$n_h$',
 'hidden_size_decoder': '$n_{\\mathrm{dec}}$',
 'hidden_size_encoder': '$n_{\\mathrm{enc}}$',
 'hidden_size_ODE': '$n_f$',
 'hidden_width': '$n_h$',
 'include_time': '$t$ input',
 'is_order_dim': '$N_P$',
 'is_RNN_hidden_dim': '$n_{\\mathrm{IS}}$',
 'is_RNN_num_layers': '$L_{\\mathrm{IS}}$',
 'latent_dim': '$d_z$',
 'latent_dim_ODE': '$D_z$',
 'laplace_scaling_factor': '$\\kappa_{\\mathrm{LP}}$',
 'layernorm': 'Layer norm',
 'learning_rate': '$\\eta$',
 'LP_trafo_type': '$\\mathscr{L}$ type',
 'ilt_alpha': '$\\alpha_{\\mathrm{ILT}}$',
 'ilt_scale': '$\\zeta$',
 'ilt_shift': '$\\tau_{\\mathrm{LP}}$',
 'ilt_terms': '$N_{\\mathrm{ILT}}$',
 'n_layers_hidden_decoder': '$L_{\\mathrm{dec}}$',
 'n_layers_hidden_ODE': '$L_f$',
 'num_layers': '$L$',
 'num_layers_encoder': '$L_{\\mathrm{enc}}$',
 'pretraining_epochs': '$E_{\\mathrm{pre}}$',
 'pretraining_lr': '$\\eta_{\\mathrm{pre}}$',
 'steps': '$N_{\\mathrm{steps}}$',
 'dlt_mode': 'DLT mode',
 'fflt_mode': 'FFLT mode',
 'transform_chunk_size': '$N_{\\mathrm{chunk}}$',
 'tf_activation': '$\\varrho_{\\mathrm{TF}}$',
 'tf_hidden_dim': '$n_{\\mathrm{TF}}$',
 'tf_num_hidden_layers': '$L_{\\mathrm{TF}}$'}

MODEL_FAMILIES = {'FC': 'discrete_baseline',
 'FC-dt': 'discrete_baseline',
 'LSTM': 'discrete_baseline',
 'GRU': 'discrete_baseline',
 'TCN': 'discrete_baseline',
 'I-LSTM': 'discrete_baseline',
 'PersistenceBaseline': 'naive_baseline',
 'LaplaceNet': 'laplace',
 'NO-Obs': 'node',
 'NO-Prior': 'node_prior',
 'NO-Aug0': 'node',
 'NO-AugH': 'node',
 'NO-Aug': 'node',
 'NO-Lat': 'node',
 'NO-Obs-t': 'node',
 'NO-Aug0-t': 'node',
 'NFDE': 'nfde',
 'NFDE-t': 'nfde',
 'NFDE_no_time': 'nfde',
 'NFDE_with_time': 'nfde',
 'NFDE_learned_no_time': 'nfde',
 'NFDE_learned_with_time': 'nfde'}

MODEL_COLORS = {'PersistenceBaseline': '#6f6f6f',
 'FC': '#4d4d4d',
 'FC-dt': '#4d4d4d',
 'LSTM': '#d62728',
 'GRU': '#8c6d31',
 'TCN': '#8c564b',
 'I-LSTM': '#e377c2',
 'LaplaceNet': '#56b4e9',
 'NO-Obs': '#ff7f0e',
 'NO-Prior': '#2ca02c',
 'NO-Aug0': '#17becf',
 'NO-AugH': '#9467bd',
 'NO-Aug': '#17becf',
 'NO-Lat': '#2ca02c',
 'NO-Obs-t': '#009E73',
 'NO-Aug0-t': '#E69F00',
 'NFDE': '#56B4E9',
 'NFDE-t': '#0072B2',
 'NFDE_no_time': '#56B4E9',
 'NFDE_with_time': '#0072B2',
 'NFDE_learned_no_time': '#56B4E9',
 'NFDE_learned_with_time': '#0072B2'}

MODEL_MARKERS = {'PersistenceBaseline': 'o',
 'FC': 's',
 'FC-dt': 's',
 'LSTM': '^',
 'GRU': 'v',
 'TCN': 'D',
 'I-LSTM': '>',
 'LaplaceNet': 'D',
 'NO-Obs': 'P',
 'NO-Prior': 'd',
 'NO-Aug0': 'X',
 'NO-AugH': '*',
 'NO-Aug': 'X',
 'NO-Lat': 'h',
 'NO-Obs-t': 'P',
 'NO-Aug0-t': 's',
 'NFDE': 's',
 'NFDE-t': 's',
 'NFDE_no_time': 'o',
 'NFDE_with_time': 'o',
 'NFDE_learned_no_time': 'o',
 'NFDE_learned_with_time': 'o'}

for alias, canonical in {**MODEL_ALIASES, **VARIANT_ALIASES}.items():
    for mapping in [MODEL_LABELS, MODEL_TABLE_LABELS, MODEL_SHORT_LABELS, MODEL_FAMILIES, MODEL_COLORS, MODEL_MARKERS]:
        if canonical in mapping:
            mapping[alias] = mapping[canonical]

LAPLACE_ODE_MODELS = ['LaplaceNet', 'NO-Obs', 'NO-Aug0', 'NO-AugH', 'NO-Lat']


def model_directory(root, dataset, model):
    parent = Path(root) / dataset
    for name in model_names(model):
        path = parent / name
        if path.is_dir():
            if name == 'LaplaceNet':
                snapshot = path / 'Seed_42/model_config.yaml'
                if snapshot.is_file():
                    import yaml
                    config = yaml.safe_load(snapshot.read_text())
                    if (config.get('model') or {}).get('response_only_history', True) is True:
                        continue
            return path
    if model_key(model) == 'LaplaceNet' and (parent / 'LaplaceNet').is_dir():
        raise ValueError(f'Only response-history Laplace results found in {parent}')
    return parent / model_key(model)


def variant_directory(root, dataset, variant):
    parent = Path(root) / dataset
    for name in variant_names(variant):
        if (parent / name).is_dir():
            return parent / name
    return parent / variant_key(variant)


def variant_result_name(root, dataset, variant):
    parent = Path(root) / dataset
    for name in variant_names(variant):
        if (parent / name).is_dir() or next(parent.glob(f'{name}_seed*.json'), None) is not None:
            return name
    return variant_key(variant)


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def model_label(model):
    return MODEL_LABELS.get(model, model)


def model_table_label(model):
    return MODEL_TABLE_LABELS.get(model, model_label(model))


def hyperparameter_label(parameter):
    return HYPERPARAMETER_LABELS.get(parameter, parameter)


def model_short_label(model):
    return MODEL_SHORT_LABELS.get(model, model_label(model))


def model_family(model):
    return MODEL_FAMILIES.get(model, 'other')
