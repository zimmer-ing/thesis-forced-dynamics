MODEL_ALIASES = {
    "M1AdaptiveNeuralODE": "NO-Obs",
    "M2AdaptiveNeuralODE": "NO-Prior",
    "M3Z": "NO-Aug0",
    "M3ZAdaptiveNeuralODE": "NO-Aug0",
    "M3H": "NO-AugH",
    "M3HAdaptiveNeuralODE": "NO-AugH",
    "M3AdaptiveNeuralODE": "NO-Aug",
    "M4AdaptiveNeuralODE": "NO-Lat",
    "LaplaceNetFull": "LaplaceNet",
    "FCSimple_t": "FC",
    "FCSimple": "FC-dt",
    "LSTMSeq2Seq": "LSTM",
    "GRUSeq2Seq": "GRU",
    "TCNEncoderDecoder": "TCN",
    "LSTMMarkov": "I-LSTM",
    "NeuralFDEFixedAlpha": "NFDE",
    "NeuralFDE": "NFDE-learned",
    "M1FixedNeuralODE": "NO-ObsFixed",
    "M2FixedNeuralODE": "NO-PriorFixed",
    "M3FixedNeuralODE": "NO-AugFixed",
    "M4FixedNeuralODE": "NO-LatFixed",
}

VARIANT_ALIASES = {
    "M1_no_time": "NO-Obs",
    "M1_with_time": "NO-Obs-t",
    "M3_zero_no_time": "NO-Aug0",
    "M3_zero_with_time": "NO-Aug0-t",
    "NFDE_fixed_no_time": "NFDE",
    "NFDE_fixed_with_time": "NFDE-t",
    "LSTMMarkov": "I-LSTM",
    "FCSimple": "FC-dt",
    "TCNEncoderDecoder": "TCN",
}


def model_key(name):
    return MODEL_ALIASES.get(name, name)


def model_names(name):
    canonical = model_key(name)
    return [canonical] + [alias for alias, target in MODEL_ALIASES.items() if target == canonical]


def variant_key(name):
    return VARIANT_ALIASES.get(name, name)


def variant_names(name):
    canonical = variant_key(name)
    return [canonical] + [alias for alias, target in VARIANT_ALIASES.items() if target == canonical]
