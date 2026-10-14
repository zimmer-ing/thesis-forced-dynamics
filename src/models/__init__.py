from .LSTM_Seq2Seq import LSTMSeq2Seq
from .GRU_Seq2Seq import GRUSeq2Seq
from .LaplaceNet import LaplaceNet as LaplaceNet
from .TCN_EncDec import TCNEncoderDecoder
from .PhysicsOnlyNODE import PhysicsOnlyAdaptiveNODE
from .PhysicsOnlyNODE import PhysicsOnlyFixedStepNODE
from .FC_Simple import FCSimple
from .NO_Obs import NODEfixedStepModel as NOObsFixedModel
from .NO_Obs import NODEadaptiveModel as NOObsAdaptiveModel
from .NO_Prior import NODEadaptiveModel as NOPriorAdaptiveModel
from .NO_Prior import NODEfixedStepModel as NOPriorFixedModel
from .NO_Aug import NODEadaptiveModel as NOAugAdaptiveModel
from .NO_Aug import NODEfixedStepModel as NOAugFixedModel
from .NO_Lat import NODEadaptiveModel as NOLatAdaptiveModel
from .NO_Lat import NODEfixedStepModel as NOLatFixedModel
from .LSTM_init_hist import LSTMMarkov as ILSTM
from .NeuralFDE import NFDEFixedAlphaModel, NFDEModel
from src.utils.model_names import MODEL_ALIASES

# Registry dictionary mapping model names to classes
MODEL_REGISTRY = {
    "LSTM": LSTMSeq2Seq,
    "GRU": GRUSeq2Seq,
    "LaplaceNet": LaplaceNet,
    "TCN": TCNEncoderDecoder,
    "FC": FCSimple,
    "FC-dt": FCSimple,
    "NO-Obs": NOObsAdaptiveModel,
    "NO-ObsFixed": NOObsFixedModel,
    "NO-Prior": NOPriorAdaptiveModel,
    "NO-PriorFixed": NOPriorFixedModel,
    "NO-Aug": NOAugAdaptiveModel,
    "NO-Aug0": NOAugAdaptiveModel,
    "NO-AugH": NOAugAdaptiveModel,
    "NO-AugFixed": NOAugFixedModel,
    "NO-Lat": NOLatAdaptiveModel,
    "NO-LatFixed": NOLatFixedModel,
    "PhysicsOnlyGreyboxAdaptiveNODE": PhysicsOnlyAdaptiveNODE,
    "PhysicsOnlyWhiteboxAdaptiveNODE": PhysicsOnlyAdaptiveNODE,
    "PhysicsOnlyGreyboxFixedNODE": PhysicsOnlyFixedStepNODE,
    "PhysicsOnlyWhiteboxFixedNODE": PhysicsOnlyFixedStepNODE,
    "I-LSTM": ILSTM,
    "NFDE-learned": NFDEModel,
    "NFDE": NFDEFixedAlphaModel,
}

MODEL_REGISTRY.update({alias: MODEL_REGISTRY[canonical] for alias, canonical in MODEL_ALIASES.items()})
MODEL_REGISTRY.update({
    "LSTMSeq2SeqResponseOnly": LSTMSeq2Seq,
    "GRUSeq2SeqResponseOnly": GRUSeq2Seq,
    "FCSimple_tResponseOnly": FCSimple,
})

__all__ = [
    "MODEL_REGISTRY", "MODEL_ALIASES", "LSTMSeq2Seq", "GRUSeq2Seq", "LaplaceNet",
    "TCNEncoderDecoder", "FCSimple", "ILSTM", "NFDEModel", "NFDEFixedAlphaModel",
    "PhysicsOnlyAdaptiveNODE", "PhysicsOnlyFixedStepNODE", "NOObsAdaptiveModel", "NOObsFixedModel",
    "NOPriorAdaptiveModel", "NOPriorFixedModel", "NOAugAdaptiveModel", "NOAugFixedModel",
    "NOLatAdaptiveModel", "NOLatFixedModel",
]
