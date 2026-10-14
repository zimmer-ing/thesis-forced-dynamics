from .RLC_physics import RLCCircuitWhiteboxPhysics, RLCCircuitGreyboxPhysics


PHYSICS_MODEL_REGISTRY = {
    "RLCDataset_Whitebox": RLCCircuitWhiteboxPhysics,
    "RLCDataset_Greybox": RLCCircuitGreyboxPhysics,
}

__all__ = list(PHYSICS_MODEL_REGISTRY.keys())
