from .exponential import ExponentialDecay
from .piecewise import PiecewiseDecay
from .learned import LearnedDecay

DECAY_REGISTRY = {
    "exponential": ExponentialDecay,
    "piecewise": PiecewiseDecay,
    "learned": LearnedDecay,
}

def get_decay(name: str, params: dict):
    if name not in DECAY_REGISTRY:
        # Default to exponential
        name = "exponential"

    return DECAY_REGISTRY[name](params)
