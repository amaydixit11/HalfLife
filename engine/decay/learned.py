"""
learned.py — LearnedDecay function (Option A: chunk-level λ predictor).

This class is used by the DecayRegistry like any other decay function.
The key difference from ExponentialDecay: λ was predicted by the MLP
at ingestion time (not hand-tuned), stored in Redis, and loaded here.

At query time this is pure exponential decay — no ML inference.
The MLP runs only at ingestion time (or when feedback updates trigger
a λ re-prediction via FeedbackUpdater).

The decay_params dict in Redis must contain:
    {"lambda": float}    — predicted by LearnedDecayEngine.predict_lambda()

If lambda is missing, falls back to the MLP's cold-start default for
generic doc type — equivalent to ExponentialDecay with λ=1e-6.
"""

import math
from datetime import datetime
from .base import DecayFunction


class LearnedDecay(DecayFunction):
    """
    Exponential decay using a MLP-predicted λ stored in params.

    Identical runtime behaviour to ExponentialDecay — the "learned"
    part is in how λ was set, not in how it's used.

    decay(Δt) = e^(-λ · Δt)

    where λ ∈ [1e-8, 1e-4] was predicted by DecayMLP from chunk features.
    """

    def compute(self, timestamp: datetime, now: datetime) -> float:
        delta_seconds = (now - timestamp).total_seconds()
        delta_seconds = max(0.0, delta_seconds)

        lambda_ = self.params.get("lambda", 1e-6)

        # Safety clamp — λ should always be in the MLP's output range,
        # but guard against stale Redis values or hand-edited metadata.
        lambda_ = max(1e-8, min(lambda_, 1e-4))

        return math.exp(-lambda_ * delta_seconds)