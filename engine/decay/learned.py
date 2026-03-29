import math
from datetime import datetime
from .base import DecayFunction
import numpy as np
from .learned_model import LEARNED_ENGINE

class LearnedDecay(DecayFunction):
    """
    Learned decay: score = f(delta_time, vector_score)
    This implementation uses a 2-layer MLP to reconciliate 
    recency with semantic quality.
    """
    def compute(self, timestamp: datetime, now: datetime) -> float:
        delta_seconds = (now - timestamp).total_seconds()
        delta_seconds = max(0, delta_seconds)
        
        # We need the original vector score to make a neural decision
        # Since the base API only passes (timestamp, now), we'll use 
        # a default placeholder or assume it's provided in params.
        vector_score = self.params.get("vector_score", 0.7)
        
        return LEARNED_ENGINE.get_score(delta_seconds, vector_score)
