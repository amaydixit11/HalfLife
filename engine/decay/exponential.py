import math
from datetime import datetime
from .base import DecayFunction

class ExponentialDecay(DecayFunction):
    """
    Exponential decay: score = e^(-lambda * delta_time)
    Good for news and fast-moving trends.
    """
    def compute(self, timestamp: datetime, now: datetime) -> float:
        delta_seconds = (now - timestamp).total_seconds()
        # Ensure delta_seconds is not negative (e.g., if there's a minor clock drift)
        delta_seconds = max(0, delta_seconds)
        
        # lambda_ is the decay constant. Default: 1e-6 (roughly half-life of 8 days)
        lambda_ = self.params.get("lambda", 1e-6)

        return math.exp(-lambda_ * delta_seconds)
