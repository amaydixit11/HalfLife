import math
from datetime import datetime
from .base import DecayFunction

class ExponentialDecay(DecayFunction):
    """
    Exponential decay: score = e^(-lambda * delta_time)
    Good for news and fast-moving trends.
    """
    def __init__(self, params: dict):
        self.lambda_val = params.get("lambda", 1e-8)

    def compute(self, timestamp: datetime, now: datetime) -> float:
        delta_seconds = (now - timestamp).total_seconds()
        # Ensure delta_seconds is not negative (e.g., if there's a minor clock drift)
        delta_seconds = max(0, delta_seconds)
        
        return math.exp(-self.lambda_val * delta_seconds)
