from datetime import datetime
from .base import DecayFunction

class PiecewiseDecay(DecayFunction):
    """
    Step-wise decay: score is determined by fixed time-intervals.
    Good for domain documents or compliance data.
    """
    def compute(self, timestamp: datetime, now: datetime) -> float:
        delta_days = (now - timestamp).days
        # Ensure delta_days is not negative
        delta_days = max(0, delta_days)

        # Thresholds can be passed in params or use defaults
        # 1.0 (week), 0.7 (year), 0.3 (older)
        if delta_days < 7:
            return 1.0
        elif delta_days < 365:
            return 0.7
        else:
            return 0.3
