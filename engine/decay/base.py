from abc import ABC, abstractmethod
from datetime import datetime

class DecayFunction(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def compute(self, timestamp: datetime, now: datetime) -> float:
        """
        Compute the decay score for a given timestamp and the current time.
        Returns a value between 0.0 and 1.0.
        """
        pass
