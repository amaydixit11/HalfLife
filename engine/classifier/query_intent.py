import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

class QueryIntentClassifier:
    """
    Classifies the temporal intent of a query and returns fusion weights.
    """

    FRESH_KEYWORDS = {
        "latest", "recent", "newest", "current", "now",
        "today", "this week", "this month", "breaking", "just",
        "updated", "new", "2024", "2025",
    }

    HISTORICAL_KEYWORDS = {
        "history", "historical", "evolution", "evolved", "origins",
        "background", "originally", "used to", "how did", "first version",
        "early", "founded", "invented", "introduced", "over the years",
        "timeline", "progression", "was", "were", "previous", "past",
        "before", "older", "archive", "ancient",
    }

    intent_weights = {
        "fresh":      {"vector": 0.3, "temporal": 0.6, "trust": 0.1},
        "historical": {"vector": 0.4, "temporal": 0.5, "trust": 0.1},
        "static":     {"vector": 0.8, "temporal": 0.1, "trust": 0.1},
    }

    def classify(self, query: str) -> Dict:
        """
        Returns:
            {
                "intent":  "fresh" | "historical" | "static",
                "weights": {"vector": float, "temporal": float, "trust": float},
            }
        """
        q = query.lower()
        
        # 1. Date Detection: If user specifies a year (1900-2023) -> Historical
        years = re.findall(r"\b(19\d{2}|20[0-1]\d|202[0-3])\b", q)
        if years:
            return {
                "intent": "historical",
                "weights": self.intent_weights["historical"],
            }

        # 2. Keyword Detection
        if any(kw in q for kw in self.FRESH_KEYWORDS):
            return {
                "intent": "fresh",
                "weights": self.intent_weights["fresh"],
            }

        if any(kw in q for kw in self.HISTORICAL_KEYWORDS):
            return {
                "intent": "historical",
                "weights": self.intent_weights["historical"],
            }

        # Default: static / time-agnostic
        return {
            "intent": "static",
            "weights": self.intent_weights["static"],
        }
