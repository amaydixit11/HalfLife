import logging
from typing import Dict

logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """
    Classifies the temporal intent of a query and returns fusion weights.

    Intent categories:
        fresh      — user wants current information ("latest", "recent")
                     β (temporal) is high, α (vector) is lower
        historical — user wants evolution or past state ("history of", "how did X evolve")
                     β is kept moderate but the reranker INVERTS temporal_score
                     so that older chunks rank higher (see reranker.py)
        static     — time-agnostic ("what is", "define", "explain")
                     α (vector) dominates, temporal signal is minimal

    The reranker consumes both 'weights' and 'intent' from this output.
    Weights alone are not enough for historical queries — the inversion
    flag is what actually surfaces old content.
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

    def classify(self, query: str) -> Dict:
        """
        Returns:
            {
                "intent":  "fresh" | "historical" | "static",
                "weights": {"vector": float, "temporal": float, "trust": float},
            }
        """
        q = query.lower()
        import re
        
        # 1. Date Detection: If user specifies a year in the past -> Historical
        years = re.findall(r"\b(19\d{2}|20[0-1]\d|202[0-3])\b", q)
        if years:
            return {
                "intent": "historical",
                "weights": {
                    "vector":   0.4,
                    "temporal": 0.5, # High weight for inversion
                    "trust":    0.1,
                },
            }

        # 2. Keyword Detection
        if any(kw in q for kw in self.FRESH_KEYWORDS):
            return {
                "intent": "fresh",
                "weights": {
                    "vector":   0.3,
                    "temporal": 0.6,
                    "trust":    0.1,
                },
            }

        if any(kw in q for kw in self.HISTORICAL_KEYWORDS):
            return {
                "intent": "historical",
                "weights": {
                    "vector":   0.4,
                    "temporal": 0.5,
                    "trust":    0.1,
                },
            }

        # Default: static / time-agnostic
        return {
            "intent": "static",
            "weights": {
                "vector":   0.8,
                "temporal": 0.1,
                "trust":    0.1,
            },
        }


if __name__ == "__main__":
    clf = QueryIntentClassifier()
    for q in [
        "latest BERT papers",
        "history of transformer architectures",
        "what is attention mechanism",
    ]:
        result = clf.classify(q)
        print(f"{q!r:45s} → intent={result['intent']}, weights={result['weights']}")
