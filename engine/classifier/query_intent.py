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
        "timeline", "progression",
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
            # NOTE: weights here look similar to static, but the reranker
            # receives intent="historical" and inverts temporal_score.
            # The result: temporal weight still matters, but it now rewards
            # old chunks instead of fresh ones.
            return {
                "intent": "historical",
                "weights": {
                    "vector":   0.5,
                    "temporal": 0.3,
                    "trust":    0.2,
                },
            }

        # Default: static / time-agnostic
        return {
            "intent": "static",
            "weights": {
                "vector":   0.7,
                "temporal": 0.1,
                "trust":    0.2,
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
