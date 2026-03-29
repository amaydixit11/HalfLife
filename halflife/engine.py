import os
import sys
from typing import List, Dict, Optional

# Root path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.fusion.reranker import Reranker
from engine.classifier.query_intent import QueryIntentClassifier
from engine.store.redis_store import RedisStore
from engine.ingestion.pipeline import HalfLifeIngestor

class HalfLife:
    """
    The primary Python SDK for HalfLife.
    Use this to integrate temporal reranking into existing RAG pipelines.
    
    Example:
        >>> from halflife import HalfLife
        >>> hl = HalfLife()
        >>> reranked = hl.rerank(query="latest GNNs", chunks=qdrant_results)
    """

    def __init__(self, qdrant_url="http://localhost:6333", redis_url="redis://localhost:6379"):
        self.store = RedisStore(url=redis_url)
        self.reranker = Reranker(self.store)
        self.classifier = QueryIntentClassifier()
        self.ingestor = HalfLifeIngestor(qdrant_url=qdrant_url, redis_url=redis_url)

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5, intent: Optional[str] = None) -> List[Dict]:
        """
        Reranks a list of chunks based on query intent and temporal decay.
        If 'intent' is provided, it overrides the automatic classifier.
        """
        if intent:
            if intent not in self.classifier.intent_weights:
                raise ValueError(f"Invalid intent: '{intent}'. Must be one of: {list(self.classifier.intent_weights.keys())}")
            weights = self.classifier.intent_weights[intent]
            classification = {"intent": intent, "weights": weights}
        else:
            # Use automatic classification
            classification = self.classifier.classify(query)

        result = self.reranker.rerank(
            query=query,
            chunks=chunks,
            intent=classification["intent"],
            weights=classification["weights"],
            top_k=top_k
        )
        return result["reranked_chunks"]

    def ingest(self, text: str, timestamp: str, doc_type: str = "generic"):
        """
        Ingests a document with its temporal metadata.
        """
        from datetime import datetime
        # Simple date parser support
        try:
            ts = datetime.fromisoformat(timestamp)
        except:
            # Fallback for plain years
            from datetime import timezone
            ts = datetime(int(timestamp), 1, 1, tzinfo=timezone.utc)
            
        return self.ingestor.ingest(text=text, timestamp=ts, doc_type=doc_type)
