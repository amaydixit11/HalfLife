import os
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field

from engine.store.redis_store import RedisStore
from engine.fusion.reranker import Reranker
from engine.classifier.query_intent import QueryIntentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HalfLife Re-ranking API", version="0.2.0")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
store      = RedisStore(url=REDIS_URL)
reranker   = Reranker(store)
classifier = QueryIntentClassifier()


# ------------------------------------------------------------------ #
#  Request / response models                                          #
# ------------------------------------------------------------------ #

class ChunkInput(BaseModel):
    id:      str
    score:   float = Field(..., ge=0.0, le=1.0)
    payload: Dict  = Field(default_factory=dict,
                           description="Qdrant payload — must include 'timestamp'")

class RerankRequest(BaseModel):
    query:   str
    chunks:  List[ChunkInput]
    top_k:   int             = Field(10, ge=1, le=100)
    weights: Optional[Dict]  = None   # override auto-weights from classifier

class MetadataIngestRequest(BaseModel):
    chunk_id:     str
    decay_type:   str   = "exponential"
    decay_params: Dict  = Field(default_factory=lambda: {"lambda": 1e-6})
    trust_score:  float = Field(0.5, ge=0.0, le=1.0)

class FeedbackRequest(BaseModel):
    chunk_id:   str
    was_useful: bool


# ------------------------------------------------------------------ #
#  Endpoints                                                          #
# ------------------------------------------------------------------ #

@app.get("/health")
def health_check():
    redis_ok = False
    try:
        redis_ok = store.client.ping() if store.client else False
    except Exception:
        pass
    return {"status": "ok", "redis": redis_ok}


@app.post("/rerank")
def rerank_endpoint(req: RerankRequest):
    """
    Main middleware endpoint.
    """
    try:
        # Classify query intent → weights + intent label
        classification = classifier.classify(req.query)
        weights = req.weights or classification["weights"]
        intent  = classification["intent"]

        chunks_as_dicts = [c.model_dump() for c in req.chunks]

        result = reranker.rerank(
            query=req.query,
            chunks=chunks_as_dicts,
            top_k=req.top_k,
            weights=weights,
            intent=intent,
        )

        return {
            **result,
            "query_intent": intent,
        }

    except Exception as e:
        logger.exception("Rerank failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/metadata")
def ingest_metadata(req: MetadataIngestRequest):
    """
    Directly write Redis metadata for a chunk.
    """
    metadata = RedisStore.build_metadata(
        chunk_id=req.chunk_id,
        decay_type=req.decay_type,
        decay_params=req.decay_params,
        trust_score=req.trust_score,
    )
    store.set_chunk(req.chunk_id, metadata)
    store.mark_dirty(req.chunk_id)
    return {"status": "ingested", "chunk_id": req.chunk_id}


@app.post("/feedback")
def feedback_endpoint(req: FeedbackRequest):
    """
    Log chunk utility signal. Marks cache dirty.
    """
    from engine.feedback.updater import FeedbackUpdater
    updater = FeedbackUpdater(store)
    updater.log_feedback(req.chunk_id, req.was_useful)
    store.mark_dirty(req.chunk_id)
    store.increment_feedback(req.chunk_id, req.was_useful)
    return {"status": "recorded", "chunk_id": req.chunk_id}


@app.get("/chunks/{chunk_id}/debug")
def debug_chunk(chunk_id: str):
    """
    Inspect the full Redis state for a chunk.
    """
    metadata = store.get_chunk(chunk_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"No metadata for chunk {chunk_id}")

    cached_score    = store.get_cached_score(chunk_id)
    feedback_counts = store.get_feedback_counts(chunk_id)

    return {
        "chunk_id":       chunk_id,
        "metadata":       metadata,
        "cached_score":   cached_score,
        "feedback_counts": feedback_counts,
        "dirty":          cached_score is None,
    }
