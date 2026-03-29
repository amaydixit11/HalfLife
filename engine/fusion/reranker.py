import logging
from datetime import datetime, timezone
from typing import Optional

from engine.decay.registry import get_decay
from engine.fusion.consistency import TemporalConsistencyChecker
from engine.store.redis_store import RedisStore

logger = logging.getLogger(__name__)


class Reranker:
    """
    HalfLife fusion layer.

    Scoring formula:
        final_score = α * vector_score
                    + β * temporal_score
                    + γ * trust_score

    Where:
        vector_score   — cosine similarity from Qdrant (0.0–1.0)
        temporal_score — decay(Δt), computed from timestamp in Qdrant
                         payload. For historical queries this is inverted:
                         temporal_score = 1.0 - decay(Δt), so older chunks
                         score higher.
        trust_score    — float from Redis metadata (0.0–1.0)

    Score cache:
        If a chunk's score_cache is warm (dirty flag not set), the cached
        score is returned without recomputing decay. This keeps rerank
        latency low when the same chunks are queried repeatedly.

    Chunk input contract:
        Each chunk dict must contain:
            id     (str)   — chunk_id
            score  (float) — vector similarity score from Qdrant
            payload (dict) — Qdrant payload with at minimum:
                               timestamp (ISO 8601 str)
        The payload is already present on Qdrant ScoredPoint objects and
        should be passed through by the caller. This eliminates the need
        to fetch timestamp from Redis, which was the original bug.
    """

    DEFAULT_WEIGHTS = {"vector": 0.6, "temporal": 0.3, "trust": 0.1}

    def __init__(self, store: RedisStore):
        self.store   = store
        self.checker = TemporalConsistencyChecker()

    # ------------------------------------------------------------------ #
    #  Public interface                                                   #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query:    str,
        chunks:   list,
        top_k:    int  = 10,
        weights:  Optional[dict] = None,
        intent:   Optional[str]  = None,
    ) -> dict:
        """
        Args:
            query:    Raw query string (used for consistency check).
            chunks:   List of dicts with keys: id, score, payload.
                      payload must include 'timestamp' (ISO 8601 str).
            top_k:    Number of results to return.
            weights:  {"vector": α, "temporal": β, "trust": γ}.
                      Defaults to DEFAULT_WEIGHTS if not provided.
            intent:   Query temporal intent — "fresh" | "historical" | "static".
                      If "historical", temporal_score is inverted so older
                      chunks rank higher. Provided by QueryIntentClassifier.

        Returns:
            {
                "reranked_chunks": [...],
                "consistency_warnings": [...],
                "applied_weights": {...},
            }
        """
        now     = datetime.now(timezone.utc)
        weights = weights or self.DEFAULT_WEIGHTS
        results = []

        for chunk in chunks:
            chunk_id     = chunk.get("id")
            vector_score = chunk.get("score", 0.0)
            payload      = chunk.get("payload", {})

            if not chunk_id:
                continue

            # ---- Score cache check ------------------------------------
            # If cache is warm, skip all computation for this chunk.
            # Cache is specific to (chunk_id, intent) to allow for
            # historical vs fresh queries to return different scores.
            cache_key = f"{chunk_id}:{intent or 'none'}"
            cached    = self.store.get_cached_score(cache_key)
            if cached is not None:
                results.append({
                    "id":             chunk_id,
                    "final_score":    cached,
                    "vector_score":   vector_score,
                    "temporal_score": None,   # not recomputed
                    "trust_score":    None,
                    "timestamp":      payload.get("timestamp"),
                    "doc_type":       payload.get("doc_type"),
                    "cache_hit":      True,
                })
                continue
            
            # ---- Redis fetch: mutable decay state only ----------------
            metadata = self.store.get_chunk(chunk_id)

            # ---- Timestamp: always from Qdrant payload ----------------
            # Never from Redis — Qdrant payload is the single source of truth.
            ts = self._parse_timestamp(payload.get("timestamp"), chunk_id, now)

            # ---- Temporal score ---------------------------------------
            if metadata:
                decay_fn = get_decay(
                    metadata.get("decay_type", "exponential"),
                    metadata.get("decay_params", {}),
                )
                trust_score = metadata.get("trust_score", 0.5)
            else:
                # No Redis entry: use exponential with default lambda.
                logger.debug(f"No Redis metadata for chunk {chunk_id}, using defaults")
                decay_fn    = get_decay("exponential", {})
                trust_score = 0.5

            raw_temporal = decay_fn.compute(timestamp=ts, now=now)

            # ---- Historical intent: invert temporal score -------------
            # For historical queries we want older = higher.
            if intent == "historical":
                temporal_score = 1.0 - raw_temporal
            else:
                temporal_score = raw_temporal

            # ---- Fusion -----------------------------------------------
            final_score = (
                weights["vector"]   * vector_score  +
                weights["temporal"] * temporal_score +
                weights["trust"]    * trust_score
            )

            # ---- Write score cache ------------------------------------
            self.store.set_cached_score(cache_key, final_score)



            results.append({
                "id":             chunk_id,
                "final_score":    round(final_score, 6),
                "vector_score":   round(vector_score, 6),
                "temporal_score": round(temporal_score, 6),
                "trust_score":    round(trust_score, 6),
                "timestamp":      ts.isoformat(),
                "doc_type":       payload.get("doc_type"),
                "cache_hit":      False,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:top_k]

        warnings = self.checker.check(top_results, intent=intent or "fresh")

        return {
            "reranked_chunks":      top_results,
            "consistency_warnings": warnings,
            "applied_weights":      weights,
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _parse_timestamp(
        self,
        ts_str:   Optional[str],
        chunk_id: str,
        fallback: datetime,
    ) -> datetime:
        """
        Parse ISO 8601 timestamp string into a tz-aware datetime.
        """
        if not ts_str:
            logger.warning(f"Missing timestamp for chunk {chunk_id}, defaulting to now")
            return fallback
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except ValueError:
            logger.warning(f"Malformed timestamp '{ts_str}' for chunk {chunk_id}, defaulting to now")
            return fallback
