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

        # ------------------------------------------------------------------ #
        #  Batch-level Scoring & Normalization                               #
        # ------------------------------------------------------------------ #

        
        # 1. First pass: compute raw scores (or use cache)
        raw_results = []
        for chunk in chunks:
            chunk_id     = chunk.get("id")
            vector_score = chunk.get("score", 0.0)
            payload      = chunk.get("payload", {})
            if not chunk_id: continue

            cache_key = f"{chunk_id}:{intent or 'none'}"
            cached    = self.store.get_cached_score(cache_key)
        start_compute = datetime.now()
        raw_results = []
        for r in chunks:
            chunk_id     = r.get("id")
            payload      = r.get("payload", {})
            vector_score = r.get("score")
            trust_score  = 0.5  # ROBUSTNESS: Default initialization
            
            # --- ROBUSTNESS: Metadata & Intent Handling ---
            raw_ts = payload.get("timestamp")
            
            if not raw_ts:
                # MESSY REALITY: Infer date from text if metadata is missing
                import re
                text_snippet = payload.get("text", "")[:500]
                # Look for years like 2024, 2025, etc.
                year_match = re.search(r"\b(19\d{2}|20[0-2]\d|2030)\b", text_snippet)
                if year_match:
                    ts = datetime(int(year_match.group(1)), 1, 1, tzinfo=timezone.utc)
                    # We give inferred dates a slightly lower trust score
                    trust_score = 0.4
                    decay_fn = get_decay("exponential", {}) # Default to exponential
                    temporal = decay_fn.compute(timestamp=ts, now=now)
                else:
                    # Neutral fallback if absolutely no date found
                    ts = None 
                    temporal = 0.5
            else:
                ts = self._parse_timestamp(raw_ts, chunk_id, now)
                
                # Metadata-driven decay selection
                metadata = self.store.get_chunk(chunk_id)
                d_type   = metadata.get("decay_type", "exponential") if metadata else "exponential"
                d_params = metadata.get("decay_params", {}) if metadata else {}
                trust_score = metadata.get("trust_score", 0.5) if metadata else 0.5
                
                # --- RESEARCH: Support Neural Lambda Overrides ---
                if "lambda" in weights:
                    d_params = {"lambda": weights["lambda"]}

                decay_fn = get_decay(d_type, d_params)
                temporal = decay_fn.compute(timestamp=ts, now=now)
                
            # Intent Inversion logic
            if intent == "historical":
                temporal = 1.0 - temporal
            
            raw_results.append({
                "id": chunk_id, "v": vector_score, "t": temporal, "tr": trust_score, "p": payload, "ts": ts
            })

        # --- Latency Tracking ---
        compute_ms = (datetime.now() - start_compute).total_seconds() * 1000
        logger.debug(f"Fusion batch for {len(chunks)} chunks took {compute_ms:.2f}ms")

        # 2. Second pass: Min-Max Normalization
        v_scores = [r["v"] for r in raw_results]
        t_scores = [r["t"] for r in raw_results]
        
        def norm(val, vals):
            if not vals or max(vals) == min(vals): return 0.5
            return (val - min(vals)) / (max(vals) - min(vals))

        final_chunks = []
        for r in raw_results:
            # Weighted fusion on normalized signals
            vn = norm(r["v"], v_scores)
            tn = norm(r["t"], t_scores)
            
            final_score = (weights["vector"] * vn + weights["temporal"] * tn + weights["trust"] * r["tr"])

            final_chunks.append({
                "id":             r["id"],
                "final_score":    round(final_score, 6),
                "vector_score":   round(r["v"], 6),
                "temporal_score": round(r["t"], 10), # Precision for debug
                "trust_score":    round(r["tr"], 6),
                "timestamp":      r["p"].get("timestamp"),
                "text":           r["p"].get("text", "---"),
            })

        final_chunks.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = final_chunks[:top_k]
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
