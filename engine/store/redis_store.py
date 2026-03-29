import redis
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# TTL constants — both the dirty flag and the score cache expire together.
# Worst-case stale score is 1 hour, acceptable for decay operating on days.
SCORE_CACHE_TTL = 3600   # seconds
DIRTY_FLAG_TTL  = 3600   # seconds


class RedisStore:
    """
    Metadata store for HalfLife chunk-level decay state.

    Field contract (what lives here and why):
      decay_type      — exponential | piecewise | learned
      decay_params    — {"lambda": float} — mutated by feedback updater
      trust_score     — float 0.0–1.0     — mutated by feedback updater
      feedback_count  — {"used": int, "ignored": int}
      dirty_flag      — bool, TTL=3600s   — set true when decay_params/trust change
      score_cache     — float, TTL=3600s  — last computed final_score, skips recompute
      last_updated    — ISO 8601 str

    What does NOT live here:
      timestamp       — lives in Qdrant payload, read from search results at query time
      doc_type        — lives in Qdrant payload, used for pre-filtering
      source_domain   — lives in Qdrant payload
      text            — lives in Qdrant payload
    """

    def __init__(self, url: str = "redis://localhost:6379", db: int = 0):
        try:
            self.client = redis.Redis.from_url(url, decode_responses=True, db=db)
            self.client.ping()
        except Exception as e:
            logger.error(f"Redis connection failed at {url}: {e}")
            self.client = None

    # ------------------------------------------------------------------ #
    #  Core read / write                                                   #
    # ------------------------------------------------------------------ #

    def set_chunk(self, chunk_id: str, data: dict) -> None:
        """
        Write full metadata dict for a chunk.
        Caller is responsible for populating all required fields.
        """
        if self.client:
            self.client.set(f"chunk:{chunk_id}", json.dumps(data))

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        if not self.client:
            return None
        raw = self.client.get(f"chunk:{chunk_id}")
        return json.loads(raw) if raw else None

    def delete_chunk(self, chunk_id: str) -> None:
        if self.client:
            self.client.delete(f"chunk:{chunk_id}")

    # ------------------------------------------------------------------ #
    #  Score cache — avoids recomputing decay on every query              #
    # ------------------------------------------------------------------ #

    def get_cached_score(self, chunk_id: str) -> Optional[float]:
        """
        Returns the cached final_score if the dirty flag is not set.
        Returns None if the cache is missing, expired, or dirty.
        """
        if not self.client:
            return None

        # If dirty flag exists, the cache is stale regardless of score_cache TTL
        if self.client.exists(f"dirty:{chunk_id}"):
            return None

        raw = self.client.get(f"score_cache:{chunk_id}")
        return float(raw) if raw is not None else None

    def set_cached_score(self, chunk_id: str, score: float) -> None:
        """
        Cache a computed score. Clears dirty flag if present.
        Both the score and the dirty flag share the same TTL window.
        """
        if not self.client:
            return
        self.client.setex(f"score_cache:{chunk_id}", SCORE_CACHE_TTL, str(score))
        self.client.delete(f"dirty:{chunk_id}")

    def mark_dirty(self, chunk_id: str) -> None:
        """
        Invalidate the cached score for a chunk.
        Called by FeedbackUpdater and EventBus after mutating decay state.
        The dirty flag itself expires after DIRTY_FLAG_TTL so a missed
        clear never permanently blocks score recomputation.
        """
        if self.client:
            self.client.setex(f"dirty:{chunk_id}", DIRTY_FLAG_TTL, "1")

    # ------------------------------------------------------------------ #
    #  Feedback helpers — called by FeedbackUpdater                       #
    # ------------------------------------------------------------------ #

    def increment_feedback(self, chunk_id: str, was_useful: bool) -> None:
        """
        Atomically increment used or ignored counter.
        Uses Redis HINCRBY for thread safety.
        """
        if not self.client:
            return
        field = "used" if was_useful else "ignored"
        self.client.hincrby(f"feedback:{chunk_id}", field, 1)

    def get_feedback_counts(self, chunk_id: str) -> dict:
        if not self.client:
            return {"used": 0, "ignored": 0}
        raw = self.client.hgetall(f"feedback:{chunk_id}")
        return {
            "used":    int(raw.get("used",    0)),
            "ignored": int(raw.get("ignored", 0)),
        }

    # ------------------------------------------------------------------ #
    #  Convenience factory — builds a well-formed metadata dict           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_metadata(
        chunk_id:     str,
        decay_type:   str   = "exponential",
        decay_params: dict  = None,
        trust_score:  float = 0.5,
    ) -> dict:
        """
        Returns a canonical metadata dict ready to pass to set_chunk().
        Use this everywhere instead of building dicts by hand — it
        enforces the field contract and prevents silent missing keys.
        """
        from datetime import datetime, timezone
        return {
            "chunk_id":    chunk_id,
            "decay_type":  decay_type,
            "decay_params": decay_params or {"lambda": 1e-6},
            "trust_score": trust_score,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
