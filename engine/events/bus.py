from engine.store.redis_store import RedisStore
import logging

class EventBus:
    """
    Handles external invalidation events and score penalties.
    """
    def __init__(self, store: RedisStore):
        self.store = store

    def invalidate(self, chunk_id: str, strategy: str = "hard", reason: str = None):
        """ Alias for handle_invalidation to match API contract. """
        return self.handle_invalidation(chunk_id, type=strategy, reason=reason)

    def handle_invalidation(self, chunk_id: str, type: str = "hard", reason: str = None):
        """
        Processes hard or soft invalidation.
        Logic:
        - hard: trust_score -> 0, lambda -> fast (accel obsolescence)
        - soft: trust_score -= 30%
        """
        metadata = self.store.get_chunk(chunk_id)
        if not metadata:
            return

        if type == "hard":
            metadata["trust_score"] = 0.0
            metadata["decay_params"] = {"lambda": 1e-3} # Extremely fast decay (obsolete within seconds)
            metadata["invalidation_reason"] = reason or "Factually superseded or retracted"
        elif type == "soft":
            metadata["trust_score"] = max(0.0, metadata.get("trust_score", 0.5) - 0.3)
            # Penalize by acceleration
            if "lambda" in metadata.get("decay_params", {}):
                metadata["decay_params"]["lambda"] *= 2.0 # 2x faster decay
        
        self.store.set_chunk(chunk_id, metadata)
        logging.info(f"Invalidated {chunk_id} ({type}): {reason}")
