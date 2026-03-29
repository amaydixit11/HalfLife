from engine.store.redis_store import RedisStore
import logging

class FeedbackUpdater:
    """
    Handles retrieval feedback for chunks.
    Uses EMA (Exponential Moving Average) to update decay parameters λ.
    
    If useful: λ = λ * (1 - alpha_learn) + target_slow * alpha_learn
    If ignored: λ = λ * (1 - alpha_learn) + target_fast * alpha_learn
    """
    def __init__(self, store: RedisStore, alpha_learn=0.1):
        self.store = store
        self.alpha_learn = alpha_learn # Learning rate for EMA

    def log_feedback(self, chunk_id: str, was_useful: bool):
        metadata = self.store.get_chunk(chunk_id)
        if not metadata:
            return

        trust_score = metadata.get("trust_score", 0.5)
        decay_params = metadata.get("decay_params", {"lambda": 1e-6})
        current_lambda = decay_params.get("lambda", 1e-6)
        
        # Targets for λ (slow vs fast)
        TARGET_SLOW = 1e-8 # 20+ year half-life (landmark)
        TARGET_FAST = 1e-4 # ~2 hour half-life (news)
        
        if was_useful:
            # Shift towards slower decay
            new_lambda = current_lambda * (1 - self.alpha_learn) + TARGET_SLOW * self.alpha_learn
            trust_score = min(1.0, trust_score + 0.05)
        else:
            # Shift towards faster decay
            new_lambda = current_lambda * (1 - self.alpha_learn) + TARGET_FAST * self.alpha_learn
            trust_score = max(0.0, trust_score - 0.05)

        decay_params["lambda"] = new_lambda
        metadata["trust_score"] = trust_score
        metadata["decay_params"] = decay_params
        
        self.store.set_chunk(chunk_id, metadata)
        logging.info(f"Feedback log for {chunk_id}: Useful={was_useful}. New λ: {new_lambda}")
