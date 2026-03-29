from typing import List, Dict
import logging
from datetime import datetime

class TemporalConsistencyChecker:
    """
    Analyzes a set of re-ranked chunks for temporal contradictions.
    Example: 
    Chunk A (2018): "BERT is SOTA"
    Chunk B (2023): "Llama-2 outperforms BERT"
    Result: Warning about superseded info.
    """
    def check(self, chunks: List[Dict], intent: str = "fresh") -> List[Dict]:
        """
        Input: list of re-ranked chunks with 'text' and 'timestamp'.
        Returns: list of consistency warnings.
        """
        warnings = []
        
        # Pattern: if the user explicitly asked for 'historical', 
        # a large temporal span is a SUCCESS, not a RISK.
        if intent == "historical":
            return warnings

        # Extract timestamps: support both flat result objects and raw Qdrant payloads
        timestamps = []
        for c in chunks:
            ts = c.get("timestamp") or c.get("payload", {}).get("timestamp")
            if ts:
                timestamps.append(ts)
        
        if len(set(timestamps)) > 1:
            try:
                # Check for potential time-gap risk
                years = [datetime.fromisoformat(ts).year for ts in timestamps if ts]
                if years and (max(years) - min(years) > 2):
                    warnings.append({
                        "type": "TEMPORAL_DRIFT_RISK",
                        "severity": "MEDIUM",
                        "message": f"Retrieved chunks span {max(years) - min(years)} years. Contradictions likely."
                    })
            except (ValueError, TypeError):
                # ISO format parse error: skip check
                pass
        
        return warnings
