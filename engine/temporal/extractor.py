import re
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class TemporalExtractor:
    """
    Temporal Signal Extraction for Unstructured RAG Corpora.
    Handles the chaotic reality where temporal metadata is often missing,
    dynamically resolving a timestamp with an associated confidence score.
    """
    
    def __init__(self):
        # High precision dates (e.g., 2024-03-01, 2024/03/01)
        self.exact_date_pattern = re.compile(r"\b(20[0-2]\d|19\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
        # Year only fallback (e.g., 2019, 2024)
        self.year_pattern = re.compile(r"\b(20[0-2]\d|19\d{2})\b")

    def resolve_timestamp(self, chunk: dict) -> Tuple[Optional[datetime], float]:
        """
        Dynamically extracts a timestamp + confidence interval from a chunk.
        
        Args:
            chunk: The chunk payload dictionary
            
        Returns:
            (datetime, float): The inferred datetime and a confidence score [0.0 - 1.0]
        """
        payload = chunk.get("payload", {})
        text = payload.get("text", "")
        
        # 1. Metadata Verification (Best Case Scenario)
        raw_ts = payload.get("timestamp") or payload.get("date")
        if raw_ts:
            try:
                # Direct ISO mapping
                ts = datetime.fromisoformat(str(raw_ts))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts, 1.0  # Perfect strict metadata confidence
            except ValueError:
                # If un-parseable, we degrade gracefully to text extraction
                pass
                
        # 2. Textual Extraction (The Practical Reality)
        # Search for exact formatted dates in text
        exact_match = self.exact_date_pattern.search(text)
        if exact_match:
            try:
                year, month, day = map(int, exact_match.groups())
                return datetime(year, month, day, tzinfo=timezone.utc), 0.8
            except ValueError:
                pass

        # Search for loose years in text
        year_match = self.year_pattern.search(text)
        if year_match:
            year = int(year_match.group(1))
            return datetime(year, 1, 1, tzinfo=timezone.utc), 0.6  # Approximate confidence
            
        # 3. Default Heuristics / Final Fallback
        # When absolutely no temporal signal exists on the chunk
        return None, 0.4
