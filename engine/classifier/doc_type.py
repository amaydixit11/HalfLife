from typing import Dict

class DocTypeClassifier:
    """
    Classifies a text chunk into a document category:
    - news: Fast decay (exponential)
    - documentation: Stable/Step decay (piecewise)
    - research: Slow/Landmark decay (exponential with slow lambda)
    - reference: Universal truth
    """
    NEWS_KEYWORDS = {"breaking", "today", "flash", "update", "newsworthy"}
    DOCS_KEYWORDS = {"version", "release", "api", "usage", "compatibility"}
    RESEARCH_KEYWORDS = {"abstract", "paper", "methodology", "citation", "experiment"}

    def classify(self, text: str) -> Dict:
        """
        Classifies the document type and returns initial decay settings.
        """
        text_lower = text.lower()
        
        # Check for News
        if any(kw in text_lower for kw in self.NEWS_KEYWORDS):
            return {
                "doc_type": "news",
                "decay_type": "exponential",
                "decay_params": {"lambda": 1e-8}, # ~2 year half-life
                "trust_score": 0.6
            }
        
        # Check for Documentation
        if any(kw in text_lower for kw in self.DOCS_KEYWORDS):
            return {
                "doc_type": "documentation",
                "decay_type": "piecewise",
                "decay_params": {}, 
                "trust_score": 0.8
            }
        
        # Check for Research
        if any(kw in text_lower for kw in self.RESEARCH_KEYWORDS):
            return {
                "doc_type": "research",
                "decay_type": "exponential",
                "decay_params": {"lambda": 1e-9}, # ~20 year half-life
                "trust_score": 0.9
            }
            
        # Default fallback
        return {
            "doc_type": "generic",
            "decay_type": "exponential",
            "decay_params": {"lambda": 5e-9}, # ~4 year half-life
            "trust_score": 0.5
        }
