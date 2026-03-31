import os
import sys
import json
from datetime import datetime, timezone
from typing import List

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from halflife import HalfLife

class ResearchAssistant:
    """
    Simulates a Research Assistant system that prioritizes finding the MOST RECENT
    breakthroughs while ensuring they are from trusted sources.
    """
    def __init__(self):
        self.hl = HalfLife()
        # Mock database of recent Arxiv papers on 'Efficient LLMs'
        self.papers = [
            {"id": "p1", "title": "LoRA: Low-Rank Adaptation of Large Language Models", "year": 2021, "summary": "LoRA reduces the number of trainable parameters by 10,000 times.", "citations": 8500, "score": 0.98},
            {"id": "p2", "title": "QLoRA: Efficient Finetuning of Quantized LLMs", "year": 2023, "summary": "QLoRA reduces memory usage enough to finetune a 65B model on a single 48GB GPU.", "citations": 1200, "score": 0.95},
            {"id": "p3", "title": "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", "year": 2024, "summary": "GaLore enables full-parameter training of a 7B model on a consumer GPU.", "citations": 150, "score": 0.92},
            {"id": "p4", "title": "DeepSeek-V3 Technical Report: Extreme Parameter Efficiency", "year": 2025, "summary": "DeepSeek-V3 introduces MoE-style efficiency breakthroughs for ultra-large models (2025).", "citations": 50, "score": 0.88}
        ]

    def find_latest_breakthroughs(self, query: str):
        print(f"\n[RESEARCH ASSISTANT] Searching for: \"{query}\"")
        
        # 1. Convert papers to HalfLife chunk format
        chunks = []
        for p in self.papers:
            chunks.append({
                "id": p["id"],
                "score": p["score"], # Standard semantic/BM25 score
                "payload": {
                    "text": f"{p['title']}: {p['summary']}",
                    "timestamp": f"{p['year']}-01-01T00:00:00Z",
                    "original_id": p["id"]
                }
            })

        # 2. Perform Temporal Rerank
        # We use high BETA for temporal relevance because 'breakthrough' implies freshness
        weights = {"vector": 0.4, "temporal": 0.5, "trust": 0.1}
        results = self.hl.rerank(
            query=query, 
            chunks=chunks, 
            top_k=3, 
            weights=weights,
            intent="fresh"
        )
        
        print(f"\n--- Top Recommended Breakthroughs (Chronological Focus) ---")
        for i, res in enumerate(results["reranked_chunks"]):
            title = [p["title"] for p in self.papers if p["id"] == res["id"]][0]
            year = res.get("inferred_year")
            print(f"#{i+1} [{year}] {title}")
            print(f"    Confidence: {res['final_score']:.3f} | Score Source: {res['temporal_source']}")

if __name__ == "__main__":
    assistant = ResearchAssistant()
    assistant.find_latest_breakthroughs("What are the most memory-efficient ways to train LLMs?")
