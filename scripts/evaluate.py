import json
import logging
import os
import sys
import time
import random
import numpy as np
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Ensure the root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ingestion.pipeline import HalfLifeIngestor
from engine.decay.learned_model import LEARNED_ENGINE
from engine.classifier.query_intent import QueryIntentClassifier
from engine.fusion.reranker import Reranker
from engine.store.redis_store import RedisStore

logger = logging.getLogger("halflife.eval")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class ResearchEvaluator:
    """
    Scientific Evaluator for HalfLife.
    Measures Top-k Accuracy, MRR, and Statistical Variance.
    """

    def __init__(self, qdrant_url="http://localhost:6333"):
        self.qdrant = QdrantClient(url=qdrant_url)
        self.store = RedisStore()
        self.reranker = Reranker(self.store)
        self.classifier = QueryIntentClassifier()
        self.ingestor = HalfLifeIngestor()
        
        print("🏛️ Loading Research Judge (Cross-Encoder NLI)...")
        self.judge = CrossEncoder('cross-encoder/nli-distilroberta-base')

    def evaluate(self, dataset_path: str, n_trials: int = 3):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        variants = ["baseline", "decay_only", "full_halflife"]
        trial_results = {v: [] for v in variants}

        print(f"\n🔬 STARTING DEFENSIVE ABLATION STUDY ({len(dataset)} samples, {n_trials} trials)")
        print(f"{'='*95}")

        for t in range(n_trials):
            print(f"Trial {t+1}/{n_trials}...")
            random.shuffle(dataset) # Ensure no ordering bias
            
            for v in variants:
                acc_top1 = 0
                acc_top3 = 0
                mrr_sum = 0
                
                for sample in dataset:
                    query = sample["query"]
                    truth = sample["ground_truth"]
                    
                    # Retrieval
                    q_vector = self.ingestor.model.encode(query).tolist()
                    q_res = self.qdrant.query_points(collection_name="halflife_chunks", query=q_vector, limit=20, with_payload=True).points
                    base_chunks = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in q_res]

                    # Variant Logic
                    if v == "baseline":
                        res = self.reranker.rerank(query, base_chunks, intent="fresh", weights={"vector": 1.0, "temporal": 0.0, "trust": 0.0}, top_k=5)
                    elif v == "decay_only":
                        res = self.reranker.rerank(query, base_chunks, intent="fresh", weights={"vector": 0.5, "temporal": 0.5, "trust": 0.0}, top_k=5)
                    elif v == "full_halflife":
                        cl = self.classifier.classify(query)
                        res = self.reranker.rerank(query, base_chunks, intent=cl["intent"], weights=cl["weights"], top_k=5)
                    
                    ranks = res["reranked_chunks"]
                    
                    # Scoring
                    is_hit_1 = self._judge_correctness(query, ranks[0], truth)
                    is_any_3 = any(self._judge_correctness(query, c, truth) for c in ranks[:3])
                    
                    if is_hit_1: acc_top1 += 1
                    if is_any_3: acc_top3 += 1
                    
                    # Calculate MRR@Answers
                    for rank, chunk in enumerate(ranks):
                        if self._judge_correctness(query, chunk, truth):
                            mrr_sum += 1.0 / (rank + 1)
                            break

                trial_results[v].append({
                    "acc1": acc_top1 / len(dataset),
                    "acc3": acc_top3 / len(dataset),
                    "mrr":  mrr_sum / len(dataset)
                })

        self._report(trial_results)

    def _judge_correctness(self, query, chunk, truth) -> bool:
        text = chunk.get("payload", {}).get("text", "")
        hypothesis = f"{truth} is the correct answer to the question: {query}"
        scores = self.judge.predict([(text, hypothesis)])
        return scores[0].argmax() == 2 # Entailment

    def _report(self, trial_results):
        print(f"\n{'='*95}")
        print(" 🏛️ RESEARCH REPORT: HALFLIFE DEFENSIVE AUDIT")
        print(f"{'='*95}")
        print(f"| Variant        | Top-1 Acc (%) | Top-3 Acc (%) | MRR (@Ans) |")
        print(f"| :---           | :---          | :---          | :---       |")
        
        for v, results in trial_results.items():
            acc1 = [r["acc1"]*100 for r in results]
            acc3 = [r["acc3"]*100 for r in results]
            mrr  = [r["mrr"] for r in results]
            
            print(f"| {v:<14} | {np.mean(acc1):>5.1f} ± {np.std(acc1):.1f} | {np.mean(acc3):>5.1f} ± {np.std(acc3):.1f} | {np.mean(mrr):.3f} |")
        print(f"{'='*95}\n")

if __name__ == "__main__":
    evaluator = ResearchEvaluator()
    evaluator.evaluate("scripts/temporal_qa.json")
