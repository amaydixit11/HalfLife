import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.store.redis_store import RedisStore
from engine.fusion.reranker import Reranker
from engine.classifier.query_intent import QueryIntentClassifier
from engine.ingestion.pipeline import HalfLifeIngestor, COLLECTION_NAME
from scripts.corpus import build_corpus, CorpusChunk, BenchmarkQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

TOP_K = 10

def ndcg_at_k(relevant_ids, retrieved_ids, k=TOP_K):
    relevant_set = set(relevant_ids)
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(retrieved_ids[:k]) if rid in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(relevant_ids, retrieved_ids):
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set: return 1.0 / (i + 1)
    return 0.0

def mean_document_age(retrieved_ids, chunk_map, k=TOP_K):
    now = datetime.now(timezone.utc)
    ages = []
    for rid in retrieved_ids[:k]:
        if not rid: continue
        chunk = chunk_map.get(rid)
        if chunk:
            age_years = (now - chunk.timestamp).total_seconds() / (86400 * 365)
            ages.append(age_years)
    return float(np.mean(ages)) if ages else 0.0

def aggregate_results(runs: List[Dict]) -> Dict:
    summary = defaultdict(lambda: defaultdict(list))
    for r in runs:
        intent = r.get("intent", "static")
        metrics = r.get("metrics", {})
        if not metrics: continue
        summary[intent]["ndcg_delta"].append(metrics["halflife"]["ndcg"] - metrics["baseline"]["ndcg"])
        summary[intent]["mrr_delta"].append(metrics["halflife"]["mrr"] - metrics["baseline"]["mrr"])
        summary[intent]["age_delta"].append(metrics["halflife"]["age"] - metrics["baseline"]["age"])
        # Individual averages
        summary[intent]["h_ndcg"].append(metrics["halflife"]["ndcg"])
        summary[intent]["b_ndcg"].append(metrics["baseline"]["ndcg"])

    agg = {}
    for intent, mets in summary.items():
        agg[intent] = {k: float(np.mean(v)) for k, v in mets.items()}
    return agg

def main(
    skip_ingest: bool = False,
    output:      Optional[str] = None,
    decay_type:  Optional[str] = None,
    qdrant_url:  str = "http://localhost:6333",
    redis_url:   str = "redis://localhost:6379",
    debug:       bool = False,
):
    """
    Main entry point for benchmarking.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("🚀 Starting HalfLife Research Benchmark...")
    chunks, queries = build_corpus()
    chunk_map = {c.chunk_id: c for c in chunks}
    
    qdrant = QdrantClient(url=qdrant_url)
    store = RedisStore(url=redis_url)
    reranker = Reranker(store)
    classifier = QueryIntentClassifier()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if not skip_ingest:
        logger.info(f"Wiping collection '{COLLECTION_NAME}' for clean benchmark...")
        try:
            qdrant.delete_collection(COLLECTION_NAME)
        except Exception:
            pass # Collection might not exist yet
            
        ingestor = HalfLifeIngestor(qdrant_url=qdrant_url, redis_url=redis_url)
        logger.info(f"Ingesting {len(chunks)} chunks...")
        id_map = {}
        for i, chunk in enumerate(chunks):
            assigned_id = ingestor.ingest(
                text=chunk.text, 
                timestamp=chunk.timestamp, 
                doc_type=chunk.doc_type, 
                decay_type=decay_type,
                original_id=chunk.chunk_id
            )
            id_map[chunk.chunk_id] = assigned_id
            if (i+1) % 50 == 0: logger.info(f"  ...ingested {i+1}")
    else:
        logger.info("Skipping ingestion — recovering id_map...")
        id_map = {c.chunk_id: c.chunk_id for c in chunks} # Placeholder for simplicity

    results = {intent: [] for intent in ["fresh", "historical", "static"]}
    for query_id, sample in queries.items():
        q_vector = embedder.encode(sample.text).tolist()
        q_points = qdrant.query_points(collection_name=COLLECTION_NAME, query=q_vector, limit=20, with_payload=True).points
        base_chunks = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in q_points]
        
        cl = classifier.classify(sample.text)
        hl_res = reranker.rerank(query=sample.text, chunks=base_chunks, intent=cl["intent"], weights=cl["weights"], top_k=TOP_K)
        
        results[sample.intent].append({
            "query":    sample.text,
            "metrics": {
                "baseline": {
                    "ndcg": ndcg_at_k(sample.target_ids, [r.payload.get("original_id") for r in q_points]),
                    "mrr":  mrr(sample.target_ids, [r.payload.get("original_id") for r in q_points]),
                    "age":   mean_document_age([r.payload.get("original_id") for r in q_points], chunk_map)
                },
                "halflife": {
                    "ndcg": ndcg_at_k(sample.target_ids, [r.get("original_id") for r in hl_res["reranked_chunks"]]),
                    "mrr":  mrr(sample.target_ids, [r.get("original_id") for r in hl_res["reranked_chunks"]]),
                    "age":   mean_document_age([r.get("original_id") for r in hl_res["reranked_chunks"]], chunk_map)
                }
            },
            "intent": cl["intent"]
        })

    # Aggregator & Summary Table
    from rich.console import Console
    from rich.table import Table
    console = Console()

    summary = Table(title=f"📊 HalfLife IR Benchmark (Mode: {decay_type or 'Default'})", box=None)
    summary.add_column("Intent", style="cyan", header_style="bold")
    summary.add_column("Metric", style="magenta", header_style="bold")
    summary.add_column("Baseline", justify="right")
    summary.add_column("HalfLife", justify="right", style="bold green")
    summary.add_column("Gain", justify="right")

    for intent, items in results.items():
        if not items: continue
        for met in ["ndcg", "mrr", "age"]:
            b_vals = [it["metrics"]["baseline"][met] for it in items]
            h_vals = [it["metrics"]["halflife"][met] for it in items]
            
            b_avg, h_avg = np.mean(b_vals), np.mean(h_vals)
            
            if met == "age":
                # For FRESH, lower age is better. For HISTORICAL, higher age is better.
                if intent == "fresh":
                    gain = ((b_avg - h_avg) / b_avg * 100) if b_avg > 0 else 0.0
                    gain_label = f"{gain:+.1f}% Freshness"
                else:
                    gain = ((h_avg - b_avg) / b_avg * 100) if b_avg > 0 else 0.0
                    gain_label = f"{gain:+.1f}% Historical"
                label = "AGE (YRS)"
            else:
                gain = ((h_avg / b_avg) - 1.0) * 100 if b_avg > 0 else 0.0
                gain_label = f"{gain:+.1f}%"
                label = met.upper()

            summary.add_row(
                intent.upper(),
                label,
                f"{b_avg:.4f}",
                f"{h_avg:.4f}",
                gain_label
            )
        summary.add_section()

    console.print("\n")
    console.print(summary)
    if output:
        with open(output, "w") as f: json.dump(results, f, indent=2)
        print(f"📄 Results written to {output}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--decay-type")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger("engine").setLevel(logging.DEBUG)
        logging.getLogger("scripts").setLevel(logging.DEBUG)
        
    main(skip_ingest=args.skip_ingest, output=args.output, decay_type=args.decay_type, debug=args.debug)
