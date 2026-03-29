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

def temporal_freshness(retrieved_ids, chunk_map, k=TOP_K):
    now = datetime.now(timezone.utc)
    scores = []
    for rid in retrieved_ids[:k]:
        chunk = chunk_map.get(rid)
        if chunk:
            age_days = (now - chunk.timestamp).total_seconds() / 86400
            scores.append(1.0 / (1.0 + age_days))
    return float(np.mean(scores)) if scores else 0.0

def main(
    skip_ingest: bool = False,
    output:      Optional[str] = None,
    decay_type:  Optional[str] = None,
    qdrant_url:  str = "http://localhost:6333",
    redis_url:   str = "redis://localhost:6379",
):
    """
    Main entry point for benchmarking.
    """
    logger.info("🚀 Starting HalfLife Research Benchmark...")
    chunks, queries = build_corpus()
    chunk_map = {c.chunk_id: c for c in chunks}
    
    qdrant = QdrantClient(url=qdrant_url)
    store = RedisStore(url=redis_url)
    reranker = Reranker(store)
    classifier = QueryIntentClassifier()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if not skip_ingest:
        ingestor = HalfLifeIngestor(qdrant_url=qdrant_url, redis_url=redis_url)
        logger.info(f"Ingesting {len(chunks)} chunks...")
        id_map = {}
        for i, chunk in enumerate(chunks):
            assigned_id = ingestor.ingest(text=chunk.text, timestamp=chunk.timestamp, doc_type=chunk.doc_type, decay_type=decay_type)
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
            "query": sample.text,
            "baseline": [str(r.id) for r in q_points],
            "halflife": [r["id"] for r in hl_res["reranked_chunks"]]
        })

    # Aggregator & Summary logic...
    # (Simplified for the CLI demo)
    print(f"\n✅ Benchmark Complete. Mode: {decay_type or 'Default'}")
    if output:
        with open(output, "w") as f: json.dump(results, f, indent=2)
        print(f"📄 Results written to {output}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--decay-type")
    args = parser.parse_args()
    main(skip_ingest=args.skip_ingest, output=args.output, decay_type=args.decay_type)
