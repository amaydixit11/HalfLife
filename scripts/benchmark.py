"""
benchmark.py — HalfLife evaluation harness.

Answers the core research question:
    Does temporal re-ranking improve retrieval quality vs. naive cosine
    similarity, and does the improvement vary by query intent?

Usage:
    # With Qdrant + Redis running (docker-compose up -d):
    python scripts/benchmark.py

    # Skip ingestion if corpus already loaded:
    python scripts/benchmark.py --skip-ingest

    # Output results to JSON for further analysis:
    python scripts/benchmark.py --output results.json

Metrics computed:
    nDCG@10  — standard ranking quality metric
    MRR      — mean reciprocal rank of first relevant result
    TF@10    — temporal freshness: mean(1 / (1 + age_days)) over top-10
               higher = fresher top-k results
               for fresh queries: TF↑ is good
               for historical queries: TF↓ is good
               for static queries: TF should be neutral (no regression)

Results are broken down by query intent so you can see exactly where
temporal re-ranking helps, hurts, and is neutral.
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOP_K = 10


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #

def ndcg_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = TOP_K) -> float:
    """
    Normalised Discounted Cumulative Gain at k.
    Binary relevance: relevant = 1, not relevant = 0.
    """
    relevant_set = set(relevant_ids)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, rid in enumerate(retrieved_ids[:k])
        if rid in relevant_set
    )
    # Ideal DCG: all relevant docs at top positions
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def temporal_freshness(
    retrieved_ids: List[str],
    chunk_map: Dict[str, CorpusChunk],
    k: int = TOP_K,
) -> float:
    """
    Temporal Freshness Score (TF@k).

    TF = mean(1 / (1 + age_days)) over top-k retrieved chunks.

    Range: (0, 1]. Score of 1.0 = all chunks published today.
    Higher = fresher top-k.
    """
    now = datetime.now(timezone.utc)
    scores = []
    for rid in retrieved_ids[:k]:
        chunk = chunk_map.get(rid)
        if chunk:
            age_days = (now - chunk.timestamp).total_seconds() / 86400
            scores.append(1.0 / (1.0 + age_days))
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------- #
#  Ingestion                                                                   #
# --------------------------------------------------------------------------- #

def ingest_corpus(
    chunks: List[CorpusChunk],
    ingestor: HalfLifeIngestor,
) -> Dict[str, str]:
    """
    Ingest benchmark corpus. Returns mapping of corpus chunk_id → Qdrant chunk_id.
    """
    logger.info(f"Ingesting {len(chunks)} chunks into Qdrant + Redis...")
    id_map = {}

    for i, chunk in enumerate(chunks):
        assigned_id = ingestor.ingest(
            text=chunk.text,
            timestamp=chunk.timestamp,
            source_domain=chunk.source_domain,
            doc_type=chunk.doc_type,
        )
        # We need to remap: the ingestor generates a new uuid, but our
        # ground truth uses corpus chunk_ids. Store the mapping.
        id_map[chunk.chunk_id] = assigned_id

        if (i + 1) % 10 == 0:
            logger.info(f"  Ingested {i+1}/{len(chunks)}")

    logger.info("Ingestion complete.")
    return id_map


# --------------------------------------------------------------------------- #
#  Single query evaluation                                                     #
# --------------------------------------------------------------------------- #

def evaluate_query(
    query:        BenchmarkQuery,
    id_map:       Dict[str, str],
    qdrant:       QdrantClient,
    reranker:     Reranker,
    classifier:   QueryIntentClassifier,
    embedder:     SentenceTransformer,
    chunk_map:    Dict[str, CorpusChunk],
) -> Dict:
    """
    Runs one query through baseline and HalfLife, returns metric dict.
    """
    # Translate ground truth corpus IDs → ingested Qdrant IDs
    relevant_qdrant_ids = [id_map[cid] for cid in query.relevant_ids if cid in id_map]

    # Embed query
    q_vector = embedder.encode(query.text).tolist()

    # ---- Baseline: naive Qdrant retrieval -------------------------
    t0 = time.perf_counter()
    naive_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vector,
        limit=20, # Fetch more for rerank
        with_payload=True,
    )
    baseline_latency = (time.perf_counter() - t0) * 1000  # ms
    
    # query_points returns a list of ScoredPoint
    naive_results = naive_response.points

    naive_ids = [str(r.id) for r in naive_results]

    # ---- HalfLife rerank -------------------------------------------------
    classification = classifier.classify(query.text)
    weights = classification["weights"]
    intent  = classification["intent"]

    chunks_for_rerank = [
        {
            "id":      str(r.id),
            "score":   r.score,
            "payload": r.payload or {},
        }
        for r in naive_results
    ]

    t0 = time.perf_counter()
    rerank_result = reranker.rerank(
        query=query.text,
        chunks=chunks_for_rerank,
        top_k=TOP_K,
        weights=weights,
        intent=intent,
    )
    halflife_latency = (time.perf_counter() - t0) * 1000  # ms

    halflife_ids = [r["id"] for r in rerank_result["reranked_chunks"]]

    # ---- Metrics ---------------------------------------------------------
    # Build reverse id_map for chunk lookup: qdrant_id → CorpusChunk
    qdrant_to_corpus = {v: chunk_map[k] for k, v in id_map.items() if k in chunk_map}

    return {
        "query_id":          query.query_id,
        "query_text":        query.text,
        "intent":            query.intent,
        "topic":             query.topic,
        "detected_intent":   intent,

        # Standard IR metrics
        "baseline_ndcg":     ndcg_at_k(relevant_qdrant_ids, naive_ids),
        "halflife_ndcg":     ndcg_at_k(relevant_qdrant_ids, halflife_ids),
        "baseline_mrr":      mrr(relevant_qdrant_ids, naive_ids),
        "halflife_mrr":      mrr(relevant_qdrant_ids, halflife_ids),

        # Temporal freshness metric
        "baseline_tf":       temporal_freshness(naive_ids,     qdrant_to_corpus),
        "halflife_tf":       temporal_freshness(halflife_ids,  qdrant_to_corpus),

        # Latency
        "baseline_latency_ms":  round(baseline_latency, 2),
        "halflife_latency_ms":  round(halflife_latency, 2),

        # Ranking shift: how many positions did results move
        "mean_rank_shift":   _mean_rank_shift(naive_ids[:TOP_K], halflife_ids),
    }


def _mean_rank_shift(before: List[str], after: List[str]) -> float:
    """Average absolute rank change across all results that appear in both lists."""
    before_rank = {rid: i for i, rid in enumerate(before)}
    shifts = []
    for new_rank, rid in enumerate(after):
        if rid in before_rank:
            shifts.append(abs(new_rank - before_rank[rid]))
    return float(np.mean(shifts)) if shifts else 0.0


# --------------------------------------------------------------------------- #
#  Results aggregation                                                         #
# --------------------------------------------------------------------------- #

def aggregate_results(per_query_results: List[Dict]) -> Dict:
    """
    Aggregates per-query metrics into per-intent summaries.
    """
    by_intent = defaultdict(list)
    for r in per_query_results:
        by_intent[r["intent"]].append(r)

    summary = {}
    for intent, results in by_intent.items():
        def mean(key): return float(np.mean([r[key] for r in results]))

        nd_base = mean("baseline_ndcg")
        nd_hl   = mean("halflife_ndcg")
        ndcg_delta = nd_hl - nd_base

        mrr_base = mean("baseline_mrr")
        mrr_hl   = mean("halflife_mrr")

        summary[intent] = {
            "n_queries":           len(results),

            "baseline_ndcg":       round(nd_base, 4),
            "halflife_ndcg":       round(nd_hl, 4),
            "ndcg_delta":          round(ndcg_delta, 4),
            "ndcg_pct_change":     round(ndcg_delta / (max(nd_base, 1e-9)) * 100, 1),

            "baseline_mrr":        round(mrr_base, 4),
            "halflife_mrr":        round(mrr_hl, 4),
            "mrr_delta":           round(mrr_hl - mrr_base, 4),

            "baseline_tf":         round(mean("baseline_tf"), 6),
            "halflife_tf":         round(mean("halflife_tf"), 6),
            "tf_delta":            round(mean("halflife_tf") - mean("baseline_tf"), 6),

            "baseline_latency_ms": round(mean("baseline_latency_ms"), 2),
            "halflife_latency_ms": round(mean("halflife_latency_ms"), 2),

            "mean_rank_shift":     round(mean("mean_rank_shift"), 2),
        }

    # --- Classification Stats ---
    hits = sum(1 for q in per_query_results if q["intent"] == q["detected_intent"])
    summary["classification_accuracy"] = hits / len(per_query_results) if per_query_results else 0
    
    confusions = defaultdict(int)
    for q in per_query_results:
        if q["intent"] != q["detected_intent"]:
            confusions[f"{q['intent']} -> {q['detected_intent']}"] += 1
    summary["confusions"] = dict(confusions)

    return summary


def print_results(summary: Dict) -> None:
    """Pretty-print the results table to stdout."""
    header = (
        f"\n{'─'*72}\n"
        f"  HALFLIFE BENCHMARK RESULTS\n"
        f"{'─'*72}"
    )
    print(header)

    col = "{:<14} {:>8} {:>8} {:>8}   {:>8} {:>8}   {:>8} {:>8}"
    print(col.format(
        "intent", "base nDCG", "HL nDCG", "Δ nDCG",
        "base MRR", "HL MRR",
        "base TF", "HL TF"
    ))
    print("─" * 72)

    for intent in ["fresh", "historical", "static"]:
        if intent not in summary:
            continue
        r = summary[intent]
        delta_str = f"{r['ndcg_delta']:+.4f}"
        print(col.format(
            intent,
            f"{r['baseline_ndcg']:.4f}",
            f"{r['halflife_ndcg']:.4f}",
            delta_str,
            f"{r['baseline_mrr']:.4f}",
            f"{r['halflife_mrr']:.4f}",
            f"{r['baseline_tf']:.4f}",
            f"{r['halflife_tf']:.4f}",
        ))

    print("─" * 72)
    print(f"\n  Classification Accuracy: {summary.get('classification_accuracy', 0):.1%}")
    if summary.get("confusions"):
        print("  Confusions:")
        for key, val in sorted(summary["confusions"].items()):
            print(f"    - {key}: {val}")

    print(
        "\n  TF interpretation:\n"
        "    fresh     → HalfLife TF > baseline TF  (✓ fresher results)\n"
        "    historical → HalfLife TF < baseline TF (✓ older results surfaced)\n"
        "    static    → HalfLife TF ≈ baseline TF  (✓ no regression)\n"
    )


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="HalfLife benchmark")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--redis-url",  default="redis://localhost:6379")
    args = parser.parse_args()

    # ---- Build corpus ---------------------------------------------------
    logger.info("Building synthetic corpus...")
    chunks, queries = build_corpus()
    chunk_map = {c.chunk_id: c for c in chunks}
    logger.info(f"  {len(chunks)} chunks, {len(queries)} queries")

    # ---- Connections ----------------------------------------------------
    qdrant    = QdrantClient(url=args.qdrant_url)
    store     = RedisStore(url=args.redis_url)
    reranker  = Reranker(store)
    classifier = QueryIntentClassifier()
    embedder  = SentenceTransformer("all-MiniLM-L6-v2")

    # ---- Ingest ---------------------------------------------------------
    if not args.skip_ingest:
        ingestor = HalfLifeIngestor(
            qdrant_url=args.qdrant_url,
            redis_url=args.redis_url,
        )
        id_map = ingest_corpus(chunks, ingestor)
    else:
        logger.info("Skipping ingest — recovering id_map from Qdrant...")
        id_map = _recover_id_map(qdrant, chunks)

    # ---- Evaluate -------------------------------------------------------
    logger.info(f"Evaluating {len(queries)} queries...")
    per_query_results = []

    for i, query in enumerate(queries):
        result = evaluate_query(
            query=query,
            id_map=id_map,
            qdrant=qdrant,
            reranker=reranker,
            classifier=classifier,
            embedder=embedder,
            chunk_map=chunk_map,
        )
        per_query_results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  Evaluated {i+1}/{len(queries)}")

    # ---- Aggregate + report --------------------------------------------
    summary = aggregate_results(per_query_results)
    print_results(summary)

    # ---- Optional JSON output ------------------------------------------
    if args.output:
        output = {
            "summary":          summary,
            "per_query":        per_query_results,
            "corpus_size":      len(chunks),
            "query_count":      len(queries),
            "top_k":            TOP_K,
            "run_timestamp":    datetime.now(timezone.utc).isoformat(),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Full results written to {args.output}")


def _recover_id_map(qdrant: QdrantClient, chunks: List[CorpusChunk]) -> Dict[str, str]:
    text_to_corpus_id = {c.text: c.chunk_id for c in chunks}
    id_map = {}

    offset = None
    while True:
        result, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        for point in result:
            text = (point.payload or {}).get("text", "")
            if text in text_to_corpus_id:
                corpus_id = text_to_corpus_id[text]
                id_map[corpus_id] = str(point.id)
        if offset is None:
            break

    logger.info(f"Recovered {len(id_map)}/{len(chunks)} chunk mappings from Qdrant")
    return id_map


if __name__ == "__main__":
    main()
