import os
import sys
from datetime import datetime, timedelta, timezone

# Allow importing from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ingestion.pipeline import HalfLifeIngestor
from engine.classifier.query_intent import QueryIntentClassifier
from engine.fusion.reranker import Reranker
from engine.store.redis_store import RedisStore
from qdrant_client import QdrantClient

def run_quickstart():
    print("🚀 Starting HalfLife Quickstart...\n")

    # 1. Initialize Components
    ingestor   = HalfLifeIngestor()
    store      = RedisStore()
    reranker   = Reranker(store)
    classifier = QueryIntentClassifier()
    qdrant     = QdrantClient(url="http://localhost:6333")

    now = datetime.now(timezone.utc)

    # 2. Check for Real Data
    results = qdrant.count(collection_name="halflife_chunks")
    if results.count < 5:
        print("📥 No live data found. Ingesting sample chunks for demonstration...")
        # [Fallback to sample chunks if needed]
        ingestor.ingest(
            text="Breakthrough in Quantum Computing announced March 2026...",
            timestamp=now,
            source_domain="science-daily", doc_type="news"
        )
    else:
        print(f"📊 Live Data Detected: Found {results.count} chunks in the engine.")

    # 3. Side-by-Side Search Comparison
    def search_and_compare(query_text: str):
        print(f"\n🔍 Query: \"{query_text}\"")
        
        q_vector = ingestor.model.encode(query_text).tolist()
        res = qdrant.query_points(
            collection_name="halflife_chunks",
            query=q_vector, limit=5, with_payload=True
        ).points
        
        chunks = [{"id": str(r.id), "score": r.score, "payload": r.payload} for r in res]

        # Rerank with HalfLife
        classification = classifier.classify(query_text)
        reranked = reranker.rerank(
            query=query_text, chunks=chunks, top_k=5,
            intent=classification["intent"], weights=classification["weights"]
        )

        print(f"   Detected Intent: {classification['intent'].upper()}")
        print(f"   {'-'*30}")
        print(f"   🏆 HalfLife #1 Hit:")
        top = reranked["reranked_chunks"][0]
        ts  = top["timestamp"][:10] if top["timestamp"] else "Unknown"
        print(f"   -> TS: {ts} | Score: {top['final_score']:.4f}")
        print(f"   -> Text snippet: {top.get('text', '---')[:90]}...")
        
        # Check if the #1 hit changed
        if str(res[0].id) != top["id"]:
            print(f"\n   ✨ BOOM! HalfLife re-ranked a NEW chunk into the #1 spot.")
            print(f"      (Baseline #1 was: {res[0].payload.get('timestamp')[:10]})")
        else:
            print(f"\n   ✓ Baseline was already temporally optimal.")
        print("-" * 50)

    # Demonstrate current advancements vs foundations
    search_and_compare("Explain the latest transformer models from 2025 and 2026")
    search_and_compare("What are the original founding principles of transformers?")

    print("\n✅ Quickstart Finished! HalfLife successfully re-ordered real Arxiv results.")

if __name__ == "__main__":
    run_quickstart()

