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
    # Ensure Docker (Qdrant + Redis) is running!
    ingestor   = HalfLifeIngestor()
    store      = RedisStore()
    reranker   = Reranker(store)
    classifier = QueryIntentClassifier()
    qdrant     = QdrantClient(url="http://localhost:6333")

    now = datetime.now(timezone.utc)

    # 2. Ingest Sample Data
    print("📥 Ingesting sample chunks...")
    
    # Chunk A: Very Recent (Latest News)
    id_a = ingestor.ingest(
        text="Breakthrough in Quantum Computing announced today, March 2026. Researchers achieved 1M qubits.",
        timestamp=now,
        source_domain="science-daily",
        doc_type="news"
    )

    # Chunk B: Old but Foundational (History)
    id_b = ingestor.ingest(
        text="The first programmable computer, ENIAC, was built in 1945. It marked the beginning of modern computing.",
        timestamp=now - timedelta(days=365*80), # 1946 approx
        source_domain="history-archives",
        doc_type="research"
    )
    print(f"   Done. (IDs: {id_a[:8]}..., {id_b[:8]}...)\n")

    # 3. Create a Helper for Reranking
    def search_and_rerank(query_text: str):
        print(f"🔍 Query: \"{query_text}\"")
        
        # --- Standard Vector Search (Qdrant) ---
        # Note: In a real app, you'd embed the query text. 
        # Here we just use the ingestor's model for consistency.
        q_vector = ingestor.model.encode(query_text).tolist()
        results = qdrant.query_points(
            collection_name="halflife_chunks",
            query=q_vector,
            limit=5,
            with_payload=True
        ).points
        
        chunks_for_rerank = [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

        # --- HalfLife Rerank ---
        classification = classifier.classify(query_text)
        reranked = reranker.rerank(
            query=query_text,
            chunks=chunks_for_rerank,
            top_k=5,
            intent=classification["intent"],
            weights=classification["weights"]
        )

        print(f"   Detected Intent: {classification['intent']}")
        print(f"   Results (Top 1):")
        top = reranked["reranked_chunks"][0]
        print(f"   -> ID: {top['id'][:8]} | Final Score: {top['final_score']:.4f} {'(Cache Hit)' if top.get('cache_hit') else ''}")
        v_score = f"{top['vector_score']:.2f}" if top['vector_score'] is not None else "---"
        t_score = f"{top['temporal_score']:.2f}" if top['temporal_score'] is not None else "CACHED"
        print(f"   -> TS: {top['timestamp']} | Base: {v_score} | Time: {t_score}")

        print("-" * 50)

    # 4. Demonstrate Different Intents
    
    # Fresh Query -> Should favor Chunk A (Quantum Computing)
    search_and_rerank("What is the latest advancement in computing?")

    # Historical Query -> Should favor Chunk B (ENIAC)
    search_and_rerank("Tell me about the history of computers.")

    print("\n✅ Quickstart Finished! HalfLife is successfully reranking based on time.")

if __name__ == "__main__":
    run_quickstart()
