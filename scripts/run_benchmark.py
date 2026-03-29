from qdrant_client import QdrantClient
from engine.store.redis_store import RedisStore
from engine.fusion.reranker import Reranker
from scripts.benchmark import HalfLifeBenchmark
import numpy as np

# Mocking connection for the script example
def run():
    qdrant = QdrantClient(url="http://localhost:6333")
    redis = RedisStore(url="redis://localhost:6379")
    reranker = Reranker(redis)
    
    benchmark = HalfLifeBenchmark(reranker, qdrant)
    
    # In a real scenario, we'd have a list of labeled queries.
    # Below is the placeholder for the user to run their benchmark.
    queries = [] 
    
    print("Starting HalfLife Benchmark Harness...")
    # benchmark.run_benchmark(queries)
    print("Note: Ingest data using engine/ingestion/pipeline.py before running.")

if __name__ == "__main__":
    run()
