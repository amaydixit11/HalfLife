import os
import sys
import json
import argparse
from typing import List, Dict

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from halflife import HalfLife

def main():
    parser = argparse.ArgumentParser(description="Test HalfLife temporal reranking on your own RAG results.")
    parser.add_argument("--query", type=str, required=True, help="Transition query (e.g. 'best NLP today')")
    parser.add_argument("--file", type=str, required=True, help="JSON file containing list of chunks: [{'text': '...', 'score': 0.9}, ...]")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        return

    with open(args.file, "r") as f:
        data = json.load(f)

    hl = HalfLife()
    
    # Prep chunks
    chunks = []
    for i, item in enumerate(data):
        # We assume messy text; HalfLife will infer timestamps automatically
        chunks.append({
            "id": f"input-{i}",
            "score": item.get("score", 0.9),
            "payload": {
                "text": item.get("text", ""),
                "timestamp": item.get("timestamp") # Optional, extractor handles it if missing
            }
        })

    results = hl.rerank(
        query=args.query,
        chunks=chunks,
        top_k=args.top_k
    )

    print(f"\n--- Results for: \"{args.query}\" ---\n")
    print(f"{'Rank':<5} | {'Final Score':<12} | {'Year':<6} | {'Source':<12} | {'Snippet'}")
    print("-" * 80)
    
    for i, res in enumerate(results["reranked_chunks"]):
        year = res.get("inferred_year", "None")
        src = res.get("temporal_source", "raw")
        text = res.get("text", "")[:60].replace("\n", " ")
        print(f"#{i+1:<4} | {res['final_score']:<12.3f} | {year:<6} | {src:<12} | {text}...")

if __name__ == "__main__":
    main()

# EXAMPLE INPUT FILE (sample.json):
# [
#   {"text": "BERT (2018) is the state of the art in NLP.", "score": 0.98},
#   {"text": "GPT-4 (2024) allows for reasoning beyond BERT.", "score": 0.85}
# ]
