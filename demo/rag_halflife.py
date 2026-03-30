from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from halflife import HalfLife
from openai import OpenAI
import os

# ----------------------------
# Setup
# ----------------------------

client = QdrantClient(":memory:")  # local in-memory DB
model = SentenceTransformer("all-MiniLM-L6-v2")
# Use the same client for HalfLife
hl = HalfLife(qdrant_client=client)

# Check for API Key
if not os.environ.get("OPENAI_API_KEY"):
    print("\n⚠️  Warning: OPENAI_API_KEY not found. LLM generation will be skipped.")
    openai_client = None
else:
    openai_client = OpenAI()

COLLECTION = "docs"

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

# ----------------------------
# Ingest data (temporal conflict)
# ----------------------------

documents = [
    {
        "id": 1,
        "text": "BERT (2019) is the state-of-the-art model for NLP tasks.",
        "timestamp": "2019"
    },
    {
        "id": 2,
        "text": "GPT-4 (2024) is currently the most advanced language model.",
        "timestamp": "2024"
    }
]

print("\n--- INGESTING DATA ---")
for doc in documents:
    emb = model.encode(doc["text"]).tolist()

    # Manual ingest to Qdrant (for baseline)
    client.upsert(
        collection_name=COLLECTION,
        points=[
            models.PointStruct(
                id=doc["id"],
                vector=emb,
                payload={
                    "text": doc["text"],
                    "timestamp": f"{doc['timestamp']}-01-01T00:00:00Z"
                }
            )
        ]
    )

    # Also register in HalfLife (it uses its own collection 'halflife_chunks')
    # Note: normally HalfLife manages everything, but we're showing integration
    hl.ingest(
        text=doc["text"],
        timestamp=doc["timestamp"]
    )

# ----------------------------
# Query
# ----------------------------

query = "What is the best NLP model today?"
print(f"🔍 Query: \"{query}\"")

query_vec = model.encode(query).tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,
    limit=5,
    with_payload=True
).points

# Format for HalfLife
chunks = [
    {
        "id": str(r.id),
        "score": r.score,
        "payload": r.payload
    }
    for r in results
]

# ----------------------------
# 🔥 Apply HalfLife
# ----------------------------

reranked = hl.rerank(query=query, chunks=chunks, top_k=2)

print("\n--- BASELINE (Standard Vector Search) ---")
for r in chunks:
    print(f"📄 {r['payload']['text']} | score: {round(r['score'], 3)}")

print("\n--- HALFLIFE (Temporal Fusion) ---")
for r in reranked:
    print(f"✅ {r['text']} | final score: {round(r['final_score'], 3)}")

# ----------------------------
# LLM Generation
# ----------------------------

if openai_client:
    context = "\n".join([r["text"] for r in reranked])

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer the user question using ONLY the provided context. If the context contains multiple facts about the same entity, prefer the most recent one."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )

        print("\n--- FINAL LLM ANSWER ---")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"\n⚠️  LLM Step failed (Likely invalid API Key): {e}")
else:
    print("\n⚠️  LLM Step skipped (no API key).")