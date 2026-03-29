# HalfLife Middleware Usage

This guide explores how to integrate HalfLife into an existing Retrieval-Augmented Generation (RAG) pipeline to introduce temporal awareness.

## 🔗 The Middleware Path

In a standard RAG flow, the chain is:
`User Query` → `Vector Search` → `Prompt` → `LLM Generation`.

With **HalfLife**, the chain becomes:
`User Query` → `Vector Search` → **`HalfLife Rerank`** → `Prompt` → `LLM Generation`.

---

## 🛠 Integrating the API

HalfLife provides a simple REST API. The most important endpoint is `/rerank`.

### 1. Simple Rerank Request
Post-vector search, send your candidates to HalfLife:

```bash
curl -X POST "http://localhost:8000/rerank" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "latest advancements in GNNs",
       "chunks": [
         {"id": "c1", "score": 0.85, "payload": {"timestamp": "2024-03-20T10:00:00Z"}},
         {"id": "c2", "score": 0.82, "payload": {"timestamp": "2021-06-15T09:00:00Z"}}
       ],
       "top_k": 5
     }'
```

The `payload` must contain a `timestamp` string in ISO 8601 format.

### 2. Manual Weight Overrides
While HalfLife classifies query intent automatically (e.g., "latest" triggers high temporal weight), you can override weights manually:

```json
{
  "query": "stable diffusion origins",
  "chunks": [...],
  "weights": {
    "vector": 0.4,
    "temporal": 0.5,
    "trust": 0.1
  }
}
```

---

## 🧪 Integration Example (Python SDK-like)

```python
import requests

# 1. Search in your vector DB (e.g. Qdrant)
search_results = qdrant.query_points(
    collection_name="docs",
    query=query_vector,
    limit=100
).points

# 2. Format for HalfLife
halflife_input = [
    {
        "id":    str(r.id),
        "score": r.score,
        "payload": r.payload
    } 
    for r in search_results
]

# 3. Rerank via HalfLife Middleware
response = requests.post(
    "http://localhost:8000/rerank",
    json={
        "query": query_text,
        "chunks": halflife_input,
        "top_k": 10
    }
)
reranked_chunks = response.json()["reranked_chunks"]

# 4. Pass top results to your LLM
context = "\n".join([c["text"] for c in reranked_chunks])
# ... generate response
```

---

## 👂 Feeding Back Groundedness

HalfLife improves over time through feedback. Let it know which chunks were actually used by the LLM:

```bash
POST /feedback
{
  "chunk_id": "c1",
  "was_useful": true
}
```
This increments the `trust_score` in Redis and may influence future rankings.
