# HalfLife

**Temporal decay–aware reranking of retrieved chunks via relevance half-life modeling**

---

## 🧠 Overview

**HalfLife** is a middleware layer for retrieval-augmented systems that models **relevance as a time-dependent signal**. Instead of treating retrieved chunks as equally valid regardless of age, HalfLife applies **per-chunk temporal decay functions** to dynamically adjust their importance during reranking.

Modern RAG pipelines rely heavily on semantic similarity, but ignore a critical dimension: **time**. In real-world systems, information evolves, becomes outdated, or gains importance depending on context. HalfLife introduces **half-life parameterization** to explicitly model how relevance changes over time.

---

## ⚡ Key Idea

Traditional retrieval assumes:

> relevance = semantic similarity

HalfLife extends this to:

> relevance = f(semantic similarity, temporal decay, trust signals)

Each chunk is assigned a **half-life**, defining how quickly its relevance decays. This enables:

* Fast-decaying content (e.g., news)
* Slow-decaying content (e.g., documentation)
* Persistent knowledge (e.g., fundamentals)

---

## 🏗️ Architecture

```
User Query
   ↓
Vector Retriever (Qdrant / FAISS / etc.)
   ↓
Retrieved Chunks (with similarity scores)
   ↓
HalfLife Middleware
   ├── Temporal Decay Engine
   ├── Metadata Store (timestamps, half-life params)
   ├── Fusion Layer (multi-signal scoring)
   ↓
Re-ranked Chunks
   ↓
LLM / Downstream System
```

---

## 🔌 Core Features

### ⏳ Temporal Decay Modeling

* Explicit **half-life functions** per chunk
* Supports:

  * Exponential decay
  * Piecewise decay
  * Custom / learned decay

---

### 🧩 Chunk-Level Scoring

* Operates at **fine granularity**
* Each chunk has:

  * timestamp
  * decay parameters
  * optional trust score

---

### ⚖️ Multi-Signal Fusion

Combines:

* semantic similarity (from retriever)
* temporal decay score
* trust / prior signals

```python
final_score = α * similarity + β * decay + γ * trust
```

---

### 🔄 Drop-in Middleware

* Works with any vector store:

  * Qdrant
  * Pinecone
  * Weaviate
  * FAISS
* No changes to existing retrieval pipelines

---

### ⚡ Low Latency

* Redis-backed metadata store
* Precomputed parameters
* Designed for real-time reranking

---

## 🧪 Example Usage

```python
results = qdrant.search(query_embedding)

reranked = engine.rerank(
    query="latest transformer architectures",
    chunks=results,
    top_k=5
)
```

### Output

```json
[
  {
    "id": "chunk_1",
    "final_score": 0.91,
    "vector_score": 0.82,
    "temporal_score": 0.76,
    "trust_score": 0.65
  }
]
```

---

## 🧠 Decay Modeling

### Half-Life Definition

Each chunk is assigned:

```python
chunk.half_life = 7 * 24 * 3600  # seconds
```

Decay function:

[
score = e^{-\lambda t}, \quad \lambda = \frac{\ln 2}{half_life}
]

---

### Interpretation

| Content Type | Half-Life    | Behavior       |
| ------------ | ------------ | -------------- |
| News         | Hours–Days   | Fast decay     |
| Blogs        | Weeks        | Moderate decay |
| Docs         | Months–Years | Slow decay     |
| Fundamentals | Years        | Near-static    |

---

## 🗂️ Metadata Schema

Each chunk stores:

```json
{
  "chunk_id": "abc123",
  "timestamp": "2024-01-01T00:00:00",
  "half_life": 604800,
  "decay_type": "exponential",
  "trust_score": 0.8,
  "doc_type": "news"
}
```

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/halflife
cd halflife
pip install -r requirements.txt
```

---

## 🐳 Running with Docker

```bash
docker-compose up
```

Includes:

* Qdrant
* Redis
* FastAPI service

---

## 🚀 API

### `POST /rerank`

#### Request

```json
{
  "query": "...",
  "chunks": [...],
  "top_k": 10
}
```

#### Response

```json
[
  {
    "id": "...",
    "final_score": 0.87,
    "vector_score": 0.91,
    "temporal_score": 0.72
  }
]
```

---

## 🧪 Evaluation

HalfLife introduces new evaluation dimensions:

* **Temporal Relevance**
  Does the system prioritize appropriate time-sensitive content?

* **Freshness Sensitivity**
  Does ranking adapt to query intent (latest vs general)?

* **Temporal Diversity**
  Does context include multiple time horizons?

---

## 🔮 Roadmap

* [ ] Query intent classifier (fresh vs static)
* [ ] Learned decay functions (neural parameterization)
* [ ] Feedback-driven half-life tuning
* [ ] Temporal knowledge graphs
* [ ] Evaluation framework + benchmarks

---

## 🧠 Design Principles

* **Time is a first-class signal**
* **Relevance is not static**
* **Retrieval needs correction, not replacement**
* **Systems should be modular and composable**

---

## 🤝 Contributing

Contributions are welcome. Please open issues or submit PRs for:

* new decay functions
* evaluation methods
* integrations

---

## 📄 License

MIT License

---

## 📌 Summary

HalfLife redefines retrieval by introducing **time-aware relevance modeling**. By treating relevance as a decaying signal rather than a static score, it enables more accurate, adaptive, and context-aware reranking in modern RAG systems.

> Relevance decays. Your retrieval should too.
