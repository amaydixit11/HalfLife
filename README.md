# HalfLife

**Temporal-Aware Chunk Re-Ranking Engine for Retrieval-Augmented Generation (RAG)**

HalfLife is a plug-and-play middleware that enhances any RAG pipeline by re-ranking retrieved chunks using **temporal signals**, **decay functions**, and **multi-factor scoring**.

Instead of relying solely on semantic similarity, HalfLife introduces a **time-aware ranking layer** that improves freshness, relevance, and contextual correctness in generated responses.

---

## ✨ Why HalfLife?

Traditional RAG systems rank documents using:

```
relevance ≈ semantic similarity(query, document)
```

HalfLife extends this to:

```
relevance = f(semantic_similarity, temporal_decay, trust, priors)
```

This enables:

* Better handling of **time-sensitive queries**
* Reduced reliance on **outdated information**
* Improved **context diversity across time**
* More **robust and explainable retrieval pipelines**

---

## 🧠 Core Idea

HalfLife sits between your retriever (e.g., Qdrant) and your LLM:

```
Retriever → HalfLife → LLM
```

It **re-scores and reorders chunks** before they are passed into the model.

---

## ⚙️ Core Features

### 🔍 1. Chunk Re-Ranking Engine

Drop-in function:

```python
results = engine.rerank(
    query="latest transformer architectures",
    chunks=[
        {"id": "1", "score": 0.82},
        {"id": "2", "score": 0.78}
    ],
    top_k=10
)
```

Returns:

```json
[
  {
    "id": "1",
    "final_score": 0.91,
    "vector_score": 0.82,
    "temporal_score": 0.76,
    "trust_score": 0.6
  }
]
```

---

### ⏳ 2. Temporal Decay Engine

Supports multiple decay strategies:

* **Exponential decay**
* **Piecewise decay**
* **(Planned) Learned decay**

Example:

```
decay(Δt) = e^(-λΔt)
```

Each chunk can have its own decay configuration.

---

### 🗂️ 3. Chunk-Level Metadata

Each chunk stores:

```json
{
  "chunk_id": "abc123",
  "timestamp": "2024-01-01T00:00:00",
  "decay_type": "exponential",
  "decay_params": {"lambda": 0.000001},
  "trust_score": 0.8
}
```

Stored in a fast-access store (Redis).

---

### ⚖️ 4. Multi-Factor Scoring

Final score is computed as:

```
final_score = α * vector_score + β * temporal_score + γ * trust_score
```

Weights are configurable and will later support **adaptive tuning**.

---

### 🔌 5. Drop-in Middleware API

HalfLife integrates with any vector database:

* Qdrant
* Pinecone
* Weaviate
* FAISS
* pgvector

Example flow:

```python
results = qdrant.search(query_embedding)

reranked = engine.rerank(
    query=query,
    chunks=results
)
```

---

### ⚡ 6. Low-Latency Design

* Redis-backed metadata
* No heavy recomputation at query time
* Designed for real-time inference pipelines

---

## 🏗️ Architecture

```
User Query
    ↓
Vector Retrieval (Qdrant)
    ↓
HalfLife Engine
    ├── Temporal Scoring (Decay)
    ├── Metadata Fetch (Redis)
    ├── Score Fusion
    ↓
Re-ranked Chunks
    ↓
LLM
```

---

## 📁 Project Structure

```
temporal-rag-engine/
├── engine/
│   ├── decay/          # Decay functions (exponential, piecewise, etc.)
│   ├── classifier/     # Document-type classification (future)
│   ├── store/          # Redis metadata store
│   ├── fusion/         # Reranking logic
│   ├── feedback/       # Online updates (future)
│   ├── events/         # Event-driven invalidation (future)
│   └── ingestion/      # Data ingestion pipeline
│
├── api/                # FastAPI interface
├── tests/
└── docker-compose.yml
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/halflife.git
cd halflife
```

---

### 2. Start services

```bash
docker-compose up -d
```

Services:

* Qdrant
* Redis
* FastAPI server

---

### 3. Run API

```bash
uvicorn api.main:app --reload
```

---

### 4. Example Request

```bash
POST /rerank
```

```json
{
  "query": "latest advancements in GNNs",
  "chunks": [
    {"id": "1", "score": 0.85},
    {"id": "2", "score": 0.80}
  ],
  "top_k": 5
}
```

---

## 🧪 Evaluation (Planned)

HalfLife introduces new evaluation dimensions:

* **Temporal Relevance**
* **Freshness Sensitivity**
* **Temporal Diversity**
* **Groundedness over time**

---

## 🧩 Roadmap

### Phase 1 (MVP)

* [x] Decay engine
* [x] Redis metadata store
* [x] Reranking API
* [x] Qdrant integration

### Phase 2

* [ ] Query temporal intent classifier
* [ ] Adaptive weighting (α, β, γ)
* [ ] Feedback-driven decay tuning

### Phase 3

* [ ] Event-driven updates
* [ ] Temporal knowledge graphs
* [ ] Learned decay models

---

## 🔬 Research Directions

HalfLife opens up exploration in:

* Temporal ranking in IR systems
* Learned decay functions
* Time-aware embeddings
* Adaptive retrieval policies

---

## 🤝 Contributing

Contributions are welcome!

Ideas:

* New decay functions
* Better fusion strategies
* Evaluation benchmarks
* Dataset integrations

---

## 📄 License

MIT License

---

## 🧠 Philosophy

HalfLife is built on a simple idea:

> Not all knowledge ages the same way.

Understanding *when* information matters is as important as understanding *what* it says.

---

## ⭐ Support

If you find this project useful, consider starring the repo.

---
