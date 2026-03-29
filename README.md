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

### 🔍 1. Plug-and-Play Reranking
HalfLife sits between your retriever (e.g., Qdrant) and your LLM. It **re-scores and reorders chunks** before they reach the model.

### ⏳ 2. Multi-Strategy Decay
Supports modular decay functions via a central registry:
*   **Exponential**: Standard time-based decay.
*   **Piecewise**: Different decay rates for recent vs. historical windows.
*   **Learned (NEW)**: Features a pure-NumPy MLP (`DecayMLP`) that predicts the optimal $\lambda$ at ingestion time based on document type, source, and feedback.

### 🧠 3. Intent-Aware Fusion
HalfLife automatically classifies user queries into **Fresh**, **Historical**, or **Static** intents and adapts its scoring weights accordingly:
*   **Fresh Query**: Penalizes older results to surface recent breakthroughs.
*   **Historical Query**: Inverts the decay signal to pull older source documents to the top.

---

## 🏗️ Architecture

```
User Query
    ↓
Intent Classifier (Fresh vs Historical)
    ↓
Vector Retrieval (Qdrant)
    ↓
HalfLife Engine
    ├── Score Fetch (Redis-backed)
    ├── Learned λ Prediction (MLP)
    └── Intent-Aware Fusion
    ↓
Re-ranked Chunks
```

---

## 🛠️ Getting Started (Developer Experience)

### 1. Install via Pip (Package Mode)
HalfLife is now a standard Python package. You can install it and use the `halflife` CLI:

```bash
git clone https://github.com/yourusername/halflife.git
pip install -e .
```

### 2. Launch Services
Infrastructure is managed via Docker:
```bash
docker-compose up -d
```

### 3. Unified CLI
Use the `halflife` command for all common tasks:
```bash
# Run the end-to-end quickstart
halflife quickstart

# Start the API server
halflife serve --port 8000

# Run evaluation benchmarks
halflife benchmark --output results.json
```

---

## 🧪 Evaluation & Rigour: The Decoy Mechanism

To ensure HalfLife's effectiveness, we built a **108-chunk synthetic corpus** containing "Decoys". For every relevant chunk, there is a decoy with **identical text but a different timestamp**. 

Because their embeddings are identical, standard cosine similarity cannot separate them. Only HalfLife's temporal engine can correctly surface the right chunk, providing a rigorous test for your RAG pipeline's time-awareness.

---

## 🧬 Learned Decay Workflow

1.  **Collect Baseline**: Run `halflife benchmark --output run_001.json`.
2.  **Train the MLP**: Run `halflife train --results run_001.json`.
3.  **Deploy**: The engine automatically loads `decay_mlp.npz` and starts predicting $\lambda$ for all new ingested chunks.

---

## 🧩 Status & Roadmap
*   [x] **Phase 1**: Core Decay Engine & Redis Metadata Store.
*   [x] **Phase 2**: Intent-Aware Fusion & Historical Inversion.
*   [x] **Phase 3**: Learned Decay MLP & Benchmark Harness.
*   [ ] **Phase 4**: Event-Driven Fact Supersession (In Progress).
*   [ ] **Phase 5**: Multi-Vector Store SDKs (Pinecone, Weaviate).

---

## 📄 License & Contributing
MIT License. Contributions are welcome for new decay functions and integration plugins!
