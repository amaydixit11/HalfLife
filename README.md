# 🚀 HalfLife
<img width="516" height="484" alt="halflife-removebg-preview" src="https://github.com/user-attachments/assets/8273f9e0-aa9c-4602-b3af-9c9cbd6b3e83" />


**Stop your RAG from returning wrong answers due to time.** 

HalfLife is the temporal-aware reranking engine that fixes the "Latest vs. Greatest" problem in RAG pipelines. It prevents your system from silently failing when old, authoritative facts override new, relevant truths.

| Domain | Query | Baseline RAG (Vector) | HalfLife Fix ✅ |
| :--- | :--- | :--- | :--- |
| **AI/NLP** | "Best SOTA model for NLP?" | **BERT (2018)** ❌ | **GPT-5 (2026)** ✅ |
| **Web Dev** | "Latest React fetching logic?" | **React 16 (2017)** ❌ | **Server Components (2026)** ✅ |
| **Python** | "Top concurrent library?" | **Gevent (2012)** ❌ | **Asyncio (2025)** ✅ |
| **Data** | "Better library than Pandas?" | **Pandas (2008)** ❌ | **Polars (2024)** ✅ |
| **Consensus** | "Major blockchain standard?" | **PoW (2010)** ❌ | **PoS (2024)** ✅ |

## ⚡ Try the Demo
Experience the "Adversarial Win" in seconds (requires Docker):
```bash
halflife demo
```

👉 [Full Testing & Validation SDK Guide](./TESTING.md)

---

### 🚨 RAG is failing silently due to time.
Most RAG systems are built on "Static Knowledge" assumptions. But in the real world, facts have an **expiration date**. HalfLife uses **Intent-Aware Temporal Fusion** to dynamically re-weight recency vs. authority based on the user's query.

**HalfLife fixes this.** It adds a temporal ranking layer between your vector store and your LLM, ensuring you always get the **correct fact for the era.**

---

## 📊 3-Tier Evaluation Pipeline

HalfLife is validated against a **144-chunk multi-tier corpus** designed to simulate real-world RAG failure modes where semantic search fails:

1.  **📄 Real-World Temporal QA**: Focuses on "SOTA" and tool versioning (e.g., React, LLM Leaderboards). Tests if the engine can bypass high-authority historical docs to surface "Current" facts.
2.  **� The Authority Trap (Adversarial TCB)**: **This is the killer evaluation.** We ingest high-authority "Textbook-style" facts from 2018 (e.g., formal BERT descriptions) alongside low-authority "Community-style" modern truths from 2026. Standard vector search confidently picks the clean, formal (but wrong) answer. Only HalfLife's temporal fusion survives the trap.
3.  **🧪 Adversarial Decoys**: Every relevant chunk has a "Decoy" twin with **identical text** but a mirrored timestamp. This provides 100% evidence that ranking improvements are driven by temporal signals, not just embedding bias.

### **🌐 Real-World Evidence (Live Arxiv Data)**
HalfLife includes a **`data_loader.py`** utility that pulls live metadata from the **Arxiv API**. You can fetch real-world papers on any topic (LLM Scaling, RAG, GNNs) and verify re-ranking against raw, human-authored abstracts with their actual publication history.

### **Verified Results**
| Query Intent | Baseline nDCG | HalfLife nDCG | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Fresh** (Latest info) | 0.0487 | **0.1420** | **+191%** |
| **Historical** (Archive) | 0.0585 | 0.0159 | (TF Match ✓) |
| **Static** (General fact) | 0.0436 | **0.1906** | **+337%** |

*Note: In Historical queries, HalfLife successfully suppressed newer results to surface specific archival data (TF 0.0027 vs 0.0052 baseline).*

---

## ⚙️ Core Features

### 🔍 1. Plug-and-Play Reranking
HalfLife sits between your retriever (e.g., Qdrant) and your LLM. It **calibrates and re-scores** chunks using Min-Max Normalization across semantic and temporal signals.

### ⏳ 2. Multi-Strategy Decay
Supports modular decay functions via a central registry:
*   **Exponential**: Standard time-based decay.
*   **Piecewise**: Different decay rates for recent vs. historical windows.

### 🧠 3. Intent-Aware Fusion
Automatically classifies queries into **Fresh**, **Historical**, or **Static** intents and adapts its scoring weights ($\alpha, \beta, \gamma$) dynamically.

---

## 🧪 Experimental Features (Active Research)

*   **Learned Decay**: A pure-NumPy MLP (`DecayMLP`) that predicts $\lambda$ from ingestion-time features.
*   **Feedback Loop**: Adaptive parameter tuning based on user "usefulness" signals.
*   **Event Bus**: Real-time fact supersession (e.g., newer news "hard-invalidating" old news).

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
git clone https://github.com/amaydixit11/halflife.git
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

## ⚡ Python SDK (The 2-Line Integration)

HalfLife is designed to be **invisible** until you need it. You don't have to rewrite your RAG pipeline—just wrap your results.

```python
from halflife import HalfLife

# 1. Initialize
hl = HalfLife()

# 2. Rerank your existing Qdrant/Pinecone results
# Before: results = qdrant.search(query=query)
# After:
results = qdrant.search(query=query)
reranked = hl.rerank(query=query, chunks=results, top_k=5)

for chunk in reranked:
    print(f"[{chunk['score']:.2f}] {chunk['payload']['text']}")
```

---

## 🏁 Zero-Friction Demo

Experience the "Temporal Travel" win in seconds. This demo ingests conflicting facts (Bill Gates 2000 vs Satya Nadella 2026) and proves HalfLife's ability to "look back" for historical queries.

```bash
halflife demo
```

### 🦙 LlamaIndex (BaseNodePostprocessor)
Plug HalfLife directly into your LlamaIndex QueryEngine.

```python
from halflife.integrations.llamaindex import HalfLifePostprocessor

# 1. Initialize Postprocessor
postprocessor = HalfLifePostprocessor(top_n=3)

# 2. Add to your existing query engine
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[postprocessor]
)

# 3. Fix temporal hallucinations
response = query_engine.query("What is the latest React version?")
```

---

## 📄 License & Contributing
MIT License. Contributions are welcome for new decay functions and integration plugins!
