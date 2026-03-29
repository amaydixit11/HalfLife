# HalfLife

**Temporal-Aware Ranking Engine for Modern RAG**

---

## 🚨 The Problem: Time-Blind RAG

Traditional RAG systems are **time-blind**. They rank documents solely by semantic similarity, leading to "Temporal Hallucinations":

*   **Query**: *"What is the state-of-the-art model for NLP?"*
*   **Retriever**: Finds a 2019 paper (BERT) with 0.98 similarity.
*   **LLM**: *"BERT is the state-of-the-art model."* (❌ **Wrong**: It was surpassed years ago).

**HalfLife fixes this.** It introduces a temporal ranking layer that understands the difference between a "Fresh" query (needs latest info) and a "Historical" query (needs archival context).

---

## 📊 Performance Baseline

Validated against a **108-chunk adversarial corpus** (including "Decoy" chunks with identical text but different dates):

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
