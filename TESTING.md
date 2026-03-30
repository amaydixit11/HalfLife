# 🧪 Testing & Validation Guide

This guide outlines how to verify HalfLife's temporal reranking logic across three layers: Proof, Scientific IR Metrics, and Unit Integrity.

## 🏗️ 0. Prerequisites
Ensure your infrastructure is running:
```bash
docker-compose up -d qdrant redis
```

## 🎭 1. The Proof (Adversarial Sweep)
Run the E2E demo to see HalfLife defeat the **"Authority Trap"** across 5 domains (AI, Web, Python, Consensus, Data).
```bash
halflife demo
```
**Goal**: Verify that GPT-5 (2026) correctly overrides BERT (2018) when the query includes "today" or "latest".

## 📊 2. Scientific Benchmarking (IR Metrics)
Run the **Temporal Confusion Benchmark (TCB)** to aggregate reproducible metrics (nDCG, MRR, TF).
```bash
halflife benchmark
```
*   **nDCG@3**: Normalized Discounted Cumulative Gain.
*   **MRR**: Mean Reciprocal Rank.
*   **TF (Temporal Freshness)**: % of results that are from the target era.

## 🔬 3. Research Ablation (Evaluate)
Compare HalfLife against 4 different temporal variants (Exponential, Linear, Learned, Baseline).
```bash
halflife evaluate --ablation
```

## 🛡️ 4. Unit & Integration Tests
Verify the core engine and store logic.
```bash
pytest
```

---

### 🔗 Integration Verification
To verify the **LangChain** or **LlamaIndex** SDKs, see the snippets in [README.md](./README.md).
