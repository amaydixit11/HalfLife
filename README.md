# HalfLife

<img width="516" height="484" alt="halflife-removebg-preview" src="https://github.com/user-attachments/assets/8273f9e0-aa9c-4602-b3af-9c9cbd6b3e83" />

**Temporal-aware reranking for RAG pipelines. Stops your system from returning outdated answers.**

> 🚨 **RAG systems return outdated answers.**
> 
> Query: `"best NLP model today"`
> 
> ❌ Without HalfLife: **#1 BERT (2019)**
> ✅ With HalfLife: **#1 GPT-4 (2024)**
> 
> **HalfLife is a drop-in middleware layer that resolves this temporal relevance gap.** **[Explore the Integration Demo](./examples/llamaindex_halflife_demo.py)** ↓

[![PyPI](https://img.shields.io/pypi/v/halflife-rag)](https://pypi.org/project/halflife-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amaydixit11/halflife/blob/main/HalfLife_RealWorld_Demo.ipynb)

---

## The Problem

Standard RAG ranks by **semantic similarity**. It has no concept of time.

Ask *"best method for LLM fine-tuning today?"* and your vector store returns whichever paper
has the closest embedding — which is often a well-cited 2020 paper with clean, formal language,
not the 2024 paper that actually answers the question.

```
Query: "Best current approach to parameter-efficient fine-tuning?"

Baseline (cosine similarity):   #1 → 2020  "How fine can fine-tuning be?"     score: 0.614
HalfLife (temporal fusion):     #1 → 2024  "DELIFT: Data Efficient LLM Fine-Tuning"  score: 0.838
```

**That's a 4-year jump on a real corpus of 120 Arxiv papers. No changes to your retriever.**

---

## Real-World Benchmark

Evaluated on **120 real Arxiv papers** (cs.CL, parameter-efficient fine-tuning),
fetched year-by-year from 2019–2024. No synthetic data.

| Query Intent | Baseline Avg Result Age | HalfLife Avg Result Age | Δ |
|:---|:---:|:---:|:---:|
| **Fresh** ("best current...", "state-of-the-art...") | 3.9 yr | **2.4 yr** | **−1.4 yr** |
| **Static** ("explain how...", "what is...") | 4.3 yr | 4.0 yr | −0.3 yr ✅ no regression |
| **Historical** ("originally...", "early NLP...") | 5.0 yr | 5.2 yr | +0.2 yr ⚠️ marginal |

For fresh queries, HalfLife surfaces results **1.4 years more recent on average**.
For static queries, behavior is unchanged — HalfLife is invisible when it doesn't need to act.
Historical inversion is marginal on this corpus (2019–2024); deeper corpora show stronger results.

**Reproduce this yourself** — the Colab notebook fetches live Arxiv data with no setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amaydixit11/halflife/blob/main/demo/HalfLife_RealWorld_Demo.ipynb)

---

## How It Works

HalfLife sits between your vector retriever and your LLM. It re-scores each retrieved
chunk using a weighted fusion of three signals:

```
final_score = α · vector_score + β · temporal_score + γ · trust_score
```

The weights **α, β, γ** are set dynamically based on query intent:

| Detected Intent | Example keywords | α (vector) | β (temporal) | γ (trust) |
|:---|:---|:---:|:---:|:---:|
| **Fresh** | "latest", "current", "today", "SOTA" | 0.3 | 0.6 | 0.1 |
| **Historical** | "originally", "history of", "early" | 0.4 | 0.5 | 0.1 |
| **Static** | "explain", "what is", "how does" | 0.8 | 0.1 | 0.1 |

For historical queries, the temporal signal is **inverted** — older chunks score higher.

```
User Query
    ↓
QueryIntentClassifier  →  sets α, β, γ
    ↓
Vector Retrieval (Qdrant / any store)
    ↓
HalfLife Reranker
    ├── Temporal decay per chunk  (exponential / piecewise / learned MLP)
    ├── Redis metadata cache      (< 1ms overhead on cached chunks)
    └── Min-Max fusion
    ↓
Re-ranked chunks  →  LLM
```

---

## Quickstart

### No infrastructure (2 minutes)

```python
pip install halflife-rag
```

```python
from halflife import HalfLife

hl = HalfLife()

# Drop into your existing retrieval results
results = qdrant.search(query=query)
reranked = hl.rerank(query=query, chunks=results, top_k=5)

for chunk in reranked:
    print(f"[{chunk['final_score']:.3f}] ({chunk['timestamp'][:4]}) {chunk['text'][:80]}")
```

### With Docker (full feature set including Redis cache)

```bash
git clone https://github.com/amaydixit11/halflife.git
cd halflife
docker-compose up -d   # starts Qdrant + Redis
pip install -e .
halflife demo          # runs the adversarial demo
```

---

## LangChain Integration

```python
from langchain.retrievers import ContextualCompressionRetriever
from halflife.integrations.langchain import HalfLifeReranker

retriever = ContextualCompressionRetriever(
    base_compressor=HalfLifeReranker(top_k=5),
    base_retriever=your_existing_retriever   # unchanged
)

docs = retriever.get_relevant_documents("Latest approach to LLM fine-tuning?")
# → surfaces 2024 papers instead of 2020 papers
```

## LlamaIndex Integration

```python
from halflife.integrations.llamaindex import HalfLifePostprocessor

query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[HalfLifePostprocessor(top_n=5)]
)

response = query_engine.query("What is the latest React version?")
```

---

## Decay Strategies

Three decay functions are available, selectable per document at ingestion time:

| Strategy | Formula | Best for |
|:---|:---|:---|
| **Exponential** | `e^(−λΔt)` | News, fast-moving fields, software versions |
| **Piecewise** | Step function (1.0 → 0.7 → 0.3) | Documentation, compliance, versioned specs |
| **Learned** | MLP-predicted λ from doc features | Mixed corpora with feedback signal |

The learned decay MLP runs **at ingestion time only** — zero ML inference at query time.

---

## Experimental Features

- **Learned Decay MLP** — pure-NumPy, predicts per-chunk λ from doc type, source domain, text length, and feedback ratio. Train on your own benchmark results with `halflife train`.
- **Feedback Loop** — adaptive λ tuning via EMA from user "was this useful?" signals.
- **Event Bus** — hard/soft invalidation when a fact is superseded (e.g. a retraction, a product discontinuation).

---

## CLI

```bash
halflife demo                              # adversarial demo (no Docker)
halflife quickstart                        # end-to-end ingest → query → rerank
halflife benchmark --output results.json   # nDCG / MRR / temporal freshness
halflife evaluate --ablation               # compare exponential / linear / learned / baseline
halflife train --results results.json      # train the decay MLP
halflife serve --port 8000                 # start the FastAPI middleware
```

---

## Roadmap

- [x] Phase 1: Core decay engine + Redis metadata store
- [x] Phase 2: Intent-aware fusion + historical inversion
- [x] Phase 3: Learned decay MLP + benchmark harness
- [x] Phase 3.5: Real-world Arxiv validation + Colab notebook
- [ ] Phase 4: Event-driven fact supersession
- [ ] Phase 5: Pinecone, Weaviate, Chroma integrations
- [ ] Phase 6: Transformer-based intent classifier (replace keyword matching)

---

## License & Contributing

MIT License. Contributions welcome — especially new decay functions, vector store integrations,
and domain-specific benchmark datasets.

If you're using HalfLife in a project, open an issue and let us know — we'd like to link to it.
