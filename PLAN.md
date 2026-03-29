# HalfLife Implementation Plan

HalfLife is a temporal-aware re-ranking engine designed to sit as a middleware between vector retrievers and LLMs. This plan outlines the step-by-step process to build the core engine, integrate it with Qdrant and Redis, and implement adaptive temporal scoring.

## 🏗 Architecture Overview

HalfLife operates on the following core components:
1.  **Decay Engine**: Computes temporal scores based on various mathematical families (Exponential, Piecewise, etc.).
2.  **Metadata Store (Redis)**: High-performance storage for chunk-level temporal metadata.
3.  **Fusion Layer**: Combines semantic similarity (from retriever) with temporal scores and priors.
4.  **API Layer (FastAPI)**: The middleware interface for external RAG systems.
5.  **Ingestion Pipeline**: Tooling to populate Qdrant and Redis with the necessary signals.

---

## 📅 Roadmap

### Phase 1: Foundation & Core Math (MVP)
*Goal: Establish the pluggable decay architecture and basic reranking API.*

1.  **Repository Setup**: 
    - Initialize Python environment (FastAPI, Redis, Qdrant-client, NumPy).
    - Create directory structure: `engine/`, `api/`, `tests/`.
2.  **Decay Engine Implementation**:
    - Implement `BaseDecayFunction` abstract class.
    - Implement `ExponentialDecay` and `PiecewiseDecay` families.
    - Create a `DecayRegistry` for dynamic selection.
3.  **Storage Layer**:
    - Implement `RedisStore` to handle chunk metadata (ID, timestamp, decay type, params).
4.  **Fusion & Reranking**:
    - Build the `Reranker` class.
    - Implement the core scoring formula: `final_score = α * sim + β * decay(t) + γ * trust`.
5.  **API Development**:
    - Create `/rerank` endpoint in FastAPI.
    - Implement basic health checks and schema validation.

### Phase 2: Integration & Ingestion
*Goal: Connect to a real vector store and build the data pipeline.*

1.  **Qdrant Integration**:
    - Setup `docker-compose.yml` with Qdrant and Redis.
    - Implement a client wrapper for Qdrant searches.
2.  **Ingestion Pipeline**:
    - Build a script to take text chunks, timestamp them, and push to both Qdrant (vectors) and Redis (metadata).
    - Add basic document-type classification (News vs. Docs) to assign initial decay priors.
3.  **Testing & Benchmarking**:
    - Create a test suite using a sample dataset (e.g., ArXiv abstracts).
    - Measure "Freshness Bias" vs. "Semantic Precision" tradeoffs.

### Phase 3: Intelligence & Adaptation
*Goal: Make the system reactive to query intent and feedback.*

1.  **Temporal Intent Classifier**:
    - Use a small model/LLM call to classify queries (e.g., "Freshness sensitive" vs. "Time agnostic").
    - Dynamically adjust fusion weights (α, β, γ) based on intent.
2.  **Feedback Hook**:
    - Implement `/feedback` endpoint.
    - Log "Retrieved vs. Used" signals for chunks.
3.  **Online Parameter Updates**:
    - Adjust decay constants (λ) in Redis based on feedback (e.g., if an old chunk is frequently selected, switch it from Exponential to Power-law or slow down its decay).

### Phase 4: Advanced Features & Refinement
*Goal: Scale and harden the system for production-grade research.*

1.  **Event-Driven Invalidation**:
    - Implement a webhook listener for external "invalidation" events.
    - Propagate score penalties for superseded knowledge.
2.  **Temporal Consistency Checker**:
    - Add a post-generation check to detect if retrieved chunks contradict each other across different timestamps.
3.  **Dashboard/Observability**:
    - Simple UI to visualize how scores decay over time for different document classes.

---

## 🛠 Tech Stack
- **Language**: Python 3.10+
- **API**: FastAPI
- **Vector DB**: Qdrant
- **Metadata Cache**: Redis
- **Math/ML**: NumPy, Scipy, Sentence-Transformers
- **Infra**: Docker, Docker Compose

## 🚀 Execution strategy
We will proceed layer by layer, starting with the `engine/decay` math and the `engine/store` Redis integration. Every step will be verified with a unit test.
