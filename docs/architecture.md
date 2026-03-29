# HalfLife Architecture

HalfLife is a temporal-aware re-ranking middleware designed for Retrieval-Augmented Generation (RAG). It enhances retrieval quality by fusing semantic similarity with temporal signals (freshness, historical relevance) and document trust priors.

## 🏗 Core Components

HalfLife operates through four primary interacting layers:

### 1. **Temporal Decay Engine (`engine/decay`)**
Computes temporal scores ($S_t$) based on mathematical families that model how information loses relevance over time.
- **Exponential Decay**: Good for news and fast-moving trends ($e^{-\lambda \Delta t}$).
- **Piecewise Decay**: Ideal for documentation or versioned data where relevance stays stable and then drops sharply.
- **Learned Decay (Future)**: Models that adapt parameters based on click-through rates.

### 2. **Metadata Store (`engine/store`)**
A high-performance **Redis-backed layer** that stores chunk-level state:
- **Decay Configuration**: Which decay family and parameters (e.g., $\lambda$) apply to each chunk.
- **Trust Scores**: Static priors reflecting the reliability of the source.
- **Score Cache**: Pre-computed temporal scores to keep rerank latency $<10\text{ms}$.

### 3. **Fusion & Reranking Layer (`engine/fusion`)**
The "brain" of HalfLife. It receives retrieved chunks from a vector store and re-scores them using the fusion formula:

$$S_{final} = \alpha \cdot S_{vector} + \beta \cdot S_{temporal} + \gamma \cdot S_{trust}$$

- **Query Intent Awareness**: Integrated with a classifier that detects if the user wants **Fresh** (high $\beta$), **Historical** (inverted $\beta$), or **Static** (high $\alpha$) content.

### 4. **API Middleware (`api/`)**
A FastAPI interface that allows any existing RAG pipeline to "drop-in" HalfLife with minimal code changes.

---

## 🔄 Data & Control Flow

1.  **Ingestion**: Documents are embedded into a Vector DB (Qdrant). Simultaneously, HalfLife's `DocTypeClassifier` determines their temporal profile and writes metadata to Redis.
2.  **Retrieval**: The user query is sent to the Vector DB. It returns the top-$K$ candidates based on cosine similarity.
3.  **Reranking**:
    - The query is classified for temporal intent.
    - HalfLife fetches metadata for the candidates from Redis.
    - Temporal scores are calculated (or retrieved from cache).
    - Fusion is performed; chunks are re-ordered.
4.  **Feedback**: The system logs which chunks were actually used in the final response, allowing for online parameter tuning.

---

## 🛠 Tech Stack
- **API**: FastAPI + Uvicorn
- **Vector DB**: Qdrant (Primary retriever)
- **Metadata Store**: Redis/Stack
- **Math**: NumPy, SciPy
- **Models**: Sentence-Transformers (Embeddings), Query Intent (Keyword-based/ML)
