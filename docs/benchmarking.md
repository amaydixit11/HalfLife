# Benchmarking HalfLife Performance

HalfLife provides a dedicated benchmarking harness (`scripts/benchmark.py`) to measure the performance improvement of temporal re-ranking over standard cosine similarity across different query intents.

## 📊 Evaluation Metrics

We compute three core Information Retrieval (IR) metrics to quantify "retrieval quality":

### 1. **nDCG@10 (Normalized Discounted Cumulative Gain)**
Standard IR quality metric. It measures the usefulness of a document based on its position in the result list.
- **Range**: 0.0 to 1.0 (Higher is better)
- **Goal**: High nDCG across all query types.

### 2. **MRR (Mean Reciprocal Rank)**
The reciprocal of the rank of the first relevant result.
- **Goal**: Find the first relevant piece of information as high as possible.

### 3. **TF@10 (Temporal Freshness Score)**
A custom metric defined as the mean of $1 / (1 + \text{age\_days})$ over the top-10 retrieved chunks.
- **Interpretation**:
  - **Fresh Queries**: We want a **high TF** (indicating very recent documents at the top).
  - **Historical Queries**: We want a **low TF** (indicating older/background documents are being surfaced).
  - **Static Queries**: TF should remain **stable** (no regression versus a baseline).

---

## 🏃 Running the Benchmark

### 1. Start Services
Ensure Qdrant and Redis are running:
```bash
docker-compose up -d
```

### 2. Run the Evaluator
To run the full benchmark (including synthetic corpus ingestion):
```bash
python scripts/benchmark.py
```

### 3. Incremental Runs
If the corpus is already ingested:
```bash
python scripts/benchmark.py --skip-ingest
```

### 4. Output to JSON
To save performance results for comparison over time:
```bash
python scripts/benchmark.py --output results/run_v0.2.json
```

---

## 🔬 Analyzing the Output

HalfLife groups results by **Query Intent**. Look for:
- **Baseline vs. HalfLife nDCG**: Is there a statistically significant lift?
- **Intent Shifts**: Does HalfLife correctly surface older papers (low TF) for historical queries? 
- **Latency**: Ensure the `halflife_latency_ms` remains below **5-10ms** per request.
