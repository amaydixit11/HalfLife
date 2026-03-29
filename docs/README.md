# HalfLife Documentation Index

Welcome to the **HalfLife** documentation. HalfLife is a temporal-aware re-ranking engine designed for RAG middleware.

## 📖 Content Index

1.  **[Architecture](./architecture.md)**: Explore the core components (Decay Engine, Redis Store, Fusion Layer, and API).
2.  **[Middleware Usage](./middleware-usage.md)**: Learn how to integrate HalfLife into an existing RAG pipeline.
3.  **[Query Intent Logic](./query-intents.md)**: Understand how the system automatically adapts to fresh and historical queries.
4.  **[Benchmarking](./benchmarking.md)**: How to measure performance using nDCG, MRR, and Temporal Freshness (TF).

---

## 🛠 Project Status
- **Core Engine (Phase 1 & 2)**: 🟢 Operational
- **Query Intent Classifier**: 🟡 Keyword-based (Transformer model planned)
- **Adaptive Decay Tuning**: 🟡 In development
- **Event-Driven Invalidation**: ⚪ Roadmap

---

## 🚀 Get Started Immediately
- **Prerequisites**: Docker & Python 3.10+
- **Commands**:
    1.  `docker-compose up -d`
    2.  `uvicorn api.main:app --reload`
    3.  Follow the [Middleware Usage Guide](./middleware-usage.md)
