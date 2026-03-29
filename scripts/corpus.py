"""
corpus.py — Synthetic benchmark corpus for HalfLife.

Builds 54 chunks across 3 topic clusters × 3 doc types × 3 vintages.
Each chunk has a known timestamp and ground-truth relevance label per
query intent (fresh / historical / static).

Design constraints:
  - Chunks within the same cluster are semantically similar so cosine
    similarity alone cannot distinguish vintages. The temporal signal
    is what separates them. This is what the benchmark actually tests.
  - Ground truth is deterministic: no human annotation required.
  - The corpus is self-contained — no external data dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List


# --------------------------------------------------------------------------- #
#  Data model                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class CorpusChunk:
    chunk_id:      str
    text:          str
    timestamp:     datetime
    doc_type:      str          # news | research | documentation
    topic:         str          # transformer_nlp | llm_scaling | vector_search
    vintage:       str          # recent | mid | old
    source_domain: str

    is_decoy:      bool = False
    
    # Ground truth relevance per intent type.
    # 1 = relevant, 0 = not relevant.
    relevant_for_fresh:      int = 0
    relevant_for_historical: int = 0
    relevant_for_static:     int = 1  # topic match is default


@dataclass
class BenchmarkQuery:
    query_id:    str
    text:        str
    intent:      str            # fresh | historical | static
    topic:       str            # must match CorpusChunk.topic
    relevant_ids: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Vintage date ranges                                                         #
# --------------------------------------------------------------------------- #

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _vintage_timestamp(vintage: str, idx: int) -> datetime:
    """
    Returns a deterministic timestamp for a given vintage and index.
    idx spreads chunks across the vintage window so they don't all
    share the exact same timestamp (which would make age ties).
    """
    now = _now()
    if vintage == "recent":
        # 1 week – 5 months ago
        base = now - timedelta(days=30)
        return base - timedelta(days=idx * 12)
    elif vintage == "mid":
        # 14–24 months ago
        base = now - timedelta(days=14 * 30)
        return base - timedelta(days=idx * 15)
    elif vintage == "old":
        # 36–54 months ago
        base = now - timedelta(days=36 * 30)
        return base - timedelta(days=idx * 20)
    raise ValueError(f"Unknown vintage: {vintage}")


# --------------------------------------------------------------------------- #
#  Chunk templates                                                             #
# --------------------------------------------------------------------------- #

_TEMPLATES = {
    "transformer_nlp": {
        "news": [
            "Researchers have released a new benchmark for evaluating transformer "
            "models on natural language understanding tasks, showing improved results "
            "over previous baselines.",
            "A major AI lab has published findings on scaling transformer attention "
            "mechanisms, reporting efficiency gains on standard NLP benchmarks.",
            "New transformer variant achieves state-of-the-art on several natural "
            "language processing tasks including question answering and summarisation.",
            "Teams studying transformer architectures report that attention head "
            "pruning reduces inference cost without significant accuracy loss.",
            "Benchmark results comparing transformer models on multilingual NLP tasks "
            "show consistent improvements with scale.",
            "Latest evaluation of transformer-based models reveals performance gaps "
            "between dense and sparse attention on long-document tasks.",
        ],
        "research": [
            "We present an analysis of multi-head self-attention in transformer "
            "architectures and its role in capturing long-range dependencies in text.",
            "This paper investigates positional encodings in transformers and proposes "
            "a relative position scheme that generalises better to unseen sequence lengths.",
            "A study of transformer depth versus width trade-offs shows that wider "
            "shallow models can match deeper narrow models on NLP benchmarks.",
            "We introduce a theoretical framework for understanding information flow "
            "through transformer layers during forward and backward passes.",
            "Empirical analysis of transformer attention patterns reveals consistent "
            "syntactic structure emergence across language model pre-training.",
            "This work examines the role of layer normalisation in transformer "
            "stability during training at large batch sizes.",
        ],
        "documentation": [
            "The Transformer class implements multi-head self-attention as described "
            "in the original architecture. Use TransformerEncoder for encoding tasks.",
            "Configuration options for the attention mechanism include num_heads, "
            "head_dim, and dropout_rate. See the API reference for full parameter list.",
            "To fine-tune a pre-trained transformer model, load the checkpoint and "
            "call model.train() before passing batches through the forward method.",
            "The tokenizer pipeline for transformer models handles subword splitting, "
            "padding, and attention mask generation automatically.",
            "Transformer model checkpoints are stored in the standard format. Use "
            "load_pretrained() to restore weights from a saved directory.",
            "Memory optimisation for transformer inference: enable gradient "
            "checkpointing and mixed-precision with the provided utility functions.",
        ],
    },

    "llm_scaling": {
        "news": [
            "A new large language model trained on a trillion tokens demonstrates "
            "strong few-shot performance across diverse language generation tasks.",
            "Researchers report emergent capabilities in language models at scale, "
            "with arithmetic reasoning appearing above a parameter threshold.",
            "Scaling experiments show that larger language models continue to improve "
            "on code generation tasks when trained with additional compute.",
            "Industry lab releases scaled language model with improved instruction "
            "following, citing reinforcement learning from human feedback.",
            "New results suggest that language model scaling laws hold across "
            "modalities when training data quality is carefully controlled.",
            "Evaluation of large language models on reasoning tasks reveals that "
            "chain-of-thought prompting significantly improves benchmark scores.",
        ],
        "research": [
            "We characterise scaling laws for language model performance as a "
            "function of model size, dataset size, and compute budget.",
            "This paper studies the emergence of in-context learning in large "
            "language models and identifies the critical scale at which it appears.",
            "A theoretical treatment of language model scaling connects loss curves "
            "to power-law behaviour in the data distribution.",
            "We investigate the relationship between language model perplexity and "
            "downstream task performance across a range of model sizes.",
            "Mechanistic interpretability methods applied to large language models "
            "reveal circuits responsible for factual recall and reasoning.",
            "This work analyses training instabilities in large language models and "
            "proposes gradient norm clipping schedules that improve stability.",
        ],
        "documentation": [
            "The language model training loop supports gradient accumulation, "
            "mixed-precision, and distributed data parallelism out of the box.",
            "To reproduce scaling experiments, set the model_size and dataset_tokens "
            "parameters in the training configuration file.",
            "Evaluation harness for large language models supports few-shot prompting, "
            "chain-of-thought, and zero-shot evaluation modes.",
            "Checkpoint management for large models: use the sharded checkpoint "
            "format to save and load models larger than GPU memory.",
            "The tokeniser for language model training supports byte-pair encoding "
            "and unigram language model vocabularies.",
            "Monitoring large language model training: log loss, gradient norm, and "
            "learning rate schedule using the provided metrics callbacks.",
        ],
    },

    "vector_search": {
        "news": [
            "A new approximate nearest neighbour index achieves sub-millisecond "
            "query latency on billion-scale vector datasets with high recall.",
            "Vector database benchmarks comparing HNSW and IVF index structures "
            "show trade-offs between build time, memory, and query throughput.",
            "Research team releases improved vector quantisation method that reduces "
            "index size by 4x with minimal recall degradation.",
            "New vector search implementation supports hybrid retrieval combining "
            "dense embeddings with sparse BM25 signals.",
            "Evaluation of vector databases at scale reveals that filtered search "
            "performance varies significantly across index implementations.",
            "Latest results on vector search benchmarks show GPU-accelerated indices "
            "outperform CPU implementations at high query-per-second loads.",
        ],
        "research": [
            "We present a theoretical analysis of approximate nearest neighbour "
            "search in high-dimensional spaces and derive recall bounds for HNSW.",
            "This paper introduces a learned index structure for vector search that "
            "adapts to the distribution of the embedding space.",
            "A study of quantisation error in product quantisation shows its effect "
            "on downstream retrieval quality in RAG pipelines.",
            "We propose a hierarchical navigable small world graph variant that "
            "improves recall on clustered embedding distributions.",
            "Empirical comparison of vector index structures across embedding models "
            "reveals that optimal index choice is embedding-distribution-dependent.",
            "This work formalises the recall-latency-memory trade-off in approximate "
            "nearest neighbour search and proposes a Pareto-optimal selection method.",
        ],
        "documentation": [
            "Creating a vector collection in Qdrant: specify the vector size and "
            "distance metric, then call create_collection() with the config object.",
            "HNSW index parameters m and ef_construction control the recall-speed "
            "trade-off. Higher values improve recall at the cost of build time.",
            "Filtered vector search allows combining payload conditions with ANN "
            "search. Use the Filter object to specify equality and range conditions.",
            "Batch upsert operations in the vector store accept lists of PointStruct "
            "objects. Use batches of 100–1000 points for optimal throughput.",
            "The search() method returns ScoredPoint objects with id, score, and "
            "payload. Set with_payload=True to include metadata in results.",
            "Collection snapshots allow point-in-time backups of vector indices. "
            "Use the snapshots API to create and restore from snapshots.",
        ],
    },
}

# --------------------------------------------------------------------------- #
#  Query templates per intent and topic                                        #
# --------------------------------------------------------------------------- #

_QUERIES = {
    "transformer_nlp": {
        "fresh": [
            "latest transformer architecture improvements",
            "recent advances in transformer attention",
            "newest transformer benchmark results this year",
            "current state of transformer NLP models",
        ],
        "historical": [
            "history of transformer attention mechanisms",
            "how did transformer architectures evolve",
            "original transformer design decisions",
            "early transformer positional encoding approaches",
        ],
        "static": [
            "how does multi-head attention work",
            "transformer encoder decoder architecture",
            "explain self-attention in transformers",
            "transformer model layer normalisation",
        ],
    },
    "llm_scaling": {
        "fresh": [
            "latest large language model scaling results",
            "recent LLM benchmark performance",
            "newest language model capabilities",
            "current LLM reasoning improvements",
        ],
        "historical": [
            "history of language model scaling laws",
            "how did LLM scaling research evolve",
            "original emergent capability findings",
            "early large language model training approaches",
        ],
        "static": [
            "what are scaling laws for language models",
            "explain in-context learning in LLMs",
            "language model training compute budget",
            "how does chain of thought prompting work",
        ],
    },
    "vector_search": {
        "fresh": [
            "latest vector search index benchmarks",
            "recent improvements to approximate nearest neighbour",
            "newest vector database performance results",
            "current hybrid vector search methods",
        ],
        "historical": [
            "history of approximate nearest neighbour algorithms",
            "how did vector search evolve",
            "original HNSW graph design",
            "early product quantisation methods",
        ],
        "static": [
            "how does HNSW index work",
            "explain product quantisation for vectors",
            "vector search recall latency trade-off",
            "how to configure vector database index",
        ],
    },
}


# --------------------------------------------------------------------------- #
#  Corpus builder                                                              #
# --------------------------------------------------------------------------- #

def build_corpus() -> tuple[list[CorpusChunk], list[BenchmarkQuery]]:
    """
    Constructs the full benchmark corpus and query set.

    Returns:
        chunks:  54 CorpusChunk objects with timestamps and relevance labels
        queries: 36 BenchmarkQuery objects (3 topics × 3 intents × 4 queries)
    """
    chunks: list[CorpusChunk] = []
    queries: list[BenchmarkQuery] = []

    vintages   = ["recent", "mid", "old"]
    doc_types  = ["news", "research", "documentation"]
    topics     = list(_TEMPLATES.keys())

    chunk_index = 0

    for topic in topics:
        for doc_type in doc_types:
            texts = _TEMPLATES[topic][doc_type]  # 6 texts
            for v_idx, vintage in enumerate(vintages):
                # 2 primary chunks per (topic, doc_type, vintage)
                for t_idx in range(2):
                    text_idx = v_idx * 2 + t_idx
                    ts = _vintage_timestamp(vintage, text_idx)

                    # Primary chunk
                    c = CorpusChunk(
                        chunk_id=f"chunk_{chunk_index:03d}",
                        text=texts[text_idx],
                        timestamp=ts,
                        doc_type=doc_type,
                        topic=topic,
                        vintage=vintage,
                        source_domain=_source_domain(doc_type),
                        relevant_for_fresh=      1 if vintage == "recent" else 0,
                        relevant_for_historical= 1 if vintage == "old"    else 0,
                    )
                    chunks.append(c)
                    chunk_index += 1

                    # Decoy shadow: Same text, DIFFERENT vintage. 
                    # If primary is recent, decoy is old. If primary is old, decoy is recent.
                    # This forces cosine to fail (as text is identical) while HalfLife wins.
                    decoy_vintage = "old" if vintage == "recent" else "recent" if vintage == "old" else "mid"
                    # Different day for decay tiebreak
                    d_ts = _vintage_timestamp(decoy_vintage, text_idx + 10) 
                    
                    decoy = CorpusChunk(
                        chunk_id=f"chunk_{chunk_index:03d}",
                        text=texts[text_idx],
                        timestamp=d_ts,
                        doc_type=doc_type,
                        topic=topic,
                        vintage=decoy_vintage,
                        source_domain=_source_domain(doc_type),
                        relevant_for_fresh=      0, # always 0 for decoys
                        relevant_for_historical= 0,
                        is_decoy=True,
                    )
                    chunks.append(decoy)
                    chunk_index += 1

    # Build queries and attach relevant_ids
    # Note: query_text matches some texts perfectly, making it even harder!
    query_index = 0
    for topic in topics:
        for intent in ["fresh", "historical", "static"]:
            # 'static' queries consider ALL chunks (including decoys) as relevant 
            # if they match the topic, as the user only cares about fact correctness.
            for q_text in _QUERIES[topic][intent]:
                relevant_ids = [
                    c.chunk_id for c in chunks
                    if c.topic == topic and _is_relevant(c, intent)
                ]
                queries.append(BenchmarkQuery(
                    query_id=f"q_{query_index:03d}",
                    text=q_text,
                    intent=intent,
                    topic=topic,
                    relevant_ids=relevant_ids,
                ))
                query_index += 1

    return chunks, queries


def _is_relevant(chunk: CorpusChunk, intent: str) -> bool:
    if intent == "fresh":
        return chunk.relevant_for_fresh == 1
    elif intent == "historical":
        return chunk.relevant_for_historical == 1
    elif intent == "static":
        return chunk.relevant_for_static == 1
    return False


def _source_domain(doc_type: str) -> str:
    return {
        "news":          "arxiv-news",
        "research":      "arxiv",
        "documentation": "github-docs",
    }.get(doc_type, "generic")


# --------------------------------------------------------------------------- #
#  Smoke test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    chunks, queries = build_corpus()

    print(f"Corpus: {len(chunks)} chunks")
    print(f"Queries: {len(queries)} queries")

    from collections import Counter
    vintage_counts = Counter(c.vintage  for c in chunks)
    type_counts    = Counter(c.doc_type for c in chunks)
    intent_counts  = Counter(q.intent   for q in queries)

    print(f"\nVintage distribution: {dict(vintage_counts)}")
    print(f"Doc type distribution: {dict(type_counts)}")
    print(f"Query intent distribution: {dict(intent_counts)}")

    sample_q = queries[0]
    print(f"\nSample query: '{sample_q.text}'")
    print(f"  intent={sample_q.intent}, topic={sample_q.topic}")
    print(f"  relevant_ids ({len(sample_q.relevant_ids)}): {sample_q.relevant_ids[:4]}...")
