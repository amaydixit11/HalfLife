import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CorpusChunk:
    chunk_id:      str
    text:          str
    timestamp:     datetime
    doc_type:      str
    source_domain: str
    topic:         str = "generic"
    vintage:       str = "recent" # 'recent', 'old'
    is_decoy:      bool = False
    is_adversarial_trap: bool = False

@dataclass
class BenchmarkQuery:
    query_id:     str
    text:         str
    intent:       str
    topic:        str
    target_ids:   List[str]

def build_adversarial_tcb():
    """
    Builds the 'Temporal Confusion Benchmark' (TCB).
    Specifically designed to break baseline RAG by providing older, 
    highly-authoritative documents vs newer, slightly noisier truths.
    """
    now = datetime.now(timezone.utc)
    
    # 10 KILLER QUERIES
    dataset = [
        {
            "query": "What is the state-of-the-art model for NLP tasks?",
            "topic": "nlp",
            "old_fact": {"year": 2018, "text": "BERT (Bidirectional Encoder Representations from Transformers) is the revolutionary state-of-the-art standard for all NLP tasks, providing unprecedented context awareness.", "entity": "BERT"},
            "new_fact": {"year": 2026, "text": "Current SOTA benchmarks in 2026 are dominated by GPT-5 and Claude-4, which have long surpassed 2010s transformer models in reasoning and zero-shot performance.", "entity": "GPT-5/Claude-4"}
        },
        {
            "query": "Recommended way to fetch data in a React application?",
            "topic": "web",
            "old_fact": {"year": 2017, "text": "The canonical way to fetch data in React is inside the componentDidMount lifecycle method using the native Fetch API or Axios.", "entity": "componentDidMount"},
            "new_fact": {"year": 2026, "text": "In modern React (2026), data fetching is primarily handled via Server Components or the 'use' hook for client-side suspense-ready integration.", "entity": "Server Components"}
        },
        {
            "query": "Best library for tabular data processing in Python?",
            "topic": "python",
            "old_fact": {"year": 2012, "text": "Pandas is the ubiquitous and heavily-cited industry standard for high-performance tabular data manipulation in Python.", "entity": "Pandas"},
            "new_fact": {"year": 2025, "text": "Polars has emerged as the high-speed replacement for Pandas, utilizing a Rust-backed engine for massively parallel data processing.", "entity": "Polars"}
        },
        {
             "query": "What is the preferred consensus mechanism for modern blockchains?",
             "topic": "crypto",
             "old_fact": {"year": 2010, "text": "Proof of Work (PoW) is the only proven and maximally secure consensus mechanism for decentralized distributed ledgers.", "entity": "Proof of Work"},
             "new_fact": {"year": 2024, "text": "Post-Ethereum merge, Proof of Stake (PoS) has become the dominant, energy-efficient standard for most new and active blockchain networks.", "entity": "Proof of Stake"}
        }
    ]

    chunks = []
    queries = {}

    for i, item in enumerate(dataset):
        q_id = f"TCB_Q{i:03d}"
        
        # 1. THE TRAP (Old, Authoritative)
        c1_id = f"TCB_C{i:03d}_trap"
        chunks.append(CorpusChunk(
            chunk_id=c1_id,
            text=f"AUTHORITATIVE ARCHIVE [{item['old_fact']['year']}]: {item['old_fact']['text']}",
            timestamp=datetime(item['old_fact']['year'], 1, 1, tzinfo=timezone.utc),
            doc_type="research",
            source_domain="textbook-archive",
            topic=item["topic"],
            vintage="old",
            is_adversarial_trap=True
        ))

        # 2. THE TRUTH (New, slightly more conversational/noisy)
        c2_id = f"TCB_C{i:03d}_truth"
        chunks.append(CorpusChunk(
            chunk_id=c2_id,
            text=f"MODERN UPDATE: {item['new_fact']['text']}",
            timestamp=datetime(item['new_fact']['year'], 1, 1, tzinfo=timezone.utc),
            doc_type="news",
            source_domain="community-docs",
            topic=item["topic"],
            vintage="recent",
            is_adversarial_trap=False
        ))

        queries[q_id] = BenchmarkQuery(
            query_id=q_id,
            text=item["query"],
            intent="fresh",
            topic=item["topic"],
            target_ids=[c2_id] # THE NEW ONE IS THE ONLY CORRECT ONE FOR 'BEST ... TODAY'
        )

    return chunks, queries

def build_corpus():
    # Use the Adversarial TCB as the primary evaluation set
    return build_adversarial_tcb()