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
    vintage:       str = "recent" # 'recent', 'mid', 'old'
    is_decoy:      bool = False

def _vintage_timestamp(vintage: str, offset: int = 0) -> datetime:
    now = datetime.now(timezone.utc)
    if vintage == "recent": return now - timedelta(days=7 + offset)
    if vintage == "mid":    return now - timedelta(days=365 + offset)
    if vintage == "old":    return now - timedelta(days=365*10 + offset)
    return now

def primary_chunks(chunks: List[CorpusChunk]) -> List[CorpusChunk]:
    return [c for c in chunks if not c.is_decoy]

def decoy_chunks(chunks: List[CorpusChunk]) -> List[CorpusChunk]:
    return [c for c in chunks if c.is_decoy]

@dataclass
class BenchmarkQuery:
    query_id:     str
    text:         str
    intent:       str
    topic:        str
    relevant_ids: List[str]

def build_tiered_corpus(dataset_path: str = "scripts/temporal_qa.json") -> (List[CorpusChunk], Dict[str, BenchmarkQuery]):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    chunks = []
    queries = {}
    
    for i, sample in enumerate(dataset):
        q_id = f"Q{i:03d}"
        q_year = int(sample["timestamp"])
        truth = sample["ground_truth"]
        q_topic = sample["topic"]
        
        # 1. THE ERA-SPECIFIC GOLD CHUNK (The one we WANT)
        c1_id = f"C{i:03d}_gold"
        chunks.append(CorpusChunk(
            chunk_id=c1_id,
            text=f"Technical Archive [{q_year}]: During this period, the officially recognized answer to {sample['query']} was {truth}. This fact is established by contemporary records.",
            timestamp=datetime(q_year, 1, 1, tzinfo=timezone.utc),
            doc_type="research",
            source_domain="gold-archive",
            topic=q_topic,
            vintage="old",
            is_decoy=False
        ))

        # 2. THE RECENT TEMPORAL DECOY (The one that DISTRACTS the baseline)
        c2_id = f"C{i:03d}_decoy"
        recent_year = 2026
        chunks.append(CorpusChunk(
            chunk_id=c2_id,
            text=f"Modern Update [{recent_year}]: Contemporary discussions on {sample['query']} now emphasize Entity_Prime_Active_Latest, replacing outdated 20th-century models.",
            timestamp=datetime(recent_year, 12, 31, tzinfo=timezone.utc),
            doc_type="news",
            source_domain="recent-updates",
            topic=q_topic,
            vintage="recent",
            is_decoy=True
        ))

        queries[q_id] = BenchmarkQuery(
            query_id=q_id,
            text=sample["query"],
            intent=sample["type"],
            topic=q_topic,
            relevant_ids=[c1_id] 
        )

    return chunks, queries

def build_corpus():
    return build_tiered_corpus()