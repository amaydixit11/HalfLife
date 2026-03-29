import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CorpusChunk:
    chunk_id:      str
    text:          str
    timestamp:     datetime
    doc_type:      str
    source_domain: str

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
        
        # 1. THE ERA-SPECIFIC GOLD CHUNK (The one we WANT)
        c1_id = f"C{i:03d}_gold"
        chunks.append(CorpusChunk(
            chunk_id=c1_id,
            text=f"Technical Archive [{q_year}]: During this period, the officially recognized answer to {sample['query']} was {truth}. This fact is established by contemporary records.",
            timestamp=datetime(q_year, 1, 1, tzinfo=timezone.utc),
            doc_type="research",
            source_domain="gold-archive"
        ))

        # 2. THE RECENT TEMPORAL DECOY (The one that DISTRACTS the baseline)
        # We use a 2026 timestamp and a DIFFERENT (wrong) entity. 
        # Standard vector search (and decay-only) will prefer this because it's fresh.
        c2_id = f"C{i:03d}_decoy"
        recent_year = 2026
        # Decoy uses the EXACT SAME query text but a fake modern entity
        chunks.append(CorpusChunk(
            chunk_id=c2_id,
            text=f"Modern Update [{recent_year}]: Contemporary discussions on {sample['query']} now emphasize Entity_Prime_Active_Latest, replacing outdated 20th-century models.",
            timestamp=datetime(recent_year, 12, 31, tzinfo=timezone.utc),
            doc_type="news",
            source_domain="recent-updates"
        ))

        queries[q_id] = BenchmarkQuery(
            query_id=q_id,
            text=sample["query"],
            intent=sample["type"],
            topic=sample["topic"],
            relevant_ids=[c1_id] 
        )

    return chunks, queries

def build_corpus():
    return build_tiered_corpus()