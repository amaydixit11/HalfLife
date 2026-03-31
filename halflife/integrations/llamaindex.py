import logging
from typing import List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from halflife import HalfLife

logger = logging.getLogger(__name__)

class HalfLifePostprocessor(BaseNodePostprocessor):
    """
    HalfLife Temporal Reranker for LlamaIndex.
    
    Prevents your RAG from returning wrong answers due to time by
    applying intent-aware temporal fusion to your retrieved nodes.
    """
    
    hl: HalfLife = None
    top_n: int = 5
    
    def __init__(self, top_n: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.hl = HalfLife()
        self.top_n = top_n

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        Rerank nodes based on HalfLife's temporal-aware logic.
        """
        if not nodes or not query_bundle:
            return nodes

        query = query_bundle.query_str

        # Convert LlamaIndex Nodes to HalfLife chunk format
        hl_chunks = []
        id_to_node = {}
        for i, node_with_score in enumerate(nodes):
            node = node_with_score.node
            score = node_with_score.score or 0.9 # Default to high if not provided
            
            chunk_id = f"li-{i}"
            id_to_node[chunk_id] = node_with_score
            
            # Extract metadata and apply 'Messy Reality' fallback if missing
            hl_chunks.append({
                "id": chunk_id,
                "score": score,
                "payload": {
                    "text": node.get_content(),
                    "timestamp": node.metadata.get("timestamp") or node.metadata.get("date"),
                    "doc_type": node.metadata.get("doc_type"),
                    "source_domain": node.metadata.get("source"),
                    "original_id": node.metadata.get("chunk_id", str(i))
                }
            })

        # Perform the rerank using the HalfLife engine
        results = self.hl.rerank(
            query=query,
            chunks=hl_chunks,
            top_k=self.top_n
        )

        # Map back to LlamaIndex NodeWithScore objects
        reranked_nodes = []
        for rank_idx, res in enumerate(results):
            # Find the original node securely by ID mapping
            node_with_score = id_to_node[res["id"]]
            
            # Update score and metadata for visibility in debug logs
            node_with_score.score = res["final_score"]
            node_with_score.node.metadata["halflife_rank"] = rank_idx + 1
            node_with_score.node.metadata["temporal_score"] = res["temporal_score"]
            node_with_score.node.metadata["temporal_source"] = res.get("temporal_source", "unknown")
            node_with_score.node.metadata["inferred_year"] = res.get("inferred_year", "None")
            
            reranked_nodes.append(node_with_score)

        return reranked_nodes

# --- Usage Example (to be included in README/Docs) ---
# postprocessor = HalfLifePostprocessor(top_n=3)
# query_engine = index.as_query_engine(
#     similarity_top_k=10,
#     node_postprocessors=[postprocessor]
# )
# response = query_engine.query("What is the latest React version?")
