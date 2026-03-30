import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from halflife import HalfLife

logger = logging.getLogger(__name__)

class HalfLifeReranker(BaseDocumentCompressor):
    """
    HalfLife Reranker for LangChain.
    
    Prevents your RAG from returning wrong answers due to time by
    applying intent-aware temporal fusion to your retrieved documents.
    """
    
    hl: HalfLife = None
    top_k: int = 5
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hl = HalfLife()

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents based on HalfLife's temporal-aware logic.
        """
        if not documents:
            return []

        # Convert LangChain Documents to HalfLife chunk format
        hl_chunks = []
        for i, doc in enumerate(documents):
            # We assume 'timestamp' exists in metadata for temporal effects
            # If not, HalfLife's internal 'Messy Reality' regex harvester will attempt extraction.
            hl_chunks.append({
                "id": f"lc-{i}",
                "score": doc.metadata.get("relevance_score", 0.9), # Default to high if not provided
                "payload": {
                    "text": doc.page_content,
                    "timestamp": doc.metadata.get("timestamp"),
                    "doc_type": doc.metadata.get("doc_type"),
                    "source_domain": doc.metadata.get("source"),
                    "original_id": doc.metadata.get("chunk_id", str(i))
                }
            })

        # Perform the rerank
        results = self.hl.rerank(
            query=query,
            chunks=hl_chunks,
            top_k=self.top_k
        )

        # Map back to LangChain Documents
        reranked_docs = []
        for rank_idx, res in enumerate(results):
            # Find the original doc by indexing (HACK: we stored id as lc-i)
            orig_idx = int(res["id"].split("-")[1])
            doc = documents[orig_idx]
            
            # Update metadata with scores
            doc.metadata["halflife_score"] = res["final_score"]
            doc.metadata["temporal_score"] = res["temporal_score"]
            doc.metadata["rank"] = rank_idx + 1
            
            reranked_docs.append(doc)

        return reranked_docs

# --- Usage Example (to be included in README/Docs) ---
# compressor = HalfLifeReranker(top_k=3)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=retriever
# )
# docs = compression_retriever.get_relevant_documents("What is the latest LLM?")
