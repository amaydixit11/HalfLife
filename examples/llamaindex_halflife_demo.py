import os
import sys

# Add project root to sys.path so 'halflife' module can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import logging

try:
    from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
except ImportError:
    print("Please install llama-index-core: pip install llama-index-core")
    sys.exit(1)

from halflife.integrations.llamaindex import HalfLifePostprocessor

# Set logging to ERROR to keep the output clean for the demo
logging.basicConfig(level=logging.ERROR)

def main():
    print("\n=============================================")
    print(" 🕰️ HalfLife vs Standard RAG Demo (LlamaIndex)")
    print("=============================================\n")
    
    query_str = "best NLP model today"
    query_bundle = QueryBundle(query_str=query_str)
    print(f'Query: "{query_str}"\n')
    
    # 1. Mock retrieved nodes (as if they just came from a Vector Store)
    # Notice: the embedding similarity for BERT is slightly higher!
    # This represents a common issue where older, highly-cited, well-structured 
    # documents have stronger semantic similarity than newer, sparser ones.
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="BERT (2019): State-of-the-art model for natural language processing tasks using bidirectional transformers.",
                metadata={"timestamp": "2019-10-15", "source": "arxiv"}
            ),
            score=0.92  # Higher Semantic similarity
        ),
        NodeWithScore(
            node=TextNode(
                text="GPT-4 (2024): Currently the most capable foundational model for advanced NLP and reasoning tasks.",
                metadata={"timestamp": "2024-03-01", "source": "arxiv"}
            ),
            score=0.88  # Lower Semantic similarity
        ),
        NodeWithScore(
            node=TextNode(
                text="Word2Vec (2013): Excellent embeddings for downstream NLP tasks.",
                metadata={"timestamp": "2013-05-10", "source": "arxiv"}
            ),
            score=0.75
        )
    ]
    
    print("1️⃣  Standard LlamaIndex (Baseline - purely vector search)")
    print("---------------------------------------------------------")
    baseline_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
    baseline_top = baseline_nodes[0]
    print(f"❌ Result: {baseline_top.node.get_content()}")
    print(f"   Date:  {baseline_top.node.metadata['timestamp'][:4]}")
    print(f"   Score: {baseline_top.score:.3f}\n")
    
    
    print("2️⃣  LlamaIndex + HalfLife (Temporal Fusion Pipeline)")
    print("---------------------------------------------------------")
    print("💡 HalfLife detects fresh intent ('today') and applies decaying vectors...\n")
    
    # Drop-in HalfLife Postprocessor
    hl_processor = HalfLifePostprocessor(top_n=3)
    
    # Process the exact same array of nodes
    reranked_nodes = hl_processor._postprocess_nodes(
        nodes=copy.deepcopy(nodes), 
        query_bundle=query_bundle
    )
    
    # Sort just in case order is not guaranteed, but _postprocess_nodes returns correctly sorted
    reranked_nodes.sort(key=lambda x: x.score, reverse=True)
    
    hl_top = reranked_nodes[0]
    print(f"✅ Result: {hl_top.node.get_content()}")
    print(f"   Date:  {hl_top.node.metadata['timestamp'][:4]}")
    print(f"   Score: {hl_top.score:.3f}\n")
    

if __name__ == "__main__":
    main()
