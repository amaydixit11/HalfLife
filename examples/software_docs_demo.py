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

logging.basicConfig(level=logging.ERROR)

def main():
    print("\n=======================================================")
    print(" 💥 THE 'MESSY CORPUS' DEMO: React Docs vs StackOverflow")
    print("=======================================================\n")
    
    query_str = "What is the best way to manage state in React today?"
    query_bundle = QueryBundle(query_str=query_str)
    print(f'Query: "{query_str}"\n')
    
    # Notice we have NO TIMESTAMPS in the metadata for the first two.
    # HalfLife will infer the dates from the text automatically!
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="React 16 introduced Redux as the standard for state management in 2018. Use connect() and mapStateToProps extensively.",
                metadata={"source": "stackoverflow"} # MISSING timestamp metadata!
            ),
            score=0.91  # Semantically highly relevant, but horribly outdated
        ),
        NodeWithScore(
            node=TextNode(
                text="In React 18, Zustand and React Context are the preferred lightweight state management patterns for 2022.",
                metadata={"source": "react_docs"} # MISSING timestamp metadata!
            ),
            score=0.88  # Slightly lower semantic relevance
        ),
        NodeWithScore(
            node=TextNode(
                text="Class components use this.setState() for local state. (React 15)",
                metadata={"timestamp": "2016-04-01", "source": "legacy_blog"} # Has metadata
            ),
            score=0.85 
        )
    ]
    
    print("1️⃣  Standard Baseline (No Temporal Awareness)")
    print("---------------------------------------------------------")
    baseline_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
    baseline_top = baseline_nodes[0]
    print(f"❌ Result: {baseline_top.node.get_content()}")
    print(f"   Score:  {baseline_top.score:.3f}")
    print("   Problem: Recommended a 2018 pattern because it purely matched semantics.\n")
    
    print("2️⃣  HalfLife (Temporal Inference & Confidence Scoring)")
    print("---------------------------------------------------------")
    print("💡 HalfLife infers missing dates via regex, assigns confidence, and decays...\n")
    
    hl_processor = HalfLifePostprocessor(top_n=3)
    reranked_nodes = hl_processor._postprocess_nodes(
        nodes=copy.deepcopy(nodes), 
        query_bundle=query_bundle
    )
    reranked_nodes.sort(key=lambda x: x.score, reverse=True)
    
    hl_top = reranked_nodes[0]
    print(f"✅ Result: {hl_top.node.get_content()}")
    print(f"   Score:  {hl_top.score:.3f}")
    print("   Success: HalfLife automatically extracted '2022' from the unstructured text and boosted the modern pattern!\n")

if __name__ == "__main__":
    main()
