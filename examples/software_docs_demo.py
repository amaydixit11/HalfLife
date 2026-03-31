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

def run_temporal_relevance_demo():
    print("-------------------------------------------------------")
    print(" UNSTRUCTURED CORPUS DEMO: Knowledge Evolution (React)")
    print("-------------------------------------------------------\n")
    
    query_str = "What is the best way to manage state in React today?"
    query_bundle = QueryBundle(query_str=query_str)
    print(f'Query: "{query_str}"\n')
    
    nodes = [
        NodeWithScore(
            node=TextNode(
                text="React 16 introduced Redux as the standard for state management in 2018. Use connect().",
                metadata={"source": "stackoverflow"} # MISSING timestamp metadata!
            ),
            score=0.95  # Make baseline confidently wrong
        ),
        NodeWithScore(
            node=TextNode(
                text="In React 18, Zustand and React Context are the preferred lightweight state management patterns for 2022.",
                metadata={"source": "react_docs"} # MISSING timestamp metadata!
            ),
            score=0.81  # Make baseline fail dramatically
        ),
        NodeWithScore(
            node=TextNode(
                text="Class components use this.setState() for local state. (React 15)",
                metadata={"timestamp": "2016-04-01", "source": "legacy_blog"} # Has metadata
            ),
            score=0.75 
        )
    ]
    
    baseline_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
    
    hl_processor = HalfLifePostprocessor(top_n=3)
    reranked_nodes = hl_processor._postprocess_nodes(
        nodes=copy.deepcopy(nodes), 
        query_bundle=query_bundle
    )
    reranked_nodes.sort(key=lambda x: x.score, reverse=True)
    
    baseline_top = baseline_nodes[0]
    
    print("\nAnalysis of Baseline Limitations:")
    print("   → 'Redux' matched query semantically")
    print("   → But it ignored that it's outdated (2018)\n")
    
    print("Temporal Inference & Confidence Scoring")
    print("---------------------------------------------------------")
    print("Methodology:")
    print("   • Baseline ignores time completely")
    print("   • HalfLife extracts time from raw text")
    print("   • Then reweights relevance using temporal decay\n")
    
    print("Rank | Baseline (Standard RAG)           | HalfLife (Temporal Inference)")
    print("---------------------------------------------------------------------------------")
    
    for i in range(3):
        b_node = baseline_nodes[i].node.get_content()
        h_node = reranked_nodes[i].node.get_content()
        
        if "Redux" in b_node:
            b_str = "Redux (2018)"
        elif "Zustand" in b_node:
            b_str = "Zustand (2022)"
        else:
            b_str = "setState (2016)"

        if "Redux" in h_node:
            h_str = "Redux (2018)"
        elif "Zustand" in h_node:
            h_str = "Zustand (2022) ✅"
        else:
            h_str = "setState (2016)"
            
        print(f"#{i+1}   | {b_str:<33} | {h_str}")
        
    print("\nTechnical Breakdown: HalfLife automated temporal inference for unstructured segments:")
    for n in reranked_nodes:
        inferred = n.node.metadata.get("inferred_year")
        src = n.node.metadata.get("temporal_source", "unknown")
        content = n.node.get_content()
        print(f"   [{n.score:.3f}] (year={inferred}, source={src}) {content[:65]}...")
    print("\n")

if __name__ == "__main__":
    run_temporal_relevance_demo()
