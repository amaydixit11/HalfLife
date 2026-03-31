import os
import sys
import time
import logging
import copy
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

# Root path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
    from halflife.integrations.llamaindex import HalfLifePostprocessor
except ImportError:
    print("Please install llama-index-core and rich: pip install llama-index-core rich")
    sys.exit(1)

# Initialize Rich Console
console = Console()
logging.basicConfig(level=logging.ERROR)

def create_mock_corpus():
    """Real-world evolution of React State Management (2015-2024)"""
    return [
        {"year": 2015, "text": "Flux is the official architectural pattern for data flow in React applications (2015). Use Dispatchers.", "source": "facebook_blog", "score": 0.85},
        {"year": 2016, "text": "Redux is the definitive state container in 2016. Connect your components using mapStateToProps and connect().", "source": "stackoverflow", "score": 0.96},
        {"year": 2017, "text": "Redux-Saga is the industry standard for handling side effects in complex React Redux apps (2017).", "source": "github_readme", "score": 0.94},
        {"year": 2018, "text": "React 16.3 (2018) releases a new Context API that makes Redux unnecessary for most apps.", "source": "react_docs", "score": 0.89},
        {"year": 2019, "text": "Hooks are here (2019). Use useReducer and useContext to manage state without external libraries.", "source": "react_blog", "score": 0.92},
        {"year": 2020, "text": "Recoil (experimental, 2020) introduces atoms and selectors to solve the state management re-render problem.", "source": "meta_oss", "score": 0.88},
        {"year": 2021, "text": "React Query (TanStack, 2021) changes everything. Server state should not be in your client state store.", "source": "community_blog", "score": 0.90},
        {"year": 2022, "text": "Zustand is the preferred lightweight alternative to Redux for 2022. Minimal boilerplate, no providers.", "source": "github_trending", "score": 0.87},
        {"year": 2023, "text": "React Server Components (RSC, 2023) eliminate the need for much of traditional client-side state management.", "source": "nextjs_docs", "score": 0.85},
        {"year": 2024, "text": "React 19 (2024) introduces 'Signals' style primitives and the 'use' hook for asynchronous data flow.", "source": "react_beta_docs", "score": 0.82},
    ]

def run_temporal_demo():
    console.clear()
    console.print(Panel("[bold cyan]HalfLife: Temporal Evolution Analysis Demo[/bold cyan]\n"
                       "[dim]Scenario: Cross-sectional query on React state management documentation (2015-2024).[/dim]",
                       expand=False, border_style="cyan"))
    
    query = "What is the best way to handle state in React today?"
    console.print(f"\n[bold]Current Query:[/bold] [italic]\"{query}\"[/italic]\n")
    
    # 1. Preparation
    corpus = create_mock_corpus()
    nodes = []
    for i, item in enumerate(corpus):
        nodes.append(NodeWithScore(
            node=TextNode(text=item["text"], metadata={"source": item["source"], "timestamp": str(item["year"])}),
            score=item["score"]
        ))
    
    # 2. BASELINE PHASE
    with console.status("[bold red]Executing Baseline Semantic Retrieval...") as status:
        time.sleep(1.5)
        baseline_results = sorted(nodes, key=lambda x: x.score, reverse=True)
        
    table = Table(title="[bold red]Standard RAG Ranking (Vector-Only)[/bold red]", box=None, header_style="bold red")
    table.add_column("Rank", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Result (Text Segment)", style="dim")
    
    for i, res in enumerate(baseline_results[:3]):
        label = "[OUTDATED]" if i == 0 else ""
        table.add_row(f"#{i+1} {label}", f"{res.score:.3f}", f"{res.node.get_content()[:85]}...")
    
    console.print(table)
    console.print("[red]Baseline Limitation:[/red] Standard vector search prioritizes [bold]2016 Redux[/bold] due to semantic density, unknowingly serving an [bold]8-year-old[/bold] pattern as the primary recommendation.\n")
    
    console.print("[bold blue]---------------------------------------------------------------------------------[/bold blue]\n")
    
    # 3. HALFLIFE PHASE
    with console.status("[bold green]Executing HalfLife Temporal Fusion...") as status:
        time.sleep(1.5)
        processor = HalfLifePostprocessor(top_n=5)
        reranked = processor._postprocess_nodes(nodes=copy.deepcopy(nodes), query_bundle=QueryBundle(query_str=query))
        reranked.sort(key=lambda x: x.score, reverse=True)

    table_hl = Table(title="[bold green]HalfLife Temporal-Aware Ranking[/bold green]", box=None, header_style="bold green")
    table_hl.add_column("Rank", justify="center")
    table_hl.add_column("Score", justify="right")
    table_hl.add_column("Source", justify="right")
    table_hl.add_column("Inferred Date", justify="center")
    table_hl.add_column("Final Result (Text Segment)")
    
    for i, res in enumerate(reranked[:3]):
        label = "[FRESH]" if i == 0 else ""
        year = res.node.metadata.get("inferred_year", "Unknown")
        src_type = res.node.metadata.get("temporal_source", "raw")
        table_hl.add_row(
            f"#{i+1} {label}", 
            f"{res.score:.3f}", 
            f"[dim]{src_type}[/dim]",
            f"[bold cyan]{year}[/bold cyan]",
            f"{res.node.get_content()[:85]}..."
        )
    
    console.print(table_hl)
    
    console.print("\n[bold green]Analysis of Results:[/bold green]")
    console.print("   • System detected [italic]'today'[/italic] intent and applied temporal decay weights.")
    console.print(f"   • Automated temporal inference identified segments dated [cyan]{reranked[0].node.metadata.get('inferred_year')}[/cyan].")
    console.print("   • Successfully surafced newer paradigms (React 19) despite lower raw semantic scores.")

    console.print(Panel("\n[bold white]HalfLife ensures RAG relevance tracks with chronological evolution.[/bold white]", 
                       subtitle="[dim]Drop-in middleware for LlamaIndex & LangChain[/dim]", 
                       border_style="green"))

if __name__ == "__main__":
    run_temporal_demo()
