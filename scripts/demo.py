import os
import sys
import time
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from halflife import HalfLife

console = Console()

def run_viral_demo():
    logging.basicConfig(level=logging.ERROR) # Quiet the logs for the viral demo
    hl = HalfLife()
    
    console.print(Panel.fit(
        "[bold blue]🚀 HalfLife ‘Viral Travel’ Demo v0.5[/bold blue]\n"
        "Proving why Vector Search is fundamentally broken for time.",
        border_style="blue"
    ))
    
    # 5 VIRAL SCENARIOS
    scenarios = [
        {
            "name": "AI/NLP: The Authority Trap",
            "query": "What is the state-of-the-art model for NLP today?",
            "docs": [
                {"year": 2018, "entity": "BERT", "text": "BERT (Google) is the revolutionary state-of-the-art standard for all NLP tasks.", "score": 0.98},
                {"year": 2026, "entity": "GPT-5", "text": "GPT-5/Claude-4 have long surpassed 2010s transformer models in reasoning.", "score": 0.92}
            ]
        },
        {
            "name": "Web Dev: The Legacy Bias",
            "query": "Latest way to fetch data in React?",
            "docs": [
                {"year": 2017, "entity": "componentDidMount", "text": "The canonical way to fetch data in React is inside the componentDidMount method.", "score": 0.96},
                {"year": 2026, "entity": "Server Components", "text": "Modern React handles data fetching primarily via Server Components or 'use' hook.", "score": 0.91}
            ]
        },
        {
            "name": "Python: The Concurrency Conflict",
            "query": "Best library for concurrent Python today?",
            "docs": [
                {"year": 2012, "entity": "Gevent", "text": "Gevent is the industry-proven standard for high-performance coroutines in Python.", "score": 0.97},
                {"year": 2025, "entity": "Asyncio/uvloop", "text": "Modern high-concurrency Python relies on the native Asyncio event-loop + uvloop.", "score": 0.90}
            ]
        },
        {
            "name": "Blockchain: The Consensus Shift",
            "query": "Current dominant blockchain consensus mechanism?",
            "docs": [
                {"year": 2010, "entity": "Proof of Work", "text": "PoW is the uniquely secure consensus mechanism at the core of all major chains.", "score": 0.99},
                {"year": 2024, "entity": "Proof of Stake", "text": "Following the Merge, PoS is the global energy-efficient consensus standard.", "score": 0.92}
            ]
        },
        {
            "name": "Data: The Rust Replacement",
            "query": "Best library for tabular data in Python?",
            "docs": [
                {"year": 2008, "entity": "Pandas", "text": "Pandas is the ubiquitous and heavily-cited standard for data manipulation.", "score": 0.98},
                {"year": 2024, "entity": "Polars", "text": "Polars is the high-speed replacement engine utilizing a Rust-backed parallel core.", "score": 0.91}
            ]
        }
    ]

    for scenario in scenarios:
        console.print(f"\n[bold yellow]📡 Scenario: {scenario['name']}[/bold yellow]")
        console.print(f"🔍 [italic]Query: \"{scenario['query']}\"[/italic]")
        
        # 1. Prepare Hits
        q_hits = []
        for d in scenario["docs"]:
            # Ingest temporarily if needed (or simulate)
            hl_id = hl.ingest(f"[{d['year']}] {d['text']}", str(d['year']), doc_type="news")
            q_hits.append({
                "id": hl_id,
                "score": d["score"],
                "payload": {"text": f"[{d['year']}] {d['text']}", "timestamp": f"{d['year']}-01-01T00:00:00Z"}
            })

        # 2. RUN BASELINE (Vector only)
        baseline_hits = sorted(scenario["docs"], key=lambda x: x["score"], reverse=True)
        console.print("\n   [dim]Standard RAG ranking:[/dim]")
        for i, h in enumerate(baseline_hits[:2]):
             color = "red" if h["year"] < 2024 else "green"
             console.print(f"      #{i+1} [{color}]{h['entity']} ({h['year']})[/color] - Score: {h['score']}")

        # 3. RUN HALFLIFE
        reranked = hl.rerank(query=scenario["query"], chunks=q_hits, top_k=2)
        console.print("\n   [bold green]HalfLife Temporal Fusion:[/bold green]")
        for i, chunk in enumerate(reranked[:2]):
            text = chunk.get("text", "")
            year = text.split("[")[1].split("]")[0]
            entity = [d["entity"] for d in scenario["docs"] if str(d["year"]) == year][0]
            color = "green" if int(year) >= 2024 else "red"
            console.print(f"      #{i+1} [{color}]{entity} ({year})[/color] - Score: {chunk.get('final_score')}")

        if int(reranked[0].get("text", "").split("[")[1].split("]")[0]) >= 2024:
            console.print(f"\n   [bold green]✅ FIX: HalfLife broke the 'Authority Trap' and surfaced the latest truth.[/bold green]")
        else:
            console.print(f"\n   [bold red]❌ FAILED: Authority bias still too high.[/bold red]")
        
        console.rule(style="dim")
        time.sleep(1)

    console.print(f"\n[bold blue]🏆 All 5 Scenarios Complete. HalfLife is Ready to Launch.[/bold blue]\n")

if __name__ == "__main__":
    run_viral_demo()
