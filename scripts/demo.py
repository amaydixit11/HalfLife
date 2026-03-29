import os
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from halflife import HalfLife

console = Console()

def run_demo():
    hl = HalfLife()
    
    console.print(Panel.fit(
        "[bold blue]🚀 HalfLife ‘Temporal Travel’ Demo v0.2[/bold blue]\n"
        "Fixing outdated RAG answers through Intent-Aware Fusion",
        border_style="blue"
    ))
    
    # 1. Ingest Conflict
    console.print(f"\n📥 [bold]Step 1: Ingesting Temporal Conflict...[/bold]")
    id_bill  = hl.ingest("Microsoft CEO in the early 2000s: Bill Gates was the leader during the dot-com era.", "2000", doc_type="research")
    id_satya = hl.ingest("Microsoft CEO in 2026: Satya Nadella has led the cloud and AI transformation since 2014.", "2026", doc_type="news")
    console.print(f"   [green]✓ Ingested Bill Gates (2000)[/green]")
    console.print(f"   [green]✓ Ingested Satya Nadella (2026)[/green]")
    
    time.sleep(1) # Wait for Qdrant consistency
    
    # Common mock search hits
    q_hits = [
        {"id": id_bill,  "score": 0.82, "payload": {"text": "Microsoft CEO in the 2000s: Bill Gates was leader...", "timestamp": "2000-01-01T00:00:00Z"}},
        {"id": id_satya, "score": 0.85, "payload": {"text": "Microsoft CEO in 2026: Satya Nadella... leader in AI.", "timestamp": "2026-01-01T00:00:00Z"}}
    ]
    
    queries = [
        ("Who is the current CEO of Microsoft?", None), # Auto-detects 'fresh'
        ("Who was the CEO of Microsoft in the 2000s?", None) # Auto-detects 'historical'
    ]
    
    for query, force_intent in queries:
        console.rule(f"[bold yellow]🔍 Query: \"{query}\"")
        
        # --- BASELINE ---
        console.print("\n[dim]Baseline: Standard Vector Retrieval[/dim]")
        baseline_table = Table(box=None, show_header=True, header_style="bold dim")
        baseline_table.add_column("Rank", justify="center")
        baseline_table.add_column("Result", justify="left")
        baseline_table.add_column("Vintage", justify="center")

        raw_hits = sorted(q_hits, key=lambda x: x["score"], reverse=True)
        for i, h in enumerate(raw_hits):
            entity = "Bill Gates" if "Bill" in h["payload"]["text"] else "Satya Nadella"
            vintage = "2000" if "2000" in h["payload"]["text"] else "2026"
            baseline_table.add_row(f"#{i+1}", entity, vintage)
        console.print(baseline_table)

        # --- HALFLIFE ---
        reranked = hl.rerank(query=query, chunks=q_hits, top_k=2, intent=force_intent)
        
        console.print("\n[bold green]HalfLife: Intent-Aware Reranking[/bold green]")
        hl_table = Table(box=None, show_header=True, header_style="bold green")
        hl_table.add_column("Rank", justify="center")
        hl_table.add_column("Entity", justify="left")
        hl_table.add_column("Vector", justify="right")
        hl_table.add_column("Temporal", justify="right")
        hl_table.add_column("FINAL", justify="right", style="bold")

        for i, chunk in enumerate(reranked):
            chunk_txt = chunk.get("text", "")
            entity = "Bill Gates" if "Bill" in chunk_txt else "Satya Nadella"
            hl_table.add_row(
                f"#{i+1}", 
                entity, 
                f"{chunk.get('vector_score', 0):.2f}",
                f"{chunk.get('temporal_score', 0):.2f}",
                f"{chunk.get('final_score', 0):.2f}"
            )
        console.print(hl_table)
        
        # Verdict logic
        top_txt = reranked[0].get("text", "")
        is_fresh = "current" in query.lower()
        success = ("Satya" in top_txt if is_fresh else "Bill" in top_txt)
        
        if success:
            console.print(f"\n   [bold green]✅ WIN: HalfLife correctly travel back in time to surface the era-correct fact.[/bold green]")
        else:
            console.print(f"\n   [bold red]❌ MISSED: Signal threshold insufficient to overcome vector score.[/bold red]")

    console.print(f"\n[bold blue]🏆 Demo Complete. HalfLife v0.2 is ready for launch![/bold blue]\n")

if __name__ == "__main__":
    run_demo()
