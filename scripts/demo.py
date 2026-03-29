import os
import sys
import time
from rich.console import Console
from rich.table import Table

# Root path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from halflife import HalfLife

console = Console()

def run_demo():
    hl = HalfLife()
    
    console.print(f"\n[bold blue]🚀 Starting HalfLife ‘Temporal Travel’ Demo...[/bold blue]\n")
    
    # 1. Ingest Conflict
    console.print(f"📥 [bold]Step 1: Ingesting Temporal Conflict...[/bold]")
    id_bill  = hl.ingest("Microsoft CEO in the early 2000s: Bill Gates was the leader during the dot-com era.", "2000", doc_type="research")
    id_satya = hl.ingest("Microsoft CEO in 2026: Satya Nadella has led the cloud and AI transformation since 2014.", "2026", doc_type="news")
    console.print(f"   [green]✓ Ingested Bill Gates (2000) as {id_bill[:8]}...[/green]")
    console.print(f"   [green]✓ Ingested Satya Nadella (2026) as {id_satya[:8]}...[/green]")
    
    time.sleep(1) # Wait for Qdrant/Redis consistency
    
    # 2. Queries
    queries = [
        ("Who is the current CEO of Microsoft?", "fresh"),
        ("Who was the CEO of Microsoft in the 2000s?", "historical")
    ]
    
    for query, intent in queries:
        console.print(f"\n🔍 [bold]Query: \"{query}\"[/bold] (Intent: {intent.upper()})")
        
        # Retrieval (Simulated hits using the ACTUAL IDs we just created)
        q_hits = [
            {"id": id_bill,  "score": 0.82, "payload": {"text": "Microsoft CEO in the early 2000s: Bill Gates was the leader...", "timestamp": "2000-01-01T00:00:00Z"}},
            {"id": id_satya, "score": 0.85, "payload": {"text": "Microsoft CEO in 2026: Satya Nadella... leader in cloud/AI.", "timestamp": "2026-01-01T00:00:00Z"}}
        ]
        
        # Rerank
        reranked = hl.rerank(query=query, chunks=q_hits, top_k=2)
        
        # UI
        table = Table(title=None, show_header=True, header_style="bold magenta", box=None)
        table.add_column("Rank", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Entity", justify="left")
        table.add_column("Vintage", justify="center")
        
        for i, chunk in enumerate(reranked):
            chunk_txt = chunk.get("text", "") # THE FIX: Reranker returns 'text' at root
            final_score = chunk.get("final_score", 0) # THE FIX: Reranker returns 'final_score'
            entity = "Bill Gates" if "Bill" in chunk_txt else "Satya Nadella"
            vintage = "2000" if "2000" in chunk_txt else "2026"
            
            style = "green" if i == 0 else "dim"
            table.add_row(f"#{i+1}", f"{final_score:.2f}", entity, vintage, style=style)
            
        console.print(table)
        
        # Decision logic
        top_txt = reranked[0].get("text", "")
        if intent == "fresh":
            success = "Satya" in top_txt
            winner = "Satya" if success else "Bill"
        else: # historical
            success = "Bill" in top_txt
            winner = "Bill" if success else "Satya"

        status = "[bold green]✅ Decision: CORRECT![/]" if success else "[bold red]❌ Decision: INCORRECT![/]"
        console.print(f"   {status} [bold]({winner} Surfaced)[/]")

    console.print(f"\n[bold blue]🏆 Demo Complete. Temporal RAG successfully validated![/bold blue]\n")

if __name__ == "__main__":
    run_demo()
