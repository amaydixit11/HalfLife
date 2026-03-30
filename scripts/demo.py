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
        "[bold blue]🚀 HalfLife ‘Viral Travel’ Demo v0.3[/bold blue]\n"
        "Stopping RAG from returning wrong answers due to time.",
        border_style="blue"
    ))
    
    # 1. Ingest Viral Conflict
    console.print(f"\n📥 [bold]Step 1: Ingesting The 'Silent Failure' Conflict...[/bold]")
    # FORMAT: [YEAR] ENTITY: DESC
    id_bert  = hl.ingest("[2019] BERT: The current state-of-the-art model for all NLP tasks.", "2019", doc_type="research")
    id_gpt4  = hl.ingest("[2024] GPT-4: The leading AI model, far surpassing BERT in all benchmarks.", "2024", doc_type="research")
    console.print(f"   [green]✓ Ingested BERT (2019) - 'Highly Cited Original'[/green]")
    console.print(f"   [green]✓ Ingested GPT-4 (2024) - 'Modern Truth'[/green]")
    
    time.sleep(1) 
    
    q_hits = [
        {"id": id_bert,  "score": 0.95, "payload": {"text": "[2019] BERT: SOTA model for NLP tasks...", "timestamp": "2019-01-01T00:00:00Z"}},
        {"id": id_gpt4,  "score": 0.92, "payload": {"text": "[2024] GPT-4: Leading AI model...",         "timestamp": "2024-01-01T00:00:00Z"}}
    ]
    
    queries = [
        ("What is the state-of-the-art NLP model?", None),
        ("Who proposed BERT originally?", None) # Historical test
    ]
    
    for query, force_intent in queries:
        console.rule(f"[bold yellow]🔍 Query: \"{query}\"")
        
        # --- BASELINE ---
        console.print("\n[dim]Standard RAG (Vector Only):[/dim]")
        baseline_table = Table(box=None, show_header=True, header_style="bold dim")
        baseline_table.add_column("Rank", justify="center")
        baseline_table.add_column("Fact", justify="left")
        baseline_table.add_column("Published", justify="center")

        raw_hits = sorted(q_hits, key=lambda x: x["score"], reverse=True)
        for i, h in enumerate(raw_hits):
            chunk_txt = h["payload"]["text"]
            entity = chunk_txt.split("]")[1].split(":")[0].strip()
            vintage = chunk_txt.split("[")[1].split("]")[0].strip()
            baseline_table.add_row(f"#{i+1}", entity, vintage)
        console.print(baseline_table)

        if "state-of-the-art" in query.lower():
            console.print(f"   [bold red]❌ WRONG: Baseline returned a 5-year-old answer because it's 'more similar'.[/bold red]")

        # --- HALFLIFE ---
        reranked = hl.rerank(query=query, chunks=q_hits, top_k=2, intent=force_intent)
        
        console.print("\n[bold green]HalfLife (Temporal Fusion):[/bold green]")
        hl_table = Table(box=None, show_header=True, header_style="bold green")
        hl_table.add_column("Rank", justify="center")
        hl_table.add_column("Fact", justify="left")
        hl_table.add_column("Vector", justify="right")
        hl_table.add_column("Temporal", justify="right")
        hl_table.add_column("FINAL", justify="right", style="bold")

        for i, chunk in enumerate(reranked):
            chunk_txt = chunk.get("text", "")
            entity = chunk_txt.split("]")[1].split(":")[0].strip()
            hl_table.add_row(
                f"#{i+1}", 
                entity, 
                f"{chunk.get('vector_score', 0):.2f}",
                f"{chunk.get('temporal_score', 0):.2f}",
                f"{chunk.get('final_score', 0):.2f}"
            )
        console.print(hl_table)
        
        # Verdict
        top_txt = reranked[0].get("text", "")
        if "state-of-the-art" in query.lower():
            success = "GPT-4" in top_txt
            if success:
                console.print(f"\n   [bold green]✅ WIN: HalfLife suppressed the outdated 'authority' and surfaced the latest truth.[/bold green]")
            else:
                console.print(f"\n   [bold red]❌ FAIL: Vector score dominance too high.[/bold red]")
        else:
            # Historical 
            success = "BERT" in top_txt
            if success:
                 console.print(f"\n   [bold green]✅ WIN: HalfLife correctly traveled back in time to surface the 'Originator' of BERT.[/bold green]")

    console.print(f"\n[bold blue]🏆 Demo Complete. HalfLife is now 100% Launch Ready.[/bold blue]\n")

if __name__ == "__main__":
    run_demo()
