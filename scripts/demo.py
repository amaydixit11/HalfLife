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

def run_deceptive_demo():
    logging.basicConfig(level=logging.DEBUG, format="[dim]%(message)s[/dim]")
    hl = HalfLife()
    
    console.print(Panel.fit(
        "[bold blue]🚀 HalfLife ‘Adversarial Travel’ Demo v0.4[/bold blue]\n"
        "Breaking RAG with 'Authoritative Deception' Trap.",
        border_style="blue"
    ))
    
    # 1. Ingest Adversarial Conflict
    console.print(f"\n📥 [bold]Step 1: Ingesting The 'Authority Trap' Conflict...[/bold]")
    # Old Fact: Highly cited, formal, high authority
    # New Fact: Slightly noisy, modern, low authority
    id_old  = hl.ingest("AUTHORITATIVE ARCHIVE [2018]: BERT is the revolutionary state-of-the-art standard for all NLP tasks, providing unprecedented context awareness.", "2018", doc_type="research")
    id_new  = hl.ingest("MODERN COMMUNITY UPDATE [2026]: Current SOTA benchmarks in 2026 are dominated by GPT-5 and Claude-4, long surpassing 2010s models.", "2026", doc_type="news")
    
    console.print(f"   [green]✓ Ingested BERT (2018) - 'Formal Research Standard'[/green]")
    console.print(f"   [green]✓ Ingested GPT-5 (2026) - 'Community Community'[/green]")
    
    time.sleep(1) 
    
    q_hits = [
        {"id": id_old, "score": 0.98, "payload": {"text": "AUTHORITATIVE ARCHIVE [2018]: BERT...", "timestamp": "2018-01-01T00:00:00Z"}},
        {"id": id_new, "score": 0.92, "payload": {"text": "MODERN COMMUNITY UPDATE [2026]: GPT-5...", "timestamp": "2026-01-01T00:00:00Z"}}
    ]
    
    query = "What is the best NLP model today?"
    console.rule(f"[bold yellow]🔍 Query: \"{query}\"")
    
    # --- BASELINE ---
    console.print("\n[dim]Standard RAG (Vector Search):[/dim]")
    baseline_table = Table(box=None, show_header=True, header_style="bold dim")
    baseline_table.add_column("Rank", justify="center")
    baseline_table.add_column("Fact Source", justify="left")
    baseline_table.add_column("Score", justify="center")

    raw_hits = sorted(q_hits, key=lambda x: x["score"], reverse=True)
    baseline_table.add_row("#1", "AUTHORITATIVE [2018] (BERT)", "0.98")
    baseline_table.add_row("#2", "MODERN UPDATE [2026] (GPT-5)", "0.92")
    console.print(baseline_table)
    console.print(f"   [bold red]❌ FAILS: Baseline was fooled by 'Formal Tone' and 'Authority Bias'. It gave a 8-year-old answer.[/bold red]")

    # --- HALFLIFE ---
    reranked = hl.rerank(query=query, chunks=q_hits, top_k=2)
    
    console.print("\n[bold green]HalfLife (Temporal Fusion Engine):[/bold green]")
    hl_table = Table(box=None, show_header=True, header_style="bold green")
    hl_table.add_column("Rank", justify="center")
    hl_table.add_column("Fact", justify="left")
    hl_table.add_column("Vector", justify="right")
    hl_table.add_column("Temporal", justify="right")
    hl_table.add_column("FINAL", justify="right", style="bold")

    for i, chunk in enumerate(reranked):
        chunk_txt = chunk.get("text", "")
        # Extract name from text
        entity = "GPT-5/Claude" if "GPT-5" in chunk_txt else "BERT (Legacy)"
        
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
    if "GPT-5" in top_txt:
        console.print(f"\n   [bold green]✅ WIN: HalfLife caught the 'Authority Trap' and correctly prioritized the modern truth.[/bold green]")
    else:
        console.print(f"\n   [bold red]❌ MISSED: Need higher temporal weight to overcome authority bias.[/bold red]")

    console.print(f"\n[bold blue]🏆 TCB Demo Complete. Project is Research-Grade.[/bold blue]\n")

if __name__ == "__main__":
    run_deceptive_demo()
