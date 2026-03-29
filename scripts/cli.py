import argparse
import sys
import os

# Ensure the root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(
        prog="halflife",
        description="HalfLife: Temporal-Aware RAG Reranking Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- CLI: benchmark ---
    benchmark_parser = subparsers.add_parser("benchmark", help="Run the evaluation benchmark")
    benchmark_parser.add_argument("--output", type=str, help="Save results to JSON")
    benchmark_parser.add_argument("--skip-ingest", action="store_true", help="Skip data ingestion")
    benchmark_parser.add_argument("--decay-type", type=str, help="Force a decay type (e.g. learned)")

    # --- CLI: train ---
    train_parser = subparsers.add_parser("train", help="Train the Learned Decay MLP")
    train_parser.add_argument("--results", required=True, help="Path to benchmark results JSON")
    train_parser.add_argument("--output", default="decay_mlp.npz", help="Output weights path")
    train_parser.add_argument("--epochs", type=int, default=500)

    # --- CLI: quickstart ---
    subparsers.add_parser("quickstart", help="Run the end-to-end integration demo")

    # --- CLI: serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI reranking server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "benchmark":
        from scripts.benchmark import main as benchmark_main
        # Re-parse sys.argv for the sub-module if needed, or pass args directly
        # For simplicity, we'll patch sys.argv and call main
        sys.argv = [sys.argv[0]]
        if args.output: sys.argv.extend(["--output", args.output])
        if args.skip_ingest: sys.argv.append("--skip-ingest")
        if args.decay_type: sys.argv.extend(["--decay-type", args.decay_type])
        benchmark_main()

    elif args.command == "train":
        from scripts.train_mlp import train
        train(results_path=args.results, output_path=args.output, epochs=args.epochs)

    elif args.command == "quickstart":
        from scripts.quickstart import run_quickstart
        run_quickstart()

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
