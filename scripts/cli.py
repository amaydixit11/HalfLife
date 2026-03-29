import argparse
import sys
import os

# Ensure root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser(
        prog="halflife",
        description="HalfLife: Temporal RAG Middleware v0.2",
        epilog="Examples:\n  halflife benchmark --output results.json\n  halflife evaluate --ablation\n  halflife serve --port 8080"
    )
    parser.add_argument("--version", action="version", version="HalfLife v0.2")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- CLI: benchmark ---
    bench_parser = subparsers.add_parser("benchmark", help="Run nDCG/MRR/TF benchmark on tiered corpus")
    bench_parser.add_argument("--output", type=str, help="Save summary to JSON")
    bench_parser.add_argument("--skip-ingest", action="store_true", help="Skip re-ingesting corpus")
    bench_parser.add_argument("--decay-type", choices=["exponential", "linear", "learned"], default="exponential")

    # --- CLI: evaluate (Research Mode) ---
    eval_parser = subparsers.add_parser("evaluate", help="Research Ablation on Temporal QA")
    eval_parser.add_argument("--dataset", default="scripts/temporal_qa.json")
    eval_parser.add_argument("--ablation", action="store_true", help="Run all 4 research variants")
    eval_parser.add_argument("--output", type=str, help="Save results to JSON")

    # --- CLI: serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the HalfLife API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    # --- CLI: train ---
    train_parser = subparsers.add_parser("train", help="Train the LearnedDecay (MLP) engine")
    train_parser.add_argument("--results", required=True, help="Input benchmark results.json")
    train_parser.add_argument("--output", default="decay_mlp.npz", help="Where to save model weights")
    train_parser.add_argument("--epochs", type=int, default=100)

    # --- CLI: quickstart ---
    quick_parser = subparsers.add_parser("quickstart", help="Complete E2E Demonstration (Ingest -> Query -> Rerank)")

    # --- CLI: demo ---
    demo_parser = subparsers.add_parser("demo", help="Zero-Friction 'Temporal Travel' Demo (Historical vs Fresh)")

    args = parser.parse_args()

    if args.command == "benchmark":
        print("🚀 Starting HalfLife Research Benchmark...")
        try:
            from scripts.benchmark import main as benchmark_func
            benchmark_func(
                output=args.output,
                skip_ingest=args.skip_ingest,
                decay_type=args.decay_type
            )
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            sys.exit(1)

    elif args.command == "evaluate":
        print("🏛️ Starting Research Evaluation Suite...")
        try:
            from scripts.evaluate import ResearchEvaluator
            evaluator = ResearchEvaluator()
            evaluator.evaluate(args.dataset)
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            sys.exit(1)

    elif args.command == "serve":
        import uvicorn
        print(f"📡 Starting HalfLife API on {args.host}:{args.port}...")
        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=True)

    elif args.command == "train":
        print("🧠 Training Neural Fusion Layer...")
        try:
            from scripts.train_mlp import train
            train(results_path=args.results, output_path=args.output, epochs=args.epochs)
        except Exception as e:
            print(f"❌ Training failed: {e}")
            sys.exit(1)

    elif args.command == "quickstart":
        print("⚡ Launching HalfLife Quickstart Demo...")
        from scripts.quickstart import main as quickstart_main
        quickstart_main()

    elif args.command == "demo":
        from scripts.demo import run_demo
        run_demo()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
