import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.ingestion.pipeline import HalfLifeIngestor

def generate_corpus():
    ingestor = HalfLifeIngestor()
    now = datetime.now(timezone.utc)
    
    print("Generating synthetic benchmark corpus...")
    
    # 1. AI Research Evolution (Transformers)
    # We want to see if "history of" surfaces the 2017 paper vs 2024 updates
    transformers_data = [
        {
            "text": "Attention Is All You Need. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
            "timestamp": datetime(2017, 6, 12, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        },
        {
            "text": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. BERT is designed to pre-train deep bidirectional representations.",
            "timestamp": datetime(2018, 10, 11, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        },
        {
            "text": "RoBERTa: A Robustly Optimized BERT Pretraining Approach. We find that BERT was significantly undertrained and can match or exceed later models.",
            "timestamp": datetime(2019, 7, 26, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        },
        {
            "text": "Language Models are Few-Shot Learners. We present GPT-3, an autoregressive language model with 175 billion parameters.",
            "timestamp": datetime(2020, 5, 28, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        },
        {
            "text": "Llama 2: Open Foundation and Fine-Tuned Chat Models. We develop and release Llama 2, a collection of pre-trained and fine-tuned LLMs.",
            "timestamp": datetime(2023, 7, 18, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        },
        {
            "text": "The Era of 1-bit LLMs: All Large Language Models are in 1.58 bits. We introduce BitNet b1.58, where every single parameter is ternary.",
            "timestamp": datetime(2024, 2, 27, tzinfo=timezone.utc),
            "doc_type": "research",
            "source_domain": "arxiv",
        }
    ]
    
    # 2. News/Current Events (GPU releases)
    # We want "latest GPUs" to surface 2024 data
    gpu_data = [
        {
            "text": "NVIDIA Launches GeForce RTX 30 Series. The ultimate play. Powered by Ampere architecture.",
            "timestamp": datetime(2020, 9, 1, tzinfo=timezone.utc),
            "doc_type": "news",
            "source_domain": "nvidia.com",
        },
        {
            "text": "NVIDIA GeForce RTX 4090 released. Beyond Fast. The world's fastest consumer GPU.",
            "timestamp": datetime(2022, 10, 12, tzinfo=timezone.utc),
            "doc_type": "news",
            "source_domain": "nvidia.com",
        },
        {
            "text": "NVIDIA Blackwell platform announced. Deep Learning training performance jumps by 25x.",
            "timestamp": datetime(2024, 3, 18, tzinfo=timezone.utc),
            "doc_type": "news",
            "source_domain": "nvidia.com",
        }
    ]
    
    # 3. Documentation (API Versions)
    api_data = [
        {
            "text": "API v1.0 Documentation. Initial release of the REST interface. Endpoints use /api/v1 prefix.",
            "timestamp": datetime(2021, 1, 1, tzinfo=timezone.utc),
            "doc_type": "documentation",
            "source_domain": "internal",
        },
        {
            "text": "API v2.0 Released. Major breaking changes. Unified GraphQL interface introduced.",
            "timestamp": datetime(2023, 6, 15, tzinfo=timezone.utc),
            "doc_type": "documentation",
            "source_domain": "internal",
        }
    ]
    
    all_chunks = transformers_data + gpu_data + api_data
    ids = ingestor.ingest_batch(all_chunks)
    
    print(f"Ingested {len(ids)} chunks into HalfLife.")
    return ids

if __name__ == "__main__":
    generate_corpus()
