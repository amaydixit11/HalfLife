import requests
import xml.etree.ElementTree as ET
import time
from datetime import datetime, timezone
import os
import sys

# Ensure the root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ingestion.pipeline import HalfLifeIngestor

class ArxivLoader:
    """
    Pulls real research data from Arxiv to build a 'Ground Truth' 
    real-world corpus for HalfLife evaluation.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query?"

    def __init__(self, ingestor: HalfLifeIngestor = None):
        self.ingestor = ingestor or HalfLifeIngestor()

    def fetch_and_ingest(self, query: str, max_results: int = 50):
        """
        Fetches papers, extracts real dates, and ingests them into HalfLife.
        """
        print(f"📡 Fetching real-world research papers for '{query}' from Arxiv...")
        
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        response = requests.get(self.BASE_URL, params=params)
        if response.status_code != 200:
            print(f"❌ Failed to fetch from Arxiv: {response.status_code}")
            return
            
        root = ET.fromstring(response.content)
        # Arxiv uses Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        print(f"📥 Found {len(entries)} real papers. Starting ingestion...")

        ingested_count = 0
        for entry in entries:
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            published_str = entry.find('atom:published', ns).text
            
            # Parse Arxiv timestamp: 2024-03-25T15:00:00Z
            published = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            
            # Construct a rich text chunk including the title for context
            full_text = f"Title: {title}\nAbstract: {summary}"
            
            try:
                self.ingestor.ingest(
                    text=full_text,
                    timestamp=published,
                    source_domain="arxiv.org",
                    doc_type="research"
                )
                ingested_count += 1
                if ingested_count % 10 == 0:
                    print(f"   ...ingested {ingested_count}/{len(entries)}")
            except Exception as e:
                print(f"   ⚠️ Error ingesting '{title[:30]}...': {e}")

        print(f"\n✅ Successfully ingested {ingested_count} REAL research papers.")
        print(f"🚀 Now ready to run semantic vs temporal reranking on live Arxiv data.")

if __name__ == "__main__":
    loader = ArxivLoader()
    # Let's pull some real data for a hot topic
    loader.fetch_and_ingest("transformer architectures", max_results=30)
