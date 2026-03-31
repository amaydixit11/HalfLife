import os
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

try:
    from fastapi import FastAPI, HTTPException, Body
    from pydantic import BaseModel, Field
except ImportError:
    print("Please install fastapi and uvicorn: pip install fastapi uvicorn")
    sys.exit(1)

# Root path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from halflife import HalfLife

app = FastAPI(
    title="HalfLife Temporal Reranker API",
    description="Production-grade temporal fusion API for RAG systems.",
    version="0.6.0"
)

# Shared HalfLife engine instance
hl_engine = HalfLife()

class ChunkPayload(BaseModel):
    text: str
    timestamp: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RerankRequest(BaseModel):
    query: str
    chunks: List[Dict[str, Any]] # Matches the internal SDK format: {id, score, payload}
    top_k: int = 5
    weights: Optional[Dict[str, float]] = None
    intent: Optional[str] = None

class RerankResponse(BaseModel):
    results: List[Dict[str, Any]]
    consistency_warnings: List[str]
    applied_weights: Dict[str, float]
    processing_ms: float

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "HalfLife v0.6.0"}

@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Core reranking endpoint. Accepts a list of chunks (typically from a vector store)
    and returns them optimally re-weighted by temporal relevance.
    """
    start_time = datetime.now()
    
    try:
        results = hl_engine.rerank(
            query=request.query,
            chunks=request.chunks,
            top_k=request.top_k,
            weights=request.weights,
            intent=request.intent
        )
        
        end_time = datetime.now()
        processing_ms = (end_time - start_time).total_seconds() * 1000
        
        return {
            "results": results["reranked_chunks"],
            "consistency_warnings": results["consistency_warnings"],
            "applied_weights": results["applied_weights"],
            "processing_ms": round(processing_ms, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
