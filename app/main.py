from typing import List

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query

from app.utils.config_loader import Config

from .models import SearchResult
from .services import HybridSearchService

config = Config()

app = FastAPI(
    title="Hybrid Search API",
    description="A FastAPI endpoint for Hybrid Search combining BM25, Vector Search, and LightGBM Reranking.",
    version="1.0.0",
)

es_host = config.get_elasticsearch_host()
es_port = config.get_elasticsearch_port()

try:
    search_service = HybridSearchService(es_host=es_host, es_port=es_port)
except Exception as e:
    raise HTTPException(
        status_code=500, detail=f"Failed to initialize search service: {e}"
    )


@app.get("/search", response_model=List[SearchResult])
async def search(
    q: str = Query(
        ..., min_length=1, max_length=200, description="The search query string."
    ),
    top_k: int = Query(
        10, ge=1, le=50, description="Number of top results to return after reranking."
    ),
) -> List[SearchResult]:
    """
    Performs a hybrid search combining BM25 and vector search, followed by LightGBM reranking.
    """
    try:
        results = await search_service.hybrid_search(query=q, top_k=top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    try:
        es_info = await search_service.es_client.info()
        return {"status": "ok", "elasticsearch": es_info.body["tagline"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")
