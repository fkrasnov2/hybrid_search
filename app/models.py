from typing import Any
from typing import Dict

from pydantic import BaseModel


class SearchResult(BaseModel):
    id: str
    text: str
    bm25_score: float
    vector_score: float
    hybrid_score: float
    rerank_score: float
    metadata: Dict[str, Any]
