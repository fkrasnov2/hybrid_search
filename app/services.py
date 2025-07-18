import logging
import os
import pickle
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer

from app.utils.config_loader import Config

from .models import SearchResult

config = Config()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HybridSearchService:
    def __init__(self, es_host: str, es_port: int, es_index: str = "documents"):
        self.es_client = AsyncElasticsearch(
            [{"host": es_host, "port": es_port, "scheme": "http"}]
        )
        self.es_index = es_index
        self.vector_model = SentenceTransformer(config.get_vector_model_name())
        self.reranker_model = self._load_reranker_model()

    def _load_reranker_model(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "..", config.get_reranker_model_path()
        )
        if not os.path.exists(model_path):
            logging.info(
                f"Warning: Reranker model not found at {model_path}. Please run scripts/train_reranker.py"
            )
            return None
        with open(model_path, "rb") as f:
            return pickle.load(f)

    async def _vector_search(
        self, query_embedding: List[float], top_n: int = 20
    ) -> List[Dict[str, Any]]:
        """Performs vector similarity search on Elasticsearch."""
        search_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding},
                    },
                }
            },
            "size": top_n,
            "_source": ["text", "metadata"],
        }
        response = await self.es_client.search(index=self.es_index, body=search_query)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "vector_score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                }
            )
        return results

    async def _bm25_search(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """Performs BM25 keyword search on Elasticsearch."""
        search_query = {
            "query": {"match": {"text": query}},
            "size": top_n,
            "_source": ["text", "metadata"],
        }
        response = await self.es_client.search(index=self.es_index, body=search_query)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(
                {
                    "id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "bm25_score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                }
            )
        return results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [0.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    async def hybrid_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Combines BM25 and vector search, then reranks using LightGBM.
        """
        query_embedding = self.vector_model.encode(query).tolist()

        bm25_results = await self._bm25_search(query, top_n=top_k * 2)
        vector_results = await self._vector_search(query_embedding, top_n=top_k * 2)

        combined_results: Dict[str, Dict[str, Any]] = {}

        bm25_scores = [res.get("bm25_score", 0.0) for res in bm25_results]
        normalized_bm25_scores = self._normalize_scores(bm25_scores)
        for i, res in enumerate(bm25_results):
            doc_id = res["id"]
            combined_results[doc_id] = {
                "id": doc_id,
                "text": res["text"],
                "bm25_score": res.get("bm25_score", 0.0),
                "vector_score": 0.0,
                "normalized_bm25_score": normalized_bm25_scores[i],
                "normalized_vector_score": 0.0,
                "metadata": res["metadata"],
            }

        vector_scores = [res.get("vector_score", 0.0) for res in vector_results]
        normalized_vector_scores = self._normalize_scores(vector_scores)
        for i, res in enumerate(vector_results):
            doc_id = res["id"]
            if doc_id not in combined_results:
                combined_results[doc_id] = {
                    "id": doc_id,
                    "text": res["text"],
                    "bm25_score": 0.0,
                    "vector_score": res.get("vector_score", 0.0),
                    "normalized_bm25_score": 0.0,
                    "normalized_vector_score": normalized_vector_scores[i],
                    "metadata": res["metadata"],
                }
            else:
                combined_results[doc_id]["vector_score"] = res.get("vector_score", 0.0)
                combined_results[doc_id][
                    "normalized_vector_score"
                ] = normalized_vector_scores[i]

        for doc_id, data in combined_results.items():
            data["hybrid_score"] = (
                0.5 * data["normalized_bm25_score"]
                + 0.5 * data["normalized_vector_score"]
            )

        candidate_results = sorted(
            combined_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[: top_k * 5]

        if not candidate_results:
            return []

        rerank_features = []
        for result in candidate_results:
            features = [
                result["normalized_bm25_score"],
                result["normalized_vector_score"],
            ]
            rerank_features.append(features)

        rerank_scores = []
        if self.reranker_model:
            rerank_scores = self.reranker_model.predict(np.array(rerank_features))
        else:
            rerank_scores = [res["hybrid_score"] for res in candidate_results]

        final_results = []
        for i, result in enumerate(candidate_results):
            final_results.append(
                SearchResult(
                    id=result["id"],
                    text=result["text"],
                    bm25_score=result["bm25_score"],
                    vector_score=result["vector_score"],
                    hybrid_score=result["hybrid_score"],
                    rerank_score=float(rerank_scores[i]),
                    metadata=result["metadata"],
                )
            )

        final_results.sort(key=lambda x: x.rerank_score, reverse=True)

        return final_results[:top_k]
