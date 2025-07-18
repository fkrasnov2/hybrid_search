import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from app.services import HybridSearchService


@pytest.fixture(scope="session")
def event_loop():
    """Create a new event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_es_client():
    """Mocks the AsyncElasticsearch client."""
    mock = AsyncMock()
    mock.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "doc1",
                    "_source": {"text": "text1", "metadata": {"source": "test"}},
                    "_score": 0.8,
                },
                {
                    "_id": "doc2",
                    "_source": {"text": "text2", "metadata": {"source": "test"}},
                    "_score": 0.5,
                },
            ]
        }
    }
    mock.indices.exists.return_value = True
    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mocks the SentenceTransformer model."""
    mock = MagicMock()
    mock.encode.return_value = np.array([0.1, 0.2, 0.3])
    return mock


@pytest.fixture
def mock_reranker_model():
    """Mocks the LightGBM reranker model."""
    mock = MagicMock()
    mock.predict.return_value = np.array([0.9, 0.7, 0.6, 0.5, 0.4])
    return mock


@pytest.fixture
def hybrid_search_service(
    mock_es_client, mock_sentence_transformer, mock_reranker_model
):
    """Fixture for HybridSearchService with mocked dependencies."""
    with patch("app.services.AsyncElasticsearch", return_value=mock_es_client), patch(
        "app.services.SentenceTransformer", return_value=mock_sentence_transformer
    ), patch("app.services.os.path.exists", return_value=True), patch(
        "app.services.pickle.load", return_value=mock_reranker_model
    ):
        service = HybridSearchService(es_host="test_host", es_port=9200)
        yield service


@pytest.mark.asyncio
async def test_vector_search(hybrid_search_service, mock_es_client):
    """Test _vector_search method."""
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "vec_doc1",
                    "_source": {"text": "vector text 1", "metadata": {"tag": "A"}},
                    "_score": 0.95,
                },
                {
                    "_id": "vec_doc2",
                    "_source": {"text": "vector text 2", "metadata": {"tag": "B"}},
                    "_score": 0.80,
                },
            ]
        }
    }
    query_embedding = [0.1, 0.2, 0.3]
    results = await hybrid_search_service._vector_search(query_embedding, top_n=2)

    assert len(results) == 2
    assert results[0]["id"] == "vec_doc1"
    assert results[0]["vector_score"] == 0.95
    mock_es_client.search.assert_called_once()
    assert "script_score" in mock_es_client.search.call_args.kwargs["body"]["query"]


@pytest.mark.asyncio
async def test_bm25_search(hybrid_search_service, mock_es_client):
    """Test _bm25_search method."""
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "bm25_doc1",
                    "_source": {"text": "bm25 text 1", "metadata": {"category": "X"}},
                    "_score": 10.5,
                },
                {
                    "_id": "bm25_doc2",
                    "_source": {"text": "bm25 text 2", "metadata": {"category": "Y"}},
                    "_score": 8.2,
                },
            ]
        }
    }
    query = "test query"
    results = await hybrid_search_service._bm25_search(query, top_n=2)

    assert len(results) == 2
    assert results[0]["id"] == "bm25_doc1"
    assert results[0]["bm25_score"] == 10.5
    mock_es_client.search.assert_called_once()
    assert "match" in mock_es_client.search.call_args.kwargs["body"]["query"]


def test_normalize_scores_normal_case(hybrid_search_service):
    """Test _normalize_scores with varied scores."""
    scores = [10, 20, 30]
    normalized = hybrid_search_service._normalize_scores(scores)
    assert normalized == [0.0, 0.5, 1.0]


def test_normalize_scores_single_score(hybrid_search_service):
    """Test _normalize_scores with a single score."""
    scores = [15]
    normalized = hybrid_search_service._normalize_scores(scores)
    assert normalized == [0.0]


def test_normalize_scores_all_same(hybrid_search_service):
    """Test _normalize_scores with all scores being the same."""
    scores = [5, 5, 5]
    normalized = hybrid_search_service._normalize_scores(scores)
    assert normalized == [0.0, 0.0, 0.0]


def test_normalize_scores_empty_list(hybrid_search_service):
    """Test _normalize_scores with an empty list."""
    scores = []
    normalized = hybrid_search_service._normalize_scores(scores)
    assert normalized == []


@pytest.mark.asyncio
async def test_hybrid_search_no_reranker(
    hybrid_search_service, mock_es_client, mock_sentence_transformer
):
    """Test hybrid_search when reranker model is not loaded."""
    hybrid_search_service.reranker_model = None

    mock_es_client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": "docA",
                        "_source": {"text": "apple banana", "metadata": {}},
                        "_score": 10.0,
                    },
                    {
                        "_id": "docB",
                        "_source": {"text": "orange grape", "metadata": {}},
                        "_score": 5.0,
                    },
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_id": "docB",
                        "_source": {"text": "orange grape", "metadata": {}},
                        "_score": 0.9,
                    },
                    {
                        "_id": "docA",
                        "_source": {"text": "apple banana", "metadata": {}},
                        "_score": 0.7,
                    },
                ]
            }
        },
    ]

    query = "fruit"
    results = await hybrid_search_service.hybrid_search(query, top_k=2)

    assert len(results) == 2
    docA_result = next((r for r in results if r.id == "docA"), None)
    docB_result = next((r for r in results if r.id == "docB"), None)

    assert docA_result is not None
    assert docB_result is not None

    assert pytest.approx(docA_result.bm25_score) == 10.0
    assert pytest.approx(docA_result.vector_score) == 0.7
    assert pytest.approx(docA_result.hybrid_score) == 0.5
    assert pytest.approx(docA_result.rerank_score) == 0.5

    assert pytest.approx(docB_result.bm25_score) == 5.0
    assert pytest.approx(docB_result.vector_score) == 0.9
    assert pytest.approx(docB_result.hybrid_score) == 0.5
    assert pytest.approx(docB_result.rerank_score) == 0.5


@pytest.mark.asyncio
async def test_hybrid_search_with_reranker(
    hybrid_search_service,
    mock_es_client,
    mock_sentence_transformer,
    mock_reranker_model,
):
    """Test hybrid_search when reranker model is loaded."""
    mock_es_client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": f"doc{i}",
                        "_source": {"text": f"text {i}", "metadata": {}},
                        "_score": 10.0 - i,
                    }
                    for i in range(10)
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_id": f"doc{i}",
                        "_source": {"text": f"text {i}", "metadata": {}},
                        "_score": 0.9 - i * 0.05,
                    }
                    for i in range(10)
                ]
            }
        },
    ]

    mock_es_client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": f"doc{i}",
                        "_source": {"text": f"text {i}", "metadata": {}},
                        "_score": 10.0 - i,
                    }
                    for i in range(5)
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_id": f"doc{i}",
                        "_source": {"text": f"text {i}", "metadata": {}},
                        "_score": 0.9 - i * 0.05,
                    }
                    for i in range(5)
                ]
            }
        },
    ]

    query = "relevant documents"
    top_k = 2
    results = await hybrid_search_service.hybrid_search(query, top_k=top_k)

    assert len(results) == top_k
    assert results[0].rerank_score == 0.9
    assert results[1].rerank_score == 0.7
    mock_reranker_model.predict.assert_called_once()
    assert mock_sentence_transformer.encode.called
    assert mock_es_client.search.call_count == 2


@pytest.mark.asyncio
async def test_hybrid_search_no_results(hybrid_search_service, mock_es_client):
    """Test hybrid_search when no results are found from either search."""
    mock_es_client.search.return_value = {"hits": {"hits": []}}

    query = "nonexistent"
    results = await hybrid_search_service.hybrid_search(query, top_k=10)

    assert len(results) == 0
    assert mock_es_client.search.call_count == 2


def test_load_reranker_model_not_found(mock_es_client, mock_sentence_transformer):
    """Test _load_reranker_model when the file is not found."""
    with patch("app.services.os.path.exists", return_value=False), patch(
        "app.services.AsyncElasticsearch", return_value=mock_es_client
    ), patch(
        "app.services.SentenceTransformer", return_value=mock_sentence_transformer
    ):
        service = HybridSearchService(es_host="test_host", es_port=9200)
        assert service.reranker_model is None
