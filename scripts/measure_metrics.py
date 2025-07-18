import json
import logging
from typing import Dict
from typing import List

import numpy as np
import requests
from sklearn.metrics import ndcg_score

from app.utils.config_loader import Config

config = Config()

FASTAPI_APP_URL = config.get_fastapi_app_url()
GROUND_TRUTH_FILE = config.get_ground_truth_file()
K_VALUES = config.get_k_values()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_search_results(query: str, top_k: int) -> List[str]:
    """
    Makes an API call to the FastAPI search endpoint and returns a list of document IDs.
    """
    try:
        response = requests.get(
            f"{FASTAPI_APP_URL}/search", params={"q": query, "top_k": top_k}
        )
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        search_results = response.json()
        return [result["id"] for result in search_results]
    except requests.exceptions.ConnectionError as e:
        logging.info(f"Connection error to FastAPI app at {FASTAPI_APP_URL}: {e}")
        logging.info("Please ensure your FastAPI app is running and accessible.")
        return []
    except requests.exceptions.RequestException as e:
        logging.info(f"Error during API call for query '{query}': {e}")
        return []


def calculate_ndcg_at_k(
    ground_truth_relevance: Dict[str, int], retrieved_doc_ids: List[str], k: int
) -> float:
    """
    Calculates NDCG@k for a single query.
    Args:
        ground_truth_relevance: Dict of doc_id -> relevance score (e.g., {"doc1": 2, "doc2": 1}).
        retrieved_doc_ids: List of document IDs retrieved by the search system, in order.
        k: The 'k' value for NDCG@k.
    Returns:
        The NDCG@k score.
    """
    if not retrieved_doc_ids or k <= 0:
        return 0.0

    # Create ideal relevance scores for NDCG calculation
    # Sort ground truth by relevance descending to get ideal order
    ideal_relevance_sorted = sorted(ground_truth_relevance.values(), reverse=True)

    # Create relevance scores for the retrieved list
    # Use 0 if a retrieved doc_id is not in ground_truth_relevance
    actual_relevance = [
        ground_truth_relevance.get(doc_id, 0) for doc_id in retrieved_doc_ids[:k]
    ]

    # Pad actual_relevance if it's shorter than k but we need to compare against ideal
    # This is important for sklearn's ndcg_score to handle lists shorter than k correctly
    actual_relevance_padded = np.asarray([actual_relevance])
    ideal_relevance_padded = np.asarray([ideal_relevance_sorted[:k]])

    # Ensure ideal_relevance_padded has at least one element if actual has
    if actual_relevance_padded.shape[1] == 0 and ideal_relevance_padded.shape[1] == 0:
        return 0.0  # No relevant documents and no retrieved documents

    # If actual_relevance_padded is empty but ideal is not, ndcg_score will be 0
    # If ideal_relevance_padded is empty but actual is not, ndcg_score will be 0
    try:
        score = ndcg_score(ideal_relevance_padded, actual_relevance_padded)
        return score
    except ValueError as e:
        logging.info(
            f"Warning: Could not calculate NDCG for k={k}. Error: {e}. Returning 0.0"
        )
        return 0.0


def calculate_recall_at_k(
    all_relevant_ids: List[str], retrieved_doc_ids: List[str], k: int
) -> float:
    """
    Calculates Recall@k for a single query.
    Args:
        all_relevant_ids: List of all known relevant document IDs for the query.
        retrieved_doc_ids: List of document IDs retrieved by the search system, in order.
        k: The 'k' value for Recall@k.
    Returns:
        The Recall@k score.
    """
    if not all_relevant_ids:
        return 1.0
    if not retrieved_doc_ids:
        return 0.0

    retrieved_at_k = set(retrieved_doc_ids[:k])
    relevant_found = len(retrieved_at_k.intersection(set(all_relevant_ids)))
    return relevant_found / len(all_relevant_ids)


async def main():
    logging.info(f"Loading ground truth data from: {GROUND_TRUTH_FILE}")
    try:
        with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
            ground_truth_data = json.load(f)
    except FileNotFoundError:
        logging.info(f"Error: Ground truth file not found at {GROUND_TRUTH_FILE}")
        logging.info(
            "Please create 'data/ground_truth.json' with your test queries and relevance judgments."
        )
        return
    except json.JSONDecodeError:
        logging.info(
            f"Error: Could not parse {GROUND_TRUTH_FILE}. Ensure it's valid JSON."
        )
        return

    total_ndcg_scores: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    total_recall_scores: Dict[int, float] = {k: 0.0 for k in K_VALUES}
    num_queries = len(ground_truth_data)

    if num_queries == 0:
        logging.info("No queries found in ground truth data. Exiting.")
        return

    logging.info(f"\n--- Running metrics evaluation for {num_queries} queries ---")

    for i, entry in enumerate(ground_truth_data):
        query = entry["query"]
        ground_truth_relevance = entry["relevant_documents"]
        all_relevant_ids = entry["all_relevant_ids"]

        max_k = max(K_VALUES)
        retrieved_doc_ids = get_search_results(query, top_k=max_k)

        logging.info(f"\nQuery {i+1}/{num_queries}: '{query}'")
        logging.info(f"  Retrieved IDs (top {max_k}): {retrieved_doc_ids}")
        logging.info(f"  Relevant IDs: {all_relevant_ids}")

        for k in K_VALUES:
            ndcg = calculate_ndcg_at_k(ground_truth_relevance, retrieved_doc_ids, k)
            recall = calculate_recall_at_k(all_relevant_ids, retrieved_doc_ids, k)

            total_ndcg_scores[k] += ndcg
            total_recall_scores[k] += recall

            logging.info(f"    NDCG@{k}: {ndcg:.4f}, Recall@{k}: {recall:.4f}")

    logging.info("\n--- Average Scores Across All Queries ---")
    for k in K_VALUES:
        avg_ndcg = total_ndcg_scores[k] / num_queries
        avg_recall = total_recall_scores[k] / num_queries
        logging.info(f"Average NDCG@{k}: {avg_ndcg:.4f}")
        logging.info(f"Average Recall@{k}: {avg_recall:.4f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
