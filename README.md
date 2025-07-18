[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)

# Hybrid Search System

This repository provides a template for a Hybrid Search system combining vector search (using `sentence-transformers`), BM25 (using Elasticsearch), and a LightGBM reranker, exposed via a FastAPI application. It uses PDM for modern dependency management.

## Features

* **Vector Search:** Leverages `sentence-transformers` to generate embeddings for semantic similarity.
* **BM25 Search:** Utilizes Elasticsearch for efficient keyword-based retrieval.
* **Hybrid Scoring:** Combines BM25 and vector scores for initial ranking.
* **LightGBM Reranking:** A machine learning model to re-rank the top hybrid results for improved relevance.
* **FastAPI Endpoint:** A high-performance web API to serve search queries.
* **PDM for Dependency Management:** Uses `pyproject.toml` and `pdm.lock` for reproducible environments.
* **Dockerized Deployment:** Easily deployable using Docker and Docker Compose.

## Installation

```bash
make install test
```

# hybrid_search
