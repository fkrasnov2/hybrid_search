import asyncio
import json
import logging
import os

from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer

from app.utils.config_loader import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

config = Config()

ELASTICSEARCH_HOST = config.get_elasticsearch_host()
ELASTICSEARCH_PORT = config.get_elasticsearch_port()
ES_INDEX = config.get_es_index()
DATA_FILE = config.get_data_file()
VECTOR_MODEL_NAME = config.get_vector_model_name()


async def create_and_index_documents():
    """
    Connects to Elasticsearch, creates/recreates the index,
    and indexes documents from the specified data file.
    """
    es = AsyncElasticsearch(
        [{"host": ELASTICSEARCH_HOST, "port": ELASTICSEARCH_PORT, "scheme": "http"}]
    )
    model = SentenceTransformer(VECTOR_MODEL_NAME)
    if await es.indices.exists(index=ES_INDEX):
        logging.info(f"Deleting existing index: {ES_INDEX}")
        await es.indices.delete(index=ES_INDEX)

    mapping = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                },
                "metadata": {"type": "object"},
            }
        }
    }
    await es.indices.create(index=ES_INDEX, body=mapping)
    logging.info(f"Index '{ES_INDEX}' created.")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    absolute_data_file_path = os.path.join(project_root, DATA_FILE)

    try:
        with open(absolute_data_file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        logging.error(
            f"Data file not found at {absolute_data_file_path}. Please ensure it exists."
        )
        await es.close()
        return
    except json.JSONDecodeError:
        logging.error(
            f"Error decoding JSON from {absolute_data_file_path}. Please check file format."
        )
        await es.close()
        return

    # Process and index each document
    for doc in documents:
        text = doc["text"]
        # Generate embedding for the document text
        embedding = model.encode(text).tolist()
        doc["embedding"] = embedding  # Add the embedding to the document
        try:
            # Index the document into Elasticsearch
            await es.index(index=ES_INDEX, id=doc["id"], document=doc)
            logging.info(f"Indexed document: {doc['id']}")
        except Exception as e:
            logging.error(f"Error indexing document {doc['id']}: {e}")

    await es.close()
    logging.info("Indexing complete.")


if __name__ == "__main__":
    asyncio.run(create_and_index_documents())
