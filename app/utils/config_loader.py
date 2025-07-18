import os
from typing import List

import yaml
from pydantic import BaseModel
from pydantic import Field


class ElasticsearchConfig(BaseModel):
    HOST: str
    PORT: int
    INDEX: str


class ModelsConfig(BaseModel):
    VECTOR_MODEL_NAME: str
    RERANKER_MODEL_PATH: str


class AppConfig(BaseModel):
    FASTAPI_APP_URL: str
    DATA_FILE: str
    GROUND_TRUTH_FILE: str


class MetricsConfig(BaseModel):
    K_VALUES: List[int] = Field(
        ..., min_length=1
    )  # Ensure K_VALUES is a list with at least one item


class ConfigData(BaseModel):
    ELASTICSEARCH: ElasticsearchConfig
    MODELS: ModelsConfig
    APP: AppConfig
    METRICS: MetricsConfig


class Config:
    _instance = None

    def __new__(cls, config_file: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_file)
        return cls._instance

    def _load_config(self, config_file: str):
        """Loads and parses the YAML configuration into the Pydantic model."""
        config_path = self._find_config_file(config_file)

        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Parse the raw YAML data into our Pydantic model
        self.config_data = ConfigData(**raw_config)

    def _find_config_file(self, config_file: str) -> str:
        """Helper to find the config file relative to project root."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        full_config_path = os.path.join(project_root, config_file)

        if not os.path.exists(full_config_path):
            if os.path.exists(config_file):
                return config_file
            else:
                raise FileNotFoundError(
                    f"Config file not found at {full_config_path} or {config_file}"
                )
        return full_config_path

    def get_elasticsearch_host(self) -> str:
        return self.config_data.ELASTICSEARCH.HOST

    def get_elasticsearch_port(self) -> int:
        return self.config_data.ELASTICSEARCH.PORT

    def get_es_index(self) -> str:
        return self.config_data.ELASTICSEARCH.INDEX

    def get_vector_model_name(self) -> str:
        return self.config_data.MODELS.VECTOR_MODEL_NAME

    def get_reranker_model_path(self) -> str:
        return self.config_data.MODELS.RERANKER_MODEL_PATH

    def get_fastapi_app_url(self) -> str:
        return self.config_data.APP.FASTAPI_APP_URL

    def get_data_file(self) -> str:
        return self.config_data.APP.DATA_FILE

    def get_ground_truth_file(self) -> str:
        return self.config_data.APP.GROUND_TRUTH_FILE

    def get_k_values(self) -> List[int]:
        return self.config_data.METRICS.K_VALUES
