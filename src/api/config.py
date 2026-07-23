from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    model: str = "mistral-7b-instruct-v0.1"
    api_key: str = "not-needed"
    temperature: float = 0.0
    timeout_seconds: float = 30.0


class DatasetConfig(BaseModel):
    qa_name: str = "ruslanmv/ai-medical-chatbot"
    qa_split: str = "train"
    qa_cache_dir: str | None = None
    pairs_name: str = "curaihealth/medical_questions_pairs"
    pairs_cache_dir: str | None = None


class RetrievalConfig(BaseModel):
    embedding_model_name: str = "paraphrase-MiniLM-L3-v2"
    device: str | None = None
    top_k: int = 5
    embedding_batch_size: int = 128
    show_progress_bar: bool = True
    embeddings_cache_dir: str | None = None


class RerankerConfig(BaseModel):
    model_name: str = "bert-base-uncased"
    saved_model_path: str | None = "saved_model"
    device: str | None = None
    top_n: int = 3


class GraphConfig(BaseModel):
    system_variant: str = "full"
    low_confidence_threshold: float = 0.7
    urgency_keywords: list[str] = Field(
        default_factory=lambda: [
            "боль",
            "кровь",
            "температура",
            "срочно",
            "скорая",
            "сердце",
            "голова",
            "передозировка",
            "потеря сознания",
            "не дышит",
        ]
    )


class TrainingConfig(BaseModel):
    output_dir: str | None = "saved_model"
    base_model_name: str = "bert-base-uncased"


class EvaluationConfig(BaseModel):
    semantic_model_name: str = "paraphrase-MiniLM-L3-v2"
    test_set_path: str = "data/test_questions.json"
    prompt_variants: list[str] = Field(
        default_factory=lambda: ["base", "few_shot", "cot", "full"]
    )


class APIConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"API config must be a YAML mapping: {path}")
    return data


def _set_if_env(
    data: dict[str, Any], section: str, key: str, env_name: str, cast=str
) -> None:
    value = os.getenv(env_name)
    if value is None:
        return
    data.setdefault(section, {})[key] = cast(value)


def _env_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def load_api_config(path: str | Path | None = None) -> APIConfig:
    load_dotenv()
    config_path = Path(path or os.getenv("API_CONFIG_PATH", "config/api.yaml"))
    data = _deep_update(APIConfig().model_dump(), _load_yaml(config_path))

    _set_if_env(data, "llm", "base_url", "LLM_BASE_URL")
    _set_if_env(data, "llm", "model", "LLM_MODEL")
    _set_if_env(data, "llm", "api_key", "LLM_API_KEY")
    _set_if_env(data, "llm", "temperature", "LLM_TEMPERATURE", float)
    _set_if_env(data, "llm", "timeout_seconds", "LLM_TIMEOUT_SECONDS", float)

    _set_if_env(data, "dataset", "qa_cache_dir", "SAVED_DATASET_QA_PATH")
    _set_if_env(data, "dataset", "pairs_cache_dir", "SAVED_DATASET_PAIRS_PATH")
    _set_if_env(data, "reranker", "saved_model_path", "SAVED_MODEL_PATH")
    _set_if_env(data, "training", "output_dir", "SAVED_MODEL_PATH")
    _set_if_env(data, "retrieval", "top_k", "RETRIEVAL_TOP_K", int)
    _set_if_env(data, "retrieval", "embedding_batch_size", "EMBEDDING_BATCH_SIZE", int)
    _set_if_env(data, "retrieval", "show_progress_bar", "EMBEDDING_SHOW_PROGRESS", _env_bool)
    _set_if_env(data, "retrieval", "embeddings_cache_dir", "EMBEDDINGS_CACHE_DIR")
    _set_if_env(data, "reranker", "top_n", "RERANKER_TOP_N", int)
    _set_if_env(data, "graph", "system_variant", "SYSTEM_VARIANT")
    _set_if_env(data, "graph", "low_confidence_threshold", "LOW_CONFIDENCE_THRESHOLD", float)

    return APIConfig.model_validate(data)


@lru_cache(maxsize=1)
def get_api_config() -> APIConfig:
    return load_api_config()
