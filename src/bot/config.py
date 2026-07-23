from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class APIClientConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    answer_timeout_seconds: float = 180.0
    feedback_timeout_seconds: float = 5.0


class TelegramConfig(BaseModel):
    token: str | None = None
    parse_mode: str = "Markdown"


class BotConfig(BaseModel):
    api: APIClientConfig = Field(default_factory=APIClientConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)


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
        raise ValueError(f"Bot config must be a YAML mapping: {path}")
    return data


def _set_if_env(
    data: dict[str, Any], section: str, key: str, env_name: str, cast=str
) -> None:
    value = os.getenv(env_name)
    if value is None:
        return
    data.setdefault(section, {})[key] = cast(value)


def load_bot_config(path: str | Path | None = None) -> BotConfig:
    load_dotenv()
    config_path = Path(path or os.getenv("BOT_CONFIG_PATH", "config/bot.yaml"))
    data = _deep_update(BotConfig().model_dump(), _load_yaml(config_path))

    _set_if_env(data, "api", "base_url", "API_URL")
    _set_if_env(data, "api", "answer_timeout_seconds", "BOT_ANSWER_TIMEOUT_SECONDS", float)
    _set_if_env(data, "api", "feedback_timeout_seconds", "BOT_FEEDBACK_TIMEOUT_SECONDS", float)
    _set_if_env(data, "telegram", "token", "TELEGRAM_TOKEN")
    _set_if_env(data, "telegram", "parse_mode", "TELEGRAM_PARSE_MODE")

    return BotConfig.model_validate(data)


@lru_cache(maxsize=1)
def get_bot_config() -> BotConfig:
    return load_bot_config()
