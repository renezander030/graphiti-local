"""Small YAML configuration model with environment-variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(3) or "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


class ServerConfig(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class GraphConfig(BaseModel):
    groups: list[str] = Field(default_factory=lambda: ["main"], min_length=1)
    workspace_dir: str = "./workspace"

    @model_validator(mode="after")
    def validate_groups(self) -> GraphConfig:
        pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("graph groups must be unique")
        invalid = [group for group in self.groups if not pattern.fullmatch(group)]
        if invalid:
            raise ValueError(f"invalid graph group: {invalid[0]}")
        return self


class OpenAIConfig(BaseModel):
    model: str
    api_url: str = "https://api.openai.com/v1"
    api_key: str = ""


class LLMConfig(OpenAIConfig):
    temperature: float | None = None
    max_tokens: int = Field(default=4096, ge=1)
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"


class EmbedderConfig(OpenAIConfig):
    dimensions: int = Field(default=1536, ge=1)


class FalkorConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=6379, ge=1, le=65535)
    username: str | None = None
    password: str | None = None


class Neo4jConfig(BaseModel):
    uri: str = "bolt://127.0.0.1:7687"
    username: str = "neo4j"
    password: str = ""


class LadybugConfig(BaseModel):
    path: str = "./workspace/graph.ladybug"


class DatabaseConfig(BaseModel):
    provider: Literal["falkordb", "neo4j", "ladybug"] = "falkordb"
    falkordb: FalkorConfig = Field(default_factory=FalkorConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    ladybug: LadybugConfig = Field(default_factory=LadybugConfig)


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4.1-mini"))
    embedder: EmbedderConfig = Field(
        default_factory=lambda: EmbedderConfig(model="text-embedding-3-small")
    )


def config_path(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get("KG_MCP_CONFIG", "config/falkordb.yaml")
    return Path(raw).expanduser().resolve()


def load_config(explicit: str | Path | None = None) -> Settings:
    path = config_path(explicit)
    if not path.is_file():
        raise FileNotFoundError(
            f"configuration not found: {path}; copy one of config/*.example.yaml"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return Settings.model_validate(_expand(raw))


def allowed_groups(requested: str | list[str] | None, settings: Settings) -> list[str] | None:
    """Normalize requested groups and enforce the configured allow-list."""
    groups = [requested] if isinstance(requested, str) else requested
    groups = groups or settings.graph.groups
    forbidden = sorted(set(groups) - set(settings.graph.groups))
    if forbidden:
        raise ValueError(f"group is not allowed by configuration: {forbidden[0]}")
    if settings.database.provider == "ladybug":
        return None
    return groups
