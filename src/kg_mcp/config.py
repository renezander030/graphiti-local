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


GROUP_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Vector width per embedding model. A mismatch between the model's real output and the
# configured width corrupts the graph silently, so a known name that disagrees is fatal.
KNOWN_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "bge-m3": 1024,
    "snowflake-arctic-embed": 1024,
}


def embedding_model_base(model: str) -> str:
    """Strip an Ollama-style ``:tag`` so ``nomic-embed-text:latest`` matches its entry."""
    return model.split(":", 1)[0].strip()


class AuthConfig(BaseModel):
    """Bearer-token gate for the network transport. Ignored by stdio."""

    token: str = ""
    groups: list[str] = Field(default_factory=list)


class ServerConfig(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @model_validator(mode="after")
    def validate_network_exposure(self) -> ServerConfig:
        if self.transport != "streamable-http":
            return self
        if not self.auth.token:
            raise ValueError(
                "server.transport 'streamable-http' requires server.auth.token; "
                "an unauthenticated network transport exposes every configured group"
            )
        if len(self.auth.token) < 16:
            raise ValueError("server.auth.token must be at least 16 characters")
        return self


class GraphConfig(BaseModel):
    groups: list[str] = Field(default_factory=lambda: ["main"], min_length=1)
    workspace_dir: str = "./workspace"

    @model_validator(mode="after")
    def validate_groups(self) -> GraphConfig:
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("graph groups must be unique")
        invalid = [group for group in self.groups if not GROUP_PATTERN.fullmatch(group)]
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

    @model_validator(mode="after")
    def validate_dimensions(self) -> EmbedderConfig:
        expected = KNOWN_EMBEDDING_DIMENSIONS.get(embedding_model_base(self.model))
        if expected is not None and expected != self.dimensions:
            raise ValueError(
                f"embedder.model '{self.model}' returns {expected}-dimensional vectors "
                f"but embedder.dimensions is {self.dimensions}; "
                "a mismatch corrupts stored embeddings silently"
            )
        return self


class RerankerConfig(BaseModel):
    """Result reranking. 'passthrough' is the default and makes no extra model call."""

    provider: Literal["passthrough", "bge", "openai", "gemini"] = "passthrough"
    model: str = ""
    api_url: str = ""
    api_key: str = ""

    @model_validator(mode="after")
    def validate_credentials(self) -> RerankerConfig:
        if self.provider in {"openai", "gemini"} and not self.api_key:
            raise ValueError(f"reranker.provider '{self.provider}' requires reranker.api_key")
        return self


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
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


def config_path(explicit: str | Path | None = None) -> Path:
    raw = (
        explicit
        or os.environ.get("GRAPHITI_LOCAL_CONFIG")
        or os.environ.get("KG_MCP_CONFIG")
        or "config/falkordb.yaml"
    )
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
