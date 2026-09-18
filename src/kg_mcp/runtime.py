"""Construct Graphiti clients from Graphiti Local's compact configuration."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from kg_mcp.config import Settings


def _is_official_openai(url: str) -> bool:
    return url.rstrip("/") in {"https://api.openai.com", "https://api.openai.com/v1"}


def uses_responses_api(settings: Settings) -> bool:
    """Resolve the configured OpenAI API family without guessing when told explicitly."""
    if settings.llm.api_mode == "responses":
        return True
    if settings.llm.api_mode == "chat":
        return False
    return _is_official_openai(settings.llm.api_url)


async def bounded(awaitable: Any, seconds: float, what: str) -> Any:
    """Bound one graph call so a hung backend fails loudly instead of blocking forever."""
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{what} timed out after {seconds:g}s; the backend did not answer"
        ) from exc


def build_reranker(settings: Settings):
    """Choose a cross-encoder. 'passthrough' stays the default: it costs no extra call.

    The concrete upstream clients are imported lazily because the bge and gemini modules
    import their third-party dependency eagerly at module load.
    """
    provider = settings.reranker.provider
    if provider == "passthrough":
        from kg_mcp.reranker import PassthroughReranker

        return PassthroughReranker()
    if provider == "bge":
        from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient

        return BGERerankerClient()

    from graphiti_core.llm_client.config import LLMConfig

    config = LLMConfig(
        api_key=settings.reranker.api_key or None,
        base_url=settings.reranker.api_url or None,
        model=settings.reranker.model or None,
    )
    if provider == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        return OpenAIRerankerClient(config=config)
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    return GeminiRerankerClient(config=config)


def cap_llm_tokens(llm: Any, maximum: int) -> Any:
    """Enforce one output ceiling even when an upstream prompt asks for more."""
    original = llm.generate_response

    async def generate_response(
        messages,
        response_model=None,
        max_tokens=None,
        model_size=None,
        group_id=None,
        prompt_name=None,
        *,
        attribute_extraction=False,
    ):
        capped = maximum if max_tokens is None else min(max_tokens, maximum)
        kwargs = {
            "response_model": response_model,
            "max_tokens": capped,
            "group_id": group_id,
            "prompt_name": prompt_name,
            "attribute_extraction": attribute_extraction,
        }
        if model_size is not None:
            kwargs["model_size"] = model_size
        return await original(messages, **kwargs)

    llm.generate_response = generate_response
    llm.max_tokens = maximum
    llm.configured_max_tokens = maximum
    return llm


def build_llm(settings: Settings):
    """Build the extraction client selected by ``llm.api_mode``."""
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    llm_config = LLMConfig(
        api_key=settings.llm.api_key or None,
        base_url=settings.llm.api_url,
        model=settings.llm.model,
        small_model=settings.llm.model,
        temperature=settings.llm.temperature,  # type: ignore[arg-type]
        max_tokens=settings.llm.max_tokens,
    )
    if uses_responses_api(settings):
        llm = OpenAIClient(config=llm_config, max_tokens=settings.llm.max_tokens)
    else:
        llm = OpenAIGenericClient(
            config=llm_config,
            max_tokens=settings.llm.max_tokens,
            structured_output_mode=settings.llm.structured_output_mode,
        )
    return cap_llm_tokens(llm, settings.llm.max_tokens)


def build_graphiti(settings: Settings, *, read_only: bool):
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("EMBEDDING_DIM", str(settings.embedder.dimensions))

    from graphiti_core import Graphiti
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    llm = build_llm(settings)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=settings.embedder.api_key or None,
            base_url=settings.embedder.api_url,
            embedding_model=settings.embedder.model,
            embedding_dim=settings.embedder.dimensions,
        )
    )

    kwargs = {
        "llm_client": llm,
        "embedder": embedder,
        "cross_encoder": build_reranker(settings),
    }
    provider = settings.database.provider
    if provider == "falkordb":
        from graphiti_core.driver.falkordb_driver import FalkorDriver

        driver_class = FalkorDriver
        if read_only:

            class ReadOnlyFalkorDriver(FalkorDriver):
                async def build_indices_and_constraints(self, delete_existing: bool = False):
                    del delete_existing

            driver_class = ReadOnlyFalkorDriver
        database = settings.database.falkordb
        driver = driver_class(
            host=database.host,
            port=database.port,
            username=database.username,
            password=database.password or None,
            database=settings.graph.groups[0],
        )
        return Graphiti(graph_driver=driver, **kwargs)
    if provider == "ladybug":
        from kg_mcp.ladybug import build_ladybug_driver

        # Readers open the file read-only so they coexist with the one writer Ladybug allows.
        driver = build_ladybug_driver(settings.database.ladybug.path, read_only=read_only)
        return Graphiti(graph_driver=driver, **kwargs)
    if provider == "neo4j":
        database = settings.database.neo4j
        return Graphiti(
            uri=database.uri,
            user=database.username,
            password=database.password,
            **kwargs,
        )
    raise ValueError(f"unsupported database provider: {provider}")
