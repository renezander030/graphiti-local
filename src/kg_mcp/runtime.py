"""Construct Graphiti clients from Graphiti Local's compact configuration."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
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


def require_all_schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a constrained-decoding schema that emits every declared object key.

    Pydantic already represents optional values as nullable types. Grammar-backed
    OpenAI-compatible servers interpret a property omitted from ``required`` as a key
    they may leave out entirely, so temporal fields can disappear before validation.
    Requiring the keys while preserving their nullable type keeps ``null`` distinct
    from an omitted field. The input schema is never mutated.
    """
    prepared = copy.deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(prepared)
    return prepared


async def bounded(awaitable: Any, seconds: float, what: str) -> Any:
    """Bound one graph call so a hung backend fails loudly instead of blocking forever."""
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"{what} timed out after {seconds:g}s; the backend did not answer"
        ) from exc


OPENAI_RERANKER_MODEL = "gpt-4.1-mini"


# Full-text fields graphiti-core indexes on FalkorDB, one DDL statement per field
# (https://github.com/getzep/graphiti/issues/1947).
FALKORDB_FULLTEXT_FIELDS = {
    "(n:Episodic)": ("n", ("content", "source", "source_description", "group_id")),
    "(n:Entity)": ("n", ("name", "summary", "group_id")),
    "(n:Community)": ("n", ("name", "group_id")),
    "()-[e:RELATES_TO]-()": ("e", ("name", "fact", "group_id")),
}
FALKORDB_FULLTEXT_INDICES = [
    f"CREATE FULLTEXT INDEX FOR {pattern} ON ({var}.{field})"
    for pattern, (var, fields) in FALKORDB_FULLTEXT_FIELDS.items()
    for field in fields
]


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

    # graphiti-core's OpenAI reranker default, gpt-4.1-nano, is retired on 2026-10-23;
    # gpt-4.1-mini accepts the same request unchanged.
    default = OPENAI_RERANKER_MODEL if provider == "openai" else None
    config = LLMConfig(
        api_key=settings.reranker.api_key or None,
        base_url=settings.reranker.api_url or None,
        model=settings.reranker.model or default,
    )
    if provider == "openai":
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

        return OpenAIRerankerClient(config=config)
    from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

    return GeminiRerankerClient(config=config)


def cap_llm_tokens(llm: Any, maximum: int) -> Any:
    """Enforce one output ceiling even when an upstream prompt asks for more.

    The client also gets its own copy of the messages: graphiti-core appends schema,
    language and retry text to the list it is given, which would change the caller's
    prompt and make a repeated identical call send a different request.
    """
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
        owned = [message.model_copy() for message in messages]
        return await original(owned, **kwargs)

    llm.generate_response = generate_response
    llm.max_tokens = maximum
    llm.configured_max_tokens = maximum
    return llm


class OutputTruncatedError(Exception):
    """The model stopped at its output ceiling, so its JSON answer is cut off.

    Sending the same request again is cut off at the same place, so this is never
    retried: raising ``llm.max_tokens`` or splitting the episode is the fix.
    """


def refuse_truncated_output(llm: Any) -> Any:
    """Raise :class:`OutputTruncatedError` when a completion ends on the token ceiling,
    instead of handing a cut-off JSON body to a parser that would retry it."""
    completions = llm.client.chat.completions
    create = completions.create

    async def guarded(*args: Any, **kwargs: Any) -> Any:
        response = await create(*args, **kwargs)
        choices = getattr(response, "choices", None) or []
        if choices and getattr(choices[0], "finish_reason", None) == "length":
            raise OutputTruncatedError(
                f"the model's answer reached max_tokens ({kwargs.get('max_tokens')}) and was "
                "cut off; raise llm.max_tokens or split the episode"
            )
        return response

    completions.create = guarded
    return llm


# Usage of the provider responses inside one ``generate_response`` call, retries included.
_call_usage: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "graphiti_local_call_usage", default=None
)


def count_provider_usage(llm: Any) -> Any:
    """Add each chat completion's reported ``usage`` to the running call's totals.

    The OpenAI-compatible client parses the answer and drops the response, so its
    token tracker never sees what the provider reported. A context variable keeps
    concurrent calls apart; ``generate_response`` records the totals afterwards.
    """
    completions = llm.client.chat.completions
    create = completions.create

    async def counted(*args: Any, **kwargs: Any) -> Any:
        response = await create(*args, **kwargs)
        usage = getattr(response, "usage", None)
        totals = _call_usage.get()
        if usage is not None and totals is not None:
            totals[0] += getattr(usage, "prompt_tokens", None) or 0
            totals[1] += getattr(usage, "completion_tokens", None) or 0
        return response

    completions.create = counted
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
        from graphiti_core.llm_client.errors import RefusalError
        from openai import APIStatusError

        class TerminalProviderError(RefusalError):
            """Carry a provider rejection through the upstream no-retry path."""

            def __init__(self, original):
                super().__init__(str(original))
                self.original = original
                self.original_cause = original.__cause__

        class GraphitiLocalOpenAIClient(OpenAIClient):
            async def _generate_response(self, *args, **kwargs):
                try:
                    return await super()._generate_response(*args, **kwargs)
                except APIStatusError as exc:
                    if exc.status_code in {400, 401, 403, 404, 422}:
                        # A rejected request is not malformed model output. Preserve
                        # upstream JSON-repair retries without resending this request.
                        raise TerminalProviderError(exc) from None
                    raise

            async def generate_response(self, *args, **kwargs):
                try:
                    return await super().generate_response(*args, **kwargs)
                except TerminalProviderError as exc:
                    raise exc.original from exc.original_cause

        llm = GraphitiLocalOpenAIClient(config=llm_config, max_tokens=settings.llm.max_tokens)
    else:
        class GraphitiLocalGenericClient(OpenAIGenericClient):
            def _build_response_format(self, response_model):
                response_format = super()._build_response_format(response_model)
                if (
                    response_model is not None
                    and self.structured_output_mode == "json_schema"
                    and settings.llm.require_all_schema_properties
                ):
                    response_format["json_schema"]["schema"] = require_all_schema_properties(
                        response_model.model_json_schema()
                    )
                return response_format

            async def generate_response(self, *args, prompt_name=None, **kwargs):
                totals = [0, 0]
                token = _call_usage.set(totals)
                try:
                    return await super().generate_response(*args, prompt_name=prompt_name, **kwargs)
                finally:
                    _call_usage.reset(token)
                    if totals[0] or totals[1]:
                        self.token_tracker.record(prompt_name, totals[0], totals[1])

        llm = refuse_truncated_output(
            count_provider_usage(
                GraphitiLocalGenericClient(
                    config=llm_config,
                    max_tokens=settings.llm.max_tokens,
                    structured_output_mode=settings.llm.structured_output_mode,
                )
            )
        )
    return cap_llm_tokens(llm, settings.llm.max_tokens)


# graphiti-core joins a multi-group Lucene filter as ``group_id:"a" OR group_id:"b" AND (q)``.
# Lucene binds AND tighter than OR, so only the last group can match. Group ids are
# restricted to [A-Za-z0-9_-], so this prefix is always the one graphiti-core wrote.
_GROUP_TERM = r'group_id:"[A-Za-z0-9_-]*"'
_UNGROUPED_FILTER = re.compile(rf"^((?:{_GROUP_TERM} OR )+{_GROUP_TERM}) AND ")


def parenthesize_group_filter(query: Any) -> Any:
    """``(group_id:"a" OR group_id:"b") AND (q)``: the fulltext query matches any group."""
    if not isinstance(query, str):
        return query
    return _UNGROUPED_FILTER.sub(r"(\1) AND ", query, count=1)


def group_filter_neo4j_driver():
    """A Neo4j driver whose fulltext queries match every requested group."""
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    class GroupFilterNeo4jDriver(Neo4jDriver):
        async def execute_query(self, cypher_query_, **kwargs):
            if "query" in kwargs and "db.index.fulltext.query" in cypher_query_:
                kwargs["query"] = parenthesize_group_filter(kwargs["query"])
            return await super().execute_query(cypher_query_, **kwargs)

    return GroupFilterNeo4jDriver


def build_graphiti(settings: Settings, *, read_only: bool):
    os.environ.setdefault("EMBEDDING_DIM", str(settings.embedder.dimensions))

    from graphiti_core import Graphiti
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

    from kg_mcp.attributes import install_nested_attribute_serialiser
    from kg_mcp.dedup import install_deterministic_tie_break

    install_deterministic_tie_break()
    install_nested_attribute_serialiser()

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
        from graphiti_core.graph_queries import get_range_indices

        class PortableFalkorDriver(FalkorDriver):
            """Full-text indexes as DDL, which FalkorDB 4.x and 6.x both accept.

            graphiti-core creates node full-text indexes with the
            ``db.idx.fulltext.createNodeIndex({label, stopwords}, ...)`` procedure form,
            which FalkorDB 6 rejects. One field per statement keeps a rebuild idempotent:
            a field already indexed is reported as such and skipped. Search strips the
            stopwords itself, so dropping the index option changes no result.
            """

            async def build_indices_and_constraints(self, delete_existing: bool = False):
                if delete_existing:
                    await self.delete_all_indexes()
                for query in get_range_indices(self.provider) + FALKORDB_FULLTEXT_INDICES:
                    await self.execute_query(query)

            def clone(self, database: str):
                if database == self._database:
                    return self
                if database == self.default_group_id:
                    return type(self)(falkor_db=self.client)
                return type(self)(falkor_db=self.client, database=database)

        driver_class = PortableFalkorDriver
        if read_only:

            class ReadOnlyFalkorDriver(PortableFalkorDriver):
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
        driver = group_filter_neo4j_driver()(database.uri, database.username, database.password)
        return Graphiti(graph_driver=driver, **kwargs)
    raise ValueError(f"unsupported database provider: {provider}")


@asynccontextmanager
async def writing(settings: Settings, *, track_fingerprint: bool = True) -> AsyncIterator[Any]:
    """A write session: the graph under the writer lock, with its indexes in place.

    Set ``session.wrote`` once something landed, so the embedder is recorded only then.
    The graph is always closed: an embedded backend left mid-write may refuse to reopen.
    """
    from kg_mcp import fingerprint
    from kg_mcp.write_lock import graph_writer_lock

    with graph_writer_lock(settings):
        # A previous writer can record a different identity while this one waits.
        # Validate under the same lock as the write, before opening the graph.
        drift = fingerprint.drift(settings) if track_fingerprint else None
        if drift:
            raise ValueError(drift)
        graph = build_graphiti(settings, read_only=False)
        try:
            await graph.build_indices_and_constraints()
            session = SimpleNamespace(graph=graph, wrote=False)
            yield session
            if session.wrote and track_fingerprint:
                fingerprint.record(settings)
        finally:
            await graph.close()
