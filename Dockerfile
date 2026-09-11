# Stdio discovery image for MCP directory evaluation; no private graph is copied.
FROM ghcr.io/astral-sh/uv:0.12.13 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY config/ollama.example.yaml ./config/ollama.example.yaml
RUN uv sync --frozen --no-dev \
    && useradd --create-home --uid 10001 mcp \
    && mkdir /data \
    && chown mcp:mcp /data
ENV PATH="/app/.venv/bin:$PATH" \
    GRAPHITI_LOCAL_CONFIG=/app/config/ollama.example.yaml \
    KG_WORKSPACE_DIR=/data \
    KG_LADYBUG_PATH=/data/graph.ladybug
USER mcp
RUN kg-ladybug-setup --database /data/graph.ladybug --apply
ENTRYPOINT ["graphiti-local"]
