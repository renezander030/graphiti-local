# MCP discovery container

The root `Dockerfile` prepares an empty LadybugDB and starts the stdio MCP server
as a non-root user. It is intended for directory introspection and reproducible
server discovery. It contains no example facts or private graph data.

```bash
docker build -t graphiti-local .
uv run --frozen python scripts/mcp_smoke.py docker run --rm -i graphiti-local
```

The smoke check performs a real MCP initialization, lists exactly six retrieval
tools, and calls `get_status`. It does not test extraction or semantic search.
The `Container MCP smoke` GitHub Actions workflow performs that check on Linux.
Check its actual result before treating a container build as verified.

The image downloads Python dependencies and Ladybug search extensions during
build. Its empty graph supports discovery without an LLM. Retrieval over populated
data still needs correctly configured inference endpoints and a prepared graph.
The default loopback Ollama address refers to the container itself; it does not
connect to the host automatically. Use the [native local walkthrough](local-quickstart.md)
for the tested Ollama inference path.

## Glama configuration

Glama generates its own Dockerfile from the fields under **Admin → Dockerfile**.
The following configuration passed its build and MCP introspection on September
11, 2026, using commit `3670d5d0532f6b157e0fab0086e2f7d5278cf2d3`:

- Base image: `debian:trixie-slim`
- Python: `3.12`; Node.js: `26` for Glama's MCP proxy
- Build steps:

```json
[
  "uv sync --frozen --no-dev --python 3.12",
  "uv run --frozen kg-ladybug-setup --database ./workspace/glama/graph.ladybug --apply"
]
```

- CMD arguments:

```json
[
  "env",
  "GRAPHITI_LOCAL_CONFIG=config/ollama.example.yaml",
  "KG_WORKSPACE_DIR=workspace/glama",
  "KG_LADYBUG_PATH=workspace/glama/graph.ladybug",
  ".venv/bin/graphiti-local"
]
```

- Environment variables JSON schema: `{"type":"object","properties":{}}`
- Placeholder parameters: `{}`

These settings expose an empty graph for discovery. They do not provide a hosted
Ollama model or a populated memory service. The root Dockerfile remains the
separately tested non-root discovery image for local use.

Sync the repository before building, inspect the successful test, and create a
release from that build. A submitted listing, a successful build, and a quality
score are separate states; only display the score badge after evaluation.
The [public listing](https://glama.ai/mcp/servers/renezander030/graphiti-local)
shows the available tools and current evaluation status.
