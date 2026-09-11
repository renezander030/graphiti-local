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

For Glama, provide the root `Dockerfile` through its server configuration when the
listing becomes available. A submitted listing, a successful build, and a quality
score are separate states; only display the score badge after evaluation.
