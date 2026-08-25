# KG-MCP

KG-MCP is a small, read-first interface to temporal knowledge graphs. It gives
agents six MCP retrieval tools, a matching `kg` command-line interface, and a
human-gated path for adding facts.

This is a standalone package. It depends on `graphiti-core`; it does not contain
Graphiti's repository, README, images, examples, or Git history.

## What is included

The MCP server registers exactly these tools:

- `search_nodes`
- `search_memory_facts`
- `get_entity_edge`
- `get_episodes`
- `get_episode_entities`
- `get_status`

There are no MCP write, delete, clear, approval, or maintenance tools.

The CLI provides:

```text
kg ask "question" [group ...]
kg nodes "query" [group ...]
kg episodes [group ...]
kg edge UUID
kg status
kg pending [group]
kg propose GROUP "fact" --type source-fact --provenance "source"
```

`propose` appends to a local JSONL queue. It does not modify the graph. A human
uses `kg-workspace approve`, reviews the dry run from `kg-workspace drain`, and
adds `--apply` only when the proposal is ready.

## Quick start

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
cp config/falkordb.example.yaml config/falkordb.yaml
export KG_MCP_CONFIG="$PWD/config/falkordb.yaml"
export OPENAI_API_KEY="..."
uv run kg status
uv run kg-mcp
```

For an MCP client using stdio:

```json
{
  "mcpServers": {
    "kg": {
      "command": "uv",
      "args": ["--directory", "/path/to/kg-mcp", "run", "kg-mcp"],
      "env": {"KG_MCP_CONFIG": "/path/to/kg-mcp/config/falkordb.yaml"}
    }
  }
}
```

The two example configurations use only synthetic group names and environment
variable references. FalkorDB and Neo4j are supported server backends. LadybugDB
provides an embedded single-file backend:

```bash
cp config/ladybug.example.yaml config/ladybug.yaml
export KG_MCP_CONFIG="$PWD/config/ladybug.yaml"
uv run kg-ladybug-setup --database ./workspace/example.ladybug
# Review the dry run, then repeat with --apply in a network-enabled environment.
```

Opening a Ladybug graph never installs extensions automatically.

## Explicit ingestion

Ingestion is a separate command and is dry-run by default:

```bash
uv run kg-ingest examples/synthetic_episodes.jsonl
uv run kg-ingest examples/synthetic_episodes.jsonl --apply
```

Inputs are UTF-8 JSONL objects with `name` and `body`; `domain`, `valid_at`, and
`provenance` are optional. A requested domain must be in `graph.groups`.

## Safety model

- Graphiti telemetry is disabled before its package is imported.
- HTTP defaults to loopback; stdio is the example default.
- Configured graph groups form an access allow-list.
- Credentials remain environment variables.
- MCP cannot write or delete.
- Ingestion requires an explicit command and `--apply`.
- Proposal approval and application are separate human actions.

Run the release gate before sharing:

```bash
uv run ruff check .
uv run pytest
uv run python scripts/release_audit.py .
```

See [PRIVACY.md](PRIVACY.md), [SECURITY.md](SECURITY.md), and
[UPSTREAM.md](UPSTREAM.md) for the deployment and provenance boundaries.

## License

Apache-2.0.
