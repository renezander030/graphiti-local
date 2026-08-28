# Graphiti Local

**Local-first temporal knowledge graph.**

Graphiti Local is a small, read-first interface that gives agents six MCP
retrieval tools, a matching `kg` command-line interface, and a human-gated path
for adding facts.

> Graphiti Local is an independent community project built on Graphiti. It is
> not affiliated with or endorsed by Zep.

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
kg doctor [--offline]
kg export [group ...] [--output PATH]
kg verify [--offline]
```

`propose` appends to a local JSONL queue. It does not modify the graph. A human
uses `kg-workspace approve`, reviews the dry run from `kg-workspace drain`, and
adds `--apply` only when the proposal is ready.

## Built for unattended use

Every command prints JSON on stdout and exits non-zero when it refuses, so a cron
job or an agent can consume a result without scraping text or guessing whether a
call succeeded. Add `-H`/`--human` for the reader-friendly form.

```bash
kg ask "what changed last week" | jq -r '.facts[].fact'

kg propose unconfigured-group "a fact"; echo $?   # 2 — refused, nothing queued
```

Before a run depends on the configuration, prove it:

```bash
kg doctor          # config, workspace, backend, LLM, and a live embedding-width probe
kg verify          # the six read tools, no write tools, and live retrieval
```

`kg doctor` probes the embedder endpoint and fails when the vector width it returns
disagrees with `embedder.dimensions`. That mismatch is otherwise silent, and it
corrupts every embedding it writes.

## Quick start

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
cp config/falkordb.example.yaml config/falkordb.yaml
export GRAPHITI_LOCAL_CONFIG="$PWD/config/falkordb.yaml"
export OPENAI_API_KEY="..."
uv run kg status
uv run graphiti-local
```

For an MCP client using stdio:

```json
{
  "mcpServers": {
    "kg": {
      "command": "uv",
      "args": ["--directory", "/path/to/graphiti-local", "run", "graphiti-local"],
      "env": {
        "GRAPHITI_LOCAL_CONFIG": "/path/to/graphiti-local/config/falkordb.yaml"
      }
    }
  }
}
```

The two example configurations use only synthetic group names and environment
variable references. FalkorDB and Neo4j are supported server backends. LadybugDB
provides an embedded single-file backend:

```bash
cp config/ladybug.example.yaml config/ladybug.yaml
export GRAPHITI_LOCAL_CONFIG="$PWD/config/ladybug.yaml"
uv run kg-ladybug-setup --database ./workspace/example.ladybug
# Review the dry run, then repeat with --apply in a network-enabled environment.
```

Opening a Ladybug graph never installs extensions automatically.

## Running without a cloud provider

`config/ollama.example.yaml` runs the graph entirely on your machine — a local model
for extraction, a local embedder, and the embedded Ladybug backend. No API key, no
data leaving the host:

```bash
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
cp config/ollama.example.yaml config/ollama.yaml
export GRAPHITI_LOCAL_CONFIG="$PWD/config/ollama.yaml"
uv run kg doctor
```

Ollama speaks the OpenAI API on `/v1`, so both clients point at it.
`structured_output_mode: json_object` suits local models, which mostly do not
implement the strict `json_schema` response format. Mind the vector width:
`nomic-embed-text` returns 768, not the 1536 an OpenAI default assumes.

## Explicit ingestion

Ingestion is a separate command and is dry-run by default:

```bash
uv run kg-ingest examples/synthetic_episodes.jsonl
uv run kg-ingest examples/synthetic_episodes.jsonl --apply
```

Inputs are UTF-8 JSONL objects with `name` and `body`; `domain`, `valid_at`, and
`provenance` are optional. A requested domain must be in `graph.groups`.

Ingestion is resumable. Each applied record is written to a content-keyed ledger in
the workspace, so re-running the same file ingests only what has not landed yet
instead of duplicating it. A record that fails is isolated and reported; the rest of
the batch still lands, and the failure sets a non-zero exit code. Use `--no-resume`
to ignore the ledger and `--fail-fast` for the old stop-at-first-error behaviour.

`SIGTERM` and `SIGINT` stop it at a record boundary rather than mid-write: it finishes
the record in flight, closes the driver, and reports `interrupted`. This matters when a
cron job wraps the run in a `timeout` — extraction is slow on a local model, and a
process killed mid-write can leave an embedded backend with a partial write it refuses
to reopen. A signal cannot help against `SIGKILL` or a power cut, so take a snapshot
with `kg export` before a long ingest into an embedded backend.

## Portability

```bash
kg export                                    # every group, to a timestamped JSONL file
kg export example --output ./snapshot.jsonl  # a named group to a chosen path
```

The snapshot is written from the graphiti models rather than backend rows, so it is
readable whichever backend produced it. Embeddings are omitted deliberately: they are
derived from the text, and a vector restored under a different embedding model would
be silently wrong.

## Safety model

- Graphiti telemetry is disabled before its package is imported.
- HTTP defaults to loopback; stdio is the example default.
- The `streamable-http` transport refuses to start without `server.auth.token`,
  and every request over it must carry that bearer token. stdio is a private pipe;
  a network port is reachable by anything that can open it.
- Configured graph groups form an access allow-list, enforced on reads *and* on
  `kg propose`. A fact addressed to an unconfigured group is refused, not queued.
- Credentials remain environment variables.
- MCP cannot write or delete.
- Ingestion requires an explicit command and `--apply`.
- Proposal approval and application are separate human actions.
- `kg-workspace drain` archives a proposal only after it actually lands. A failed
  ingest leaves it approved so the next drain retries it.

Run the release gate before sharing:

```bash
uv run ruff check .
uv run pytest
uv run python scripts/release_audit.py .
```

See [PRIVACY.md](PRIVACY.md), [SECURITY.md](SECURITY.md), and
[UPSTREAM.md](UPSTREAM.md) for the deployment and provenance boundaries.

## Related tools

The propose/approve/drain boundary here is one instance of a pattern used across a few
sibling projects:

- [agent-approval-gate](https://github.com/renezander030/agent-approval-gate) — the same
  draft/validate/approve boundary as a standalone pattern, without a graph behind it.
- [skillgate](https://github.com/renezander030/skillgate) — deterministic finish-line
  gates for agent output, the check that runs before something ships.
- [action-mcp-test](https://github.com/renezander030/action-mcp-test) — a GitHub Action
  that tests MCP servers in CI for protocol compliance and schema validation.
- [agentic-task-system](https://github.com/renezander030/agentic-task-system) — a task
  layer for agent context that can read this graph over the same `kg` CLI.

## License

Apache-2.0.
