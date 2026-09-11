# Command and deployment reference

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
kg duplicates [group ...]
kg verify [--offline]
kg --version
```

`propose` appends to a local JSONL queue. It does not modify the graph. A human
uses `kg-workspace approve`, reviews the dry run from `kg-workspace drain`, and
adds `--apply` only when the proposal is ready.

## Built for unattended use

Every command prints JSON on stdout and exits non-zero when it refuses, so a cron
job or an agent can consume a result without scraping text or guessing whether a
call succeeded. Add `-H`/`--human` for the reader-friendly form, before or after the
subcommand.

```bash
kg ask "what changed last week" | jq -r '.facts[].fact'

kg propose unconfigured-group "a fact"; echo $?   # 2 — refused, nothing queued
```

Every graph call is bounded by `graph.query_timeout_seconds` (default 30). A backend
that hangs makes `kg` exit `3` with the reason, and an MCP tool return an error object,
instead of an agent waiting on a call that never returns.

Before a run depends on the configuration, prove it:

```bash
kg doctor          # versions, config, workspace, backend, LLM, and a live embedding-width probe
kg verify          # the six read tools, no write tools, and live retrieval
```

`kg doctor` probes the embedder endpoint and fails when the vector width it returns
disagrees with `embedder.dimensions`. That mismatch is otherwise silent, and it
corrupts every embedding it writes. It also compares the configured embedder with the
one recorded by the first write to this database: a model change at the same width is
just as silent, and a later ingest or restore refuses it with exit code `2`.

On the embedded Ladybug backend the readers (`graphiti-local`, `kg ask`, `kg export`,
`kg verify`) open the file read-only, so an ingest or a drain runs while the server is
up, and the server picks up what landed without a restart.

## Backend choices

The [local walkthrough](local-quickstart.md) uses
`config/ollama.example.yaml`: Ollama, a 768-dimension `nomic-embed-text` embedder,
and embedded LadybugDB. All inference endpoints in that example are loopback.

For an existing database service, use `config/falkordb.example.yaml` or
`config/neo4j.example.yaml` and configure its connection and model credentials.
`config/ladybug.example.yaml` pairs embedded storage with a cloud model provider.
These are separate deployment choices; local storage alone does not make inference local.

An embedded database needs its schema, search extensions, and four full-text
indexes before the read-only server opens it. `kg-ladybug-setup --apply` prepares
them explicitly. Opening a reader never installs extensions automatically.
For a database created by 0.2.x, run setup again to add the indexes.

## Explicit ingestion

Ingestion is a separate command and is dry-run by default:

```bash
uv run kg-ingest examples/local_memory_demo.jsonl
uv run kg-ingest examples/local_memory_demo.jsonl --apply
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

```bash
kg-ingest ./snapshot.jsonl --restore                    # dry run: what would land where
kg-ingest ./snapshot.jsonl --restore --apply            # replay into the configured backend
kg-ingest ./snapshot.jsonl --restore --group team-a --apply   # remap every record to one group
```

A restore re-embeds every entity name and fact under the configured embedder and saves
nodes before the edges that reference them. Records keep their UUIDs, so restoring the
same snapshot twice updates rather than duplicates. This is also the path from one
backend to another, and the path to a new embedder: export, point the configuration at
an empty database, restore.

```bash
kg duplicates            # entities whose names collide once casing and punctuation are ignored
```

Graphiti resolves duplicates by embedding similarity, so `ACME Corp` and `acme corp.`
can end up as two entities. The report shows what split; a merge is a proposal like any
other correction.

## Safety model

- Graphiti telemetry is disabled before its package is imported.
- HTTP defaults to loopback; stdio is the example default.
- The `streamable-http` transport refuses to start without `server.auth.token`,
  and every request over it must carry that bearer token. stdio is a private pipe;
  a network port is reachable by anything that can open it.
- Behind a reverse proxy, `server.allowed_hosts` lists the Host names the MCP SDK's
  DNS-rebinding check accepts (`["kg.example.internal:*"]`). Without it the SDK default
  applies: loopback names only on a loopback host, no check elsewhere.
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

See [PRIVACY.md](../PRIVACY.md), [SECURITY.md](../SECURITY.md), and
[UPSTREAM.md](../UPSTREAM.md) for the deployment and provenance boundaries.

For a directory evaluation container, see [MCP discovery container](container.md).

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
