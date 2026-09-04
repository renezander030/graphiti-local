# Changelog

## 0.3.0

0.2.x could run unattended, but not on the fully local profile: on Ladybug every search
failed, and a running server locked every other command out of the file. This release
makes the embedded backend hold up under the same rules as the server backends, and it
closes the loop that `kg export` opened.

### Breaking

- **graphiti-core is pinned to 0.30.1** (was 0.29.3). Upstream fixes that reach every
  user: node attributes are no longer cleared when no entity type applies, the edge
  reranker shortlist is merged correctly, and Neo4j queries go to the configured
  database. Neo4j Enterprise users with a renamed home database: read the upstream
  0.30.0 notes, queries now target `neo4j` unless `database` is set.
- **Readers open Ladybug read-only and require a prepared database.** `graphiti-local`,
  `kg ask`, `kg export`, `kg verify` and the other read commands no longer take the
  write lock, so an ingest or a drain runs while the server is up. They refuse to open a
  file without the full-text indexes and name the fix. Existing databases: run
  `kg-ladybug-setup --database PATH --apply` once; it creates the missing indexes and
  does not touch the data.
- **A write refuses an embedder change.** The first ingest or restore records the
  embedder for its database in `workspace/embedders.json`. A later `kg-ingest`,
  `kg-workspace drain` or restore with a different `embedder.model` or width exits `2`
  instead of mixing two vector spaces in one graph. To move to a new embedder:
  `kg export`, an empty database, `kg-ingest SNAPSHOT --restore --apply`.
- **`kg` exits `3` when a graph call exceeds `graph.query_timeout_seconds`** (default
  30). The MCP tools return an error object for the same case.

### Added

- `kg-ingest SNAPSHOT --restore [--group G] [--apply]` replays a `kg export` snapshot
  into the configured backend: entity names and facts are re-embedded under the
  configured embedder, nodes are saved before the edges that reference them, and records
  keep their UUIDs, so a second restore updates rather than duplicates. An edge whose
  endpoint is not in the graph is reported as failed, not silently skipped. `--group`
  remaps every record, which is how a Ladybug snapshot lands on FalkorDB or Neo4j.
- `kg duplicates [group ...]` lists entities whose names collide once casing, whitespace
  and punctuation are ignored, with UUIDs and summaries. Read-only; a merge is a
  proposal like any other correction.
- `graph.query_timeout_seconds` bounds every graph call in the CLI, the six MCP tools and
  `kg verify`. A backend that hangs produces an error an agent can read instead of a
  tool call that never returns.
- The MCP server reopens its read-only Ladybug handle when the file changes, so facts
  landed by a drain are served without a restart.
- `server.allowed_hosts` for `streamable-http` behind a reverse proxy: the Host names the
  MCP SDK's DNS-rebinding check accepts. Without it the SDK default applies, which is
  loopback names only on a loopback host and no check elsewhere.
- `kg --version`, and `-H`/`--human` is accepted after the subcommand as well as before
  it (`kg verify --offline -H`).
- `kg doctor` reports the installed versions, the recorded embedder for the configured
  database, and whether a Ladybug file has its extensions and full-text indexes.
- `kg-ladybug-setup --apply` creates the schema and the four full-text indexes as well as
  installing the extensions, so a fresh database answers `kg ask` before anything is
  ingested.

### Fixed

- Every search on a Ladybug graph raised `Binder exception: Table RelatesToNode_ doesn't
  have an index with name edge_name_and_fact` (#1). graphiti-core's Kuzu driver declares
  its full-text indexes but its index builder is a no-op, so they were never created;
  the edgeless graph in the report was a symptom. Read-write opens and the setup command
  create them now, and an empty graph returns no facts.
- `kg-workspace drain` and `kg-ingest` report a refused write or a backend that did not
  open as an error with an exit code instead of a traceback.
- `__version__` reported 0.1.0 whatever the release; it comes from the package metadata.

## 0.2.2

- `kg-ingest` handles `SIGTERM` and `SIGINT`: it finishes the record in flight, closes
  the driver, and reports `interrupted: true` so a re-run continues from the ledger.
  Previously a wrapped `timeout` killed the process mid-write. On the embedded Ladybug
  backend that could leave a partial write the database then refused to reopen
  (`Storage exception: Checksum verification failed, the WAL file is corrupted`), which
  took out every later read as well. This was found by an end-to-end run, not in review.

  A signal cannot save you from `SIGKILL` or a power cut. Take a snapshot with
  `kg export` before a long ingest into an embedded backend.

## 0.2.1

- `kg doctor` gains an `embedding-dim-binding` check. graphiti reads `EMBEDDING_DIM`
  from the environment once, at import time, and uses that constant to build zero
  vectors during search. If anything imports graphiti before the value is set, searches
  are built at the wrong width while the embedder returns the right one, and nothing
  reports it. The check compares the constant graphiti actually resolved against
  `embedder.dimensions`. This matters most on the local profile, where
  `nomic-embed-text` is 768 and graphiti's own default is 1024.
- The export header no longer implies an import command exists; embeddings are omitted
  because they are derived from the text under whichever embedder is in use.

## 0.2.0 — Runs unattended

v0.1.0 was read-first and human-gated, but it assumed a human at the keyboard: output
was decorated text, a refused write still exited 0, configuration errors surfaced as
tracebacks mid-run, and a half-finished ingest could not resume. This release makes the
tool safe to hand to a cron job or an agent.

### Breaking

- **`kg` prints JSON on stdout by default.** Pass `-H`/`--human` for the previous
  reader-friendly output. Failures now go to stderr and never appear on stdout as a
  success-shaped object.
- **`kg propose` enforces the group allow-list.** A fact addressed to a group that
  `graph.groups` does not declare is refused with exit code `2` and is not queued.
  Reads always enforced this; the write path did not.
- **`server.transport: streamable-http` requires `server.auth.token`** (16 characters
  or more). An unauthenticated network transport exposed every configured group.
- **A known embedding model whose width disagrees with `embedder.dimensions` is now a
  configuration error.** For example `nomic-embed-text` with `1536`. The mismatch was
  previously silent and corrupted every embedding it wrote.
- `kg-ingest` and `kg-workspace` also default to JSON and accept `-H`.

### Added

- `kg doctor [--offline]` — preflight for configuration, workspace, backend, LLM and
  embedder. The embedder check calls the live endpoint and compares the real vector
  width against the configured one.
- `kg verify [--offline]` — asserts the six read tools are registered, that no write
  tool is exposed (a negative control), and that live retrieval works. Run it before
  and after any model, config or backend change.
- `kg export [group ...] [--output PATH]` — a backend-independent JSONL snapshot
  written from the graphiti models. Embeddings are omitted because they are derived
  from the text and restoring them under a different model would be silently wrong.
- `reranker:` configuration with `passthrough` (default, no extra model call), `bge`
  (local, `uv sync --extra rerank`), `openai` and `gemini`. Upstream ships these
  cross-encoders; there was previously no way to select one.
- `config/ollama.example.yaml` — a fully local profile: local model, local embedder,
  embedded Ladybug backend, no API key.
- Resumable ingestion: every applied record is recorded in a content-keyed ledger, so
  re-running a file ingests only what has not landed. `--no-resume` ignores the ledger.
- Per-record fault isolation in ingestion: one failing episode no longer costs the
  batch. Failures are reported and set a non-zero exit code. `--fail-fast` restores the
  previous behaviour. `--skip-invalid` tolerates malformed input lines.

### Fixed

- `kg-workspace drain` archived a proposal even when its ingestion did not succeed,
  which lost the proposal. It now archives only what landed and leaves the rest
  approved for the next drain.
- The proposal queue rejected group names that `graph.groups` accepts (it required
  lowercase and disallowed dots). A configurable group is now always proposable.
- `kg doctor` reports the resolved absolute Ladybug path; a relative path is read from
  the working directory, which is rarely what the reader assumes.

## 0.1.0

Initial standalone release: six read-only MCP tools, the `kg` CLI, explicit dry-run
ingestion, and the human-gated proposal queue.
