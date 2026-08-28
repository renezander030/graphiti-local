# Changelog

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
