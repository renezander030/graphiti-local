# Local agent memory with Ollama and LadybugDB

This walkthrough starts with a new checkout and ends with a fact retrieved over
MCP. It uses only the repository's synthetic examples. Agents can retrieve stored
facts; proposed updates stay in a queue until a human approves and applies them.

## Requirements

- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a running
  [Ollama](https://docs.ollama.com/quickstart) server on `127.0.0.1:11434`.
- Python 3.10 or newer; uv can provision Python when needed.
- Disk space for dependencies and models: the `qwen2.5:7b` model download alone
  is about 4.7 GB. This walkthrough was tested on an Apple M5 with 32 GiB memory;
  that is a measured environment, not a claimed minimum specification.
- Internet during dependency, model, and database-extension installation.
  After setup, this configuration sends inference to local Ollama and stores
  graph data in a local file. No cloud API key is required.

The tested revision uses schema-constrained JSON output and temperature zero.
Older copies of the Ollama config used plain JSON mode, which returned invalid
schema objects during testing. See [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

The commands use bash/zsh on macOS or Linux. The recorded run was on macOS;
Windows and Linux end-to-end inference were not tested in that run.

## 1. Install the tested revision

```bash
git clone https://github.com/renezander030/graphiti-local.git
cd graphiti-local
git checkout 246d88eea50d2e191b4172bfc43874ecf32a509b
uv sync --frozen
ollama list
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

If `ollama list` cannot connect, start the Ollama application, or run `ollama serve`
in another terminal. Keep it running. These are local models, not cloud variants.

## 2. Create a separate demo database

Use a new workspace so your first run cannot mix with existing graph data.
Run the remaining commands in the same terminal, from the checkout directory.

```bash
export GRAPHITI_LOCAL_CONFIG="$PWD/config/ollama.example.yaml"
export KG_WORKSPACE_DIR="$PWD/workspace/local-demo"
export KG_LADYBUG_PATH="$KG_WORKSPACE_DIR/graph.ladybug"
uv run --frozen kg-ladybug-setup --database "$KG_LADYBUG_PATH"
uv run --frozen kg-ladybug-setup --database "$KG_LADYBUG_PATH" --apply
uv run --frozen kg doctor
```

Setup installs the FTS and VECTOR extensions and creates four search indexes.
The first call previews the work; `--apply` performs it. Doctor must show a
reachable database and `nomic-embed-text returns 768 dimensions`. A fresh database
can warn that no embedder has been recorded yet; its first ingest records that.

## 3. Ingest one synthetic project note and ask a question

```bash
uv run --frozen kg-ingest examples/local_memory_demo.jsonl
uv run --frozen kg-ingest examples/local_memory_demo.jsonl --apply
uv run --frozen kg ask "Which database does Aurora Analytics use?" example
```

The preview plans one record. The applied run reports `ingested: 1` and no failed
records. The query should return a fact about DuckDB.
Ingestion resumes by content: rerunning the same file can report skipped records.
Do not use `--no-resume` merely to make those counters look like a first run.

## 4. Observe the human approval boundary

Propose one deliberately distinctive synthetic fact:

```bash
uv run --frozen kg propose example "Aurora Analytics uses PostgreSQL as its database." \
  --type source-fact --provenance "synthetic demo fixture"
uv run --frozen kg pending example
uv run --frozen kg ask "Which database does Aurora Analytics use?" example
uv run --frozen kg-workspace drain
```

The proposal appears in `pending`, not in stored `facts`. The dry run has no
planned writes because nothing is approved. A pending proposal is visible as a
proposal and must not be treated as graph truth.

**Human step:** review the exact synthetic text above. Copy the returned
`proposal-...` identifier, then run these commands yourself when you want to
approve and apply it. Replace `PROPOSAL_ID` with that identifier.

```bash
uv run --frozen kg-workspace approve PROPOSAL_ID
uv run --frozen kg-workspace drain
uv run --frozen kg-workspace drain --apply
uv run --frozen kg ask "Which database does Aurora Analytics use?" example
uv run --frozen kg verify
```

The second dry run now plans the approved proposal. An `ingested` count confirms
the episode write, not that the model extracted every relationship. Verify the
actual query result after applying. After `--apply`, the query
should return the PostgreSQL database as a stored fact. The MCP interface
has no approval or write tool. A human approving this fixture is a demonstration,
not permission for an agent to approve other facts.

## 5. Connect an MCP client over stdio

Resolve the checkout directory with `pwd`, and find uv with `command -v uv`.
Replace `/absolute/path/to/uv` and **every** `/absolute/path/to/graphiti-local`
below. Absolute workspace paths avoid differences in the client's working directory.

```json
{
  "mcpServers": {
    "graphiti-local": {
      "command": "/absolute/path/to/uv",
      "args": ["--directory", "/absolute/path/to/graphiti-local", "run", "--frozen", "graphiti-local"],
      "env": {
        "GRAPHITI_LOCAL_CONFIG": "/absolute/path/to/graphiti-local/config/ollama.example.yaml",
        "KG_WORKSPACE_DIR": "/absolute/path/to/graphiti-local/workspace/local-demo",
        "KG_LADYBUG_PATH": "/absolute/path/to/graphiti-local/workspace/local-demo/graph.ladybug"
      }
    }
  }
}
```

This is the common `mcpServers` JSON shape; your client may use a different
configuration file or syntax. The recorded run used the Python MCP SDK over
stdio, including initialization, tool discovery, `get_status`, and a real
`search_memory_facts` call. It does not claim every desktop client was tested.

Expect exactly six tools: `search_nodes`, `search_memory_facts`, `get_entity_edge`,
`get_episodes`, `get_episode_entities`, and `get_status`. Ask the client to call
`search_memory_facts` with query `Which database does Aurora Analytics use?`
and `group_ids: ["example"]`. It should return the approved synthetic fact.

Stdio uses a local process pipe and needs no bearer token. The optional
`streamable-http` transport requires a configured bearer token; this guide does
not expose a network server.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Connection refused or model not found | Keep Ollama running; check `ollama list` and pull both named models. |
| Missing full-text indexes | Run `kg-ladybug-setup --database "$KG_LADYBUG_PATH" --apply` before starting readers. |
| Embedding-width mismatch | Use the shipped 768-dimension config with `nomic-embed-text`; do not reuse vectors from another model. |
| First query has no expected fact | Check ingest's `failed` array and `kg episodes example`; extraction is model-dependent. |
| Fact appears only under pending | A human must approve it, inspect the drain preview, and apply it. |
| MCP client sees an empty or different graph | Use the same absolute config, workspace, database paths, and `example` group. |
| A local extraction takes time | Let it finish. Interruptions wait for the current record; avoid force-killing an active embedded write. |

[Measured outputs and timings](local-demo-results.md) ·
[Report a successful or blocked setup](https://github.com/renezander030/graphiti-local/issues/new?template=setup-result.yml)

Report only synthetic output, your OS, model versions, and the failing step.
Do not attach a private graph, credentials, or client data. If the project helps
you, a [repository star](https://github.com/renezander030/graphiti-local) helps
other local-agent builders find it.
