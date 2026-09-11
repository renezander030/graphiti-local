# Local demo results

Measured on 2026-09-11 on an Apple M5 with 32 GiB memory, macOS, Python 3.10.20,
Graphiti Local 0.3.0, graphiti-core 0.30.1, Ladybug 0.19.1, and MCP SDK 1.29.1.
Models: local `qwen2.5:7b` and `nomic-embed-text` (768 dimensions). The extraction
config uses `json_schema` and temperature zero. These are observations from one
machine and synthetic fixture, not performance or extraction-quality benchmarks.

The final walkthrough uses one initial database decision and one human-approved
update in a fresh workspace. The 42-second GIF is a presentation of captured
output excerpts with pauses condensed; it is not a real-time terminal recording.

## Final run

| Command | Observed seconds | Exit code |
| --- | ---: | ---: |
| setup | 1.321 | 0 |
| doctor | 0.548 | 0 |
| ingest | 34.398 | 0 |
| initial query | 1.329 | 0 |
| propose | 0.167 | 0 |
| before approval | 1.206 | 0 |
| unapproved drain | 0.156 | 0 |
| approve fixture | 0.150 | 0 |
| apply fixture | 44.596 | 0 |
| after approval | 1.328 | 0 |
| verify | 1.302 | 0 |

Times exclude downloading models/dependencies. The first doctor warning about an
unrecorded embedder is expected on a fresh database. `kg verify` returned warnings for its generic retrieval probe (no matches) and
its invalidation check (no edges found under the named group on Ladybug). The
specific CLI and MCP queries above passed. This run does not claim
that every verification check passed or that temporal correctness was benchmarked.

The actual MCP SDK client initialized the stdio server, found exactly six read
tools, called `get_status`, and retrieved the approved PostgreSQL fact with
`search_memory_facts`. See [full MCP responses](evidence/mcp.json).

### Before approval

The proposed PostgreSQL update was absent from stored facts. It was separately
visible in the pending queue; the unapproved drain preview was empty.

```json
{
  "query": "Which database does Aurora Analytics use?",
  "groups": [
    "example"
  ],
  "facts": [
    {
      "fact": "Aurora Analytics uses DuckDB as its database.",
      "uuid": "b374bcc8-ab24-49b7-bb8a-5a03244ff908",
      "group_id": "",
      "valid_at": "2026-01-10T00:00:00",
      "invalid_at": null
    }
  ],
  "pending": [
    {
      "id": "proposal-08345f9455e2",
      "domain": "example",
      "text": "Aurora Analytics uses PostgreSQL as its database."
    }
  ]
}
```

### After human approval and application

```json
{
  "query": "Which database does Aurora Analytics use?",
  "groups": [
    "example"
  ],
  "facts": [
    {
      "fact": "Aurora Analytics uses DuckDB as its database.",
      "uuid": "b374bcc8-ab24-49b7-bb8a-5a03244ff908",
      "group_id": "",
      "valid_at": "2026-01-10T00:00:00",
      "invalid_at": "2026-09-11T00:00:00"
    },
    {
      "fact": "Aurora Analytics uses PostgreSQL as its database.",
      "uuid": "7290d5fd-5f29-4568-a8fe-8010de57e172",
      "group_id": "",
      "valid_at": "2026-09-11T00:00:00",
      "invalid_at": null
    }
  ],
  "pending": []
}
```

Search can return historical or loosely related facts. Read `valid_at` and
`invalid_at`; do not treat every returned row as a currently valid answer.

## Failures encountered during preparation

Three earlier attempts failed. They are retained here to make the limits visible:

- The plain-JSON configuration omitted the generic name "Example Project" during
  entity extraction in two proposal examples. Upstream then discarded the relation
  because its source entity was missing. The episode write still reported ingestion.
- With clearer entity names, plain-JSON mode also returned a schema object where
  `EdgeDuplicate` data was required. That ingestion correctly reported a failure.
- A subsequent schema-constrained run completed, but the small model incorrectly
  invalidated a reporting-database fact when an unrelated documentation fact arrived.
  Schema constraints ensure output shape; they do not guarantee semantic correctness.

The published fixture uses a named synthetic organization and one clear database
change. Always inspect retrieved facts and timestamps after ingestion. The config
change and revised fixture make this walkthrough work; they do not eliminate the
need to review model extraction on your own data.

[Final command transcript](evidence/commands.json) ·
[First failed attempt](evidence/commands-first-attempt.json) ·
[Second failed attempt](evidence/commands-second-attempt.json) ·
[Schema parsing failure](evidence/commands-third-attempt.json) ·
[Two-note schema run](evidence/commands-schema-two-notes.json)

All records are synthetic. Local checkout paths were replaced with `$CHECKOUT`.
The existing automated suite also passed: 78 tests and Ruff.
