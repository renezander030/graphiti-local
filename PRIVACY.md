# Privacy defaults

- MCP registers six read-only tools. It does not register ingestion, deletion,
  approval, or graph-maintenance tools.
- `GRAPHITI_TELEMETRY_ENABLED=false` is set before Graphiti is imported.
- The default HTTP bind address is `127.0.0.1`.
- Configuration contains environment-variable references, never credentials.
- Group names are an allow-list. A caller cannot select an unconfigured graph.
- Proposals remain in a local JSONL workspace until a human approves them and
  explicitly runs `kg-workspace drain --apply`.
- Ladybug extensions are installed only by `kg-ladybug-setup --apply`; opening a
  graph for reading does not download extensions.

The graph itself may contain sensitive data. Authentication, network exposure,
database access controls, backups, and retention remain deployment concerns.

