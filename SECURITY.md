# Security

Please report vulnerabilities privately to the repository owner before opening a
public issue.

The server is intended to run on stdio or a loopback HTTP interface. If it is
placed behind a network endpoint, add authentication and TLS at the gateway and
restrict database access independently. Treat returned graph content as
sensitive even though the MCP surface is read-only.

