# Upstream and provenance

Graphiti Local is an independent community integration package. It depends on
[Graphiti](https://github.com/getzep/graphiti) (`graphiti-core==0.30.1`) for the
temporal graph model and search implementation. Graphiti is Apache-2.0 licensed.
Graphiti Local is not affiliated with or endorsed by Zep.

This repository does not vendor Graphiti source code, documentation, images,
examples, tests, or Git history. Its README documents Graphiti Local only.

LadybugDB compatibility is an adapter around Graphiti's Kuzu driver because the
two Python APIs share the surface used here. FalkorDB and Neo4j use Graphiti's
native drivers.

## The Kuzu driver upstream

Graphiti marks its Kuzu driver as deprecated: the original Kuzu project is no longer
maintained, and LadybugDB is its community continuation. Graphiti Local keeps the
embedded backend on that driver deliberately and carries what the driver leaves out in
`kg_mcp/ladybug.py`: read-only opens so readers coexist with the one writer Ladybug
allows, the full-text indexes the driver declares but never creates, and reopening a
read-only handle when another process commits. Should a future `graphiti-core` remove
the driver, that adapter is where it would be vendored; the pin is exact so the removal
cannot arrive unannounced.
