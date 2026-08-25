# Upstream and provenance

Graphiti Local is an independent community integration package. It depends on
[Graphiti](https://github.com/getzep/graphiti) (`graphiti-core==0.29.3`) for the
temporal graph model and search implementation. Graphiti is Apache-2.0 licensed.
Graphiti Local is not affiliated with or endorsed by Zep.

This repository does not vendor Graphiti source code, documentation, images,
examples, tests, or Git history. Its README documents Graphiti Local only.

LadybugDB compatibility is an adapter around Graphiti's Kuzu driver because the
two Python APIs share the surface used here. FalkorDB and Neo4j use Graphiti's
native drivers.
