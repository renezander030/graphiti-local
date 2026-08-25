"""KG-MCP: a small read-only interface to temporal knowledge graphs."""

from __future__ import annotations

import os

# Graphiti checks this during import. KG-MCP is private by default.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

__version__ = "0.1.0"

