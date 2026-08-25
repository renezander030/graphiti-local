"""Graphiti Local: local-first temporal knowledge graphs for agents."""

from __future__ import annotations

import os

# Graphiti checks this during import. Graphiti Local is private by default.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

__version__ = "0.1.0"
