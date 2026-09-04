"""Graphiti Local: local-first temporal knowledge graphs for agents."""

from __future__ import annotations

import os
from importlib import metadata

# Graphiti checks this during import. Graphiti Local is private by default.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

try:
    __version__ = metadata.version("graphiti-local")
except metadata.PackageNotFoundError:  # a source checkout that was never installed
    __version__ = "0.0.0+unknown"
