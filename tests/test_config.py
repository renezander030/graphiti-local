from pathlib import Path

import pytest

from kg_mcp.config import Settings, allowed_groups, load_config


def test_loads_yaml_and_expands_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "config.yaml"
    path.write_text(
        """
graph:
  groups: [public]
database:
  provider: falkordb
llm:
  model: test-model
  api_key: ${TEST_KEY:}
embedder:
  model: test-embedder
  dimensions: 8
  api_key: ${TEST_KEY:}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_KEY", "synthetic-key")
    settings = load_config(path)
    assert settings.llm.api_key == "synthetic-key"
    assert settings.graph.groups == ["public"]


def test_group_allow_list_rejects_unknown_group():
    settings = Settings.model_validate({"graph": {"groups": ["allowed"]}})
    with pytest.raises(ValueError, match="not allowed"):
        allowed_groups("other", settings)


def test_ladybug_uses_embedded_scope():
    settings = Settings.model_validate(
        {"graph": {"groups": ["local"]}, "database": {"provider": "ladybug"}}
    )
    assert allowed_groups("local", settings) is None
    with pytest.raises(ValueError, match="not allowed"):
        allowed_groups("other", settings)

