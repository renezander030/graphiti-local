"""The v0.2.0 automation contract: honest exit codes, machine-readable output, resumability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_mcp.config import Settings
from kg_mcp.output import CommandError, emit, fail

CONFIG = """
server:
  transport: stdio
graph:
  groups: [allowed]
  workspace_dir: {workspace}
database:
  provider: falkordb
llm:
  model: test-model
embedder:
  model: test-embedder
  dimensions: 8
"""


@pytest.fixture()
def configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.format(workspace=workspace), encoding="utf-8")
    monkeypatch.setenv("GRAPHITI_LOCAL_CONFIG", str(config))
    monkeypatch.setenv("KG_WORKSPACE_DIR", str(workspace))
    return workspace


def run_cli(*argv: str) -> None:
    """Invoke the real entry point the way a shell would."""
    import sys

    from kg_mcp.cli import main

    original = sys.argv
    sys.argv = ["kg", *argv]
    try:
        main()
    finally:
        sys.argv = original


# --- item 2: the write path enforces the allow-list and says so with an exit code ---


def test_propose_rejects_a_group_the_config_does_not_declare(configured, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_cli("propose", "not-configured", "a fact")
    assert exit_info.value.code == 2, "a refused proposal must not exit 0"
    assert "not allowed" in capsys.readouterr().err


def test_refused_proposal_is_never_queued(configured):
    with pytest.raises(SystemExit):
        run_cli("propose", "not-configured", "a fact")
    assert not (configured / "pending.jsonl").exists()


def test_accepted_proposal_queues_and_exits_zero(configured, capsys):
    run_cli("propose", "allowed", "a real fact")
    payload = json.loads(capsys.readouterr().out)
    assert payload["domain"] == "allowed"
    assert payload["proposed"].startswith("proposal-")
    assert (configured / "pending.jsonl").exists()


def test_a_configured_group_name_is_always_proposable(configured):
    """GraphConfig and the queue must accept the same names."""
    from kg_mcp.config import GROUP_PATTERN
    from kg_mcp.workspace import DOMAIN_PATTERN

    for name in ("Example.v2", "team-a", "main_2"):
        assert GROUP_PATTERN.fullmatch(name)
        assert DOMAIN_PATTERN.fullmatch(name), f"{name} is configurable but not proposable"


# --- item 4: JSON by default, human on request, failures on stderr ---


def test_emit_defaults_to_json(capsys):
    emit({"a": 1}, human=lambda: ["a is 1"], as_json=True)
    assert json.loads(capsys.readouterr().out) == {"a": 1}


def test_emit_human_uses_the_renderer(capsys):
    emit({"a": 1}, human=lambda: ["a is 1"], as_json=False)
    assert capsys.readouterr().out.strip() == "a is 1"


def test_failures_go_to_stderr_and_exit_nonzero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        fail("broken", code=2, as_json=True)
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "", "stdout must never carry a success-shaped failure"
    assert json.loads(captured.err) == {"error": "broken"}


def test_command_error_carries_its_exit_code():
    assert CommandError("x", code=2).code == 2


# --- item 1: the dimension guard ---


def test_known_model_with_wrong_dimensions_is_rejected():
    with pytest.raises(ValueError, match="corrupts stored embeddings"):
        Settings.model_validate({"embedder": {"model": "nomic-embed-text", "dimensions": 1536}})


def test_ollama_tag_is_stripped_before_the_dimension_check():
    settings = Settings.model_validate(
        {"embedder": {"model": "nomic-embed-text:latest", "dimensions": 768}}
    )
    assert settings.embedder.dimensions == 768


def test_unknown_model_is_left_alone():
    settings = Settings.model_validate({"embedder": {"model": "custom-thing", "dimensions": 99}})
    assert settings.embedder.dimensions == 99


# --- item 8: the network transport cannot be opened without a token ---


def test_streamable_http_without_a_token_is_refused():
    with pytest.raises(ValueError, match=r"requires server\.auth\.token"):
        Settings.model_validate({"server": {"transport": "streamable-http"}})


def test_streamable_http_rejects_a_trivial_token():
    with pytest.raises(ValueError, match="at least 16 characters"):
        Settings.model_validate(
            {"server": {"transport": "streamable-http", "auth": {"token": "short"}}}
        )


def test_stdio_needs_no_token():
    assert Settings.model_validate({"server": {"transport": "stdio"}}).server.auth.token == ""


# --- item 6: the reranker is configurable and still defaults to passthrough ---


def test_reranker_defaults_to_passthrough():
    assert Settings().reranker.provider == "passthrough"


def test_hosted_reranker_requires_a_key():
    with pytest.raises(ValueError, match=r"requires reranker\.api_key"):
        Settings.model_validate({"reranker": {"provider": "openai"}})


def test_passthrough_reranker_preserves_input_order():
    import asyncio

    from kg_mcp.reranker import PassthroughReranker

    ranked = asyncio.run(PassthroughReranker().rank("q", ["first", "second", "third"]))
    assert [passage for passage, _ in ranked] == ["first", "second", "third"]
    assert ranked[0][1] > ranked[-1][1]
