from pathlib import Path

import pytest

from kg_mcp.config import load_config

SECRET = "synthetic-secret-value-0123456789"


def _write_config(tmp_path: Path, api_key: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
graph:
  groups: [public]
database:
  provider: falkordb
llm:
  model: test-model
  api_key: "{api_key}"
embedder:
  model: test-embedder
  dimensions: 8
  api_key: "{api_key}"
""",
        encoding="utf-8",
    )
    return path


def test_file_reference_reads_secret_and_strips_one_newline(tmp_path: Path):
    secret = tmp_path / "openai_key"
    secret.write_text(SECRET + "\n", encoding="utf-8")
    settings = load_config(_write_config(tmp_path, f"${{file:{secret}}}"))
    assert settings.llm.api_key == SECRET
    assert settings.embedder.api_key == SECRET


def test_file_reference_keeps_inner_whitespace_and_strips_only_one_newline(tmp_path: Path):
    secret = tmp_path / "key"
    secret.write_text("a b\n\n", encoding="utf-8")
    settings = load_config(_write_config(tmp_path, f"${{file:{secret}}}"))
    assert settings.llm.api_key == "a b\n"


def test_relative_file_reference_resolves_against_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key").write_text(SECRET, encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    settings = load_config(_write_config(tmp_path, "${file:secrets/key}"))
    assert settings.llm.api_key == SECRET


def test_missing_secret_file_is_refused_clearly(tmp_path: Path):
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="secret file") as info:
        load_config(_write_config(tmp_path, f"${{file:{missing}}}"))
    assert str(missing) in str(info.value)


def test_unreadable_secret_file_error_never_shows_the_secret(tmp_path: Path):
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    with pytest.raises(ValueError, match="secret file") as info:
        load_config(_write_config(tmp_path, f"${{file:{directory}}}"))
    assert SECRET not in str(info.value)

    binary = tmp_path / "binary"
    binary.write_bytes(SECRET.encode() + b"\xff\xfe")
    with pytest.raises(ValueError, match="secret file") as info:
        load_config(_write_config(tmp_path, f"${{file:{binary}}}"))
    assert SECRET not in str(info.value)
    assert "synthetic-secret" not in str(info.value)


def test_environment_references_still_expand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_KEY", SECRET)
    settings = load_config(_write_config(tmp_path, "${TEST_KEY:}"))
    assert settings.llm.api_key == SECRET
