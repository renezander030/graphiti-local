import importlib.util
from pathlib import Path


def _module():
    script = Path(__file__).parents[1] / "scripts" / "release_audit.py"
    spec = importlib.util.spec_from_file_location("release_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def _audit_function():
    return _module().audit


def test_github_noreply_identities_are_share_safe():
    """A pull_request run checks out a merge commit GitHub signs with its own identity."""
    sensitive = _module().sensitive_identities
    # Assembled so this file does not itself contain an address the audit would flag.
    github = "GitHub <noreply" + "@" + "github.com>"
    user = "Someone <12345+someone" + "@" + "users.noreply.github.com>"
    person = "Real Person <real.person" + "@" + "example.com>"
    assert sensitive(github) == []
    assert sensitive(user) == []
    assert sensitive("KG-MCP Builder <noreply@localhost>") == []
    assert sensitive("\n".join([github, person])) == [person]


def test_release_audit_detects_home_paths(tmp_path: Path):
    path = "/" + "Users/example/private"
    (tmp_path / "bad.txt").write_text(path, encoding="utf-8")
    assert _audit_function()(tmp_path) == ["bad.txt: absolute home path"]


def test_release_audit_accepts_generic_source(tmp_path: Path):
    (tmp_path / "ok.md").write_text("Synthetic example only", encoding="utf-8")
    assert _audit_function()(tmp_path) == []
