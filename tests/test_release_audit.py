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


def _git(root: Path, *args: str, who: str = "KG-MCP Builder <noreply@localhost>") -> str:
    import subprocess

    name, email = who[: who.index(" <")], who[who.index("<") + 1 : -1]
    env = {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
        "HOME": str(root),
        "PATH": "/usr/bin:/bin",
    }
    return subprocess.run(
        ["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    """A repository whose first commit carries a personal address, as old history can."""
    root = tmp_path / "repo"
    root.mkdir()
    # Assembled so this file does not itself contain an address the audit would flag.
    person = "Real Person <real.person" + "@" + "example.com>"
    _git(root, "init", "-q")
    (root / "a.txt").write_text("one\n")
    _git(root, "add", "a.txt")
    _git(root, "commit", "-q", "-m", "old", who=person)
    (root / "a.txt").write_text("two\n")
    _git(root, "commit", "-q", "-am", "new")
    return root


def test_a_published_tip_is_audited_without_its_public_history(tmp_path: Path):
    """main after a merge: HEAD is the published base, so only the tip is checked."""
    root = _repo(tmp_path)
    _git(root, "branch", "published")
    assert _audit_function()(root, allow_remote=True, identity_base="published") == []


def test_a_personal_identity_on_the_published_tip_is_still_refused(tmp_path: Path):
    root = _repo(tmp_path)
    person = "Real Person <real.person" + "@" + "example.com>"
    (root / "a.txt").write_text("three\n")
    _git(root, "commit", "-q", "-am", "tip", who=person)
    _git(root, "branch", "published")
    findings = _audit_function()(root, allow_remote=True, identity_base="published")
    assert findings == ["git metadata: sensitive author identity"]


def test_unpublished_commits_are_all_audited(tmp_path: Path):
    """A branch: every commit since the published base is checked, not only the tip."""
    root = _repo(tmp_path)
    first = _git(root, "rev-list", "--max-parents=0", "HEAD")
    _git(root, "branch", "published", first)
    _git(root, "reset", "-q", "--hard", first)
    person = "Real Person <real.person" + "@" + "example.com>"
    (root / "a.txt").write_text("unpublished\n")
    _git(root, "commit", "-q", "-am", "mine", who=person)
    (root / "a.txt").write_text("later\n")
    _git(root, "commit", "-q", "-am", "later")
    findings = _audit_function()(root, allow_remote=True, identity_base="published")
    assert findings == ["git metadata: sensitive author identity"]
