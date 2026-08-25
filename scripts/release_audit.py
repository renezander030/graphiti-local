"""Fail when a release tree contains likely secrets, PII, or local residue."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

IGNORED_PARTS = {".git", ".venv", ".ruff_cache", ".pytest_cache", "__pycache__"}
TEXT_SUFFIXES = {
    "",
    ".md",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".txt",
    ".example",
}
PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "absolute home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
}


def audit(root: Path) -> list[str]:
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(root)}: {name}")
    if (root / ".git").exists():
        remote = subprocess.run(
            ["git", "remote", "-v"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if remote:
            findings.append("git metadata: remote configured")
        identities = subprocess.run(
            ["git", "log", "--format=%an <%ae>"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if any(pattern.search(identities) for pattern in PATTERNS.values()):
            findings.append("git metadata: sensitive author identity")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    findings = audit(args.root.resolve())
    if findings:
        print("release audit failed:")
        for finding in findings:
            print(f" - {finding}")
        raise SystemExit(1)
    print("release audit passed: no likely secrets, PII, local paths, or remotes")


if __name__ == "__main__":
    main()

