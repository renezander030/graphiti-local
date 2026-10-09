"""Deterministic checks a proposal passes before it can enter the review queue.

Every check is pure: it reads text (and, for rejections, the archive the caller
passes in) and returns what it found. ``workspace.add_proposal`` decides what a
finding means and which exit code a refusal carries.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any

# Commit hashes, wallet addresses and e-mail account identifiers are deliberately not
# here: they are legitimate fact content.
SECRET_SHAPES = (
    (
        "api key",
        r"\b(sk-(ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}"
        r"|hf_[A-Za-z0-9]{30,}|fw_[A-Za-z0-9]{20,})",
    ),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}"),
    ("bot token", r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b"),
    (
        "credential assignment",
        r"(?i)\b(password|passwort|passwd|pwd|secret|api[_ -]?key|bearer)\b"
        r"\s*(is|=|:)\s*[\"']?[^\s\"']{8,}",
    ),
    ("iban", r"\b[A-Z]{2}\d{2}(?: ?\d{4}){3,7}(?: ?\d{1,4})?\b"),
)


def secret_shapes(*texts: str) -> list[str]:
    """Names of the secret shapes found in the texts. Never returns the matched value."""
    joined = "\n".join(text or "" for text in texts)
    return [name for name, pattern in SECRET_SHAPES if re.search(pattern, joined)]


# Tokens before a period that do not end a sentence (EN + DE), lower case, no final dot.
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "approx", "ca", "no", "nr", "dr", "mr", "mrs", "ms",
    "prof", "jr", "sr", "st", "inc", "ltd", "co", "corp", "u.s", "u.k", "a.m", "p.m", "min",
    "max", "sec", "fig", "vol", "dept", "est", "al", "z.b", "bzw", "ggf", "usw", "vgl", "str",
    "tel", "evtl", "u.a", "d.h", "z.t", "s.o", "s.u",
}
_BOUNDARY = re.compile(r"([.!?]+)(\s+)(?=[\"'(\[]?[A-Z0-9ÄÖÜ])")


def sentences(text: str) -> list[str]:
    """Split text into sentences deterministically.

    A boundary is ``[.!?]`` + whitespace + an upper-case letter or digit, unless the
    token before it is a known abbreviation or a single initial. Decimals, versions,
    URLs and dates contain no whitespace after the dot, so they never split.
    """
    normalized = re.sub(r"\.{2,}", " ", text)
    found, start = [], 0
    for match in _BOUNDARY.finditer(normalized):
        before = normalized[start : match.start()]
        last = re.search(r"([A-Za-z\.]+)$", before)
        token = last.group(1).lower() if last else ""
        if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
            continue
        found.append(normalized[start : match.end(1)].strip())
        start = match.end()
    found.append(normalized[start:].strip())
    return [sentence for sentence in found if sentence]


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def word_overlap(first: str, second: str) -> float:
    """Shared words over all words (Jaccard). A changed entity name lowers it sharply,
    so a correction of a rejected fact is not mistaken for the same claim."""
    a, b = _words(first), _words(second)
    return len(a & b) / len(a | b) if a or b else 1.0


def rejected_matches(
    domain: str, text: str, rejected: Iterable[dict[str, Any]], threshold: float
) -> list[tuple[float, dict[str, Any]]]:
    """Rejected proposals of the same domain that ``text`` resembles, closest first."""
    matches = []
    for item in rejected:
        if item.get("domain") != domain:
            continue
        candidates = item.get("facts") or [item.get("text", "")]
        ratio = max(word_overlap(text, str(candidate)) for candidate in candidates)
        if ratio >= threshold:
            matches.append((ratio, item))
    matches.sort(key=lambda match: -match[0])
    return matches


# The fields a reviewer approves. Status, decision and drain bookkeeping are excluded.
CONTENT_FIELDS = (
    "id",
    "domain",
    "operation",
    "text",
    "facts",
    "fact_type",
    "provenance",
    "source",
    "link",
    "supersedes",
    "supersedes_rejection",
    "valid_at",
    "learned_at",
)


def content_digest(item: dict[str, Any]) -> str:
    """SHA-256 over the reviewed content of a proposal, in canonical JSON."""
    content = {field: item.get(field) for field in CONTENT_FIELDS if item.get(field) is not None}
    canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
