"""Machine-readable output with a human fallback.

JSON on stdout is the default so an agent or cron job can parse a result without
scraping decorated text; ``-H``/``--human`` switches to the reader-friendly form.
Failures never print a success-shaped object: the message goes to stderr and the
process exits non-zero, so the exit code alone is a trustworthy signal.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

EXIT_ERROR = 1
EXIT_REJECTED = 2
EXIT_TIMEOUT = 3
# Proposal gates: each refusal has its own code so a caller can react without parsing.
EXIT_NOT_ATOMIC = 4
EXIT_RESEMBLES_REJECTED = 5
EXIT_SECRET_SHAPED = 6


class CommandError(Exception):
    """A failure that should reach the user as a message plus a non-zero exit code."""

    def __init__(self, message: str, *, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


def emit(
    payload: Any,
    *,
    human: Callable[[], Iterable[str]] | None = None,
    as_json: bool = True,
) -> None:
    """Write one command result to stdout."""
    if as_json or human is None:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        sys.stdout.write("\n")
        return
    for line in human():
        print(line)


def fail(message: str, *, code: int = EXIT_ERROR, as_json: bool = True) -> None:
    """Report a failure on stderr and exit non-zero."""
    if as_json:
        json.dump({"error": message}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
    else:
        print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


@contextmanager
def refusals(*, as_json: bool = True) -> Iterator[None]:
    """Turn a refused call into its message and exit code instead of a traceback."""
    try:
        yield
    except CommandError as exc:
        fail(str(exc), code=exc.code, as_json=as_json)
    except ValueError as exc:
        fail(str(exc), code=EXIT_REJECTED, as_json=as_json)
    except TimeoutError as exc:
        fail(str(exc), code=EXIT_TIMEOUT, as_json=as_json)
    except (FileNotFoundError, RuntimeError) as exc:
        fail(str(exc), as_json=as_json)
