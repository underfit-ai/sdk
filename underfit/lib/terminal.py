"""Console capture utilities."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any, Callable


class _InterceptedStream:
    def __init__(self, stream: IO[str], callback: Callable[[str], None]) -> None:
        self._stream = stream
        self._callback = callback

    def write(self, data: str) -> int:
        written = self._stream.write(data)
        if written and data:
            self._callback(data)
        return written

    def writelines(self, lines: list[str]) -> None:
        self._stream.writelines(lines)
        for line in lines:
            if line:
                self._callback(line)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


@contextmanager
def capture(write_callback: Callable[[str, str], None]) -> Iterator[None]:
    """Capture stdout and stderr writes with a callback."""
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _InterceptedStream(sys.stdout, lambda data: write_callback("stdout", data))
    sys.stderr = _InterceptedStream(sys.stderr, lambda data: write_callback("stderr", data))
    try:
        yield
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
