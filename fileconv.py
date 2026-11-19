"""Proxy-side file reading: turn PDFs and Office files into Markdown.

Claude Code's Read tool sends PDFs as base64 documents (expensive: text plus a
page image each) and cannot meaningfully read .docx/.pptx/.xlsx. markitdown
converts them to compact Markdown instead; the optimizer injects the result.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

CONVERTIBLE = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".epub", ".msg"}
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_CHARS = 200_000   # ~50k tokens; longer documents are truncated with a notice
MIN_CHARS = 200       # near-empty output (scanned PDF) -> leave it to Claude's own reader

_lock = threading.Lock()
_converter = None


def is_convertible(path: str) -> bool:
    return Path(path).suffix.lower() in CONVERTIBLE


def _convert(path: Path):
    global _converter
    try:
        with _lock:
            if _converter is None:
                from markitdown import MarkItDown

                _converter = MarkItDown()
            text = (_converter.convert(str(path)).text_content or "").strip()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # markitdown's FileConversionException derives from BaseException
        return None
    if len(text) < MIN_CHARS:
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + (
            f"\n\n[... truncated: showing the first {MAX_CHARS:,} of {len(text):,} characters]"
        )
    return text


def convert_path(path: str):
    """Markdown for a local file, or None if it is missing, too big or unreadable."""
    p = Path(path).expanduser()
    try:
        if p.suffix.lower() not in CONVERTIBLE or not p.is_file() or p.stat().st_size > MAX_FILE_BYTES:
            return None
    except OSError:
        return None
    return _convert(p)


def convert_bytes(data: bytes, ext: str):
    """Markdown for in-memory file content (e.g. a base64 PDF block)."""
    if ext not in CONVERTIBLE or len(data) > MAX_FILE_BYTES:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"input{ext}"
        p.write_bytes(data)
        return _convert(p)
