"""Prompt shrinking: lossless cleanup plus LLMLingua-2 token pruning.

Every function here is deterministic -- the same text always shrinks to the
same bytes -- which is what lets the optimizer keep Anthropic's prompt cache
valid (a block, once rewritten, is replayed verbatim on every later request).
"""

from __future__ import annotations

import json
import os
import re
import threading
import warnings

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODELS = {
    "small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    "large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
}

# ---------------------------------------------------------------- tokens

_encoder = None


def count_tokens(text: str) -> int:
    """Approximate token count (tiktoken cl100k; ~chars/4 if unavailable)."""
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _encoder = False
    if _encoder:
        return len(_encoder.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


# ------------------------------------------------------- lossless cleanup

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


_HEX = re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{8,}\b", re.I)
_NUM = re.compile(r"\d+")
FOLD_MIN_RUN = 6
_FOLD_MARKER = "[… {n} similar lines omitted: same text, only numbers/ids differ …]"


def _template(line: str):
    """Line with numbers and ids masked, or None if it is not safe to fold."""
    if len(line) > 300 or line.lstrip().startswith(("|", "[… ", "[previous line")) or line.count("\t") > 1:
        return None
    masked = _NUM.sub("#", _HEX.sub("#", line))
    if "#" not in masked or sum(c.isalpha() for c in masked) < 4:
        return None
    return " ".join(masked.split())


def fold_similar(text: str) -> str:
    """Fold runs of log lines that differ only in numbers or ids.

    "[12/300] downloading pkg-3 ... done" x 300 becomes the first line, one
    marker, and the last line. Errors, URLs and file locations are never folded
    (they break a run), tables are skipped, and a run needs FOLD_MIN_RUN lines.
    """
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        key = None if protected(lines[i]) else _template(lines[i])
        j = i
        if key is not None:
            while j + 1 < len(lines) and not protected(lines[j + 1]) and _template(lines[j + 1]) == key:
                j += 1
        run = j - i + 1
        if key is not None and run >= FOLD_MIN_RUN:
            out += [lines[i], _FOLD_MARKER.format(n=run - 2), lines[j]]
        else:
            out += lines[i:j + 1]
        i = j + 1
    return "\n".join(out)


def clean(text: str, fold: bool = False) -> str:
    """Remove noise without changing meaning.

    Strips ANSI colour codes, keeps only the last frame of \\r progress bars,
    trims trailing whitespace, collapses repeated lines / blank-line runs and
    minifies JSON documents. With fold=True (command output only) runs of lines
    that differ only in numbers or ids are folded too.
    """
    text = _ANSI.sub("", text.replace("\r\n", "\n"))
    lines = []
    for line in text.split("\n"):
        if "\r" in line:
            frames = [part for part in line.split("\r") if part]
            line = frames[-1] if frames else ""
        lines.append(line.rstrip())

    out, i = [], 0
    while i < len(lines):
        j = i
        while j + 1 < len(lines) and lines[j + 1] == lines[i]:
            j += 1
        repeats = j - i + 1
        if lines[i] and repeats >= 3:
            out += [lines[i], f"[previous line repeated {repeats - 1} more times]"]
        else:
            out += lines[i:j + 1]
        i = j + 1

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    if text[:1] in ("{", "["):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            pass
    return fold_similar(text) if fold else text


def clean_prompt(text: str) -> str:
    """Strictly lossless cleanup for text the user wrote or pasted.

    Gentler than clean(): never collapses repeated lines and leaves fenced code
    blocks byte-for-byte alone (apart from terminal escape codes). Removes ANSI
    codes, keeps the last frame of \\r progress bars, trims trailing spaces and
    blank-line runs outside code fences, and minifies a prompt that is one JSON
    document.
    """
    text = _ANSI.sub("", text.replace("\r\n", "\n"))
    out, fenced, blanks = [], False, 0
    for line in text.split("\n"):
        if "\r" in line:
            frames = [part for part in line.split("\r") if part]
            line = frames[-1] if frames else ""
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            line = line.rstrip()
            blanks = 0
        elif not fenced:
            line = line.rstrip()
            blanks = blanks + 1 if not line else 0
            if blanks > 1:
                continue
        out.append(line)
    text = "\n".join(out)
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            return json.dumps(json.loads(stripped), ensure_ascii=False, separators=(",", ":"))
        except ValueError:
            pass
    return text


_TABLE_SEP = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def compact_markdown(text: str) -> str:
    """Lossless layout cleanup for Markdown the model wrote: tables lose their column padding
    (rendering ignores it) and separator rows shrink to `---`. Fenced code is left alone."""
    out, fenced = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
        elif not fenced and line.lstrip().startswith("|") and line.count("|") >= 3:
            if _TABLE_SEP.match(line):
                cells = line.strip().strip("|").split("|")
                line = "|" + "|".join(
                    (":" if c.strip().startswith(":") else "") + "---" + (":" if c.strip().endswith(":") else "")
                    for c in cells) + "|"
            else:
                line = re.sub(r" {2,}", " ", line)
        out.append(line)
    return "\n".join(out)


# ------------------------------------------------------ LLMLingua-2 pruning

# Lines that must survive verbatim. Locations and URLs are protected at any length;
# error keywords only on short, log-style lines (a long prose line that merely
# mentions "failed" would otherwise shield a whole paragraph).
_ALWAYS = re.compile(r"https?://|\w\.\w{1,6}:\d+|^\s*File \"|^\s*at \S+ \(")
_KEYWORDS = re.compile(
    r"(?i)\b(error|exception|traceback|fatal|panic|failed|failure|denied|exit code|assert\w*)\b"
    r"|no such file|not found"
)
_SHORT_LINE = 240
_WORD = re.compile(r"^[\w'’\-]*[A-Za-z][\w'’\-]*[.,;:!?)\"”]?$")


def protected(line: str) -> bool:
    return bool(_ALWAYS.search(line)) or (len(line) < _SHORT_LINE and bool(_KEYWORDS.search(line)))


def prose_share(text: str) -> float:
    """Share of characters sitting in sentence-like lines (>= 7 words, mostly plain words).

    Natural-language text scores near 1; source code, tables, diffs and data score low.
    """
    total = good = 0
    for line in text[:30000].split("\n"):
        if not line:
            continue
        total += len(line)
        words = line.split()
        if len(words) >= 7 and sum(1 for w in words if _WORD.match(w)) >= 0.75 * len(words):
            good += len(line)
    return good / total if total else 0.0


class Compressor:
    """Lazy-loading wrapper around LLMLingua-2 (token-classification pruning)."""

    def __init__(self, model="small", rate=0.6, device="cpu", min_run_tokens=80,
                 min_prose=0.6, max_chars=150_000, load_timeout=90.0):
        self.model_id = MODELS.get(model, model)
        self.rate = rate
        self.device = device
        self.min_run_tokens = min_run_tokens
        self.min_prose = min_prose
        self.max_chars = max_chars
        self.load_timeout = load_timeout
        self.error = None
        self._pc = None
        self._ready = threading.Event()
        self._infer = threading.Lock()

    def start(self):
        threading.Thread(target=self._load, daemon=True, name="llmlingua-load").start()

    def _load(self):
        try:
            from llmlingua import PromptCompressor

            self._pc = PromptCompressor(
                model_name=self.model_id, use_llmlingua2=True, device_map=self.device
            )
        except Exception as exc:  # missing package, offline, out of memory ...
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

    @property
    def state(self) -> str:
        if not self._ready.is_set():
            return "loading"
        return "ready" if self._pc else "unavailable"

    def squeeze(self, text: str):
        """Prune low-information tokens. Returns None when it can't or shouldn't."""
        if len(text) > self.max_chars or not self._ready.wait(self.load_timeout) or not self._pc:
            return None
        pieces, run = [], []

        def flush():
            if run:
                pieces.append(self._prune("\n".join(run)))
                run.clear()

        for line in text.split("\n"):
            if protected(line):
                flush()
                pieces.append(line)
            else:
                run.append(line)
        flush()
        return "\n".join(pieces)

    def _prune(self, chunk: str) -> str:
        if count_tokens(chunk) < self.min_run_tokens or prose_share(chunk) < self.min_prose:
            return chunk
        try:
            with self._infer:
                out = self._pc.compress_prompt_llmlingua2(
                    [chunk],
                    rate=self.rate,
                    force_tokens=["\n", ".", "?", "!"],
                    chunk_end_tokens=[".", "\n"],
                    drop_consecutive=True,
                    force_reserve_digit=True,
                )
            pruned = (out.get("compressed_prompt") or "").strip()
        except Exception:
            return chunk
        return pruned or chunk
