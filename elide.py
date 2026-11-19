"""Abridge the model's own tool-call arguments once they have run.

What the assistant *wrote* into a tool call -- a whole file for Write, a script
heredoc for Bash -- is the biggest single slice of context in a long session,
and it is dead weight after the call has executed: the file is on disk, the
command's effect is in its result. Each such block is abridged the first time
it is seen (when it is still in the not-yet-cached tail, so it costs nothing to
change) and replayed byte-for-byte afterwards, like every other rewrite here.

Stubs keep what the model needs to carry on without re-reading: where the
thing is (path), how big it was, how it starts and ends, and, for code, an
outline of its definitions with line numbers. Everything is deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Optional

import targeted
from shrink import count_tokens

TAG = "proxy"
NOTE_START = "Note: to save tokens, a proxy may abridge"
SYSTEM_NOTE = (
    f"{NOTE_START} older tool calls in this conversation's history, for example a Write call's `content` "
    "replaced by a bracketed [proxy: ...] summary. Those markers exist only in the displayed history; the files "
    "and command results themselves are intact. Never write such markers into files, commands or replies. "
    "Read a file when you need its exact current text."
)

TAIL_NOTE = f"<system-reminder>\n{SYSTEM_NOTE}\n</system-reminder>"   # same note, for a conversation joined midway

EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"}
MIN_GAIN = 150          # tokens an abridgement must save to be worth doing
MIN_RATIO = 0.4         # ... and the fraction of the block it must remove
OUTLINE_MAX = 40
HEAD_LINES = 6
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][\w]*)\1")


def _clip(line: str, n: int) -> str:
    return line if len(line) <= n else line[: n - 1] + "…"


def _marker(text: str) -> str:
    return f"[{TAG}: {text}]"


def file_stub(path: Optional[str], content: str) -> str:
    """Summary of a file body: size, opening lines, outline of definitions, closing lines."""
    lines = content.split("\n")
    n = len(lines)
    tokens = count_tokens(content)
    out = [_marker(f"{n}-line body (~{tokens} tokens) omitted from this history to save tokens; "
                   "the file on disk has all of it, Read it for exact text")]
    out.append("Start:")
    out += [_clip(line, 140) for line in lines[:HEAD_LINES]]
    ext = "." + path.rsplit(".", 1)[-1].lower() if path and "." in path.rsplit("/", 1)[-1] else ""
    units = None
    if ext in targeted.CODE_EXTENSIONS:
        try:
            units = targeted.split_units(lines, ext)
        except Exception:
            units = None
    shown = [u for u in (units or []) if u.kind != "misc" and u.start >= HEAD_LINES]
    if shown:
        out.append("Outline:")
        for u in shown[:OUTLINE_MAX]:
            out.append(f"L{u.start + 1} {_clip(u.sig, 100)}")
        if len(shown) > OUTLINE_MAX:
            out.append(f"… +{len(shown) - OUTLINE_MAX} more definitions")
    if n > 2 * HEAD_LINES:
        out.append("End:")
        out += [_clip(line, 140) for line in lines[-2:]]
    return "\n".join(out)


def text_stub(text: str) -> str:
    """Head and tail of a long string argument."""
    lines = text.split("\n")
    if len(lines) <= 8:  # one huge line: cut by characters
        head, tail = text[:600], text[-200:]
        return f"{head}\n{_marker(f'{len(text) - 800} characters omitted from this history')}\n{tail}"
    omitted = len(lines) - 5
    return "\n".join(lines[:3] + [_marker(f"{omitted} lines omitted from this history")] + lines[-2:])


def command_stub(command: str) -> str:
    """A long shell command: keep the command line itself, shorten the script body."""
    lines = command.split("\n")
    for i, line in enumerate(lines):
        m = _HEREDOC.search(line)
        if not m:
            continue
        tag = m.group(2)
        end = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == tag), None)
        if end is None or end - i <= 6:
            break
        body = lines[i + 1:end]
        kept = [_clip(b, 160) for b in body[:3]]
        return "\n".join(lines[:i + 1] + kept
                         + [_marker(f"{len(body) - 3} more script lines omitted from this history; "
                                    "the command ran as written")]
                         + lines[end:])
    return text_stub(command)


def elide_input(name: str, args: dict, min_tokens: int = 400, bash_min_tokens: int = 250):
    """Return (new_args, before_tokens, after_tokens) or None if nothing is worth abridging."""
    if not isinstance(args, dict):
        return None
    new = dict(args)
    before = after = 0

    def swap(key, make, floor):
        nonlocal before, after
        value = new.get(key)
        if not isinstance(value, str) or len(value) < floor * 3 or value.lstrip().startswith(f"[{TAG}:"):
            return
        b = count_tokens(value)
        if b < floor:
            return
        stub = make(value)
        a = count_tokens(stub)
        if b - a >= max(MIN_GAIN if floor >= 300 else MIN_GAIN // 2, b * MIN_RATIO):
            new[key] = stub
            before += b
            after += a

    if name == "Write":
        path = args.get("file_path") if isinstance(args.get("file_path"), str) else None
        swap("content", lambda v: file_stub(path, v), min_tokens)
    elif name == "Bash":
        swap("command", command_stub, bash_min_tokens)
    elif name == "Edit":
        swap("old_string", text_stub, min_tokens)
        swap("new_string", text_stub, min_tokens)
    elif name == "NotebookEdit":
        swap("new_source", text_stub, min_tokens)
    elif name == "MultiEdit" and isinstance(args.get("edits"), list):
        edits = []
        for edit in args["edits"]:
            if not isinstance(edit, dict):
                edits.append(edit)
                continue
            saved = dict(new)
            new = dict(edit)
            swap("old_string", text_stub, min_tokens)
            swap("new_string", text_stub, min_tokens)
            edits.append(new)
            new = saved
        new["edits"] = edits
    else:
        return None
    return (new, before, after) if after < before else None


def has_note(system) -> bool:
    text = system if isinstance(system, str) else "".join(
        b.get("text", "") for b in system if isinstance(b, dict)) if isinstance(system, list) else ""
    return NOTE_START in text


def annotate_system(system):
    """Append SYSTEM_NOTE to the last text block of a system prompt (str or block list)."""
    if isinstance(system, str):
        return system if NOTE_START in system else f"{system}\n\n{SYSTEM_NOTE}"
    if isinstance(system, list):
        for blk in reversed(system):
            if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                if NOTE_START not in blk["text"]:
                    blk["text"] = f"{blk['text']}\n\n{SYSTEM_NOTE}"
                return system
    return system


def json_of(args: dict) -> str:
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
