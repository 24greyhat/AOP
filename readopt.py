"""Read-level optimization: treat every file read in a conversation as one history of that file.

Claude Code re-reads files constantly: after an edit, after a context jump, because the model forgot it
already had the file. Every copy is resent on every later request. This module indexes the whole
conversation's reads per path (before any rewriting) and answers two questions, both exactly:

  anchor(ref)       Is this whole-file Read byte-identical to an earlier whole-file Read of the same
                    path that is still in the history unchanged? Then the new copy adds nothing and
                    can be replaced by a short pointer (done once, at first sight, so cache-safe).

  stale_reason(ref) Is this whole-file Read of a file that was since overwritten (a later Write) or read
                    again with different content (a later intact Read)? Then the old copy is wrong or
                    redundant and can be reduced to an outline (done only when the cache is cold).

What is never done: ranged Reads (offset/limit), failed Reads, and Reads whose later copy was itself
narrowed are left alone, because dropping the only complete view of a file would force a re-Read.
No I/O and no model calls: pure functions over the request.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import elide
import targeted
from shrink import count_tokens

RESULT_HEAD = "<system-reminder>\nResult of calling the Read tool:\n"
RESULT_TAIL = "\n</system-reminder>"
_CALLED = re.compile(
    r'\A<system-reminder>\s*Called the Read tool with the following input:\s*(\{.*\})\s*</system-reminder>\s*\Z',
    re.S,
)
POINTER_START = "[proxy: identical to the earlier Read"


def called_input(text: str) -> Optional[dict]:
    """Input of a 'Called the Read tool' attachment marker, or None if `text` is not one."""
    m = _CALLED.match(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def called_path(text: str) -> Optional[str]:
    path = (called_input(text) or {}).get("file_path")
    return path if isinstance(path, str) else None


def is_pointer(text) -> bool:
    if not isinstance(text, str):
        return False
    if text.startswith(RESULT_HEAD):
        text = text[len(RESULT_HEAD):]
    return text.startswith(POINTER_START)


def pointer(path: str, lines: int, tokens: int) -> str:
    name = path.rsplit("/", 1)[-1] or path
    return (f"{POINTER_START} of {name}: the file has not changed since, so this copy ({lines} lines, "
            f"~{tokens} tokens) is omitted from this history to save tokens. Use that earlier result.]")


def wrap_attachment(inner: str) -> str:
    return f"{RESULT_HEAD}{inner}{RESULT_TAIL}"


@dataclass
class ReadRef:
    seq: int            # order of appearance among reads
    mi: int             # index of the message holding the result
    path: str
    orig: str           # the result exactly as Claude Code sent it
    blk: dict           # the live block: its `field` may be rewritten while the request is processed
    field: str          # "content" (tool_result) or "text" (an @-attached file)
    ident: str          # tool_use_id, or a synthetic id for attachments
    attach: bool = False

    @property
    def live(self):
        return self.blk.get(self.field)

    @property
    def intact(self) -> bool:
        return self.live == self.orig

    @property
    def tokens(self) -> int:
        return count_tokens(self.orig)


@dataclass
class ReadIndex:
    refs: list = field(default_factory=list)
    by_block: dict = field(default_factory=dict)    # id(live block) -> ReadRef
    by_path: dict = field(default_factory=dict)     # path -> [ReadRef] in order
    writes: dict = field(default_factory=dict)      # path -> [message index] of whole-file Writes

    @classmethod
    def build(cls, messages) -> "ReadIndex":
        """Index the reads of a request. Call before anything in `messages` is rewritten."""
        idx = cls()
        calls = {}        # tool_use_id -> (path, ranged)
        pending = None    # (path, ranged) named by the latest "Called the Read tool" marker
        for mi, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if msg.get("role") == "assistant" and isinstance(content, list):
                for blk in content:
                    if not (isinstance(blk, dict) and blk.get("type") == "tool_use"):
                        continue
                    args = blk.get("input") if isinstance(blk.get("input"), dict) else {}
                    path = args.get("file_path")
                    if not isinstance(path, str):
                        continue
                    if blk.get("name") == "Read" and isinstance(blk.get("id"), str):
                        calls[blk["id"]] = (path, args.get("offset") is not None or args.get("limit") is not None)
                    elif blk.get("name") == "Write":
                        idx.writes.setdefault(path, []).append(mi)
            elif msg.get("role") == "user":
                if isinstance(content, str):
                    marker = called_input(content)
                    if marker and isinstance(marker.get("file_path"), str):
                        pending = marker
                    continue
                for blk in content if isinstance(content, list) else []:
                    if not isinstance(blk, dict):
                        continue
                    if blk.get("type") == "tool_result":
                        call = calls.get(blk.get("tool_use_id"))
                        if call and not call[1] and not blk.get("is_error") and isinstance(blk.get("content"), str):
                            idx._add(mi, call[0], blk, "content", str(blk.get("tool_use_id")), False)
                    elif blk.get("type") == "text" and isinstance(blk.get("text"), str):
                        text = blk["text"]
                        marker = called_input(text)
                        if marker and isinstance(marker.get("file_path"), str):
                            pending = marker
                        elif text.startswith(RESULT_HEAD) and pending:
                            if pending.get("offset") is None and pending.get("limit") is None:
                                idx._add(mi, pending["file_path"], blk, "text", f"attach:{mi}:{len(idx.refs)}", True)
                            pending = None
        return idx

    def _add(self, mi, path, blk, field_name, ident, attach):
        ref = ReadRef(len(self.refs), mi, path, blk[field_name], blk, field_name, ident, attach)
        self.refs.append(ref)
        self.by_block[id(blk)] = ref
        self.by_path.setdefault(path, []).append(ref)

    # ------------------------------------------------------------ queries

    def ref_for(self, blk) -> Optional[ReadRef]:
        return self.by_block.get(id(blk))

    def anchor(self, ref: ReadRef) -> Optional[ReadRef]:
        """The earlier whole-file Read of the same file that `ref` merely repeats, if it is still in
        the history unchanged. Looks only at the nearest earlier reads: a different copy in between
        means the file changed, and the model should see the new one."""
        for other in reversed(self.by_path.get(ref.path, [])):
            if other.seq >= ref.seq:
                continue
            if other.orig != ref.orig:
                return None
            if other.intact:
                return other
            if is_pointer(other.live):
                continue          # a repeat itself: its own anchor is further back
            return None           # narrowed or converted: the model never saw all of it
        return None

    def stale_reason(self, ref: ReadRef, before_mi: int) -> Optional[str]:
        """Why a whole-file Read no longer shows the file, or None. The replacement (a later Read with
        different, intact content, or a later Write) must itself lie before `before_mi`, so that
        nothing that points at this copy sits in the part of the history that stays untouched."""
        for other in self.by_path.get(ref.path, []):
            if (other.seq > ref.seq and other.mi < before_mi and other.orig != ref.orig
                    and other.intact and not is_pointer(other.live)):
                return "the file was read again later with different content"
        for mi in self.writes.get(ref.path, []):
            if ref.mi < mi < before_mi:
                return "the file was overwritten later"
        return None


def stale_stub(path: str, text: str, reason: str) -> str:
    """Outline of an old Read result: size, opening lines, definitions with line numbers, closing lines."""
    inner = text[len(RESULT_HEAD):-len(RESULT_TAIL)] if text.startswith(RESULT_HEAD) and text.endswith(RESULT_TAIL) \
        else text
    parsed = targeted.parse_read_result(inner)
    body = "\n".join(t for _, _, t in parsed[0]) if parsed else inner
    summary = elide.file_stub(path, body).split("\n", 1)
    n = body.count("\n") + 1
    head = (f"[proxy: Read result of {path} ({n} lines, ~{count_tokens(text)} tokens) omitted from this "
            f"history because {reason}. Read the file again for its current text]")
    out = f"{head}\n{summary[1]}" if len(summary) > 1 else head
    return wrap_attachment(out) if text.startswith(RESULT_HEAD) else out
