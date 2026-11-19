"""Rewrite /v1/messages requests to carry fewer tokens without breaking caching.

Anthropic's prompt cache matches on an exact prefix, so changing anything that
was already sent invalidates everything after it. The rule that keeps both
goals intact:

    shrink a block once, the first time it is seen, and replay that exact
    result forever after.

A block seen for the first time sits in the not-yet-cached tail, so shrinking
it costs nothing. Every later request resends the memoized bytes, so the cached
prefix never changes. Decisions to leave a block alone are memoized as well. If
the proxy joins a conversation midway, history that may already be cached is
frozen as-is and only new content is shrunk.

The user's own messages get lossless cleanup (escape codes, progress-bar
frames, blank-line runs, trailing spaces; fenced code is kept verbatim) when
long enough to matter. Lossy pruning is limited to big pasted prose: long
paragraphs that are not the first or last paragraph, not code, and read as
natural language. Short paragraphs -- where instructions live -- stay verbatim.

Whole-file Reads of big source files are narrowed to the parts that match the task
(targeted.py): kept lines are byte-identical, elided ranges are marked, and any
Read with offset/limit is left alone so the model can always fetch the rest.

What the assistant itself wrote into Write/Bash/Edit calls is abridged once the call has
run (elide.py): that is the largest slice of a long session's context and dead weight
after execution. The fixed prefix -- tool definitions and system prompt -- is cleaned
once, when a conversation starts, and slimmed for MCP tools that go unused (prefix.py).
Assistant narration gets lossless cleanup (lossy pruning is opt-in: models imitate
the style of their own history).

Never touched: thinking blocks (they carry signatures), the results of Edit/Grep/Glob
and ranged Reads. A conversation the proxy joins midway keeps its (probably cached) history
and prefix exactly as sent; only turns from then on are rewritten.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cold
import economics
import elide
import fileconv
import prefix
import readopt
import targeted
from shrink import clean, clean_prompt, compact_markdown, count_tokens, prose_share

MEMO_VERSION = "1"
# Tool output that is prose/log-like and safe to prune lossily. Everything else
# (Read, Edit, Write, Grep, Glob, ...) is left exactly as the tool returned it.
LOSSY_TOOLS = {"Bash", "BashOutput", "WebFetch", "WebSearch", "Agent", "Task"}
LOSSY_NOTE = "[Condensed by proxy: filler words removed. Re-run the command if exact text matters.]"
# Command output is log-like, so runs of near-identical lines can be folded. Other tools
# (MCP, WebFetch...) may return tables or data rows where every line matters.
FOLDED_TOOLS = {"Bash", "BashOutput"}
PASTE_NOTE = "[condensed by proxy: filler words removed]"
MIN_GAIN_TOKENS = 24
PDF_PAGE_TOKENS = 2250  # text + page image, per Anthropic's PDF pricing notes

_RESULT_HEAD = readopt.RESULT_HEAD
_RESULT_TAIL = readopt.RESULT_TAIL


@dataclass
class Entry:
    text: Optional[str]  # None = leave the block exactly as it arrived
    before: int = 0
    after: int = 0


@dataclass
class Event:
    label: str
    before: int
    after: int
    fresh: bool          # decided on this request (vs. replayed from memo)
    converted: bool = False


@dataclass
class Result:
    body: bytes
    events: list = field(default_factory=list)
    conv: str = ""
    model: str = ""
    orig_pieces: Optional[list] = None   # estimated tokens per piece (prefix, then each message),
    new_pieces: Optional[list] = None    # original vs rewritten: lets economics.py price both sides
    cold: bool = False                   # the cache had expired: history was compacted on this request

    @property
    def changed(self):
        return bool(self.events)

    @property
    def saved(self):  # net tokens this request drops versus the untouched original
        return sum(e.before - e.after for e in self.events)

    @property
    def fresh_saved(self):
        return sum(e.before - e.after for e in self.events if e.fresh)


class Memo:
    """Persistent block-hash -> replacement store (survives proxy restarts)."""

    def __init__(self, path=None, max_age=7 * 86400):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._db = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(
                "CREATE TABLE IF NOT EXISTS memo(key TEXT PRIMARY KEY, text TEXT, before INTEGER, after INTEGER, ts REAL);"
                "CREATE TABLE IF NOT EXISTS convs(key TEXT PRIMARY KEY, ts REAL);"
                "CREATE TABLE IF NOT EXISTS conv_state(key TEXT PRIMARY KEY, ts REAL, epoch INTEGER);"
                "CREATE TABLE IF NOT EXISTS tool_seen(conv TEXT, name TEXT, ts REAL, PRIMARY KEY(conv, name));"
                "CREATE TABLE IF NOT EXISTS tool_used(conv TEXT, name TEXT, ts REAL, PRIMARY KEY(conv, name));"
            )
            cutoff = time.time() - max_age
            self._db.execute("DELETE FROM memo WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM convs WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM conv_state WHERE ts < ?", (cutoff,))
            stats_cutoff = time.time() - 60 * 86400   # tool usage is learned over weeks, not days
            self._db.execute("DELETE FROM tool_seen WHERE ts < ?", (stats_cutoff,))
            self._db.execute("DELETE FROM tool_used WHERE ts < ?", (stats_cutoff,))
            self._db.commit()
        self._noted = set()

    def get(self, key):
        with self._lock:
            row = self._db.execute("SELECT text, before, after, ts FROM memo WHERE key=?", (key,)).fetchone()
            now = time.time()
            if row and now - row[3] > 86400:   # sliding expiry: a block still replayed must not age out mid-session
                self._db.execute("UPDATE memo SET ts=? WHERE key=?", (now, key))
                self._db.commit()
        return Entry(*row[:3]) if row else None

    def put(self, key, entry):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO memo VALUES (?,?,?,?,?)",
                (key, entry.text, entry.before, entry.after, time.time()),
            )
            self._db.commit()

    def known_conv(self, key) -> bool:
        """True if this conversation went through the proxy before; registers it otherwise."""
        with self._lock:
            seen = self._db.execute("SELECT 1 FROM convs WHERE key=?", (key,)).fetchone() is not None
            if not seen:
                self._db.execute("INSERT INTO convs VALUES (?,?)", (key, time.time()))
                self._db.commit()
        return seen

    def touch(self, key, now, ttl, detect=True):
        """Record a request of conversation `key` at time `now`. Returns (cold, epoch): `cold` is True when
        the previous request is older than the cache TTL (so the cache is gone); `epoch` counts how many
        cold restarts the conversation has been through, this one included."""
        with self._lock:
            row = self._db.execute("SELECT ts, epoch FROM conv_state WHERE key=?", (key,)).fetchone()
            was_cold, epoch = False, 0
            if row:
                epoch = row[1]
                was_cold = detect and cold.is_cold(now - row[0], ttl)
                epoch += was_cold
            self._db.execute("INSERT OR REPLACE INTO conv_state VALUES (?,?,?)", (key, now, epoch))
            self._db.commit()
        return was_cold, epoch

    def epoch(self, key) -> int:
        with self._lock:
            row = self._db.execute("SELECT epoch FROM conv_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else 0

    def note_tools(self, conv, seen, used):
        """Learn which tools exist in conversations (`seen`, from their first request) and which get called."""
        now = time.time()
        with self._lock:
            for table, names in (("tool_seen", seen), ("tool_used", used)):
                for name in names:
                    if (table, conv, name) not in self._noted:
                        self._noted.add((table, conv, name))
                        self._db.execute(f"INSERT OR IGNORE INTO {table} VALUES (?,?,?)", (conv, name, now))
            self._db.commit()

    def tool_unused(self, name, min_convs) -> bool:
        """True once a tool has been offered in at least `min_convs` conversations and never called."""
        with self._lock:
            seen = self._db.execute("SELECT COUNT(*) FROM tool_seen WHERE name=?", (name,)).fetchone()[0]
            used = self._db.execute("SELECT COUNT(*) FROM tool_used WHERE name=?", (name,)).fetchone()[0]
        return seen >= min_convs and used == 0


def _strip_cc(obj):
    """A copy of `obj` without cache_control markers, which move between requests."""
    if isinstance(obj, dict):
        return {k: _strip_cc(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cc(v) for v in obj]
    return obj


def _conv_key(messages) -> str:
    """Identity of a conversation's start: its first message, ignoring the cache_control marker, which
    sits on the newest messages and so moves off the first one after the first request."""
    return _hash("conv", json.dumps(_strip_cc(messages[0]), sort_keys=True))


def _hash(*parts) -> str:
    h = hashlib.sha256(MEMO_VERSION.encode())
    for part in parts:
        h.update(b"\x00")
        h.update(part if isinstance(part, bytes) else str(part).encode("utf-8", "surrogatepass"))
    return h.hexdigest()


def _cache_breakpoints(messages):
    """Indices of messages that carry a cache_control marker."""
    found = []
    for i, msg in enumerate(messages):
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list) and any(isinstance(b, dict) and "cache_control" in b for b in content):
            found.append(i)
    return found


def _pdf_tokens(data: bytes) -> int:
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data)) or max(1, len(data) // 60000)
    return pages * PDF_PAGE_TOKENS


class Optimizer:
    def __init__(self, compressor=None, memo=None, convert=True, min_tokens=120, lossy_min_tokens=300,
                 clean_prompts=True, prune_prompts=True, prompt_prune_tokens=400, paragraph_tokens=150,
                 targeted_reads=True, read_min_tokens=1500, elide_args=True, elide_min_tokens=400,
                 bash_min_tokens=250, prefix_rewrite=True, slim_after=5, narration=True,
                 prune_narration=False, narration_min_tokens=150, cold_cache=True, cache_ttl=None,
                 cold_keep=10, cold_min_tokens=1000, cold_read_min_tokens=300, cold_elide_min_tokens=150,
                 cold_bash_min_tokens=120, cold_through_thinking=False, dedupe_reads=True, dedupe_min_tokens=150, clock=time.time):
        self.compressor = compressor
        self.cold_cache = cold_cache            # compact history when a conversation returns after its cache expired
        self.cache_ttl = cache_ttl              # seconds; None = read it from the request's cache_control
        self.cold_keep = cold_keep              # newest messages a compaction never touches
        # Signed thinking blocks are bound to the prefix before them; editing that prefix can 400.
        self.cold_through_thinking = cold_through_thinking
        self.cold_min_tokens = cold_min_tokens  # smallest old tool output worth aging
        self.cold_read_min_tokens = cold_read_min_tokens
        self.cold_elide_min_tokens = cold_elide_min_tokens
        self.cold_bash_min_tokens = cold_bash_min_tokens
        self.dedupe_reads = dedupe_reads        # point repeated whole-file Reads at the identical earlier one
        self.dedupe_min_tokens = dedupe_min_tokens
        self.clock = clock
        self._conv, self._cold_now, self._epoch = "", False, 0   # per-request state, set under the lock
        self._reads = None
        self._poisoned = set()
        self.elide_args = elide_args
        self.elide_min_tokens = elide_min_tokens
        self.bash_min_tokens = bash_min_tokens
        self.prefix_rewrite = prefix_rewrite
        self.slim_after = slim_after
        self.narration = narration
        self.prune_narration = prune_narration
        self.narration_min_tokens = narration_min_tokens
        self.memo = memo or Memo()
        self.convert = convert
        self.clean_prompts = clean_prompts
        self.prune_prompts = prune_prompts
        self.targeted_reads = targeted_reads
        self.read_min_tokens = read_min_tokens
        self.prompt_prune_tokens = prompt_prune_tokens
        self.paragraph_tokens = paragraph_tokens
        self.min_tokens = min_tokens
        self.lossy_min_tokens = lossy_min_tokens
        self._lock = threading.RLock()

    # ------------------------------------------------------------ entry point

    def optimize(self, raw: bytes) -> Result:
        """Return the (possibly rewritten) request body. Never raises: on any
        problem the original bytes are forwarded untouched."""
        try:
            data = json.loads(raw)
            messages = data.get("messages") if isinstance(data, dict) else None
            if not isinstance(messages, list) or not messages:
                return Result(raw)
            with self._lock:
                result = self._run(raw, data, messages)
            result.model = data.get("model") or ""
            return result
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # third-party converters may raise non-Exception errors
            return Result(raw)

    def _identify(self, messages) -> str:
        """Conversation id: the first message, plus the second once one exists. Two sessions that open with the
        same text then diverge at their first reply and get separate memo state instead of sharing one."""
        base = _conv_key(messages)
        if len(messages) < 2:
            return base
        second = _hash("conv2", json.dumps(_strip_cc(messages[1]), sort_keys=True))
        key = _hash("conv-second", base)
        entry = self.memo.get(key)
        if entry is None:
            self.memo.put(key, Entry(second))
            return base
        return base if entry.text == second else _hash("conv-fork", base, second)

    def poison(self, conv):
        """Stop rewriting a conversation (the API rejected a rewritten request of it)."""
        self._poisoned.add(conv)

    def _run(self, raw, data, messages) -> Result:
        conv = self._identify(messages)
        if conv in self._poisoned:
            return Result(raw, conv=conv)
        known = self.memo.known_conv(conv)
        # Cold cache: the conversation's previous request is older than the cache TTL, so this request
        # re-writes the whole prefix at the write price whatever we do. History can be compacted now.
        ttl = self.cache_ttl or cold.request_ttl(data)
        cold_now, epoch = self.memo.touch(conv, self.clock(), ttl, detect=self.cold_cache and len(messages) > 1)
        self._conv, self._cold_now, self._epoch = conv, cold_now, epoch
        reads = readopt.ReadIndex.build(messages)   # before anything is rewritten: the originals
        self._reads = reads
        # Joining: the proxy first meets this conversation partway through (new to it, or seen
        # only by an older version that never recorded a mode for it).
        joining = len(messages) > 1 and (not known or self.memo.get(_hash("prefix-mode", conv)) is None)
        freeze_upto = -1
        if joining:
            # Everything up to the previous request's last cache breakpoint may already be
            # cached upstream, so leave it alone; only the newest turn is still free to change.
            marks = _cache_breakpoints(messages)
            if marks:
                freeze_upto = (marks[-2] if len(marks) > 1 else marks[-1]) - 1

        events = []
        tools, errors, called = {}, set(), set()
        for msg in messages:
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
                continue
            for blk in msg["content"]:
                if not isinstance(blk, dict):
                    continue
                if msg.get("role") == "assistant" and blk.get("type") == "tool_use":
                    tools[blk.get("id")] = (blk.get("name"), blk.get("input") or {})
                    called.add(blk.get("name"))
                elif blk.get("type") == "tool_result" and blk.get("is_error"):
                    errors.add(blk.get("tool_use_id"))

        first_request = len(messages) == 1
        mode = self._prefix_mode(conv, data)   # "note" | "plain" | "tail:<i>" | "off"
        if mode in ("note", "plain"):
            self._prefix(data, conv, mode, first_request, events)
        elif mode.startswith("tail:"):
            self._tail_note(messages, int(mode[5:]), joining, events)
        self.memo.note_tools(
            conv,
            [t["name"] for t in data.get("tools") or [] if prefix.is_custom_tool(t)] if first_request else [],
            [n for n in called if isinstance(n, str)])
        elide_on = self.elide_args and (mode == "note" or mode.startswith("tail:"))
        # A conversation compacted at a cold restart under a mode with no abridgement note gets one in
        # its newest user message at that time; later requests replay it from the memo.
        note_entry = self.memo.get(_hash("cold-note", conv))
        if note_entry is not None and note_entry.text is not None:
            self._tail_note(messages, int(note_entry.text), False, events)

        pending_path = None  # file named by the latest "Called the Read tool" attachment
        prompts, notes = [], []  # what the task is about: typed prompts, assistant text and tool calls
        for mi, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                notes += self._assistant_notes(msg.get("content"))  # from the original, before any abridging
                if isinstance(msg.get("content"), list):
                    self._assistant(msg, mi <= freeze_upto, events, errors, elide_on)
                continue
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            prompts += self._typed_text(msg.get("content"))
            frozen = mi <= freeze_upto
            content = msg.get("content")
            if isinstance(content, str):
                called = self._called_path(content)
                pending_path = called or pending_path
                if not called:
                    new = self._prompt(content, frozen, events)
                    if new is not None:
                        msg["content"] = new
                continue
            if not isinstance(content, list):
                continue
            if self.convert:
                self._documents(content, frozen, events, "PDF")
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                kind = blk.get("type")
                if kind == "tool_result":
                    self._tool_result(blk, tools, frozen, events, [(t, 2.0) for t in prompts[-2:]] + [(t, 1.0) for t in notes[-6:]])
                elif kind == "text":
                    text = blk.get("text") or ""
                    called = self._called_path(text)
                    if called:
                        pending_path = called
                    elif text.startswith(_RESULT_HEAD) and pending_path:
                        self._attachment(blk, text, pending_path, frozen, events)
                        pending_path = None
                    elif not text.lstrip().startswith("<system-reminder>"):
                        new = self._prompt(text, frozen, events)
                        if new is not None:
                            blk["text"] = new

        if self.cold_cache and epoch:
            before = len(events)
            self._cold_pass(messages, tools, reads, events)
            if cold_now and len(events) > before and note_entry is None and not mode.startswith("tail:") \
                    and mode != "note" and messages[-1].get("role") == "user":
                idx = len(messages) - 1
                self.memo.put(_hash("cold-note", conv), Entry(str(idx)))
                self._tail_note(messages, idx, True, events)

        if not events:
            return Result(raw, conv=conv, cold=cold_now)
        body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return Result(body, events, conv, orig_pieces=economics.pieces(json.loads(raw)),
                      new_pieces=economics.pieces(data), cold=cold_now)

    # ---------------------------------------------------------------- memoize

    def _decide(self, key, frozen, events, label, produce, converted=False, cold_produce=None):
        """Memoized decision for one block; returns replacement text or None.

        A block left alone while the cache was warm (frozen history, or too small to bother with) is
        reconsidered once per cold restart, when changing it is free; that second decision lives under
        its own key and is replayed like any other."""
        entry = self.memo.get(key)
        fresh = False
        if entry is None:
            entry = Entry(None) if frozen else (produce() or Entry(None))
            self.memo.put(key, entry)
            fresh = not frozen
        elif entry.text is None and self._epoch:
            ckey = _hash("cold1", self._conv, key)
            again = self.memo.get(ckey)
            if self._cold_now and (again is None or again.text is None):
                new = (cold_produce or produce)() or Entry(None)
                if again is None or new.text is not None:
                    self.memo.put(ckey, new)
                    again, fresh = new, new.text is not None
            if again is not None:
                entry = again
        if entry.text is not None:
            events.append(Event(label, entry.before, entry.after, fresh, converted))
        return entry.text

    # ------------------------------------------------- prefix: system + tools

    def _prefix_mode(self, conv, data) -> str:
        """Decided once, on a conversation's very first request, then replayed:
        "note"  -> prefix cleaned, system prompt carries the abridgement note, tool-call args may be abridged
        "plain" -> prefix cleaned only
        "tail:i"-> joined midway: the prefix is likely cached upstream, so it is left alone for good, but
                   new tool-call args are still abridged (they sit in the uncached tail) and the
                   note rides in user message i, the newest one when the proxy joined
        "off"   -> nothing to do"""
        key = _hash("prefix-mode", conv)
        entry = self.memo.get(key)
        if entry is None:
            mode = "off"
            messages = data.get("messages") or []
            names = {t.get("name") for t in data.get("tools") or [] if isinstance(t, dict)}
            if len(messages) == 1:
                if self.elide_args and data.get("system") and names & elide.EDIT_TOOLS:
                    mode = "note"
                elif self.prefix_rewrite and (data.get("system") or names):
                    mode = "plain"
            elif self.elide_args and names & elide.EDIT_TOOLS and isinstance(messages[-1], dict) \
                    and messages[-1].get("role") == "user":
                mode = f"tail:{len(messages) - 1}"
            entry = Entry(mode)
            self.memo.put(key, entry)
        return entry.text

    @staticmethod
    def _tail_note(messages, idx, fresh, events):
        """Put the abridgement note in user message `idx` (the same one on every request)."""
        msg = messages[idx] if 0 <= idx < len(messages) else None
        if not isinstance(msg, dict) or msg.get("role") != "user":
            return
        content = msg.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            return
        block = {"type": "text", "text": elide.TAIL_NOTE}
        if content and isinstance(content[-1], dict) and "cache_control" in content[-1]:
            block["cache_control"] = content[-1].pop("cache_control")  # the breakpoint stays on the last block
        content.append(block)
        msg["content"] = content
        events.append(Event("note", 0, count_tokens(elide.TAIL_NOTE), fresh))

    def _prefix(self, data, conv, mode, first_request, events):
        if self.prefix_rewrite:
            system = data.get("system")
            if isinstance(system, str):
                new = self._decide(_hash("sys", system), False, events, "system", lambda: self._shrink_system(system))
                if new is not None:
                    data["system"] = new
            elif isinstance(system, list):
                for blk in system:
                    if isinstance(blk, dict) and blk.get("type") == "text" and isinstance(blk.get("text"), str):
                        text = blk["text"]
                        new = self._decide(_hash("sys", text), False, events, "system",
                                           lambda text=text: self._shrink_system(text))
                        if new is not None:
                            blk["text"] = new
            self._tool_defs(data, conv, first_request, events)
        if mode == "note" and data.get("system"):
            data["system"] = elide.annotate_system(data["system"])
            events.append(Event("note", 0, count_tokens(elide.SYSTEM_NOTE), first_request))

    @staticmethod
    def _shrink_system(text) -> Optional[Entry]:
        before = count_tokens(text)
        out = clean_prompt(text)
        after = count_tokens(out)
        return Entry(out, before, after) if before - after >= 8 else None

    def _tool_defs(self, data, conv, first_request, events):
        tools = data.get("tools")
        if not isinstance(tools, list):
            return
        dumps = lambda obj: json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        for i, tool in enumerate(tools):
            if not prefix.is_custom_tool(tool):
                continue
            name = tool["name"]
            # Slim only tools offered from the conversation's first request: one that shows up
            # later was loaded on purpose and the model is about to use it.
            skey = _hash("slim", conv, name)
            decision = self.memo.get(skey)
            if decision is None:
                slim = first_request and prefix.slimmable(name) and self.memo.tool_unused(name, self.slim_after)
                decision = Entry("1" if slim else "0")
                self.memo.put(skey, decision)
            slim = decision.text == "1"

            def produce(tool=tool, slim=slim):
                new = (prefix.slim_tool if slim else prefix.clean_tool)(tool)
                before, after = count_tokens(dumps(tool)), count_tokens(dumps(new))
                return Entry(dumps(new), before, after) if after < before else None

            key = _hash("tooldef", "slim" if slim else "clean", json.dumps(tool, sort_keys=True, ensure_ascii=False))
            new = self._decide(key, False, events, "tools·slim" if slim else "tools", produce)
            if new is not None:
                tools[i] = json.loads(new)

    # ------------------------------------------------- assistant turns

    def _assistant(self, msg, frozen, events, errors, elide_on):
        """Abridge executed tool-call arguments and tidy long narration, once, on first sight."""
        for blk in msg["content"]:
            if not isinstance(blk, dict):
                continue
            kind = blk.get("type")
            if kind == "tool_use" and elide_on and blk.get("name") in elide.EDIT_TOOLS \
                    and isinstance(blk.get("id"), str) and blk["id"] not in errors \
                    and isinstance(blk.get("input"), dict):
                args, name = blk["input"], blk["name"]

                def produce(args=args, name=name):
                    r = elide.elide_input(name, args, self.elide_min_tokens, self.bash_min_tokens)
                    return Entry(elide.json_of(r[0]), r[1], r[2]) if r else None

                def produce_cold(args=args, name=name):   # a cold restart: smaller arguments are worth abridging too
                    r = elide.elide_input(name, args, self.cold_elide_min_tokens, self.cold_bash_min_tokens)
                    return Entry(elide.json_of(r[0]), r[1], r[2]) if r else None

                key = _hash("elide", name, blk["id"], json.dumps(args, sort_keys=True, ensure_ascii=False))
                new = self._decide(key, frozen, events, f"args:{name}", produce, cold_produce=produce_cold)
                if new is not None:
                    blk["input"] = json.loads(new)
            elif kind == "text" and self.narration and isinstance(blk.get("text"), str) \
                    and len(blk["text"]) >= 4 * self.narration_min_tokens:
                text = blk["text"]
                new = self._decide(_hash("narr", text), frozen, events, "narration",
                                   lambda text=text: self._clean_narration(text))
                if new is not None:
                    blk["text"] = new

    def _clean_narration(self, text) -> Optional[Entry]:
        before = count_tokens(text)
        if before < self.narration_min_tokens:
            return None
        out = compact_markdown(clean_prompt(text))
        if self.prune_narration and self.compressor and count_tokens(out) >= self.prompt_prune_tokens:
            out = self._prune_pasted(out)
        after = count_tokens(out)
        if before - after < max(MIN_GAIN_TOKENS // 2, before * 0.02):
            return None
        return Entry(out, before, after)

    # -------------------------------------------------------------- producers

    def _shrink_text(self, text, fold=False) -> Optional[Entry]:
        """Tool output -> cleaned (and, if big and prose-like, pruned) text."""
        before = count_tokens(text)
        if before < self.min_tokens:
            return None
        out = clean(text, fold=fold)
        if self.compressor and count_tokens(out) >= self.lossy_min_tokens:
            pruned = self.compressor.squeeze(out)
            if pruned:
                out = f"{LOSSY_NOTE}\n{pruned}"
        after = count_tokens(out)
        if before - after < max(MIN_GAIN_TOKENS, before * 0.03):
            return None
        return Entry(out, before, after)

    @staticmethod
    def _assistant_notes(content):
        if isinstance(content, str):
            return [content[:2000]]
        notes = []
        for blk in content if isinstance(content, list) else []:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "text":
                notes.append((blk.get("text") or "")[:2000])
            elif blk.get("type") == "tool_use" and blk.get("name") != "Read":
                notes.append(json.dumps(blk.get("input") or {}, ensure_ascii=False)[:1000])
        return notes

    @staticmethod
    def _typed_text(content):
        """Text the user typed (not tool results or system reminders)."""
        items = [content] if isinstance(content, str) else [
            b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"
        ] if isinstance(content, list) else []
        return [t[:4000] for t in items if t.strip() and not t.lstrip().startswith("<system-reminder>")]

    def _prompt(self, text, frozen, events):
        """The user's own text: lossless cleanup, and only when it saves something real."""
        if not self.clean_prompts or len(text) < 4 * self.min_tokens:
            return None  # too short to be worth a token count
        return self._decide(_hash("prompt2", text), frozen, events, "prompt", lambda: self._clean_prompt(text))

    def _clean_prompt(self, text) -> Optional[Entry]:
        before = count_tokens(text)
        if before < self.min_tokens:
            return None
        out = clean_prompt(text)
        if self.prune_prompts and self.compressor and count_tokens(out) >= self.prompt_prune_tokens:
            out = self._prune_pasted(out)
        after = count_tokens(out)
        if before - after < max(MIN_GAIN_TOKENS, before * 0.03):
            return None
        return Entry(out, before, after)

    def _prune_pasted(self, text) -> str:
        """Prune only big pasted prose; everything the user is likely to have typed stays verbatim."""
        paragraphs, current, fenced = [], [], False  # paragraph = (lines, protected)
        for line in text.split("\n"):
            marker = line.lstrip().startswith(("```", "~~~"))
            if marker:
                fenced = not fenced
            if not line and not fenced and not marker:
                if current:
                    paragraphs.append(current)
                    current = []
                paragraphs.append(None)  # blank separator
            else:
                current.append(line)
        if current:
            paragraphs.append(current)

        real = [i for i, p in enumerate(paragraphs) if p]
        edges = {real[0], real[-1]} if real else set()  # opening and closing paragraphs often hold the ask
        for i in real:
            if i in edges:
                continue
            body = "\n".join(paragraphs[i])
            if "```" in body or "~~~" in body or count_tokens(body) < self.paragraph_tokens:
                continue
            if prose_share(body) < 0.6:
                continue
            pruned = self.compressor.squeeze(body)
            if pruned and count_tokens(pruned) < count_tokens(body):
                paragraphs[i] = [PASTE_NOTE, *pruned.split("\n")]
        return "\n".join("" if p is None else "\n".join(p) for p in paragraphs)

    def _shrink_doc(self, markdown, name, before, require_gain) -> Optional[Entry]:
        """Converted document -> Markdown block (cleaned, pruned when it is prose)."""
        if not markdown:
            return None
        body = clean(markdown)
        pruned = False
        if self.compressor and count_tokens(body) >= self.lossy_min_tokens:
            squeezed = self.compressor.squeeze(body)
            if squeezed:
                body, pruned = squeezed, True
        note = "; filler words removed" if pruned else ""
        text = f"[{name} converted to Markdown by proxy{note}]\n\n{body}"
        after = count_tokens(text)
        if require_gain and after >= before * 0.9:
            return None
        return Entry(text, before, after)

    # ---------------------------------------------------------------- handlers

    def _tool_result(self, blk, tools, frozen, events, query=()):
        name, args = tools.get(blk.get("tool_use_id"), (None, {}))
        if not name:
            return
        content = blk.get("content")
        if name == "Read":
            path = args.get("file_path")
            if self._repeat_read(blk, frozen, events):
                return
            if self.convert and isinstance(path, str) and fileconv.is_convertible(path):
                key = _hash("read", path, json.dumps(content, sort_keys=True))
                ext = os.path.splitext(path)[1].lower()
                label = f"{ext[1:]}→md"
                before = self._original_tokens(content)

                def produce():
                    return self._shrink_doc(fileconv.convert_path(path), os.path.basename(path),
                                            before, require_gain=(ext == ".pdf"))

                new = self._decide(key, frozen, events, label, produce, converted=True)
                if new is not None:
                    blk["content"] = new
                    blk.pop("is_error", None)
                    return
            if self.targeted_reads and isinstance(path, str) and isinstance(content, str) \
                    and not blk.get("is_error") and args.get("offset") is None and args.get("limit") is None:
                def produce_excerpt():
                    text = targeted.targeted(content, path, query, min_tokens=self.read_min_tokens)
                    return Entry(text, count_tokens(content), count_tokens(text)) if text else None

                new = self._decide(_hash("tread", blk.get("tool_use_id"), content), frozen, events,
                                   "Read→excerpt", produce_excerpt)
                if new is not None:
                    blk["content"] = new
                return
            if self.convert and isinstance(content, list):
                self._documents(content, frozen, events, "PDF")
            return
        if blk.get("is_error") or not (name in LOSSY_TOOLS or name.startswith("mcp__")):
            return
        if isinstance(content, str):
            new = self._decide(_hash("tool", name, content), frozen, events, name,
                               lambda: self._shrink_text(content, name in FOLDED_TOOLS))
            if new is not None:
                blk["content"] = new
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    text = item["text"]
                    new = self._decide(_hash("tool", name, text), frozen, events, name,
                                       lambda text=text: self._shrink_text(text, name in FOLDED_TOOLS))
                    if new is not None:
                        item["text"] = new

    def _attachment(self, blk, text, path, frozen, events):
        """An @-mentioned file that Claude Code inlined as a 'Result of calling the Read tool' block."""
        if self._repeat_read(blk, frozen, events):
            return
        if not (self.convert and fileconv.is_convertible(path)):
            return
        key = _hash("attach", path, text)
        before = count_tokens(text)
        ext = os.path.splitext(path)[1].lower()

        def produce():
            entry = self._shrink_doc(fileconv.convert_path(path), os.path.basename(path),
                                     before, require_gain=(ext == ".pdf"))
            if entry:
                entry.text = f"{_RESULT_HEAD}{entry.text}{_RESULT_TAIL}\n"
            return entry

        new = self._decide(key, frozen, events, f"@{ext[1:]}→md", produce, converted=True)
        if new is not None:
            blk["text"] = new

    def _documents(self, content, frozen, events, label):
        """Replace base64 PDF document blocks in a content list by extracted Markdown."""
        for i, item in enumerate(content):
            if not (isinstance(item, dict) and item.get("type") == "document"):
                continue
            src = item.get("source") or {}
            if (src.get("type") != "base64" or src.get("media_type") != "application/pdf"
                    or (item.get("citations") or {}).get("enabled")):
                continue
            b64 = src.get("data")
            if not isinstance(b64, str):
                continue
            name = item.get("title") or "attached PDF"

            def produce(b64=b64, name=name):
                data = base64.b64decode(b64)
                return self._shrink_doc(fileconv.convert_bytes(data, ".pdf"), name,
                                        _pdf_tokens(data), require_gain=True)

            new = self._decide(_hash("pdf", b64), frozen, events, f"{label}→md", produce, converted=True)
            if new is not None:
                replacement = {"type": "text", "text": new}
                if "cache_control" in item:
                    replacement["cache_control"] = item["cache_control"]
                content[i] = replacement

    # ------------------------------------------------- read-level optimization

    def _repeat_read(self, blk, frozen, events) -> bool:
        """A whole-file Read that repeats an earlier one byte for byte (and that earlier copy is still in
        the history as the model saw it) becomes a short pointer. Decided once, on first sight."""
        ref = self._reads.ref_for(blk) if self.dedupe_reads and self._reads is not None else None
        if ref is None or not ref.intact:
            return False

        def produce():
            if ref.tokens < self.dedupe_min_tokens or self._reads.anchor(ref) is None:
                return None
            text = readopt.pointer(ref.path, ref.orig.count("\n") + 1, ref.tokens)
            if ref.attach:
                text = readopt.wrap_attachment(text)
            after = count_tokens(text)
            return Entry(text, ref.tokens, after) if after <= ref.tokens * 0.5 else None

        new = self._decide(_hash("rdup", self._conv, ref.ident, ref.orig), frozen, events, "read·repeat", produce)
        if new is None:
            return False
        blk[ref.field] = new
        if not ref.attach:
            blk.pop("is_error", None)
        return True

    # ------------------------------------------------- cold-cache compaction

    def _cold_pass(self, messages, tools, reads, events):
        """Second stage, after every block has had its normal first-sight treatment. On a cold request
        it decides (and memoizes) which old blocks to reduce; on every later request it replays those
        decisions. Reads whose file has since changed become outlines; big old command/web/MCP output
        keeps its head, tail and error lines. The newest `cold_keep` messages are never touched."""
        protect = max(0, len(messages) - self.cold_keep)
        floor = -1 if self.cold_through_thinking else self._last_thinking(messages)
        for mi, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                continue
            # New compaction never reaches back past the newest signed thinking block; replays are unaffected.
            create = self._cold_now and mi < protect and mi > floor
            for blk in msg["content"]:
                if not isinstance(blk, dict):
                    continue
                ref = reads.ref_for(blk)
                if ref is not None:
                    self._cold_stale(ref, protect, create, events)
                elif blk.get("type") == "tool_result" and not blk.get("is_error"):
                    name = tools.get(blk.get("tool_use_id"), (None, None))[0]
                    if cold.agable(name):
                        self._cold_age(blk, name, create, events)

    @staticmethod
    def _last_thinking(messages) -> int:
        """Index of the newest assistant message holding a (signed or redacted) thinking block, else -1."""
        last = -1
        for i, m in enumerate(messages):
            c = m.get("content") if isinstance(m, dict) and m.get("role") == "assistant" else None
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking")
                                           for b in c):
                last = i
        return last

    def _cold_decide(self, key, create, events, label, produce):
        entry = self.memo.get(key)
        fresh = False
        if entry is None:
            if not create:
                return None      # a warm request never starts a compaction, it only replays one
            entry = produce() or Entry(None)
            self.memo.put(key, entry)
            fresh = True
        if entry.text is None:
            return None
        events.append(Event(label, entry.before, entry.after, fresh))
        return entry.text

    def _cold_stale(self, ref, protect, create, events):
        live = ref.live
        repeat = readopt.is_pointer(live)
        if not isinstance(live, str) or not (ref.intact or repeat):
            return          # already narrowed or converted: leave it

        def produce():
            reason = self._reads.stale_reason(ref, protect)
            if not reason:
                return None
            if repeat:      # its anchor goes with it, so a pointer to that anchor must not stay behind
                stub = (f"[proxy: repeated Read result of {ref.path} omitted from this history because {reason}. "
                        "Read the file again for its current text]")
                if ref.attach:
                    stub = readopt.wrap_attachment(stub)
                return Entry(stub, count_tokens(live), count_tokens(stub))
            if ref.tokens < self.cold_read_min_tokens:
                return None
            stub = readopt.stale_stub(ref.path, ref.orig, reason)
            before, after = count_tokens(live), count_tokens(stub)
            return Entry(stub, before, after) if after <= before * 0.5 else None

        new = self._cold_decide(_hash("cold2", self._conv, ref.ident, live), create, events, "cold:read", produce)
        if new is not None:
            ref.blk[ref.field] = new

    def _cold_age(self, blk, name, create, events):
        content, tid = blk.get("content"), blk.get("tool_use_id")
        if isinstance(content, str):
            items = [(None, content)]
        elif isinstance(content, list):
            items = [(i, it["text"]) for i, it in enumerate(content)
                     if isinstance(it, dict) and it.get("type") == "text" and isinstance(it.get("text"), str)]
        else:
            return
        for i, text in items:
            def produce(text=text):
                before = count_tokens(text)
                if before < self.cold_min_tokens:
                    return None
                stub = cold.aged_stub(name, text)
                after = count_tokens(stub)
                return Entry(stub, before, after) if after <= before * 0.5 else None

            new = self._cold_decide(_hash("cold2", self._conv, tid, i, text), create, events, f"cold:{name}", produce)
            if new is not None:
                if i is None:
                    blk["content"] = new
                else:
                    content[i]["text"] = new

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _called_path(text):
        return readopt.called_path(text)

    @staticmethod
    def _original_tokens(content) -> int:
        """Rough token cost of what Claude Code originally put in a tool result."""
        if isinstance(content, str):
            return count_tokens(content)
        total = 0
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                total += count_tokens(item.get("text") or "")
            elif item.get("type") == "document":
                src = item.get("source") or {}
                if src.get("media_type") == "application/pdf" and isinstance(src.get("data"), str):
                    try:
                        total += _pdf_tokens(base64.b64decode(src["data"]))
                    except ValueError:
                        pass
            elif item.get("type") == "image":
                total += 1500
        return total
