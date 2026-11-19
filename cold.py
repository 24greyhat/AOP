"""Cold-cache compaction: shrink history at the one moment it costs nothing.

Anthropic's prompt cache expires after its TTL (5 minutes, or 1 hour when Claude Code asks for it).
A session resumed after that gap re-writes its whole context at the cache-write price, which is
many times the read price. On measured sessions those restarts were ~18% of spend. And because the
whole prefix is being rewritten anyway, that request is the only time already-sent history can be
changed without breaking a warm cache.

So when a conversation comes back after its cache has expired, the optimizer runs a compaction pass
over its history (optimizer.py, `_cold_pass`): old blocks the model no longer needs verbatim are
reduced, and the result is memoized so every later request resends exactly the same bytes. This
module holds the decisions that do not depend on the request machinery: how to tell the cache is
cold, which tool results may be aged, and what the aged stub looks like.

Never aged: the newest messages, thinking blocks, error results, user text, assistant text, and results
of Edit/Grep/Glob/Agent/Task/WebSearch (small, or knowledge the model produced or searched for).
"""

from __future__ import annotations

import re

from shrink import count_tokens

TTL_5M = 300
TTL_1H = 3600
MARGIN = 60          # seconds of slack: a reply can take a while, and the TTL counts from request time

AGED_TOOLS = {"Bash", "BashOutput", "WebFetch"}   # plus every MCP tool
HEAD_LINES, TAIL_LINES, KEEP_IMPORTANT = 6, 4, 10
_IMPORTANT = re.compile(r"error|fail|traceback|exception|fatal|panic|denied|not found|warning:|https?://", re.I)


def request_ttl(data) -> int:
    """Cache lifetime this client asks for: 1 hour if any cache_control says so, else 5 minutes."""
    def walk(node):
        if isinstance(node, dict):
            cc = node.get("cache_control")
            if isinstance(cc, dict) and cc.get("ttl") == "1h":
                return True
            return any(walk(v) for v in node.values() if isinstance(v, (dict, list)))
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False
    return TTL_1H if walk(data) else TTL_5M


def is_cold(gap_seconds: float, ttl: int) -> bool:
    """True when the previous request of a conversation is too old for its cache to be alive."""
    return gap_seconds > ttl + MARGIN


def agable(tool_name) -> bool:
    return isinstance(tool_name, str) and (tool_name in AGED_TOOLS or tool_name.startswith("mcp__"))


def aged_stub(name: str, text: str) -> str:
    """Head, tail and any error/URL lines of an old tool output, with a marker for the rest."""
    lines = text.split("\n")
    n, tokens = len(lines), count_tokens(text)
    if n <= HEAD_LINES + TAIL_LINES + 2:      # few, very long lines: cut by characters
        cut = len(text) - 1000
        if cut <= 0:
            return text
        return (f"{text[:700]}\n[proxy: {cut} characters (~{tokens} tokens in all) of this {name} result omitted "
                f"from this history to save tokens; re-run it if exact output matters]\n{text[-300:]}")
    middle = lines[HEAD_LINES:n - TAIL_LINES]
    kept = [ln[:300] for ln in middle if _IMPORTANT.search(ln)][:KEEP_IMPORTANT]
    omitted = len(middle) - len(kept)
    marker = (f"[proxy: {omitted} lines (~{tokens} tokens in all) of this {name} result omitted from this "
              f"history to save tokens; error and URL lines kept; re-run it if exact output matters]")
    return "\n".join(lines[:HEAD_LINES] + kept + [marker] + lines[n - TAIL_LINES:])
