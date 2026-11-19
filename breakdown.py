"""Where a request's tokens go: system, tools, your text, narration, tool-call args and results by tool.

Used by `--profile FILE` (one JSON line per request: before, after, saved per lever).
Counts are tokenizer estimates, good for comparing slices.
"""

from __future__ import annotations

import json
from collections import Counter

from shrink import count_tokens


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(item.get("text") or "")
            elif item.get("type") in ("image", "document"):
                parts.append("")
    return "\n".join(parts)


def composition(data: dict) -> dict:
    """Token counts per category for one /v1/messages request body (already parsed)."""
    c = Counter()
    system = data.get("system")
    if system:
        c["system"] = count_tokens(system if isinstance(system, str) else
                                   "".join(b.get("text", "") for b in system if isinstance(b, dict)))
    for tool in data.get("tools") or []:
        c["tools"] += count_tokens(json.dumps(tool, ensure_ascii=False, separators=(",", ":")))
    names = {}
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            c["assistant_text" if msg.get("role") == "assistant" else "user_text"] += count_tokens(content)
            continue
        for blk in content if isinstance(content, list) else []:
            if not isinstance(blk, dict):
                continue
            kind = blk.get("type")
            if kind == "text":
                n = count_tokens(blk.get("text") or "")
                if msg.get("role") == "assistant":
                    c["assistant_text"] += n
                elif (blk.get("text") or "").lstrip().startswith("<system-reminder>"):
                    c["reminders"] += n
                else:
                    c["user_text"] += n
            elif kind == "thinking":
                c["thinking"] += count_tokens(blk.get("thinking") or "")
            elif kind == "tool_use":
                names[blk.get("id")] = blk.get("name") or "?"
                c["args:" + _short(blk.get("name"))] += count_tokens(
                    json.dumps(blk.get("input") or {}, ensure_ascii=False, separators=(",", ":")))
            elif kind == "tool_result":
                c["result:" + _short(names.get(blk.get("tool_use_id")))] += count_tokens(_text_of(blk.get("content")))
    return dict(c)


def _short(name) -> str:
    name = name or "?"
    return "mcp" if name.startswith("mcp__") else name


def record(raw: bytes, result, conv: str = "") -> dict:
    """One profile row: composition before and after the optimizer, plus saved tokens by lever."""
    try:
        before = composition(json.loads(raw))
        after = composition(json.loads(result.body)) if result.changed else before
    except ValueError:
        return {}
    saved = Counter()
    for e in result.events:
        saved[e.label] += e.before - e.after
    return {"conv": conv[:8], "before": before, "after": after, "saved_by": dict(saved),
            "total_before": sum(before.values()), "total_after": sum(after.values())}


def append_jsonl(path: str, row: dict):
    if row:
        with open(path, "a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
