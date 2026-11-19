"""The fixed prefix of every request: tool definitions and system prompt.

Roughly a fifth of a session's context cost is the ~37k-token prefix Claude Code
resends on every call. It is rewritten once, when a conversation starts (nothing
is cached yet, so it is free), and the same bytes are replayed for the whole
conversation.

Tool definitions
  * all tools: schema annotations that carry no meaning for the model are
    dropped (`$schema`, `title`), descriptions get lossless whitespace cleanup;
  * MCP tools the proxy has watched go unused across several conversations are
    *slimmed*: first sentence of the description, no per-parameter prose. Names,
    types, enums and `required` stay, so the tool still works if the model picks
    it, and its next conversation gets the full definition back.
  Built-in tools are never slimmed, and a tool that first appears mid-conversation
  (for example one the model just loaded on purpose) is never slimmed.

System prompt
  Lossless whitespace cleanup only; Claude Code's own instructions are not pruned.
"""

from __future__ import annotations

import re
from typing import Any

from shrink import clean_prompt

_DROP = {"$schema", "title"}
_MAPS = {"properties", "patternProperties", "definitions", "$defs", "dependentSchemas"}
_SUBSCHEMAS = {"items", "additionalProperties", "not", "if", "then", "else", "contains",
               "propertyNames", "unevaluatedProperties", "additionalItems"}
_LISTS = {"anyOf", "oneOf", "allOf", "prefixItems"}
_SENTENCE = re.compile(r"(?s)\A(.{20,}?[.!?])(?:\s|\Z)")


def _walk(node: Any, slim: bool) -> Any:
    """Rewrite one JSON-schema node (not a data value) and everything below it."""
    if isinstance(node, list):
        return [_walk(n, slim) for n in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in _DROP or (slim and key in {"description", "examples"}):
            continue
        if key in _MAPS and isinstance(value, dict):
            out[key] = {name: _walk(sub, slim) for name, sub in value.items()}
        elif key in _SUBSCHEMAS and isinstance(value, (dict, list)):
            out[key] = _walk(value, slim)
        elif key in _LISTS and isinstance(value, list):
            out[key] = [_walk(sub, slim) for sub in value]
        else:
            out[key] = value
    return out


def first_sentence(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    m = _SENTENCE.match(text)
    sentence = m.group(1) if m else text
    return sentence if len(sentence) <= limit else sentence[: limit - 1] + "…"


def clean_tool(tool: dict) -> dict:
    """Lossless: same meaning, fewer bytes."""
    out = dict(tool)
    if isinstance(out.get("description"), str):
        out["description"] = clean_prompt(out["description"]).strip()
    if isinstance(out.get("input_schema"), dict):
        out["input_schema"] = _walk(out["input_schema"], slim=False)
    return out


def slim_tool(tool: dict) -> dict:
    """Lossy: name, parameter names/types/enums/required kept; prose dropped."""
    out = clean_tool(tool)
    if isinstance(out.get("description"), str):
        out["description"] = first_sentence(out["description"]) + " (abridged)"
    if isinstance(out.get("input_schema"), dict):
        out["input_schema"] = _walk(out["input_schema"], slim=True)
    return out


def is_custom_tool(tool: Any) -> bool:
    """User-defined tools carry an input_schema; server tools (web_search ...) do not."""
    return isinstance(tool, dict) and isinstance(tool.get("name"), str) and isinstance(tool.get("input_schema"), dict)


def slimmable(name: str) -> bool:
    return name.startswith("mcp__")
