"""Live-API checks for the two claims the unit tests cannot settle. Costs a few cents; needs ANTHROPIC_API_KEY.

    python3 verify_live.py                     # both checks
    python3 verify_live.py thinking            # only: does editing history before a signed thinking block 400?
    python3 verify_live.py cache               # only: does a rewritten (shorter) older block still hit the cache?
    --model ID  --thinking '{"type":"enabled","budget_tokens":2048}'   # thinking parameter shape for the model

It talks to the API directly (not through the proxy) and sends only the synthetic prompts below.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://api.anthropic.com/v1/messages"
FILLER = "\n".join(f"line {i}: the quick brown fox {i * 7919 % 1000} jumps over the lazy dog {i * 104729 % 1000}"
                   for i in range(400))          # ~8k tokens, enough to be cacheable
SYSTEM = [{"type": "text", "text": "You are terse. " + FILLER[:4000], "cache_control": {"type": "ephemeral"}}]
TOOLS = [{"name": "lookup", "description": "Look up a value by key.",
          "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}]


def call(body, key):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {
        "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def base(model, messages, thinking=None):
    body = {"model": model, "max_tokens": 3000, "system": SYSTEM, "tools": TOOLS, "messages": messages}
    if thinking:
        body["thinking"] = thinking
    return body


def mark(messages):
    """cache_control on the last block of the last message, like Claude Code."""
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return messages


def thinking_check(model, thinking, key):
    print("== thinking: edit history before a signed thinking block ==")
    user = {"role": "user", "content": f"Reference notes:\n{FILLER}\n\nCall the lookup tool with key 'alpha', then say done."}
    status, r1 = call(base(model, mark([dict(user)]), thinking), key)
    if status != 200:
        return print(f"setup request failed: {status} {r1}")
    assistant = {"role": "assistant", "content": r1["content"]}
    if not any(b["type"] == "tool_use" for b in r1["content"]):
        return print("inconclusive: the model did not call the tool")
    print("thinking blocks in reply:", sum(b["type"] in ("thinking", "redacted_thinking") for b in r1["content"]))
    tool_use = next(b for b in r1["content"] if b["type"] == "tool_use")
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "42"}]}

    def second(first_user):
        return call(base(model, mark([first_user, assistant, dict(result)]), thinking), key)

    status, _ = second(user)
    print("unmodified history      ->", status)
    edited = {"role": "user", "content": user["content"].replace(FILLER, FILLER[:len(FILLER) // 2] + "\n[omitted]")}
    status, body = second(edited)
    print("history edited before the thinking block ->", status, body.get("error", {}).get("message", "")[:120])
    print("=> 400 means cold compaction must stay frozen before thinking blocks; 200 means it is safe on this model.")


def cache_check(model, key):
    print("== cache: does a rewritten older block keep the cache warm on later requests? ==")
    tool_block = {"type": "text", "text": "Output:\n" + FILLER}
    shrunk = {"type": "text", "text": "Output:\n" + FILLER[:1500] + "\n[proxy: omitted]"}

    def req(first_block, n):
        msgs = [{"role": "user", "content": [first_block, {"type": "text", "text": "Reply with one word."}]}]
        for i in range(n):
            msgs += [{"role": "assistant", "content": [{"type": "text", "text": f"ok{i}"}]},
                     {"role": "user", "content": f"again {i}"}]
        return call(base(model, mark(msgs)), key)

    def usage(label, resp):
        status, body = resp
        u = body.get("usage", {})
        print(f"{label:34s} {status}  read={u.get('cache_read_input_tokens')} write={u.get('cache_creation_input_tokens')}")
        return u

    usage("1. rewritten block, first send", req(shrunk, 0))
    u2 = usage("2. same bytes replayed, +1 turn", req(shrunk, 1))
    usage("3. ORIGINAL block (control: should miss)", req(tool_block, 1))
    ok = (u2.get("cache_read_input_tokens") or 0) > 0
    print("=>", "replaying memoized rewritten bytes hits the cache" if ok
          else "NO cache read on replay: memoized rewrites are not preserving the cache")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", choices=["thinking", "cache"])
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--thinking", default='{"type":"enabled","budget_tokens":2048}')
    a = ap.parse_args()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("set ANTHROPIC_API_KEY")
    if a.which in (None, "thinking"):
        thinking_check(a.model, json.loads(a.thinking), key)
    if a.which in (None, "cache"):
        cache_check(a.model, key)


if __name__ == "__main__":
    main()
