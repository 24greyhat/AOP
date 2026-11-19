"""Plain log lines on stderr: one per request when it is rewritten, one when its cost is known."""

from __future__ import annotations

import sys
import time

_prev_ctx = {}   # conversation -> context size of its previous request


def usd(x: float) -> str:
    return f"${x:.4f}"


def human(n: float) -> str:
    n = abs(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{int(n):,}"


def say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def request_line(result) -> str:
    """What the proxy changed in one request, grouped by lever."""
    stamp = time.strftime("%H:%M:%S")
    if not result.changed:
        return f"{stamp} no change"
    by_lever = {}
    for e in result.events:
        if e.fresh:
            n, delta = by_lever.get(e.label, (0, 0))
            by_lever[e.label] = (n + 1, delta + e.before - e.after)
    parts = [f"{label}{f'×{n}' if n > 1 else ''} −{human(delta)}" for label, (n, delta) in by_lever.items()]
    head = f"saved {human(result.fresh_saved)} tok" if result.fresh_saved else "replayed"
    if getattr(result, "cold", False):
        head = f"cold cache, history compacted: {head}"
    return f"{stamp} {head}  {' · '.join(parts) or 'earlier blocks'}  (request total −{human(result.saved)})"


def money_line(row) -> str:
    """Cost of one finished request, and the estimated saving against the unrewritten one."""
    saved, orig = row["saved"], row["without_proxy"]
    verb = "would save" if row["dry_run"] else "est. saved"
    return (f"{time.strftime('%H:%M:%S')} {usd(row['billed'])} billed  {verb} {usd(saved)} "
            f"({100 * saved / orig if orig else 0.0:.1f}%)  "
            f"read {human(row['read'])} write {human(row['write'])} out {human(row['output'])}")


def cache_miss(usage, conv) -> str | None:
    """A warning when a request read back under half of what the conversation's previous request held."""
    if not usage:
        return None
    read = usage.get("cache_read_input_tokens") or 0
    ctx = read + (usage.get("cache_creation_input_tokens") or 0) + (usage.get("input_tokens") or 0)
    prev, _prev_ctx[conv] = _prev_ctx.get(conv, 0), ctx
    if prev >= 2000 and read < prev * 0.5:
        return f"cache read {human(read)} of {human(ctx)} tokens (previous request: {human(prev)})"
    return None
