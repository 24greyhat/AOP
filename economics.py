"""Dollars: what a request cost, and what it would have cost with / without the proxy.

Cost of a request comes from the API's own usage report (cache read / cache write / fresh /
output tokens) priced per model. The *other* side of the comparison keeps the same cache
boundary: the tokens that were a cache hit stay a hit, and the proxy's rewrite only changes how
many tokens sit before and after that boundary. That is the cache-aware part: a token removed
from the cached prefix saves its cache-read price on this request, a token removed from the
uncached tail saves its (much dearer) cache-write price once and then keeps saving reads.

Per-piece sizes are estimates (chars / 3.6) scaled so their total matches the tokens the API
actually billed, so absolute numbers are the API's and only the *proportions* are estimated.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

CHARS_PER_TOKEN = 3.6
WRITE_5M, WRITE_1H = 1.25, 2.0          # cache-write multipliers on the input price
DEFAULT_READ = 0.1                       # cache-read multiplier when a model's read price is not listed

# (input, output, cache read) in $ per million tokens. Anthropic first-party list prices
# (claude-api skill, cached 2026-11-03). Cache reads: listed for the models that state one,
# else DEFAULT_READ x input. Haiku 5.5 is priced for prompts up to 100K tokens.
_PRICES = {
    "claude-fable-5-2": (8.0, 40.0, 0.20), "claude-mythos-5-2": (8.0, 40.0, 0.20),
    "claude-fable-5-1": (10.0, 50.0, None), "claude-mythos-5-1": (10.0, 50.0, None),
    "claude-fable-5": (10.0, 50.0, None),
    "claude-opus-5-5": (4.0, 20.0, 0.20), "claude-opus-5": (5.0, 25.0, None),
    "claude-opus-4-8": (5.0, 25.0, None), "claude-opus-4-7": (5.0, 25.0, None),
    "claude-opus-4-6": (5.0, 25.0, None),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10), "claude-sonnet-5": (2.0, 10.0, None),
    "claude-sonnet-4-6": (3.0, 15.0, None),
    "claude-haiku-5-5": (0.10, 0.50, None), "claude-haiku-4-5": (1.0, 5.0, None),
}


@dataclass(frozen=True)
class Price:
    model: str
    inp: float
    out: float
    read: float

    @property
    def write5(self):
        return self.inp * WRITE_5M

    @property
    def write1h(self):
        return self.inp * WRITE_1H


def price_for(model: Optional[str]) -> Optional[Price]:
    """Price of a model id (date suffixes tolerated); None for models without a listed price."""
    if not model:
        return None
    best = max((k for k in _PRICES if model.startswith(k)), key=len, default=None)
    if best is None:
        return None
    inp, out, read = _PRICES[best]
    return Price(best, inp, out, read if read is not None else inp * DEFAULT_READ)


@dataclass
class Cost:
    read: float = 0.0
    write: float = 0.0
    fresh: float = 0.0
    output: float = 0.0

    @property
    def total(self):
        return self.read + self.write + self.fresh + self.output

    @property
    def context(self):
        return self.read + self.write + self.fresh


def tokens_of(usage: dict):
    """(read, write_5m, write_1h, fresh, output) token counts from an API usage object."""
    usage = usage or {}
    cc = usage.get("cache_creation") or {}
    if cc:
        w5, w1 = cc.get("ephemeral_5m_input_tokens") or 0, cc.get("ephemeral_1h_input_tokens") or 0
    else:
        w5, w1 = usage.get("cache_creation_input_tokens") or 0, 0
    return (usage.get("cache_read_input_tokens") or 0, w5, w1,
            usage.get("input_tokens") or 0, usage.get("output_tokens") or 0)


def usage_cost(usage: dict, price: Price) -> Cost:
    r, w5, w1, f, o = tokens_of(usage)
    m = 1e-6
    return Cost(read=r * price.read * m, write=(w5 * price.write5 + w1 * price.write1h) * m,
                fresh=f * price.inp * m, output=o * price.out * m)


# ------------------------------------------------------------------ piece sizes


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def pieces(req: dict) -> list:
    """Estimated tokens of a request in cache order: [tools + system, message 1, message 2, ...]."""
    est = lambda obj: len(json.dumps(_plain(obj), ensure_ascii=False, separators=(",", ":"))) / CHARS_PER_TOKEN
    return [est({"tools": req.get("tools"), "system": req.get("system")})] + \
           [est(m) for m in req.get("messages") or []]


# ------------------------------------------------------------------ comparison


def compare(usage: dict, price: Price, actual_pieces: list, other_pieces: list):
    """Price one request twice: as it was sent (`actual_pieces`, billed per `usage`) and as the other
    version (`other_pieces`) would have been, keeping the same cache boundary.

    Returns (cost_actual, cost_other). Output cost is the same on both sides. The other side is the
    real bill plus the *difference* between two identically-built estimates, so requests the rewrite
    did not change come out with exactly zero difference whatever the estimate's rounding.
    """
    r, w5, w1, f, o = tokens_of(usage)
    actual = usage_cost(usage, price)
    total = r + w5 + w1 + f
    n = min(len(actual_pieces), len(other_pieces))
    est_total = sum(actual_pieces[:n])
    if total <= 0 or est_total <= 0 or n == 0:
        return actual, Cost(actual.read, actual.write, actual.fresh, actual.output)
    scale = total / est_total

    # Cache boundary: the piece boundary whose cumulative size is closest to what was read.
    best, best_gap, cum = 0, abs(r), 0.0
    for k in range(1, n + 1):
        cum += actual_pieces[k - 1] * scale
        if abs(cum - r) < best_gap:
            best, best_gap = k, abs(cum - r)
    if r == 0:
        best = 0

    uncached = w5 + w1 + f
    m = 1e-6
    if uncached > 0:   # keep the real mix of 5m writes / 1h writes / fresh input
        write_rate = (w5 * price.write5 + w1 * price.write1h) / uncached
        fresh_rate = f * price.inp / uncached
    else:
        write_rate, fresh_rate = price.write5, 0.0

    def estimate(pieces):
        cached = sum(pieces[:best]) * scale
        rest = max(0.0, sum(pieces[:n]) * scale - cached)
        return (cached * price.read * m, rest * write_rate * m, rest * fresh_rate * m)

    ea, eo = estimate(actual_pieces), estimate(other_pieces)
    other = Cost(read=max(0.0, actual.read + eo[0] - ea[0]), write=max(0.0, actual.write + eo[1] - ea[1]),
                 fresh=max(0.0, actual.fresh + eo[2] - ea[2]), output=actual.output)
    return actual, other


# ------------------------------------------------------------------ live ledger


@dataclass
class Totals:
    requests: int = 0
    orig: float = 0.0      # what the requests cost / would have cost unrewritten
    new: float = 0.0       # what they cost / would cost rewritten
    model: str = ""
    unpriced: int = 0

    @property
    def saved(self):
        return self.orig - self.new

    @property
    def pct(self):
        return 100 * self.saved / self.orig if self.orig else 0.0


class Ledger:
    """Thread-safe per-request and per-conversation money accounting for the proxy."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.lock = threading.Lock()
        self.convs = {}
        self.all = Totals()
        self._prev = {}     # conv -> (ts, cacheable tokens of its previous request)

    def add(self, conv, model, usage, result, dry_run=False, fell_back=False):
        """Record one finished request. Returns the row, or None when it cannot be priced."""
        price = price_for(model)
        if not usage or price is None:
            with self.lock:
                self.all.unpriced += 1
            return None
        orig_p, new_p = (result.orig_pieces, result.new_pieces) if result is not None and result.orig_pieces \
            else (None, None)
        if orig_p is None:
            # nothing was rewritten for this request: both sides cost exactly what was billed
            cost = usage_cost(usage, price)
            orig_cost = new_cost = cost
            actual = cost
        elif dry_run:   # forwarded unchanged: billed == original, rewritten side is an estimate
            actual, other = compare(usage, price, orig_p, new_p)
            orig_cost, new_cost = actual, other
        else:           # forwarded rewritten: billed == rewritten, original side is an estimate
            actual, other = compare(usage, price, new_p, orig_p)
            orig_cost, new_cost = other, actual
        r, w5, w1, f, o = tokens_of(usage)
        row = {
            "ts": round(time.time(), 1), "conv": (conv or "")[:8], "model": price.model, "dry_run": dry_run,
            "read": r, "write": w5 + w1, "fresh": f, "output": o,
            "billed": round(actual.total, 6), "without_proxy": round(orig_cost.total, 6),
            "with_proxy": round(new_cost.total, 6), "saved": round(orig_cost.total - new_cost.total, 6),
        }
        # Cache audit: a request soon after the previous one should read back most of what that one cached.
        ctx = r + w5 + w1
        with self.lock:
            prev = self._prev.get(row["conv"])
            self._prev[row["conv"]] = (row["ts"], ctx)
        row["fell_back"] = fell_back          # API rejected the rewritten request; original resent
        row["cold"] = bool(getattr(result, "cold", False))
        row["rewritten"] = orig_p is not None and not dry_run
        row["prev_ctx"] = prev[1] if prev else 0
        row["gap"] = round(row["ts"] - prev[0], 1) if prev else None
        with self.lock:
            for t in (self.convs.setdefault(row["conv"], Totals(model=price.model)), self.all):
                t.requests += 1
                t.orig += orig_cost.total
                t.new += new_cost.total
            if self.path:
                with open(self.path, "a") as fh:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        return row

    def snapshot(self):
        with self.lock:
            return Totals(**self.all.__dict__), {k: Totals(**v.__dict__) for k, v in self.convs.items()}
