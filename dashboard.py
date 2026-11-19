"""Terminal UI: scrolling request log with a pinned savings panel at the bottom."""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from collections import deque

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_SPARK = "▁▂▃▄▅▆▇█"
_PARTIAL = " ▏▎▍▌▋▊▉"


def vlen(text: str) -> int:
    return len(_ANSI.sub("", text))


def fit(text: str, width: int) -> str:
    """Pad or truncate to exactly `width` visible columns (colour codes preserved)."""
    if vlen(text) <= width:
        return text + " " * (width - vlen(text))
    plain = _ANSI.sub("", text)
    return plain[: max(0, width - 1)] + "…"


def usd(x: float) -> str:
    """Dollars, always four decimals: $0.0000, $0.0005, $0.0531, $45.8123."""
    return f"${x:.4f}"


def human(n: float) -> str:
    n = abs(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    return f"{int(n):,}"


class Stats:
    """Thread-safe running totals shown in the panel."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.saved_total = 0      # per-request savings summed (what you would have resent)
        self.saved_unique = 0     # each shrunken block counted once
        self.blocks = 0
        self.files = 0
        self.sent_total = 0       # real input tokens the API reported
        self.cache_read = 0
        self.cache_write = 0
        self.history = deque(maxlen=48)
        self.last_saved = 0
        self.last_ctx = 0         # real context size of the latest request (post-shrink)
        self.last_hit = None      # cache hit rate of the latest request
        self._prev_ctx = {}       # conversation -> previous context size
        self.by_lever = {}        # lever label -> tokens saved (each block counted once)

    def request(self, result):
        with self.lock:
            self.requests += 1
            self.saved_total += result.saved
            self.saved_unique += result.fresh_saved
            self.last_saved = result.saved
            self.history.append(result.saved)
            for e in result.events:
                if e.fresh:
                    self.blocks += 1
                    self.files += e.converted
                    self.by_lever[e.label] = self.by_lever.get(e.label, 0) + e.before - e.after

    def usage(self, usage, conv=""):
        """Record the API's own token accounting; returns a warning string on a cache miss."""
        if not usage:
            return None
        read = usage.get("cache_read_input_tokens") or 0
        write = usage.get("cache_creation_input_tokens") or 0
        fresh = usage.get("input_tokens") or 0
        ctx = read + write + fresh
        with self.lock:
            self.sent_total += ctx
            self.cache_read += read
            self.cache_write += write
            self.last_ctx = ctx
            self.last_hit = read / ctx if ctx else None
            prev = self._prev_ctx.get(conv, 0)
            self._prev_ctx[conv] = ctx
        if prev >= 2000 and read < prev * 0.5:
            return f"cache read {human(read)} of {human(ctx)} tokens (previous request: {human(prev)})"
        return None

    def snapshot(self):
        with self.lock:
            return dict(self.__dict__, history=list(self.history), by_lever=dict(self.by_lever))


class Dashboard:
    HEIGHT = 9

    def __init__(self, stats, stream=None):
        self.stats = stats
        self.out = stream or sys.stderr
        self.tty = self.out.isatty() and os.environ.get("TERM") != "dumb"
        self.color = self.tty and "NO_COLOR" not in os.environ
        self.lock = threading.Lock()
        self.active = False
        self.ledger = None   # economics.Ledger, set by the proxy
        self.dry_run = False
        self._rows = 0

    # --------------------------------------------------------------- styling

    def c(self, code, text):
        return f"\x1b[{code}m{text}\x1b[0m" if self.color else str(text)

    def green(self, t): return self.c("38;5;78", t)
    def cyan(self, t): return self.c("38;5;80", t)
    def amber(self, t): return self.c("38;5;215", t)
    def red(self, t): return self.c("38;5;203", t)
    def dim(self, t): return self.c("38;5;243", t)
    def bold(self, t): return self.c("1", t)

    def bar(self, frac, width, color):
        frac = min(1.0, max(0.0, frac))
        eighths = round(frac * width * 8)
        full, part = divmod(eighths, 8)
        fill = "█" * full + (_PARTIAL[part] if part and full < width else "")
        rest = "░" * max(0, width - vlen(fill))
        return color(fill) + self.dim(rest)

    def spark(self, values, width):
        values = values[-width:]
        top = max(values) if values else 0
        if top <= 0:
            return self.dim("▁" * max(1, len(values)))
        return self.green("".join(_SPARK[min(7, int(v / top * 7.999))] for v in values))

    # ----------------------------------------------------------------- panel

    def panel(self, width):
        s = self.stats.snapshot()
        inner = width - 4
        clock = time.strftime("%H:%M:%S")
        up = int(time.time() - s["started"])
        title = f" ◆ AOP · token saver "
        right = f" up {up // 3600}h{up % 3600 // 60:02d}m · {clock} "
        top = "╭─" + self.bold(self.cyan(title)) + self.dim("─" * max(1, inner - len(title) - len(right)) + right) + "─╮"
        sent, saved = s["sent_total"], s["saved_total"]
        pct = saved / (saved + sent) * 100 if saved + sent else 0.0

        line1 = (f"{self.bold('SAVED')} {self.bold(self.green('−' + human(saved) + ' tok'))}  "
                 f"{self.green(f'{pct:.1f}%')} of context sent   "
                 f"{self.dim('unique')} {human(s['saved_unique'])}   "
                 f"{self.dim('reqs')} {s['requests']}  {self.dim('blocks')} {s['blocks']}  {self.dim('files→md')} {s['files']}")

        ctx = s["last_ctx"]
        before = ctx + s["last_saved"]
        bw = max(8, min(30, inner - 46))
        if ctx:
            line2 = (f"{self.dim('last ')} {self.bar(ctx / before, bw, self.cyan)}  "
                     f"{human(before)} → {self.bold(human(ctx))} tok  "
                     f"{self.green('−' + str(round(100 * s['last_saved'] / before)) + '%')}")
        else:
            line2 = f"{self.dim('last ')} {self.dim('waiting for first response…')}"

        total_in = s["cache_read"] + s["cache_write"]
        hit = s["last_hit"]
        if hit is None:
            line3 = f"{self.dim('cache')} {self.dim('no data yet')}"
        else:
            shade = self.green if hit >= 0.8 else self.amber if hit >= 0.4 else self.red
            line3 = (f"{self.dim('cache')} {self.bar(hit, bw, shade)}  {shade(f'{hit * 100:.0f}% hit')}   "
                     f"{self.dim('session read')} {human(s['cache_read'])}  {self.dim('written')} {human(s['cache_write'])}")
        line4 = f"{self.dim('trend')} {self.spark(s['history'], min(48, inner - 8))}"
        levers = sorted(s["by_lever"].items(), key=lambda kv: -kv[1])[:4]
        line5 = (f"{self.dim('levers')} " + "  ".join(f"{name} {self.green('−' + human(v))}" for name, v in levers)
                 if levers else f"{self.dim('levers')} {self.dim('none yet')}")
        if self.ledger is not None and self.ledger.all.requests:
            t, _ = self.ledger.snapshot()
            billed = t.orig if self.dry_run else t.new
            line6 = (f"{self.dim('money')} billed {self.bold(usd(billed))}  "
                     f"{self.dim('without proxy')} {usd(t.orig)}  "
                     f"{self.green(('would save' if self.dry_run else 'est. saved') + f' {usd(t.saved)} ({t.pct:.1f}%)')}  "
                     f"{self.dim('reqs')} {t.requests}")
        else:
            line6 = f"{self.dim('money')} {self.dim('waiting for first priced response…')}"
        rows = [line1, line2, line3, line4, line5, line6]
        body = ["│ " + fit(r, inner) + " │" for r in rows]
        return [top, *body, "╰" + "─" * (inner + 2) + "╯"]

    def _size(self):
        size = shutil.get_terminal_size((100, 30))
        return size.lines, min(max(size.columns, 60), 110)

    def _draw(self):
        rows, cols = self._size()
        lines = self.panel(cols)
        self.out.write("\x1b7")  # save cursor
        for i, text in enumerate(lines):
            self.out.write(f"\x1b[{rows - self.HEIGHT + 1 + i};1H\x1b[2K{text}")
        self.out.write("\x1b8")  # restore cursor
        self.out.flush()

    def _sync(self):
        rows, _ = self._size()
        if rows != self._rows:
            self._rows = rows
            self.out.write(f"\x1b[{rows - self.HEIGHT + 1};1H\x1b[J")      # clear the panel area
            self.out.write(f"\x1b[1;{rows - self.HEIGHT}r\x1b[{rows - self.HEIGHT};1H")

    # ----------------------------------------------------------------- public

    def start(self):
        if not self.tty:
            return
        with self.lock:
            self.out.write("\n" * self.HEIGHT)  # scroll existing output up to make room
            self._rows = 0
            self._sync()
            self.active = True
            self._draw()

    def stop(self):
        with self.lock:
            if self.active:
                rows, _ = self._size()
                self.out.write(f"\x1b[r\x1b[{rows};1H\n")
                self.out.flush()
                self.active = False

    def log(self, text=""):
        with self.lock:
            if not self.active:
                print(text, file=self.out, flush=True)
                return
            self._sync()
            _, cols = self._size()
            self.out.write(fit(text, cols - 1).rstrip() + "\n")
            self._draw()

    def refresh(self):
        with self.lock:
            if self.active:
                self._sync()
                self._draw()

    # ------------------------------------------------------------ log lines

    def request_line(self, result):
        """One line per request: tokens saved now, with a per-source breakdown."""
        stamp = self.dim(time.strftime("%H:%M:%S"))
        if not result.changed:
            return f"{stamp} {self.dim('▸ no change')}"
        groups = {}
        for e in result.events:
            if e.fresh:
                n, delta = groups.get(e.label, (0, 0))
                groups[e.label] = (n + 1, delta + e.before - e.after)
        parts = []
        for label, (n, delta) in groups.items():
            tag = label + (f"×{n}" if n > 1 else "")
            parts.append(f"{tag} {self.green('−' + human(delta)) if delta >= 0 else self.amber('+' + human(delta))}")
        fresh = result.fresh_saved
        head = self.green(f"▸ saved {human(fresh)} tok") if fresh else self.dim("▸ replayed")
        if getattr(result, "cold", False):
            head = f"{self.amber('❄ cold cache: history compacted')} {head}"
        tail = self.dim(f"(request total −{human(result.saved)})")
        return f"{stamp} {head}  {' · '.join(parts) or self.dim('earlier blocks')}  {tail}"

    def money_line(self, row):
        """One line per finished request: what it cost and what the proxy saved, cache-aware."""
        stamp = self.dim(time.strftime("%H:%M:%S"))
        saved, orig = row["saved"], row["without_proxy"]
        pct = 100 * saved / orig if orig else 0.0
        verb = "would save" if row["dry_run"] else "est. saved"
        tint = self.green if saved > 0 else self.dim
        return (f"{stamp} {usd(row['billed'])} billed  {tint(f'{verb} {usd(saved)} ({pct:.1f}%)')}  "
                f"{self.dim('read')} {human(row['read'])} {self.dim('write')} {human(row['write'])} "
                f"{self.dim('out')} {human(row['output'])}")
