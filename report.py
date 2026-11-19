"""Money report from a proxy ledger (`python3 proxy.py --ledger ledger.jsonl`).

    python3 report.py ledger.jsonl                  # one line per conversation, then the total
    python3 report.py ledger.jsonl --requests ab12  # one row per request of a conversation (id prefix)

Each request is priced from the API's own usage report (cache read / write / fresh / output tokens
at that model's prices). "Without proxy" is the same request as Claude Code sent it, priced with the
same cache boundary; "saved" is the difference, an estimate: the unmodified request is never actually billed. Under `--dry-run` nothing is rewritten, so
"billed" equals "without proxy" and "saved" is what the proxy *would* have saved.
"""

from __future__ import annotations

import argparse
import collections
import json
import time


def load(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def pct(saved, base):
    return f"{100 * saved / base:5.1f}%" if base else "    - "


def cache_audit(rows, window=280, floor=2000, hit=0.8):
    """Did Anthropic actually reuse the cached prefix? Looks at requests that follow their conversation's
    previous request within `window` seconds (inside both the 5 min and 1 h TTL) and count it a hit when it
    reads back at least `hit` of what the previous request held. Rewritten and untouched requests are
    reported apart: a lower rate for rewritten ones means rewrites are breaking the cache.
    Needs a ledger written by a proxy version that records prev_ctx / gap / rewritten."""
    groups = {True: [0, 0], False: [0, 0]}      # rewritten -> [eligible, hits]
    misses = []
    for r in rows:
        if r.get("gap") is None or r["gap"] > window or r.get("prev_ctx", 0) < floor:
            continue
        g = groups[bool(r.get("rewritten"))]
        g[0] += 1
        if r["read"] >= hit * r["prev_ctx"]:
            g[1] += 1
        else:
            misses.append(r)
    for rewritten, name in ((True, "rewritten"), (False, "untouched")):
        n, h = groups[rewritten]
        print(f"{name:10s} {n:5d} follow-up requests, {h:5d} read the cache back ({pct(h, n).strip()})")
    for r in misses[-10:]:
        print(f"  miss {time.strftime('%H:%M:%S', time.localtime(r['ts']))} conv {r['conv']} "
              f"read {r['read']:,d} of previous {r['prev_ctx']:,d}{' (rewritten)' if r.get('rewritten') else ''}")
    if not any(g[0] for g in groups.values()):
        print("no eligible requests (ledger lacks prev_ctx/gap, or no follow-ups within the window)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ledger")
    ap.add_argument("--requests", metavar="CONV", help="list the requests of one conversation (id prefix)")
    ap.add_argument("--cache-audit", action="store_true",
                    help="check that rewritten requests still read back the previous request's cache")
    a = ap.parse_args()
    rows = load(a.ledger)
    if not rows:
        print("ledger is empty")
        return
    if a.requests:
        rows = [r for r in rows if r["conv"].startswith(a.requests)]
        print(f"conversation {a.requests}: {len(rows)} requests\n")
        print("  time      cache read  cache write   output |   billed  without proxy    saved      %")
        for r in rows:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(r['ts']))} {r['read']:12,d} {r['write'] + r['fresh']:12,d} {r['output']:8,d} |"
                  f" {r['billed']:8.4f}  {r['without_proxy']:12.4f} {r['saved']:9.4f} {pct(r['saved'], r['without_proxy'])}")
        return
    if a.cache_audit:
        return cache_audit(rows)
    convs = collections.OrderedDict()
    for r in rows:
        c = convs.setdefault(r["conv"], collections.Counter())
        c["n"] += 1
        c["billed"] += r["billed"]
        c["without"] += r["without_proxy"]
        c["saved"] += r["saved"]
        c["model"] = r["model"]
        c["dry"] += r.get("dry_run", False)
    print(f"{'conversation':14s} {'model':18s} {'reqs':>5s} {'billed':>10s} {'without proxy':>14s} {'saved':>10s} {'%':>6s}")
    tot = collections.Counter()
    for conv, c in convs.items():
        tag = " (dry-run)" if c["dry"] == c["n"] else ""
        print(f"{conv:14s} {c['model']:18s} {c['n']:5d} {c['billed']:10.4f} {c['without']:14.4f} {c['saved']:10.4f} {pct(c['saved'], c['without'])}{tag}")
        tot.update({k: c[k] for k in ("n", "billed", "without", "saved")})
    print("-" * 82)
    print(f"{'TOTAL':14s} {'':18s} {tot['n']:5d} {tot['billed']:10.4f} {tot['without']:14.4f} {tot['saved']:10.4f} {pct(tot['saved'], tot['without'])}")


if __name__ == "__main__":
    main()
