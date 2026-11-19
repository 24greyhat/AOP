"""Summarize an A/B quality run: success rate and cost per successful task, proxy on vs off.

    python3 ab_summary.py results.csv --on ledger_on.jsonl --off ledger_off.jsonl

results.csv columns: task,arm,start,end,passed   (arm = on|off; start/end = "YYYY-MM-DD HH:MM:SS" local
time or epoch seconds; passed = 1/0 from your own acceptance check). Cost of a run is the sum of `billed`
in that arm's ledger rows inside [start, end]. See EVALUATION.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import time


def when(v):
    try:
        return float(v)
    except ValueError:
        return time.mktime(time.strptime(v.strip(), "%Y-%m-%d %H:%M:%S"))


def ledger(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    a = ap.parse_args()
    ledgers = {"on": ledger(a.on), "off": ledger(a.off)}
    stats = {"on": [0, 0, 0.0], "off": [0, 0, 0.0]}     # runs, passed, cost
    per_task = {}
    for r in csv.DictReader(open(a.results)):
        arm, lo, hi = r["arm"].strip(), when(r["start"]), when(r["end"])
        cost = sum(x["billed"] for x in ledgers[arm] if lo <= x["ts"] <= hi)
        ok = int(r["passed"])
        s = stats[arm]
        s[0] += 1
        s[1] += ok
        s[2] += cost
        per_task.setdefault(r["task"], {})[arm] = (ok, cost)
    print(f"{'task':24s} {'on: pass $':>14s} {'off: pass $':>14s}")
    for t, arms in per_task.items():
        cell = lambda k: f"{arms[k][0]}  {arms[k][1]:8.3f}" if k in arms else "-"
        print(f"{t:24s} {cell('on'):>14s} {cell('off'):>14s}")
    print()
    for arm in ("on", "off"):
        n, ok, cost = stats[arm]
        per = f"${cost / ok:.3f}" if ok else "n/a"
        print(f"{arm:4s} runs {n:3d}  passed {ok:3d} ({100 * ok / n if n else 0:.0f}%)  cost ${cost:.3f}  per success {per}")
    if stats["on"][1] and stats["off"][1]:
        ratio = (stats["on"][2] / stats["on"][1]) / (stats["off"][2] / stats["off"][1])
        print(f"\ncost per successful task, proxy on / off = {ratio:.2f}  (below 1.00 = proxy is cheaper per success)")
    print("Small samples: with under ~20 runs per arm treat differences in pass rate as anecdotes, not results.")


if __name__ == "__main__":
    main()
