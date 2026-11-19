# Does the proxy hurt task quality? A/B protocol

Question: does AOP lower the **cost per successfully completed task**, not just tokens per request?
Works on a Claude subscription: the ledger prices usage at API rates, which is fine for comparing arms.

## Setup
- Pick 10-20 tasks you actually do, each with an objective pass check you can run without judgement
  (tests pass, a script's output matches, a specific bug no longer reproduces). Include some where the answer
  depends on details deep in a big file or old command output; those are where lossy reads would bite.
- Keep each task in its own git branch/worktree at a fixed starting commit, so both arms start identically.
- Two proxy runs, same flags except the arm:
  - **on:**  `python3 proxy.py --ledger ledger_on.jsonl`
  - **off:** `python3 proxy.py --dry-run --ledger ledger_off.jsonl`   (forwards everything unchanged)
- Same model, same settings (thinking on/off) in both arms.

## Procedure (per task)
1. Reset the task's worktree. Point Claude Code at the proxy for the arm.
2. Note the start time. Give the **same prompt** (paste it from a file). No follow-up hints.
3. Stop at pass, or at a fixed budget (e.g. 30 min or 40 turns), whichever comes first. Note the end time.
4. Run the pass check **after** the run, blind to the arm if someone else can do it. Record 1 or 0.
5. Alternate the arm order across tasks (on, off, off, on, ...) and run each task in both arms, ideally twice,
   because agent runs vary a lot between identical attempts.
6. For at least some tasks, let the session idle past the cache lifetime (over 1 h) midway and resume, so cold
   compaction is exercised. Cold compaction is the lever most likely to remove evidence.

## Record
`results.csv`:

    task,arm,start,end,passed
    fix-date-parse,on,2026-11-20 10:02:00,2026-11-20 10:19:30,1
    fix-date-parse,off,2026-11-20 11:00:00,2026-11-20 11:14:10,1

## Read the result
    python3 ab_summary.py results.csv --on ledger_on.jsonl --off ledger_off.jsonl
    python3 report.py ledger_on.jsonl --cache-audit

Decide on **cost per success** and pass rate together. Judge it like this:
- Pass rate lower on, same or higher cost per success: lossy levers are hurting. Re-run with `--full-reads`,
  `--no-cold`, `--keep-args` one at a time to find which lever.
- Pass rate equal, cost per success lower on: the proxy is paying for itself.
- Failures in the on arm: open the transcript and look for a `[proxy: ... omitted` marker near where the agent
  went wrong. That is direct evidence of a lossy cut causing the error.
- Also check the cache audit: if rewritten requests read the cache back less often than untouched ones, or any
  request fell back after a 400, fix that first; it makes the cost comparison unreliable.

## Caveats
- Under 20 runs per arm, differences in pass rate are noise. Report them as such.
- Costs are estimated from token counts at API prices, not your subscription's real accounting.
- The off arm is `--dry-run`, so the proxy's own logic still runs; only its rewrites are withheld.
