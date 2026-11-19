# Agent Optimization Proxy

A proxy between Claude Code and Anthropic's API. It shrinks your requests so you pay less. It never breaks prompt caching, and it shows you an estimate of what it saved in dollars (a counterfactual: the unmodified request is never billed, and token counts are approximations).

## Run

```
pip install -r requirements.txt
python3 proxy.py
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

Try `--dry-run` first. It measures savings and forwards everything unchanged.

To use it every time (CLI and VS Code), put this in `~/.claude/settings.json` and keep `proxy.py` running:

```json
{ "env": { "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080" } }
```

Delete that line to go back to the direct API.

## Live results

Real requests through the proxy, real API bills. From `ledger.jsonl`:

| | |
|---|---|
| Requests | 52, one conversation, about 22 min |
| Billed | $2.4453 |
| Would have cost without the proxy | $2.7570 |
| Saved | **$0.3117 (11.3%)** |

- "Billed" is exact. "Without the proxy" is an estimate: the same request priced with the same cache boundary.
- 2 requests cost slightly more than they would have ($0.0012 total).
- It's one session. Don't read 11.3% as a promise.
- Cold-cache compaction and repeat-read dedupe are **not** checked live. No cache went cold during that run, and I can't confirm the running proxy had that code. Treat them as untested against the real API.

Check your own numbers:

```
python3 proxy.py --ledger ledger.jsonl
python3 report.py ledger.jsonl
python3 report.py ledger.jsonl --requests ab12   # one conversation, per request
```

Judge it by dollars, not the token counts in the log. Tokens cut from cached history save the cheap read price. Tokens cut from the new tail save the expensive write price.

## How it stays cache-safe

Anthropic's cache matches an exact prefix. Change anything already sent and everything after it gets re-billed.

So every block is rewritten once, when it first shows up (it's not cached yet, so that's free). The result is saved in `~/.cache/aop-proxy/memo.sqlite` and the same bytes are sent on every later request. If the proxy joins a conversation midway, the existing history is left alone.

If the API rejects a rewritten request (400), the proxy resends the original and stops touching that conversation.

## What it changes

| Content | What happens |
|---|---|
| `Bash`, `WebFetch`, `Agent`, MCP output | Strip ANSI and noise, minify JSON, prune prose with LLMLingua-2, fold long runs of near-identical log lines |
| PDFs, docx/pptx/xlsx/epub reads | Converted to Markdown with markitdown |
| Your messages | Lossless cleanup. Big pasted prose gets pruned. Short paragraphs and code stay as is |
| Big whole-file `Read` | Keeps the parts that match your task, with original line numbers, and marks what's skipped |
| Repeated whole-file `Read` | If it's identical to an earlier read, it becomes a short pointer |
| `Write`/`Edit`/`Bash` args the assistant already ran | Big ones become a stub: path, size, outline. The file is on disk |
| System prompt and tool definitions | Whitespace cleanup. Unused MCP tools get slimmed |
| Cold cache | See below |

Never touched: thinking blocks, `Edit`/`Grep`/`Glob` results, ranged reads, error results, code, tables, diffs, URLs.

## Cold cache

The cache dies after its TTL (1 hour, or 5 minutes). Come back after that and the whole context gets re-billed at the write price. That request is the one time you can change old history for free.

When a conversation comes back after its TTL, the proxy compacts old history once and replays it from then on:

- old `Bash` / `WebFetch` / MCP output keeps its head, tail and error lines
- old reads of files that were since overwritten or re-read differently become an outline
- the newest 10 messages are never touched

`--no-cold` turns it off.

## Flags

| Flag | Effect |
|---|---|
| `--port N` | Local port (default 8080) |
| `--dry-run` | Measure only, forward originals |
| `--no-optimize` | Pure pass-through |
| `--ledger FILE` | One JSON line per request with its real cost and the cost without the proxy |
| `--no-cold` | No cold-cache compaction |
| `--cache-ttl SECONDS` | Cache lifetime to assume (default: read from the request) |
| `--cold-keep N` | Newest messages cold compaction never touches (10) |
| `--keep-repeat-reads` | Don't replace repeated reads with pointers |
| `--no-compress` | Lossless cleanup only, no LLMLingua |
| `--no-convert` | Skip markitdown |
| `--full-reads` | Always send whole files |
| `--keep-prompts` / `--lossless-prompts` | Never touch / only losslessly clean your messages |
| `--keep-args` | Don't abridge executed Write/Bash/Edit args |
| `--keep-prefix` | Leave system prompt and tool definitions as sent |
| `--model`, `--rate`, `--device` | LLMLingua-2 model, keep rate (0.6), torch device |
| `--memo PATH` | Memo location (`""` = memory only) |
| `--upstream URL` | API origin |

More flags: `python3 proxy.py --help`.

## Reading the code

About 3,000 lines of source. Read in this order and stop whenever you have enough:

1. `proxy.py` (about 400 lines): the HTTP server. Reads a request, asks the optimizer to rewrite it, forwards it, sniffs the token usage out of the reply, and resends the original if the API returns a 400.
2. `optimizer.py`, top docstring and `_run`: the whole rewrite pipeline. Everything is decided once per block and stored in SQLite (`_decide`), so later requests resend identical bytes and the prompt cache stays valid.
3. The levers it calls, each small and independent: `prefix.py` (system prompt and tools), `elide.py` (long tool-call arguments), `shrink.py` (tool output and prompts), `targeted.py` (large file reads), `readopt.py` (repeated and stale reads), `cold.py` (history compaction after the cache has expired).
4. `economics.py`, `log.py`, `report.py`: pricing, log lines, and the ledger report. Nothing here changes requests.

The idea to hold onto: **a block is rewritten the first time it is seen, never again.** The one exception is a cold restart, when the cache is gone anyway.

## Files

| File | Role |
|---|---|
| `proxy.py` | HTTP proxy, usage sniffing, 400 fallback |
| `optimizer.py` | Cache-safe request rewriting, memo |
| `readopt.py` | Repeated and stale file reads |
| `cold.py` | Cold-cache detection and old-output stubs |
| `elide.py` | Abridging executed tool-call args |
| `prefix.py` | System prompt and tool definition cleanup |
| `shrink.py` | Cleanup and LLMLingua-2 |
| `targeted.py` | Targeted file reads |
| `fileconv.py` | markitdown wrapper |
| `economics.py` | Prices and cache-aware cost |
| `report.py` | Money report from a ledger |
| `log.py` | One-line request and cost log |
| `breakdown.py` | Token composition per request (`--profile`) |
| `test_aop.py`, `test_cold.py` | `python3 -m unittest test_aop test_cold` |

Every file in detail: [technical.md](technical.md).
