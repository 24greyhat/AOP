# AOP technical reference

AOP (Agent Optimization Proxy) is a local HTTP proxy that sits between Claude Code and the Anthropic API.
It rewrites `/v1/messages` request bodies so they carry fewer tokens, without ever invalidating Anthropic's
prompt cache, and prices every request so the saving is measured in dollars rather than guessed in tokens.

This document explains the architecture first, then every file. For usage and live results see
[README.md](README.md).

## Contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [The one invariant: cache safety](#2-the-one-invariant-cache-safety)
3. [Request lifecycle](#3-request-lifecycle)
4. [The optimizer in depth](#4-the-optimizer-in-depth)
5. [Cold-cache compaction](#5-cold-cache-compaction)
6. [Read-level optimization](#6-read-level-optimization)
7. [The memo: schema and key catalogue](#7-the-memo-schema-and-key-catalogue)
8. [File by file](#8-file-by-file)
9. [Measurement tooling](#9-measurement-tooling)
10. [Testing](#10-testing)
11. [Failure handling](#11-failure-handling)
12. [Known limitations](#12-known-limitations)

---

## 1. Architecture at a glance

```
 Claude Code ──HTTP──▶ proxy.py ──▶ optimizer.py ──▶ rewritten body ──HTTPS──▶ api.anthropic.com
 (ANTHROPIC_BASE_URL)     │             │  ▲                                         │
                          │             │  └── memo.sqlite (every decision, replayed)│
                          │             ├── shrink.py     cleanup + LLMLingua-2      │
                          │             ├── targeted.py   narrow big file reads      │
                          │             ├── readopt.py    per-file read history      │
                          │             ├── cold.py       cache-expiry compaction    │
                          │             ├── elide.py      abridge executed tool args │
                          │             ├── prefix.py     system prompt + tool defs  │
                          │             └── fileconv.py   PDF/Office → Markdown      │
                          │                                                           │
                          ◀───────── response relayed byte for byte ◀─────────────────┘
                          │   UsageSniffer reads `usage` out of the stream
                          ▼
                   economics.Ledger ─▶ log.py (log lines) ─▶ --ledger file ─▶ report.py
```

Three properties define the design:

* **The proxy is transparent.** Non-`/v1/messages` traffic and any request it cannot parse are forwarded
  untouched. Responses are relayed as they arrive (SSE included); only the token `usage` is observed.
* **Every rewrite is a memoized, deterministic decision** about a block of the request, stored in SQLite and
  replayed byte for byte. That is what keeps the cache valid (section 2).
* **Every rewrite is optional.** Each lever has a flag, and a 400 from the API on a rewritten request makes the
  proxy resend the original and stop rewriting that conversation.

## 2. The one invariant: cache safety

Anthropic's prompt cache matches an exact byte prefix of the request. Changing anything that was already sent
invalidates everything after it, and the tokens after the change are re-written at the cache-write price
(1.25x input for 5-minute entries, 2x for 1-hour entries) instead of read at about 0.1x.

The rule that follows:

> Shrink a block **once, the first time it is seen**, when it sits in the not-yet-cached tail and changing it is
> free. Store the result. Resend exactly those bytes on every later request.

Corollaries implemented in the code:

| Situation | Behavior |
|---|---|
| A block is new | Decide now, store the decision (including "leave alone"). |
| A block was decided before | Replay the stored decision; never recompute. |
| The proxy meets a conversation midway | Everything up to the previous request's last cache breakpoint may already be cached upstream, so it is **frozen**: stored as "leave alone". Only the newest turn is shrunk (`joining`, `freeze_upto` in `_run`). |
| The conversation key | Hash of the first message with `cache_control` stripped. The marker sits on the newest messages, so it moves off `messages[0]` after request 1; hashing it would make request 2 look like a different conversation. |
| The proxy restarts | Decisions live in SQLite, so a restarted proxy (even with a different compressor) replays identical bytes. |
| The cache has expired | The one case where history may be changed: the whole prefix is being re-written at the write price anyway. See section 5. |

Never rewritten at all: `thinking` blocks (they carry signatures), `Edit`/`Grep`/`Glob` results, ranged `Read`
results, repeated `<system-reminder>` blocks (the newest copy is the live nudge), and anything in an error result.

## 3. Request lifecycle

Implemented in [proxy.py](proxy.py) (`ProxyHandler.proxy_request`):

1. **Read the body.** `read_body` accepts a single `Content-Length` or `Transfer-Encoding: chunked` and rejects
   anything ambiguous with a 400 (request-smuggling hygiene).
2. **Classify.** Only `POST /v1/messages` is a "messages request". Everything else is a plain pass-through.
3. **Optimize** (`OPTIMIZER.optimize(body)`), skipped when the request has a `Content-Encoding` header or the
   proxy runs with `--no-optimize`. `--dry-run` computes the rewrite and the statistics but forwards the original.
   `--dump` appends the original body to a JSON file; `--profile` appends a token-composition row.
4. **Forward.** `send_upstream` opens a fresh HTTPS (or HTTP for a custom `--upstream`) connection, copies headers
   except hop-by-hop ones, and for messages requests forces `Accept-Encoding: identity` so the usage block can be
   read from an uncompressed stream. Connections are closed per request (`Connection: close`).
5. **Fallback.** If a *rewritten* request gets HTTP 400, the proxy logs the reason, calls `OPTIMIZER.poison(conv)`
   (that conversation is never rewritten again) and resends the original body.
6. **Relay.** `relay_response` streams the upstream body with `read1`, re-chunking when needed, feeding every
   piece to `UsageSniffer`.
7. **Account.** After the stream ends, `UsageSniffer.finish()` yields the usage object (input side from
   `message_start`, output total from the last `message_delta`). `log.cache_miss` flags cache misses,
   `Ledger.add` prices the request with and without the proxy, and `log.money_line` prints it.

`main()` wires flags to `Optimizer(...)`, starts the LLMLingua model in a background thread, installs a SIGTERM
handler that follows the Ctrl+C shutdown path, and warns if a local `llmlingua.py`/`markitdown.py` shadows the
real package.

## 4. The optimizer in depth

[optimizer.py](optimizer.py) is the core. `Optimizer.optimize(raw)` never raises: any exception returns the
original bytes. It takes a re-entrant lock, so requests are optimized one at a time; decisions are therefore
consistent even when subagents and the main thread send requests concurrently.

### 4.1 `_run`, phase by phase

1. **Identify and time the conversation.** `_conv_key`, then `Memo.touch(conv, now, ttl)` records the request
   time and reports whether the cache expired since the previous one (section 5). `ReadIndex.build(messages)`
   indexes the reads of the request *before anything is rewritten* (section 6).
2. **Joining check.** `joining` is true when the conversation is new to the proxy and has more than one message,
   or was seen only by an older version that recorded no prefix mode. It sets `freeze_upto`.
3. **Collect context.** One pass over the messages gathers `tools` (tool_use id → name, input) and the set of
   failed tool_use ids (`errors`). Failed calls are never abridged.
4. **Prefix mode** (`_prefix_mode`, decided once per conversation and replayed):

   | Mode | Meaning |
   |---|---|
   | `note` | Seen from request 1 with an editing tool and a system prompt: prefix cleaned, the system prompt gets the abridgement note, executed tool-call arguments may be abridged. |
   | `plain` | Seen from request 1: prefix cleaned only. |
   | `tail:<i>` | Joined midway: prefix left alone for good (it is probably cached), tool-call arguments still abridged, the note rides in user message `i`. |
   | `off` | Nothing to do. |

   `note` and `tail:` exist because abridged arguments appear in history as `[proxy: …]` markers; the model must
   be told they are display-only and must never be written into files (`elide.SYSTEM_NOTE`).
5. **Walk the messages in order.**
   * assistant: collect narration and tool-call text as "what the task is about", then `_assistant` abridges
     executed `Write`/`Edit`/`MultiEdit`/`NotebookEdit`/`Bash` arguments and lightly cleans long narration.
   * user, string content: an attachment marker or a typed prompt (`_prompt`).
   * user, block content: PDFs (`_documents`), `tool_result` blocks (`_tool_result`), `text` blocks (typed text,
     or an `@`-attached file that Claude Code inlined as a "Result of calling the Read tool" block).
6. **Cold pass** (`_cold_pass`) when the conversation has ever gone cold (section 5).
7. **Serialize** with compact separators, or return the original bytes if no event happened (so an untouched
   request is byte-identical, not just JSON-equal). The result carries per-piece token estimates for pricing.

### 4.2 `_decide`: the memoization primitive

```python
_decide(key, frozen, events, label, produce, converted=False, cold_produce=None) -> replacement | None
```

* No stored entry: unless `frozen`, call `produce()`; store the entry (`Entry(None)` = leave alone); the event is
  `fresh`. If `frozen`, store `Entry(None)` without calling `produce`.
* Stored entry with text: replay it (`fresh=False`).
* Stored "leave alone" entry in a conversation that has been through a cold restart: on the cold request itself,
  reconsider once under key `cold1` (using `cold_produce` if given, e.g. lower abridgement thresholds); on later
  requests replay that second decision.

`Event(label, before, after, fresh)` feeds the log line and the profile rows.

### 4.3 What each handler does

| Content | Handler | Decision |
|---|---|---|
| `Bash`, `BashOutput`, `WebFetch`, `WebSearch`, `Agent`, `Task`, `mcp__*` results | `_tool_result` → `_shrink_text` | `shrink.clean`; fold near-identical log runs (Bash only); LLMLingua-2 on prose-like output above `lossy_min_tokens`. Requires a minimum absolute and relative gain. |
| `Read` result of a PDF/Office path | `_tool_result` | markitdown conversion (`fileconv`), cleaned and pruned. |
| `Read` of a big source file, whole-file | `_tool_result` → `targeted.targeted` | Keep the functions/classes that match the task, verbatim, with a marker for the rest. |
| `Read` repeating an earlier identical `Read` | `_repeat_read` | Short pointer (section 6). |
| PDF `document` blocks, `@`-attached office files | `_documents`, `_attachment` | Converted to Markdown. |
| Typed user text (long) | `_prompt` | Lossless cleanup; pasted prose paragraphs pruned (`_prune_pasted`), first/last paragraph, short paragraphs and code kept. |
| Executed `Write`/`Edit`/`Bash` arguments | `_assistant` → `elide.elide_input` | Stub with path, size, head, outline, tail. |
| Assistant narration | `_clean_narration` | `clean_prompt` + `compact_markdown` (lossless). Lossy pruning only with `--prune-narration`. |
| System prompt | `_prefix` → `_shrink_system` | Lossless whitespace cleanup. |
| Tool definitions | `_tool_defs` | Drop `$schema`/`title`, clean descriptions; slim unused MCP tools. |

Query for targeted reads: the last two typed prompts (weight 2.0) and the last six assistant notes/tool-call
inputs (weight 1.0).

### 4.4 Targeted reads

[targeted.py](targeted.py) splits a code file into *units* (Python through `ast`; other languages by a
start-of-definition heuristic plus indentation-level member splitting for blocks of 50+ lines), scores each
unit against the query with an IDF-weighted term overlap (identifier words are split on `_` and camelCase; a
unit whose exact name is mentioned scores a large bonus), and keeps: small leading statements (imports,
constants), the best-scoring units (up to a token budget of about 45% of the file), and the class header lines
around any kept method. Gaps under about 90 tokens are kept rather than replaced because the marker costs as
much. Every other run of units becomes one marker naming what it holds and the `offset`/`limit` to fetch it.
Kept lines are byte-identical to what the tool returned. It returns `None` (send whole) when the file is small,
unparsable, nothing clearly matches (one stray shared word is not enough), or the result would not be at least
30% smaller.

### 4.5 Abridging executed arguments

[elide.py](elide.py): what the assistant wrote into a tool call is dead weight once the call ran. The stub
keeps what the model needs to continue without re-reading: where the thing is (path), its size, how it starts
and ends and, for code, an outline of definitions with line numbers (`file_stub`); a shell command keeps its
command line and first script lines (`command_stub`); long edit strings keep head and tail (`text_stub`). A
stub must save at least `MIN_GAIN` tokens and remove `MIN_RATIO` of the block or it is not used. Calls whose
result was an error are skipped.

## 5. Cold-cache compaction

### 5.1 Why

The prompt cache entry lives 5 minutes, or 1 hour when Claude Code requests the 1-hour TTL. Each request
refreshes it. A session resumed after the TTL has no cache: the whole context is re-written at the write price.
In the measured transcripts five such restarts cost 17.9% of total spend. Because the entire prefix is being
re-written on that request regardless, it is the one moment already-sent history can be changed for free.

### 5.2 Detection

* `cold.request_ttl(data)`: 3600 s if any `cache_control` in the request has `"ttl": "1h"`, else 300 s.
  `--cache-ttl` overrides.
* `cold.is_cold(gap, ttl)`: `gap > ttl + 60` seconds. The margin absorbs the time a reply takes, because the TTL
  counts from when the previous request was processed.
* `Memo.touch(conv, now, ttl, detect)` stores `(last request time, epoch)` in `conv_state` on **every** request
  and returns `(cold_now, epoch)`. `epoch` counts the cold restarts the conversation has been through and
  survives proxy restarts. `detect` is false for single-message requests and when `--no-cold` is set (the
  timestamp is still kept so enabling it later works).

A conversation the proxy has no earlier timestamp for cannot be judged cold; a conversation first met after a
long gap is treated as a normal join.

### 5.3 What the compaction does

Two mechanisms, both memoized and replayed:

**Reconsideration (inside `_decide`).** Blocks that were left alone while the cache was warm (history frozen at
join time, or blocks below the first-sight size thresholds) are decided again once per cold restart under key
`cold1`. Executed-argument abridging uses lower thresholds here (`cold_elide_min_tokens=150`,
`cold_bash_min_tokens=120`).

**Second stage (`_cold_pass`)**, run after every block has had its normal treatment:

| Target | Rule |
|---|---|
| Old `Bash`/`BashOutput`/`WebFetch`/`mcp__*` results ≥ `cold_min_tokens` (1000) | `cold.aged_stub`: first 6 and last 4 lines, up to 10 error/URL lines from the middle, and a marker with the omitted line count. A few very long lines are cut by characters. Must remove at least half the block. |
| Whole-file `Read` of a file later overwritten or re-read with different content | `readopt.stale_stub` (section 6). |

Protections: the newest `cold_keep` (10) messages are never created or changed; error results, thinking blocks,
user text, assistant text and `Edit`/`Grep`/`Glob`/`Agent`/`Task`/`WebSearch` results are never touched.

### 5.4 Replay safety

Second-stage decisions are keyed `cold2 | conv | tool_use id (| item index) | hash of the block's current text`.
The "current text" is the output of the first stage, which is itself memoized, so on every later request the same
key is found and the same bytes are produced. A warm request never creates a second-stage decision
(`_cold_decide(create=False)`); it only replays. Decisions made at a later cold restart only touch what the
earlier restart protected or left alone, so earlier compacted bytes never change again.

### 5.5 The abridgement note

If the conversation's prefix does not already carry the note (`plain` or `off` mode), the first cold compaction
that produces any event appends it as a `<system-reminder>` block to the newest user message and stores that
message's index under `cold-note`. Later requests re-add it at the same index, so the bytes are stable.

### 5.6 Accounting

A cold request is flagged on `Result.cold`. The log line is prefixed "cold cache, history compacted". Savings are priced by
`economics.compare` at the write price, because the usage of a cold request shows the context as cache writes.

## 6. Read-level optimization

[readopt.py](readopt.py) treats every file read in a request as one history of that file. `ReadIndex.build`
walks the original messages once and records, per path, every whole-file read (a `Read` tool result, or an
`@`-attached "Result of calling the Read tool" block) as a `ReadRef`, and every whole-file `Write` by message
index. Ranged reads (`offset`/`limit`) and failed reads are excluded. The index holds the *live* block, so it can
tell whether a block has since been rewritten (`ref.intact`).

### 6.1 Repeat pointer (first sight)

`ReadIndex.anchor(ref)` scans the earlier reads of the same path from the nearest backwards:

* nearest earlier copy has different content → `None` (the file changed; the model must see the new copy);
* identical and still intact → that copy is the anchor;
* identical but itself already a pointer → keep scanning (its anchor is further back);
* identical but narrowed or converted → `None` (the model never saw the whole file).

If an anchor exists and the read is at least `dedupe_min_tokens` (150), the result becomes a short pointer
(`readopt.pointer`) when that is at most half the size. Decision key: `rdup | conv | tool_use id | original text`.
The sequence A, B, A is deliberately **not** deduplicated.

### 6.2 Stale outline (cold restart only)

`ReadIndex.stale_reason(ref, protect)` says a read is stale if, before the protected window, the same file was
read later with different intact content, or was overwritten by a later `Write`. A stale read of at least
`cold_read_min_tokens` (300) becomes an outline: size, first lines, definitions with line numbers (from
`elide.file_stub`), closing lines, and the reason. A repeat pointer in the same stale range becomes a tiny
marker, because its anchor goes with it. Requiring the replacement to lie before the protected window
guarantees nothing that points at a stale copy stays behind in the untouched tail.

## 7. The memo: schema and key catalogue

`Memo` ([optimizer.py](optimizer.py)) is one SQLite connection guarded by a lock (`:memory:` when `--memo ""`).

| Table | Columns | Purpose |
|---|---|---|
| `memo` | `key` PK, `text`, `before`, `after`, `ts` | Every decision. `text` NULL = leave the block as sent. |
| `convs` | `key` PK, `ts` | Conversations the proxy has seen (join detection). |
| `conv_state` | `key` PK, `ts`, `epoch` | Last request time and cold-restart count per conversation. |
| `tool_seen` / `tool_used` | `conv`, `name`, `ts` | Which tools were offered / called, for slimming unused MCP tools. |

Rows older than 7 days are removed on open from `memo`, `convs`, `conv_state`; tool statistics keep 60 days.
Keys are `sha256(MEMO_VERSION \0 part \0 part …)`:

| Key prefix | Parts | Decides |
|---|---|---|
| `conv` | first message without `cache_control` | conversation identity |
| `prefix-mode` | conv | `note` / `plain` / `tail:i` / `off` |
| `sys` | system text | cleaned system prompt |
| `slim` | conv, tool name | whether an MCP tool is slimmed |
| `tooldef` | `slim`\|`clean`, tool JSON | tool definition |
| `elide` | tool name, id, args JSON | abridged tool-call arguments |
| `narr`, `prompt2`, `tool` | text (and tool name) | narration, typed prompt, tool result |
| `read`, `tread`, `attach`, `pdf` | path/id + content | converted documents, targeted excerpts |
| `rdup` | conv, ident, original text | repeat-read pointer |
| `cold1` | conv, original key | second look at a block left alone while warm |
| `cold2` | conv, id (, index), current text | cold-pass aging and stale outlines |
| `cold-note` | conv | index of the user message carrying the note |

Bumping `MEMO_VERSION` invalidates every decision (and therefore every conversation's cached prefix once).

## 8. File by file

### Runtime

**[proxy.py](proxy.py)**: the HTTP server (`ThreadingHTTPServer` on `127.0.0.1`), described in section 3.
`UsageSniffer` extracts token usage from JSON or SSE without altering the stream. `forwarded_headers` drops
hop-by-hop headers and anything named in `Connection`. `dump_request` implements `--dump`. Module globals hold
the optimizer, ledger and log. Flags are defined in `main()`.

**[optimizer.py](optimizer.py)**: `Entry`, `Event`, `Result`, `Memo`, `Optimizer` (sections 4, 5, 7). Helpers:
`_conv_key`, `_hash`, `_cache_breakpoints` (indices of messages carrying `cache_control`), `_pdf_tokens`.

**[shrink.py](shrink.py)**: deterministic text shrinking.
* `count_tokens`: tiktoken `cl100k_base`, falling back to chars/4.
* `clean`: strip ANSI/OSC codes, keep the last frame of `\r` progress bars, trim trailing whitespace, collapse
  runs of 3+ identical lines into one plus a counter, collapse blank-line runs, minify JSON; with `fold=True`
  fold runs of 6+ lines that differ only in numbers or ids (`fold_similar`).
* `clean_prompt`: gentler, for text the user wrote: never collapses repeated lines, leaves fenced code verbatim.
* `compact_markdown`: strips table padding and shortens separator rows outside code fences.
* `protected(line)`: lines that must survive pruning (URLs, `file:line` locations, tracebacks, short lines with
  error keywords). `prose_share(text)`: share of characters in sentence-like lines; code, tables and data score low.
* `Compressor`: lazy LLMLingua-2 wrapper loaded on a background thread. `squeeze` splits text into protected
  lines and runs of prose, prunes only prose-heavy runs of at least 80 tokens, and returns `None` when the model
  is unavailable or the text exceeds 150k characters. Deterministic for a given model.

**[targeted.py](targeted.py)**: section 4.4. Pure functions, no I/O: `parse_read_result` (recognizes the
`cat -n` format and the trailing system-reminder), `split_units`, `terms`, `score_units`, `targeted`.

**[elide.py](elide.py)**: section 4.5 plus the abridgement note (`SYSTEM_NOTE`, `TAIL_NOTE`), `annotate_system`
and `has_note`.

**[prefix.py](prefix.py)**: the fixed prefix (about a fifth of a long session's context cost). `clean_tool`
drops `$schema`/`title` and cleans descriptions (lossless). `slim_tool` keeps names, parameter names/types/enums
and `required` but reduces the description to one sentence and drops parameter prose. Only `mcp__*` tools are
slimmable, only when offered from the conversation's first request, and only after `--slim-after` conversations
offered them without a single call. Built-in tools and server tools without an `input_schema` are never touched.

**[fileconv.py](fileconv.py)**: markitdown wrapper for `.pdf .docx .pptx .xlsx .xls .epub .msg`. Refuses files
over 40 MB, truncates output at 200k characters with a notice, and returns `None` for near-empty output (scanned
PDFs) so Claude's own reader is used. Catches `BaseException` because markitdown's conversion exceptions derive
from it.

**[readopt.py](readopt.py)**: section 6.

**[cold.py](cold.py)**: section 5: TTL detection, the cold test, the aged-tool allowlist and `aged_stub`.

### Observability and money

**[economics.py](economics.py)**: per-model list prices (`_PRICES`, longest-prefix match), `tokens_of` and
`usage_cost` (price a usage object), `pieces` (estimated tokens of tools+system and of each message),
`compare` (price a request twice with the same cache boundary: the real bill plus the *difference* between two
identically-built estimates, so an unchanged request yields exactly zero difference), and `Ledger` (thread-safe
per-conversation totals plus the optional JSONL file). Token sizes are estimated at 3.6 chars/token and scaled to
the billed total, so absolute dollars are the API's and only proportions are estimated.

**[log.py](log.py)**: plain stderr lines: `request_line` (what was rewritten, per lever), `money_line` (cost and
estimated saving of a finished request) and `cache_miss` (a warning when a request reads back under half of the
previous one's context).

**[breakdown.py](breakdown.py)**: `composition` counts tokens per category of a request (system, tools, user
text, assistant text, thinking, args by tool, results by tool); `record` makes one `--profile` row (before,
after, saved per lever).

**[report.py](report.py)**: reads a `--ledger` file and prints per-conversation or per-request money tables.

### Tests and metadata

**[test_aop.py](test_aop.py)** (75 tests) and **[test_cold.py](test_cold.py)** (30 tests): section 10.
**[requirements.txt](requirements.txt)**: `markitdown[all]`, `llmlingua`, `tiktoken`.
**[README.md](README.md)**: usage, flags, live results. **ledger.jsonl**: output of a `--ledger` run (one JSON object per
request). **\_\_pycache\_\_/**: Python bytecode cache; not source.

## 9. Measurement tooling

* **Live:** `python3 proxy.py --ledger ledger.jsonl`, then `python3 report.py ledger.jsonl`. Under `--dry-run`
  "billed" equals "without proxy" and "saved" is what the proxy would have saved.
* **Composition:** `--profile FILE` shows which slice of the context each lever attacks.

Judge by dollars, not tokens: a token cut from already-cached history saves the cheap read price, one cut from
the new tail saves the write price first, and output tokens are not reduced at all.

## 10. Testing

`python3 -m unittest -v test_aop test_cold` (105 tests, a few seconds, no network, no model; a deterministic
`FakeCompressor` stands in for LLMLingua).

| Test class | What it pins down |
|---|---|
| `OptimizerTests` | shrinking, **byte-identical prefix across requests**, replay without new work, joining freezes history, restart replay, fail-open |
| `TargetedReadTests`, `FoldTests`, `PromptTests`, `FileConversionTests`, `HelperTests` | the individual levers and helpers |
| `ElisionTests`, `PrefixTests`, `NarrationTests` | argument abridging and the note, system/tool rewriting and slimming, narration cleanup |
| `InstrumentationTests`, `EconomicsTests` | stats, profiling, usage sniffing, pricing, ledger and report |
| `FallbackTests` | the 400 → resend original → poison path |
| `ReadLevelTests` | repeat pointers: nearest copy decides, anchors, small/ranged/failed reads, replay, attachments |
| `ColdDetectionTests` | TTL detection, margin, epoch persistence across a reopened memo, aged stub contents |
| `ColdCompactionTests` | warm conversations untouched, compaction on a cold request, **replay of compacted bytes on every later request**, second restart, restart persistence, note placement, frozen joins reconsidered, protected window |
| `ColdReadLevelTests` | stale outlines, kept latest read, protected-window rule, pointers never dangle |

The central cache-safety assertion (used in both files): for requests n and n+1 of one conversation, the
`cache_control`-stripped messages of n are an exact prefix of those of n+1.

## 11. Failure handling

| Failure | Behavior |
|---|---|
| Unparseable body, no `messages`, any exception in the optimizer | Original bytes forwarded. |
| API 400 on a rewritten request | Original resent; conversation poisoned (never rewritten again); warning logged. |
| Upstream connection error | 502 to the client. |
| LLMLingua missing or still loading | `squeeze` returns `None`; lossless cleanup still applies. |
| markitdown missing or failing | Conversion returns `None`; the original block is kept. |
| Tokenizer missing | Falls back to chars/4. |
| Rewrite not worth it (too small, too little gained) | Stored as "leave alone" so it is not retried. |

## 12. Known limitations

* **Memo expiry.** Entries are deleted 7 days after they were last *used*: `Memo.get` refreshes the timestamp
  once it is a day old, so a block still being replayed does not age out mid-session. Entries for conversations
  that go idle for a week are still dropped.
* **Conversation identity.** A conversation is its first message, plus its second once one exists (`_identify`).
  Two sessions that open with identical text split at their first reply instead of sharing memo state. Sessions
  that stay identical through the second message still collide.
* **Signed thinking blocks.** Cold compaction never creates new edits at or before the newest assistant message
  with a `thinking`/`redacted_thinking` block (`--cold-through-thinking` lifts this). Whether editing that
  history really 400s is untested against the live API.
* **Savings are estimates.** The unmodified request is never billed; its cost is priced from token counts.
  `report.py --cache-audit` checks the other claim: that rewritten requests still read back the previous
  request's cache, compared with untouched ones. Quality of lossy reads and aged stubs is unmeasured.
* **Cold detection is based on what this proxy saw.** If requests reached the API another way (another client, a
  proxy restart on a different memo), the gap can be misjudged. Wrongly assuming cold costs one prefix rewrite;
  wrongly assuming warm only forgoes a saving. The 60-second margin and per-request TTL detection reduce this.
* **Cold compaction is lossy by design.** Aged output keeps head, tail and error/URL lines; stale reads keep an
  outline. The model is told to re-run or re-read for exact text. Whether answers get worse after a compaction has
  not been measured (see the README's quality signals). `--no-cold` disables it; `--cold-keep` and
  `--cold-min-tokens` tune it.
* **Read-level dedupe is unmeasured on real data.** The sample sessions contain 13 whole-file Reads and no
  repeats, so its dollar effect is 0.00% there; it is covered by unit tests only.
* **Not verified against the live API:** that a modified `tool_use` input in an assistant turn carrying signed
  thinking blocks is accepted. The 400 fallback bounds the damage to lost savings.
* **Replay is an estimate**, not a live A/B: one user, one model, a handful of sessions, token sizes estimated
  and scaled, list prices assumed, and the largest session dominates spend.
* **Joined conversations** keep their prefix as sent for good (system/tool cleanup needs a conversation seen from
  its first request); only the cold restart can reconsider their history.
* Concurrency: optimization is serialized by one lock; very heavy parallel traffic queues behind
  LLMLingua inference.
