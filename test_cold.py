"""Tests for read-level optimization (readopt.py) and cold-cache compaction (cold.py).

Run: python -m unittest -v test_cold
"""

import os
import tempfile
import time
import unittest

import cold
import readopt
from optimizer import Memo, Optimizer
from test_aop import (FakeCompressor, NOISY, _module, messages_of, numbered, request, strip_cache_control,
                      tool_turn)

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar "
         "papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
IDLE = 4000.0       # seconds: longer than the 1 h cache TTL plus its margin
POINTER = readopt.POINTER_START


def distinct_output(n=140, seed=1):
    """Log-like output whose lines never fold or compress away: every line is different words."""
    out, x = [], seed
    for _ in range(n):
        line = []
        for _ in range(10):
            x = (x * 1103515245 + 12345) % (2 ** 31)
            line.append(WORDS[x % len(WORDS)])
        out.append(" ".join(line))
    return "\n".join(out)


def read_turn(tid, path, text):
    """An exact Read exchange (tool_turn appends text to the result, which would break equality)."""
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": "Read", "input": {"file_path": path}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]},
    ]


def result_text(msgs, i):
    content = msgs[i]["content"][0]["content"]
    return content if isinstance(content, str) else content[0]["text"]


class ConversationIdentityTests(unittest.TestCase):
    def test_same_opening_different_reply_forks_the_conversation(self):
        opt = Optimizer(None, Memo())
        first = [{"role": "user", "content": "go"}]
        a = first + [{"role": "assistant", "content": [{"type": "text", "text": "A"}]}, {"role": "user", "content": "x"}]
        b = first + [{"role": "assistant", "content": [{"type": "text", "text": "B"}]}, {"role": "user", "content": "x"}]
        ida0, ida, idb = opt._identify(first), opt._identify(a), opt._identify(b)
        self.assertEqual(ida0, ida)
        self.assertNotEqual(ida, idb)
        self.assertEqual(opt._identify(a), ida)

    def test_memo_entry_in_use_does_not_expire(self):
        m = Memo()
        from optimizer import Entry
        m.put("k", Entry("v", 1, 1))
        m._db.execute("UPDATE memo SET ts = ts - 200000")
        self.assertEqual(m.get("k").text, "v")
        self.assertGreater(m._db.execute("SELECT ts FROM memo").fetchone()[0], time.time() - 100)


class CacheAuditTests(unittest.TestCase):
    def test_audit_separates_rewritten_from_untouched(self):
        import io, contextlib, report
        rows = [{"gap": 30, "prev_ctx": 9000, "read": 8900, "rewritten": True, "ts": 1, "conv": "c"},
                {"gap": 30, "prev_ctx": 9000, "read": 100, "rewritten": True, "ts": 2, "conv": "c"},
                {"gap": 30, "prev_ctx": 9000, "read": 9000, "rewritten": False, "ts": 3, "conv": "c"},
                {"gap": 4000, "prev_ctx": 9000, "read": 0, "rewritten": True, "ts": 4, "conv": "c"}]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            report.cache_audit(rows)
        lines = out.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("rewritten") and "2 follow-up" in lines[0] and "1 read" in lines[0].replace("    ", " "))
        self.assertIn("untouched", lines[1])
        self.assertTrue(any("miss" in ln and "(rewritten)" in ln for ln in lines))


class ThinkingFreezeTests(unittest.TestCase):
    def test_last_thinking_index(self):
        m = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"}]},
             {"role": "user", "content": "y"}]
        self.assertEqual(Optimizer._last_thinking(m), 1)
        self.assertEqual(Optimizer._last_thinking(m[:1] + m[2:]), -1)


START = [{"role": "user", "content": [{"type": "text", "text": "fix the build"}]}]
FILE_V1 = numbered(_module(14))
FILE_V2 = numbered(_module(15))


class ReadLevelTests(unittest.TestCase):
    def setUp(self):
        self.opt = Optimizer(None, Memo(), targeted_reads=False, cold_cache=False)

    def run_reads(self, texts):
        """Feed the reads one request at a time, as a real conversation does (a history handed over
        in one piece would look like a conversation joined midway, whose old blocks are frozen)."""
        history, result = list(START), None
        for i, text in enumerate(texts):
            history += read_turn(f"r{i}", "/p/app.py", text)
            result = self.opt.optimize(request(history))
        return history, result

    def test_identical_reread_becomes_pointer(self):
        history, result = self.run_reads([FILE_V1, FILE_V1])
        out = messages_of(result.body)
        self.assertEqual(result_text(out, 2), FILE_V1)              # the first copy is untouched
        self.assertTrue(result_text(out, 4).startswith(POINTER))
        self.assertGreater(result.saved, 1000)
        self.assertEqual([e.label for e in result.events], ["read·repeat"])

    def test_changed_file_is_not_deduped(self):
        _, result = self.run_reads([FILE_V1, FILE_V2])
        self.assertFalse(result.changed)

    def test_nearest_copy_decides(self):
        """A, B, A: the file changed in between, so the last copy must stay in full."""
        _, result = self.run_reads([FILE_V1, FILE_V2, FILE_V1])
        self.assertFalse(result.changed)

    def test_third_identical_copy_points_back_to_an_intact_one(self):
        _, result = self.run_reads([FILE_V1, FILE_V1, FILE_V1])
        out = messages_of(result.body)
        self.assertEqual(result_text(out, 2), FILE_V1)
        self.assertTrue(result_text(out, 4).startswith(POINTER))
        self.assertTrue(result_text(out, 6).startswith(POINTER))

    def test_small_files_and_ranged_reads_are_left_alone(self):
        _, result = self.run_reads(["     1\tx = 1", "     1\tx = 1"])
        self.assertFalse(result.changed)
        history = list(START) + read_turn("a", "/p/app.py", FILE_V1)
        history += read_turn("b", "/p/app.py", FILE_V1)
        history[3]["content"][0]["input"] = {"file_path": "/p/app.py", "offset": 1, "limit": 500}
        self.assertFalse(Optimizer(None, Memo(), targeted_reads=False).optimize(request(history)).changed)

    def test_failed_reads_are_left_alone(self):
        history = list(START) + read_turn("a", "/p/app.py", FILE_V1) + read_turn("b", "/p/app.py", FILE_V1)
        history[4]["content"][0]["is_error"] = True
        self.assertFalse(self.opt.optimize(request(history)).changed)

    def test_decision_is_replayed_and_prefix_is_stable(self):
        previous = None
        for n in range(1, 6):
            history, result = self.run_reads([FILE_V1] * n)
            current = strip_cache_control(messages_of(result.body))
            if previous is not None:
                self.assertEqual(previous, current[:len(previous)], f"prefix changed at read {n}")
            previous = current
        again = self.opt.optimize(request(history))
        self.assertEqual(again.fresh_saved, 0)

    def test_disabled(self):
        opt = Optimizer(None, Memo(), targeted_reads=False, cold_cache=False, dedupe_reads=False)
        history = list(START) + read_turn("a", "/p/app.py", FILE_V1) + read_turn("b", "/p/app.py", FILE_V1)
        self.assertFalse(opt.optimize(request(history)).changed)

    def test_attached_file_repeat(self):
        """An @-mentioned file inlined by Claude Code twice."""
        def attach(text):
            return [
                {"type": "text", "text": '<system-reminder>\nCalled the Read tool with the following input: '
                                         '{"file_path":"/p/app.py"}\n</system-reminder>'},
                {"type": "text", "text": f"{readopt.RESULT_HEAD}{text}{readopt.RESULT_TAIL}"},
            ]
        history = [{"role": "user", "content": attach(FILE_V1) + [{"type": "text", "text": "look"}]},
                   {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                   {"role": "user", "content": attach(FILE_V1) + [{"type": "text", "text": "again"}]}]
        out = messages_of(self.opt.optimize(request(history)).body)
        self.assertIn(FILE_V1, out[0]["content"][1]["text"])
        self.assertTrue(out[2]["content"][1]["text"].startswith(readopt.RESULT_HEAD + POINTER))

    def test_anchor_requires_the_earlier_copy_to_be_intact(self):
        history = list(START) + read_turn("a", "/p/app.py", FILE_V1) + read_turn("b", "/p/app.py", FILE_V1)
        idx = readopt.ReadIndex.build(history)
        first, second = idx.refs
        self.assertIs(idx.anchor(second), first)
        first.blk["content"] = "[narrowed]"
        self.assertIsNone(idx.anchor(second))


class ColdDetectionTests(unittest.TestCase):
    def test_ttl_is_read_from_the_request(self):
        self.assertEqual(cold.request_ttl({"messages": [{"content": [{"cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]}),
                         3600)
        self.assertEqual(cold.request_ttl({"messages": [{"content": [{"cache_control": {"type": "ephemeral"}}]}]}), 300)
        self.assertEqual(cold.request_ttl({}), 300)

    def test_margin(self):
        self.assertFalse(cold.is_cold(3600, 3600))
        self.assertFalse(cold.is_cold(3600 + cold.MARGIN, 3600))
        self.assertTrue(cold.is_cold(3600 + cold.MARGIN + 1, 3600))

    def test_memo_counts_epochs_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.sqlite")
            now = time.time()   # a reopened Memo forgets conversations older than a week
            memo = Memo(path)
            self.assertEqual(memo.touch("c", now, 3600), (False, 0))
            self.assertEqual(memo.touch("c", now + 100, 3600), (False, 0))
            self.assertEqual(memo.touch("c", now + 100 + IDLE, 3600), (True, 1))
            memo = Memo(path)
            self.assertEqual(memo.touch("c", now + 110 + IDLE, 3600), (False, 1))
            self.assertEqual(memo.touch("c", now + 100 + 2 * IDLE, 3600), (True, 2))
            self.assertEqual(memo.touch("c", now + 10 ** 6, 3600, detect=False), (False, 2))

    def test_aged_stub_keeps_head_tail_and_errors(self):
        lines = [f"line {i} " + "word " * 12 for i in range(100)]
        lines[50] = "ERROR: build failed in module core"
        lines[60] = "see https://example.com/log for details"
        stub = cold.aged_stub("Bash", "\n".join(lines))
        for kept in (lines[0], lines[5], lines[-1], lines[-4], lines[50], lines[60]):
            self.assertIn(kept, stub)
        self.assertNotIn(lines[30], stub)
        self.assertIn("[proxy:", stub)


class ColdCompactionTests(unittest.TestCase):
    def setUp(self):
        self.t = [time.time()]   # real time: a Memo reopened from disk drops state older than a week
        self.opt = Optimizer(None, Memo(), targeted_reads=False, cold_through_thinking=True, clock=lambda: self.t[0])

    def feed(self, history, gap=5.0):
        """Send every prefix of `history` that ends in a user message, like a live session."""
        result = None
        for n in range(1, len(history) + 1):
            if history[n - 1]["role"] == "user":
                result = self.send(history[:n], gap=gap)
        return result

    def convo(self, turns):
        history = list(START)
        for i in range(turns):
            history += tool_turn(i, "Bash", {"command": f"run {i}"}, distinct_output(140, i + 1))
        return history

    def send(self, history, gap=5.0, opt=None):
        self.t[0] += gap
        return (opt or self.opt).optimize(request(history))

    def test_default_leaves_history_before_signed_thinking_alone(self):
        safe = Optimizer(None, Memo(), targeted_reads=False, clock=lambda: self.t[0])
        history = self.convo(14)
        for n in range(1, len(history) + 1):
            if history[n - 1]["role"] == "user":
                self.send(history[:n], gap=5.0, opt=safe)
        self.t[0] += IDLE
        result = self.send(history, opt=safe)
        self.assertTrue(result.cold)
        self.assertFalse([e for e in result.events if e.label.startswith("cold:")])

    def test_warm_conversation_is_never_compacted(self):
        previous = None
        for n in range(1, 16):
            result = self.send(self.convo(n), gap=30)
            self.assertFalse(result.cold)
            self.assertFalse(result.changed)
            current = strip_cache_control(messages_of(result.body))
            if previous is not None:
                self.assertEqual(previous, current[:len(previous)])
            previous = current

    def test_cold_request_compacts_old_output_and_keeps_the_recent_turns(self):
        for n in range(1, 15):
            self.send(self.convo(n))
        history = self.convo(15)
        result = self.send(history, gap=IDLE)
        self.assertTrue(result.cold)
        out = messages_of(result.body)
        protect = len(out) - self.opt.cold_keep
        aged = [i for i in range(2, protect, 2) if "[proxy:" in result_text(out, i)]
        self.assertEqual(aged, list(range(2, protect, 2)))
        for i in range(protect, len(out) - 1):                   # the recent window is verbatim (the newest
            # message also carries the abridgement note, checked in its own test)
            self.assertEqual(strip_cache_control(out[i]), strip_cache_control(history[i]))
        self.assertTrue(all(e.label == "cold:Bash" for e in result.events if e.label.startswith("cold")))
        self.assertGreater(result.saved, 8000)

    def test_compacted_bytes_are_replayed_on_every_later_request(self):
        for n in range(1, 14):
            self.send(self.convo(n))
        cold_out = strip_cache_control(messages_of(self.send(self.convo(14), gap=IDLE).body))
        previous = cold_out
        for n in range(15, 20):
            result = self.send(self.convo(n), gap=20)
            self.assertFalse(result.cold)
            current = strip_cache_control(messages_of(result.body))
            self.assertEqual(previous, current[:len(previous)], f"prefix changed at turn {n}")
            self.assertEqual(result.fresh_saved, 0)
            previous = current

    def test_second_cold_restart_compacts_what_was_protected_before(self):
        for n in range(1, 14):
            self.send(self.convo(n))
        first = strip_cache_control(messages_of(self.send(self.convo(14), gap=IDLE).body))
        for n in range(15, 24):
            self.send(self.convo(n), gap=20)
        second = strip_cache_control(messages_of(self.send(self.convo(24), gap=IDLE).body))
        self.assertEqual(first[:len(first) - self.opt.cold_keep], second[:len(first) - self.opt.cold_keep])
        newly = len(first) - self.opt.cold_keep + 1
        self.assertIn("[proxy:", result_text(second, newly if newly % 2 == 0 else newly + 1))

    def test_disabled(self):
        opt = Optimizer(None, Memo(), targeted_reads=False, cold_cache=False, cold_through_thinking=True, clock=lambda: self.t[0])
        for n in range(1, 15):
            self.send(self.convo(n), opt=opt)
        result = self.send(self.convo(15), gap=IDLE, opt=opt)
        self.assertFalse(result.cold)
        self.assertFalse(result.changed)

    def test_errors_and_small_outputs_are_kept(self):
        history = self.convo(14)
        history[2]["content"][0]["is_error"] = True
        history[4]["content"][0]["content"] = "ok"
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        self.assertEqual(result_text(out, 2), history[2]["content"][0]["content"])
        self.assertEqual(result_text(out, 4), "ok")
        self.assertIn("[proxy:", result_text(out, 6))

    def test_unlisted_tools_are_not_aged(self):
        history = list(START)
        for i, name in enumerate(("Grep", "Glob", "Agent", "Edit", "WebSearch")):
            history += tool_turn(100 + i, name, {"pattern": "x"}, distinct_output(140, i + 1))
        history += self.convo(8)[1:]
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        for i in (2, 4, 6, 8, 10):
            self.assertNotIn("[proxy:", result_text(out, i))

    def test_restart_replays_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memo.sqlite")
            first = Optimizer(None, Memo(path), targeted_reads=False, cold_through_thinking=True, clock=lambda: self.t[0])
            for n in range(1, 14):
                self.send(self.convo(n), opt=first)
            cold_out = strip_cache_control(messages_of(self.send(self.convo(14), gap=IDLE, opt=first).body))
            second = Optimizer(None, Memo(path), targeted_reads=False, cold_through_thinking=True, clock=lambda: self.t[0])
            result = self.send(self.convo(15), gap=20, opt=second)
            self.assertFalse(result.cold)
            self.assertEqual(cold_out, strip_cache_control(messages_of(result.body))[:len(cold_out)])

    def test_abridgement_note_rides_in_the_newest_message_at_the_cold_restart(self):
        for n in range(1, 14):
            self.send(self.convo(n))
        result = self.send(self.convo(14), gap=IDLE)
        last = messages_of(result.body)[-1]["content"]
        self.assertTrue(any("Note: to save tokens" in b.get("text", "") for b in last))
        later = messages_of(self.send(self.convo(15), gap=20).body)
        notes = [i for i, m in enumerate(later) if m["role"] == "user" and isinstance(m["content"], list)
                 and any("Note: to save tokens" in b.get("text", "") for b in m["content"])]
        self.assertEqual(notes, [len(messages_of(result.body)) - 1])

    def test_blocks_frozen_when_joining_are_reconsidered_at_the_cold_restart(self):
        opt = Optimizer(FakeCompressor(), Memo(), cold_through_thinking=True, clock=lambda: self.t[0])

        def noisy(n):
            history = list(START)
            for i in range(n):
                history += tool_turn(i)
            return history
        joined = self.send(noisy(6), opt=opt)
        self.assertIn("\x1b[", result_text(messages_of(joined.body), 2))     # history of a joined conversation stays
        warm = self.send(noisy(7), gap=20, opt=opt)
        self.assertIn("\x1b[", result_text(messages_of(warm.body), 2))
        result = self.send(noisy(8), gap=IDLE, opt=opt)
        self.assertTrue(result.cold)
        self.assertNotIn("\x1b[", result_text(messages_of(result.body), 2))   # ... until the cache is cold anyway
        after = self.send(noisy(9), gap=20, opt=opt)
        self.assertEqual(result_text(messages_of(after.body), 2), result_text(messages_of(result.body), 2))


class ColdReadLevelTests(unittest.TestCase):
    def setUp(self):
        self.t = [time.time()]
        self.opt = Optimizer(None, Memo(), targeted_reads=False, cold_through_thinking=True, clock=lambda: self.t[0])

    def feed(self, history, gap=5.0):
        """Send every prefix of `history` that ends in a user message, like a live session."""
        result = None
        for n in range(1, len(history) + 1):
            if history[n - 1]["role"] == "user":
                result = self.send(history[:n], gap=gap)
        return result

    def history(self, tail=14):
        history = list(START)
        history += read_turn("r1", "/p/app.py", FILE_V1)
        history += tool_turn(50, "Edit", {"file_path": "/p/app.py"}, "edited")
        history += read_turn("r2", "/p/app.py", FILE_V2)
        history += read_turn("r3", "/p/other.py", FILE_V1)
        for i in range(tail):
            history += tool_turn(i, "Grep", {"pattern": "x"}, "match")
        return history

    def send(self, history, gap=5.0):
        self.t[0] += gap
        return self.opt.optimize(request(history))

    def test_old_read_of_a_changed_file_becomes_an_outline(self):
        history = self.history()
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        stale = result_text(out, 2)
        self.assertTrue(stale.startswith("[proxy: Read result of /p/app.py"))
        self.assertIn("read again later with different content", stale)
        self.assertIn("handler_", stale)                           # the outline names the definitions
        self.assertEqual(result_text(out, 6), FILE_V2)             # the current copy is whole
        self.assertEqual(result_text(out, 8), FILE_V1)             # another file's read is not touched

    def test_a_later_write_makes_a_read_stale(self):
        history = list(START) + read_turn("r1", "/p/app.py", FILE_V1)
        history += tool_turn(9, "Write", {"file_path": "/p/app.py", "content": "x"}, "written")
        for i in range(14):
            history += tool_turn(i, "Grep", {"pattern": "x"}, "match")
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        self.assertIn("overwritten later", result_text(out, 2))

    def test_the_latest_read_is_kept_when_nothing_replaces_it(self):
        history = list(START) + read_turn("r1", "/p/app.py", FILE_V1)
        for i in range(14):
            history += tool_turn(i, "Grep", {"pattern": "x"}, "match")
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        self.assertEqual(result_text(out, 2), FILE_V1)

    def test_replacement_inside_the_protected_window_does_not_count(self):
        history = list(START) + read_turn("r1", "/p/app.py", FILE_V1) + read_turn("r2", "/p/app.py", FILE_V2)
        self.send(history)
        out = messages_of(self.send(history, gap=IDLE).body)
        self.assertEqual(result_text(out, 2), FILE_V1)

    def test_repeat_pointers_never_dangle(self):
        """A, A(pointer), B: A and its pointer go together; nothing is left pointing at an outline."""
        history = list(START) + read_turn("r1", "/p/app.py", FILE_V1) + read_turn("r2", "/p/app.py", FILE_V1)
        history += read_turn("r3", "/p/app.py", FILE_V2)
        for i in range(14):
            history += tool_turn(i, "Grep", {"pattern": "x"}, "match")
        warm = messages_of(self.feed(history).body)
        self.assertTrue(result_text(warm, 4).startswith(POINTER))
        out = messages_of(self.send(history, gap=IDLE).body)
        self.assertIn("[proxy: Read result of /p/app.py", result_text(out, 2))
        self.assertFalse(result_text(out, 4).startswith(POINTER))
        self.assertIn("[proxy: repeated Read result", result_text(out, 4))
        self.assertEqual(result_text(out, 6), FILE_V2)

    def test_stale_outline_is_replayed_byte_for_byte(self):
        history = self.history()
        self.send(history)
        cold_out = strip_cache_control(messages_of(self.send(history, gap=IDLE).body))
        later = history + tool_turn(99, "Grep", {"pattern": "y"}, "hit")
        out = strip_cache_control(messages_of(self.send(later, gap=20).body))
        self.assertEqual(cold_out, out[:len(cold_out)])


if __name__ == "__main__":
    unittest.main()
