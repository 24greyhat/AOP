"""Tests for the cache-safe optimizer. Run: python -m unittest -v test_aop"""

import base64
import copy
import io
import json
import os
import re
import tempfile
import unittest
import zipfile

from shrink import clean, clean_prompt, count_tokens, fold_similar
import log
from optimizer import Memo, Optimizer
from targeted import targeted
from proxy import UsageSniffer


class FakeCompressor:
    """Deterministic stand-in for LLMLingua: drops every third word."""

    def squeeze(self, text):
        lines = []
        for line in text.split("\n"):
            words = line.split(" ")
            lines.append(" ".join(w for i, w in enumerate(words) if i % 3 != 2))
        return "\n".join(lines)


NOISY = "\n".join(
    f"\x1b[32mstep {i}\x1b[0m processing item {chr(97 + i % 26)}{chr(97 + i // 26)} with plenty of descriptive filler words here"
    for i in range(120)
)


def strip_cache_control(obj):
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(v) for v in obj]
    return obj


def tool_turn(i, name="Bash", args=None, output=NOISY):
    return [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"thought {i}", "signature": f"sig{i}"},
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": args or {"command": f"run {i}"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"{output}\nturn {i}"},
        ]},
    ]


def request(messages):
    """Mimic Claude Code: cache_control on the last block of the last two messages."""
    msgs = copy.deepcopy(messages)
    for m in msgs[-2:]:
        if isinstance(m["content"], list):
            m["content"][-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return json.dumps({"model": "m", "system": [{"type": "text", "text": "sys"}], "messages": msgs}).encode()


def messages_of(result_body):
    return json.loads(result_body)["messages"]


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.opt = Optimizer(FakeCompressor(), Memo())
        self.start = [{"role": "user", "content": [{"type": "text", "text": "fix the build"}]}]

    def conversation(self, turns):
        history = list(self.start)
        for i in range(turns):
            history += tool_turn(i)
        return history

    def test_shrinks_noisy_bash_output(self):
        result = self.opt.optimize(request(self.conversation(1)))
        self.assertTrue(result.changed)
        self.assertGreater(result.saved, 100)
        text = messages_of(result.body)[2]["content"][0]["content"]
        self.assertNotIn("\x1b[", text)
        self.assertTrue(text.startswith("[Condensed by proxy"))

    def test_prefix_is_byte_identical_across_requests(self):
        """The heart of cache safety: nothing already sent may ever change."""
        previous = None
        for turns in range(1, 7):
            result = self.opt.optimize(request(self.conversation(turns)))
            current = strip_cache_control(messages_of(result.body))
            if previous is not None:
                self.assertEqual(previous, current[: len(previous)], f"prefix changed at turn {turns}")
            previous = current

    def test_second_request_replays_without_new_work(self):
        first = self.opt.optimize(request(self.conversation(2)))
        again = self.opt.optimize(request(self.conversation(2)))
        self.assertEqual(first.body, again.body)
        self.assertEqual(again.fresh_saved, 0)
        self.assertEqual(first.saved, again.saved)

    def test_untouchable_content_is_left_alone(self):
        history = list(self.start)
        history += tool_turn(0, "Read", {"file_path": "/x/app.py"})
        history += tool_turn(1, "Edit", {"file_path": "/x/app.py"})
        history += tool_turn(2, "Grep", {"pattern": "a"})
        raw = request(history)
        result = self.opt.optimize(raw)
        self.assertFalse(result.changed)
        self.assertEqual(result.body, raw)  # byte-for-byte, not even re-serialized

    def test_thinking_and_assistant_text_survive(self):
        result = self.opt.optimize(request(self.conversation(1)))
        assistant = messages_of(result.body)[1]
        self.assertEqual(assistant["content"][0], {"type": "thinking", "thinking": "thought 0", "signature": "sig0"})

    def test_joining_mid_conversation_freezes_possibly_cached_history(self):
        history = self.conversation(4)
        result = self.opt.optimize(request(history))
        out = messages_of(result.body)
        # history before the previous breakpoint stays as received; only the newest turn shrinks
        for i in (2, 4, 6):
            self.assertEqual(out[i]["content"][0]["content"], history[i]["content"][0]["content"])
        self.assertTrue(out[-1]["content"][0]["content"].startswith("[Condensed"))
        # and those frozen blocks stay frozen on the next request
        later = messages_of(self.opt.optimize(request(self.conversation(5))).body)
        for i in (2, 4, 6):
            self.assertEqual(later[i]["content"][0]["content"], history[i]["content"][0]["content"])

    def test_restart_replays_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memo.sqlite")
            first = Optimizer(FakeCompressor(), Memo(path)).optimize(request(self.conversation(3)))
            # a restarted proxy with a *different* compressor must still send the same bytes
            second = Optimizer(None, Memo(path)).optimize(request(self.conversation(3)))
            self.assertEqual(first.body, second.body)

    def test_fails_open(self):
        for junk in (b"not json", b"[]", b'{"messages": 5}', b'{"messages": []}'):
            self.assertEqual(self.opt.optimize(junk).body, junk)

    def test_error_results_and_small_outputs_are_kept(self):
        history = list(self.start) + tool_turn(0, output="ok")
        history[2]["content"][0]["is_error"] = True
        history += tool_turn(1, output=NOISY)
        history[4]["content"][0]["is_error"] = True
        self.assertFalse(self.opt.optimize(request(history)).changed)


def _module(n=12):
    """A source file with n functions, each ~15 lines, plus a class."""
    parts = ["import os", "import json", "", ""]
    for i in range(n):
        parts += [f"def handler_{i}(request, retries={i}):", f'    """Handle kind {i} requests."""']
        parts += [f"    value_{i}_{k} = request.get('field_{i}_{k}', {k}) * {k + 1}  # step {k}" for k in range(12)]
        parts += [f"    return value_{i}_0", "", ""]
    return "\n".join(parts)


def numbered(src):
    return "\n".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(src.split("\n")))


class TargetedReadTests(unittest.TestCase):
    SRC = _module()

    def read(self, *, path="/x/api.py", args=None, content=None, prompt="fix the bug in handler_7"):
        opt = Optimizer(None, Memo())
        history = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        history += tool_turn(0, "Read", args or {"file_path": path}, output=content or numbered(self.SRC))
        history[2]["content"][0]["content"] = content or numbered(self.SRC)
        return opt.optimize(request(history)), history

    def test_keeps_relevant_function_verbatim_and_elides_the_rest(self):
        result, history = self.read()
        out = messages_of(result.body)[2]["content"][0]["content"]
        original = history[2]["content"][0]["content"].split("\n")
        self.assertTrue(out.startswith("[Condensed by proxy"))
        self.assertIn("def handler_7(request, retries=7):", out)
        self.assertNotIn("def handler_2(", out.split("elided")[0])  # others are only named in markers
        self.assertIn("def handler_2(request, retries=2):", out)    # ... and the outline still lists them
        self.assertLess(count_tokens(out), count_tokens("\n".join(original)) * 0.6)
        verbatim = set(original)
        for line in out.split("\n")[1:]:
            self.assertTrue(line.startswith("[…") or line in verbatim, line)  # Edit-safe: no altered lines

    def test_markers_give_a_working_read_range(self):
        out = messages_of(self.read()[0].body)[2]["content"][0]["content"]
        marker = next(line for line in out.split("\n") if line.startswith("[…") and "handler_2(" in line)
        first, count = [int(n) for n in re.search(r"offset=(\d+) limit=(\d+)", marker).groups()]
        lines = self.SRC.split("\n")
        self.assertTrue(lines[first - 1].startswith("def handler_") or lines[first - 1].startswith("def"))
        self.assertTrue(lines[first - 1 + count - 1] == "" or lines[first - 1 + count - 1].startswith("    "))

    def test_whole_file_when_nothing_matches_or_file_is_small_or_not_code(self):
        raw = self.read(prompt="what is the weather like")[0]
        self.assertFalse(raw.changed)
        small = numbered("\n".join(self.SRC.split("\n")[:40]))
        self.assertFalse(self.read(content=small)[0].changed)
        self.assertFalse(self.read(path="/x/notes.md")[0].changed)

    def test_ranged_reads_and_failed_parses_are_never_touched(self):
        self.assertFalse(self.read(args={"file_path": "/x/api.py", "offset": 10, "limit": 80})[0].changed)
        self.assertFalse(self.read(args={"file_path": "/x/api.py", "offset": 1})[0].changed)
        self.assertFalse(self.read(content=self.SRC)[0].changed)  # no line numbers -> unrecognised

    def test_replay_is_byte_identical_and_flag_disables(self):
        opt = Optimizer(None, Memo())
        history = [{"role": "user", "content": [{"type": "text", "text": "fix the bug in handler_7"}]}]
        history += tool_turn(0, "Read", {"file_path": "/x/api.py"}, output=numbered(self.SRC))
        history[2]["content"][0]["content"] = numbered(self.SRC)
        first = opt.optimize(request(history))
        history += [{"role": "assistant", "content": [{"type": "text", "text": "Looking at handler_2 now"}]},
                    {"role": "user", "content": [{"type": "text", "text": "ok now handler_2 please"}]}]
        later = opt.optimize(request(history))  # the context changed, the replayed block must not
        self.assertEqual(strip_cache_control(messages_of(first.body)[2]), strip_cache_control(messages_of(later.body)[2]))
        off = Optimizer(None, Memo(), targeted_reads=False)
        self.assertFalse(off.optimize(request(history[:3])).changed)

    def test_non_python_files_use_the_generic_splitter(self):
        funcs = []
        for i in range(10):
            funcs += [f"function handler{i}(req) {{"] + [f"  const v{k} = req.field{i}_{k} * {k};" for k in range(12)] + ["}", ""]
        src = "\n".join(['import x from "x";', ""] + funcs)
        result, _ = self.read(path="/x/api.js", content=numbered(src), prompt="handler3 has a bug")
        out = messages_of(result.body)[2]["content"][0]["content"]
        self.assertIn("function handler3(req) {", out.split("\n[…")[0] + out)
        self.assertIn("elided by proxy", out)
        self.assertLess(count_tokens(out), count_tokens(numbered(src)) * 0.6)


class FoldTests(unittest.TestCase):
    LOG = "\n".join(f"[{i}/300] downloading pkg-{i % 7} ... done" for i in range(1, 301))

    def test_numbered_log_runs_are_folded_keeping_first_and_last(self):
        out = fold_similar("start\n" + self.LOG + "\nfinished")
        lines = out.split("\n")
        self.assertEqual(lines[0], "start")
        self.assertEqual(lines[1], "[1/300] downloading pkg-1 ... done")
        self.assertIn("298 similar lines omitted", lines[2])
        self.assertEqual(lines[3:], ["[300/300] downloading pkg-6 ... done", "finished"])

    def test_errors_break_runs_and_stay_verbatim(self):
        log = [f"worker {i} handled request {i * 3} ok" for i in range(10)]
        log.insert(5, "worker 5 failed with error code 12")
        out = fold_similar("\n".join(log))
        self.assertIn("worker 5 failed with error code 12", out)
        self.assertEqual(out.count("similar lines omitted"), 0)  # both halves are under the run minimum

    def test_short_runs_tables_and_plain_numbers_are_left_alone(self):
        short = "\n".join(f"retry {i} of 9 pending" for i in range(5))
        table = "\n".join(f"| row {i} | {i * 2} | label |" for i in range(20))
        numbers = "\n".join(f"{i},{i * 2},{i * 3}" for i in range(20))
        for text in (short, table, numbers):
            self.assertEqual(fold_similar(text), text)

    def test_folding_is_off_by_default_and_skips_json(self):
        self.assertEqual(clean(self.LOG), self.LOG)
        pretty = json.dumps([{"count": i} for i in range(30)], indent=2)
        self.assertEqual(clean(pretty, fold=True), json.dumps([{"count": i} for i in range(30)], separators=(",", ":")))

    def test_only_command_output_is_folded(self):
        opt = Optimizer(None, Memo())
        msgs = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        for name in ("Bash", "mcp__db__query"):
            history = msgs + tool_turn(0, name, output=FoldTests.LOG)
            body = opt.optimize(request(history)).body.decode()
            self.assertEqual("similar lines omitted" in body, name == "Bash", name)


class PromptTests(unittest.TestCase):
    PASTE = "\n".join(f"\x1b[31mline {i}\x1b[0m   with some pasted log text that is long enough" for i in range(60))
    CODE = "```\nx = 1   \n\n\n\ny = 2\n```"

    def setUp(self):
        self.opt = Optimizer(FakeCompressor(), Memo())

    def ask(self, text, opt=None):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return (opt or self.opt).optimize(request(msgs))

    def test_pasted_noise_is_cleaned_losslessly(self):
        result = self.ask("why does this fail?\n" + self.PASTE)
        out = messages_of(result.body)[0]["content"][0]["text"]
        self.assertTrue(out.startswith("why does this fail?\nline 0   with"))  # instruction untouched
        self.assertNotIn("\x1b", out)
        self.assertNotIn("Condensed", out)  # never lossy
        self.assertGreater(result.saved, 0)

    def test_short_prompts_and_code_fences_are_untouched(self):
        raw = request([{"role": "user", "content": [{"type": "text", "text": "fix the build"}]}])
        self.assertEqual(self.opt.optimize(raw).body, raw)
        self.assertEqual(clean_prompt(self.CODE), self.CODE)
        self.assertNotIn("\x1b", clean_prompt("```\n\x1b[1mbold\x1b[0m\n```"))

    def test_prompt_replay_is_stable_and_system_reminders_skipped(self):
        reminder = {"type": "text", "text": "<system-reminder>\n" + self.PASTE + "\n</system-reminder>"}
        msgs = [{"role": "user", "content": [reminder, {"type": "text", "text": "q\n" + self.PASTE}]}]
        first = self.opt.optimize(request(msgs))
        again = self.opt.optimize(request(msgs))
        self.assertEqual(first.body, again.body)
        out = messages_of(first.body)[0]["content"]
        self.assertEqual(out[0]["text"], reminder["text"])
        self.assertNotIn("\x1b", out[1]["text"])

    ARTICLE = " ".join(f"Sentence {i} explains that the quarterly report shows a steady and gradual improvement overall." for i in range(25))

    def test_only_big_pasted_prose_is_pruned(self):
        ask, tail = "Summarize the article below in one line please.", "Thanks, keep it short and neutral."
        small = "A short middle note that is plain prose but far below the size limit for pruning at all."
        code = "```\n" + self.ARTICLE + "\n```"
        text = "\n\n".join([ask, self.ARTICLE, small, code, self.ARTICLE.replace("Sentence", "Item"), tail])
        out = messages_of(self.ask(text).body)[0]["content"][0]["text"]
        parts = out.split("\n\n")
        self.assertEqual(parts[0], ask)       # opening paragraph verbatim
        self.assertEqual(parts[-1], tail)     # closing paragraph verbatim
        self.assertEqual(parts[2], small)     # short paragraph verbatim
        self.assertEqual(parts[3], code)      # fenced block verbatim
        self.assertTrue(parts[1].startswith("[condensed by proxy"))
        self.assertTrue(parts[4].startswith("[condensed by proxy"))
        self.assertLess(count_tokens(out), count_tokens(text) * 0.9)

    def test_pruned_prompt_replays_identically_and_lossless_flag_disables_it(self):
        text = "Please review.\n\n" + self.ARTICLE + "\n\nThanks."
        first, again = self.ask(text), self.ask(text)
        self.assertEqual(first.body, again.body)
        self.assertIn("[condensed by proxy", first.body.decode())
        lossless = self.ask(text, Optimizer(FakeCompressor(), Memo(), prune_prompts=False))
        self.assertNotIn("condensed", lossless.body.decode())

    def test_keep_prompts_flag(self):
        raw = request([{"role": "user", "content": [{"type": "text", "text": self.PASTE}]}])
        self.assertEqual(Optimizer(None, Memo(), clean_prompts=False).optimize(raw).body, raw)


class FileConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.opt = Optimizer(None, Memo())
        self.words = " ".join(f"Quarterly revenue grew in region {i} thanks to new customers." for i in range(30))

    def make_docx(self):
        path = os.path.join(self.tmp.name, "report.docx")
        body = "".join(f"<w:p><w:r><w:t>{s}.</w:t></w:r></w:p>" for s in self.words.split(".") if s.strip())
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("[Content_Types].xml",
                       '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                       '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                       '<Default Extension="xml" ContentType="application/xml"/>'
                       '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
            z.writestr("_rels/.rels",
                       '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                       '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
            z.writestr("word/document.xml",
                       '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                       f"<w:body>{body}</w:body></w:document>")
        return path

    def make_pdf(self, pages=3):
        objs, kids = [], []
        for p in range(pages):
            kids.append(f"{4 + 2 * p} 0 R")
        objs.append("<< /Type /Catalog /Pages 2 0 R >>")
        objs.append(f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {pages} >>")
        objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        for p in range(pages):
            lines = "".join(f"BT /F1 11 Tf 40 {760 - 14 * i} Td ({self.words[i * 70:(i + 1) * 70]}) Tj ET\n" for i in range(20))
            objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * p} 0 R "
                        f"/Resources << /Font << /F1 3 0 R >> >> >>")
            objs.append(f"<< /Length {len(lines)} >>\nstream\n{lines}endstream")
        out, offsets = bytearray(b"%PDF-1.4\n"), []
        for n, obj in enumerate(objs, 1):
            offsets.append(len(out))
            out += f"{n} 0 obj\n{obj}\nendobj\n".encode()
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
        return bytes(out)

    def test_read_tool_on_docx_is_replaced_by_markdown(self):
        path = self.make_docx()
        history = [{"role": "user", "content": [{"type": "text", "text": "summarise"}]}]
        history += tool_turn(0, "Read", {"file_path": path}, output="Error: cannot read binary file")
        history[2]["content"][0]["is_error"] = True
        result = self.opt.optimize(request(history))
        block = messages_of(result.body)[2]["content"][0]
        self.assertIn("converted to Markdown", block["content"])
        self.assertIn("Quarterly revenue grew in region 7", block["content"])
        self.assertNotIn("is_error", block)
        self.assertTrue(result.events[0].converted)

    def test_pdf_document_block_becomes_text(self):
        pdf = self.make_pdf()
        doc = {"type": "document", "title": "q3.pdf",
               "source": {"type": "base64", "media_type": "application/pdf", "data": base64.b64encode(pdf).decode()}}
        history = [{"role": "user", "content": [doc, {"type": "text", "text": "what grew?"}]}]
        result = self.opt.optimize(request(history))
        first = messages_of(result.body)[0]["content"][0]
        self.assertEqual(first["type"], "text")
        self.assertIn("q3.pdf converted to Markdown", first["text"])
        self.assertGreater(result.saved, 1000)  # 3 pages * ~2k tokens vs. a few hundred of text

    def test_attached_file_block_is_converted_in_place(self):
        path = self.make_docx()
        called = f'<system-reminder>\nCalled the Read tool with the following input: {json.dumps({"file_path": path})}\n</system-reminder>'
        history = [{"role": "user", "content": [
            {"type": "text", "text": called},
            {"type": "text", "text": "<system-reminder>\nResult of calling the Read tool:\nPK\x03\x04 binary junk\n</system-reminder>\n"},
            {"type": "text", "text": "please summarise"},
        ]}]
        out = messages_of(self.opt.optimize(request(history)).body)[0]["content"]
        self.assertTrue(out[1]["text"].startswith("<system-reminder>\nResult of calling the Read tool:\n[report.docx converted"))
        self.assertTrue(out[1]["text"].rstrip().endswith("</system-reminder>"))
        self.assertEqual(out[2]["text"], "please summarise")

    def test_unreadable_pdf_falls_back_to_original(self):
        doc = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                              "data": base64.b64encode(b"%PDF-1.4 garbage").decode()}}
        raw = request([{"role": "user", "content": [doc]}])
        self.assertEqual(self.opt.optimize(raw).body, raw)


class HelperTests(unittest.TestCase):
    def test_clean(self):
        text = "\x1b[31mred\x1b[0m\nprogress 10%\rprogress 100%\n\n\n\n" + "dup\n" * 5 + "tail   "
        out = clean(text)
        self.assertNotIn("\x1b", out)
        self.assertIn("progress 100%", out)
        self.assertNotIn("progress 10%", out)
        self.assertIn("[previous line repeated 4 more times]", out)
        self.assertNotIn("\n\n\n", out)

    def test_clean_minifies_json(self):
        self.assertEqual(clean('{\n  "a": [1, 2],\n  "b": "x"\n}'), '{"a":[1,2],"b":"x"}')

    def test_usage_sniffer_reads_sse_split_across_chunks(self):
        event = ('event: message_start\ndata: {"type":"message_start","message":{"usage":'
                 '{"input_tokens":5,"cache_read_input_tokens":900,"cache_creation":{"a":1}}}}\n\n').encode()
        sniffer = UsageSniffer(sse=True)
        for i in range(0, len(event), 17):
            sniffer.feed(event[i:i + 17])
        self.assertEqual(sniffer.finish()["cache_read_input_tokens"], 900)

    def test_usage_sniffer_reads_plain_json(self):
        sniffer = UsageSniffer(sse=False)
        sniffer.feed(b'{"usage": {"input_tokens": 7}}')
        self.assertEqual(sniffer.finish(), {"input_tokens": 7})

    def test_cache_miss_warning(self):
        self.assertIsNone(log.cache_miss({"cache_read_input_tokens": 9000, "input_tokens": 100}, "c"))
        self.assertIsNone(log.cache_miss({"cache_read_input_tokens": 9500, "input_tokens": 200}, "c"))
        self.assertIn("cache read 0", log.cache_miss({"cache_creation_input_tokens": 9800, "input_tokens": 50}, "c"))

    def test_prose_detector_separates_prose_from_code(self):
        from shrink import prose_share
        prose = ("The new design splits the pipeline into three stages and the first stage builds only "
                 "the images whose source directories changed.\n") * 3
        code = "def main():\n    parser = argparse.ArgumentParser(description=__doc__)\n    args = parser.parse_args()\n"
        table = "| Region | Q1 | Q2 |\n|---|---|---|\n| North | 3 | 4 |\n"
        self.assertGreater(prose_share(prose), 0.9)
        self.assertLess(prose_share(code), 0.3)
        self.assertLess(prose_share(table), 0.3)

    def test_keyword_protects_log_lines_but_not_whole_paragraphs(self):
        from shrink import protected
        self.assertTrue(protected("tests/test_billing.py::test_total FAILED"))
        self.assertTrue(protected("see https://example.com/docs for details " + "x " * 400))
        self.assertFalse(protected("the build failed once because of staging " + "word " * 80))

    def test_tokens_counted(self):
        self.assertGreater(count_tokens("hello world, this is a test"), 3)


# ---------------------------------------------------------------- levers 1, 2, 4

import breakdown
import elide
import prefix

TOOLS = [{"name": n, "description": f"{n} tool.", "input_schema": {"type": "object", "properties": {}}}
         for n in ("Bash", "Write", "Edit", "Read")]
SYSTEM = "You are Claude Code.   \n\n\n\nBe brief.  \n" + "\n\n\n\nFollow the rules carefully.    \n" * 12


def full_request(messages, tools=None, system=SYSTEM):
    """A request like Claude Code's: system, tools, cache_control on the last two messages."""
    msgs = copy.deepcopy(messages)
    for m in msgs[-2:]:
        if isinstance(m["content"], list):
            m["content"][-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    return json.dumps({"model": "m", "system": [{"type": "text", "text": system}],
                       "tools": copy.deepcopy(TOOLS if tools is None else tools), "messages": msgs}).encode()


def parsed(result):
    return json.loads(result.body)


def strip_cc(obj):
    return strip_cache_control(obj)


BIG_FILE = _module(14)
SCRIPT = "python3 - <<'EOF'\n" + "\n".join(f"print('line {i} of a long analysis script', {i} * 3)" for i in range(60)) + "\nEOF"


class ElisionTests(unittest.TestCase):
    def setUp(self):
        self.opt = Optimizer(None, Memo())
        self.start = [{"role": "user", "content": [{"type": "text", "text": "build the api"}]}]
        self.opt.optimize(full_request(self.start))  # the conversation's first request

    def history(self, turns, name="Write", args=None, output="File created successfully"):
        h = list(self.start)
        for i in range(turns):
            h += tool_turn(i, name, args or {"file_path": f"/x/app{i}.py", "content": BIG_FILE}, output=output)
        return h

    def test_executed_write_is_abridged_with_outline_on_first_sight(self):
        out = parsed(self.opt.optimize(full_request(self.history(1))))
        args = out["messages"][1]["content"][1]["input"]
        self.assertEqual(args["file_path"], "/x/app0.py")
        self.assertTrue(args["content"].startswith("[proxy:"))
        self.assertIn("Outline:", args["content"])
        self.assertIn("def handler_9(", args["content"])
        self.assertLess(count_tokens(args["content"]), count_tokens(BIG_FILE) * 0.3)
        self.assertIn(elide.NOTE_START, out["system"][0]["text"])

    def test_nothing_sent_earlier_ever_changes(self):
        """Cache safety across the whole request sequence, including the cache_control marker
        moving off the first message after request 1."""
        previous = None
        for turns in range(0, 7):
            out = parsed(self.opt.optimize(full_request(self.history(turns))))
            current = (strip_cc(out["system"]), strip_cc(out["tools"]), strip_cc(out["messages"]))
            if previous is not None:
                self.assertEqual(previous[0], current[0], f"system changed at turn {turns}")
                self.assertEqual(previous[1], current[1], f"tools changed at turn {turns}")
                self.assertEqual(previous[2], current[2][: len(previous[2])], f"history changed at turn {turns}")
            previous = current
        self.assertIn(elide.NOTE_START, previous[0][0]["text"])

    def test_replay_is_byte_identical_and_survives_restart(self):
        first = self.opt.optimize(full_request(self.history(3)))
        again = self.opt.optimize(full_request(self.history(3)))
        self.assertEqual(first.body, again.body)
        self.assertEqual(again.fresh_saved, 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.sqlite")
            a = Optimizer(None, Memo(path))
            a.optimize(full_request(self.start))
            one = a.optimize(full_request(self.history(3)))
            two = Optimizer(None, Memo(path)).optimize(full_request(self.history(3)))
            self.assertEqual(one.body, two.body)

    def test_failed_and_small_calls_are_kept(self):
        h = self.history(1)
        h[2]["content"][0]["is_error"] = True            # a failed Write may be retried verbatim
        h += tool_turn(1, "Write", {"file_path": "/x/s.py", "content": "print('hi')"}, output="ok")
        out = parsed(self.opt.optimize(full_request(h)))
        self.assertEqual(out["messages"][1]["content"][1]["input"]["content"], BIG_FILE)
        self.assertEqual(out["messages"][3]["content"][1]["input"]["content"], "print('hi')")

    def test_bash_heredoc_keeps_the_command_line_and_drops_the_body(self):
        out = parsed(self.opt.optimize(full_request(self.history(1, "Bash", {"command": SCRIPT}, output="done"))))
        cmd = out["messages"][1]["content"][1]["input"]["command"]
        self.assertTrue(cmd.startswith("python3 - <<'EOF'\n"))
        self.assertTrue(cmd.endswith("\nEOF"))
        self.assertIn("more script lines omitted", cmd)
        self.assertIn("line 0 of a long analysis script", cmd)
        self.assertNotIn("line 40 of a long analysis script", cmd)

    def test_short_commands_reads_and_greps_are_untouched(self):
        h = list(self.start) + tool_turn(0, "Bash", {"command": "ls -la"}, output="ok")
        h += tool_turn(1, "Read", {"file_path": "/x/a.py"}, output="x")
        raw = full_request(h)
        out = parsed(self.opt.optimize(raw))
        self.assertEqual(strip_cc(out["messages"]), strip_cc(json.loads(raw)["messages"]))

    def test_joining_midway_abridges_only_new_calls_and_carries_the_note_in_the_tail(self):
        """The usual case: a long session that predates the proxy. Its cached history and prefix
        must stay as sent; only the newest turn is still free to change."""
        opt = Optimizer(None, Memo())
        out = parsed(opt.optimize(full_request(self.history(3))))
        for i in (1, 3):                                   # frozen: may already be cached upstream
            self.assertEqual(out["messages"][i]["content"][1]["input"]["content"], BIG_FILE)
        self.assertTrue(out["messages"][5]["content"][1]["input"]["content"].startswith("[proxy:"))
        self.assertEqual(out["system"][0]["text"], SYSTEM)  # prefix untouched
        self.assertEqual(out["tools"], TOOLS)
        last = out["messages"][-1]["content"]
        self.assertEqual(last[-1]["text"], elide.TAIL_NOTE)
        self.assertIn("cache_control", last[-1])            # breakpoint stays on the last block
        self.assertNotIn("cache_control", last[-2])

    def test_a_joined_conversation_never_changes_what_was_already_sent(self):
        opt = Optimizer(None, Memo())
        previous = None
        for turns in range(3, 8):
            out = parsed(opt.optimize(full_request(self.history(turns))))
            current = strip_cc(out["messages"])
            if previous is not None:
                self.assertEqual(previous, current[: len(previous)], f"history changed at turn {turns}")
            previous = current
        abridged = [m["content"][1]["input"]["content"].startswith("[proxy:")
                    for m in previous if m["role"] == "assistant"]
        self.assertEqual(abridged, [False, False, True, True, True, True, True][: len(abridged)])
        self.assertEqual(sum(
            1 for m in previous for b in m["content"] if b.get("text") == elide.TAIL_NOTE), 1)

    def test_conversation_known_to_an_older_version_is_adopted_the_same_way(self):
        from optimizer import _conv_key
        opt = Optimizer(None, Memo())
        h = self.history(3)
        opt.memo.known_conv(_conv_key(h))                  # registered earlier, no mode recorded
        out = parsed(opt.optimize(full_request(h)))
        self.assertTrue(out["messages"][5]["content"][1]["input"]["content"].startswith("[proxy:"))
        self.assertEqual(out["messages"][1]["content"][1]["input"]["content"], BIG_FILE)

    def test_joining_with_the_flag_off_changes_nothing(self):
        opt = Optimizer(None, Memo(), elide_args=False)
        raw = full_request(self.history(3))
        result = opt.optimize(raw)
        self.assertFalse(result.changed)
        self.assertEqual(result.body, raw)

    def test_flag_off_leaves_args_alone_and_adds_no_note(self):
        opt = Optimizer(None, Memo(), elide_args=False)
        opt.optimize(full_request(self.start))
        out = parsed(opt.optimize(full_request(self.history(1))))
        self.assertEqual(out["messages"][1]["content"][1]["input"]["content"], BIG_FILE)
        self.assertNotIn(elide.NOTE_START, out["system"][0]["text"])

    def test_poisoned_conversation_passes_through_untouched(self):
        raw = full_request(self.history(2))
        result = self.opt.optimize(raw)
        self.assertTrue(result.changed)
        self.opt.poison(result.conv)
        again = self.opt.optimize(raw)
        self.assertFalse(again.changed)
        self.assertEqual(again.body, raw)

    def test_stub_is_smaller_and_names_where_the_file_is(self):
        stub = elide.file_stub("/x/app.py", BIG_FILE)
        self.assertIn("Read it for exact text", stub)
        self.assertIn("Outline:", stub)
        small = elide.elide_input("Write", {"file_path": "/x/a.py", "content": "x = 1\n" * 20})
        self.assertIsNone(small)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.start = [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]

    def tool(self, name, desc="Search the shared drive for files matching a query. Longer explanation follows here for the model."):
        return {"name": name, "description": desc, "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "title": "Args",
            "properties": {"title": {"type": "string", "title": "Title", "description": "The title to use."},
                           "mode": {"type": "string", "enum": ["a", "b"], "description": "Which mode."}},
            "required": ["title"]}}

    def first(self, opt, tools):
        return parsed(opt.optimize(full_request(self.start, tools)))

    def test_schema_annotations_dropped_but_property_named_title_kept(self):
        out = self.first(Optimizer(None, Memo()), [self.tool("Bash")])["tools"][0]
        schema = out["input_schema"]
        self.assertNotIn("$schema", schema)
        self.assertNotIn("title", schema)
        self.assertIn("title", schema["properties"])             # a parameter, not an annotation
        self.assertNotIn("title", schema["properties"]["title"])
        self.assertEqual(schema["properties"]["mode"]["enum"], ["a", "b"])
        self.assertEqual(schema["required"], ["title"])
        self.assertIn("The title to use.", json.dumps(out))      # prose kept for built-ins

    def test_system_prompt_gets_lossless_cleanup(self):
        out = self.first(Optimizer(None, Memo()), TOOLS)
        text = out["system"][0]["text"]
        self.assertTrue(text.startswith("You are Claude Code.\n\nBe brief.\n\nFollow the rules carefully.\n\nFollow"))
        self.assertNotIn("  \n", text)

    def test_unused_mcp_tool_is_slimmed_after_enough_conversations(self):
        opt = Optimizer(None, Memo(), slim_after=5)
        for i in range(5):
            opt.memo.note_tools(f"c{i}", ["mcp__x__idle", "mcp__x__busy", "Bash"], [])
        opt.memo.note_tools("cu", [], ["mcp__x__busy"])
        tools = [self.tool("Bash"), self.tool("mcp__x__idle"), self.tool("mcp__x__busy")]
        out = {t["name"]: t for t in self.first(opt, tools)["tools"]}
        idle = out["mcp__x__idle"]
        self.assertTrue(idle["description"].endswith("(abridged)"))
        self.assertNotIn("Longer explanation", idle["description"])
        props = idle["input_schema"]["properties"]
        self.assertEqual(sorted(props), ["mode", "title"])        # names and types stay
        self.assertEqual(props["mode"]["enum"], ["a", "b"])
        self.assertNotIn("description", props["mode"])
        self.assertEqual(idle["input_schema"]["required"], ["title"])
        self.assertIn("Longer explanation", out["mcp__x__busy"]["description"])   # it gets used
        self.assertIn("Longer explanation", out["Bash"]["description"])           # built-ins never slimmed

    def test_not_enough_evidence_means_no_slimming(self):
        opt = Optimizer(None, Memo(), slim_after=5)
        for i in range(4):
            opt.memo.note_tools(f"c{i}", ["mcp__x__idle"], [])
        out = self.first(opt, [self.tool("mcp__x__idle")])["tools"][0]
        self.assertIn("Longer explanation", out["description"])

    def test_tool_that_appears_mid_conversation_is_never_slimmed(self):
        opt = Optimizer(None, Memo(), slim_after=2)
        for i in range(3):
            opt.memo.note_tools(f"c{i}", ["mcp__x__late"], [])
        opt.optimize(full_request(self.start, TOOLS))
        later = self.start + tool_turn(0, "Read", {"file_path": "/x/a"}, output="x")
        out = parsed(opt.optimize(full_request(later, TOOLS + [self.tool("mcp__x__late")])))
        self.assertIn("Longer explanation", out["tools"][-1]["description"])

    def test_decision_is_stable_for_the_whole_conversation(self):
        opt = Optimizer(None, Memo(), slim_after=2)
        for i in range(3):
            opt.memo.note_tools(f"c{i}", ["mcp__x__idle"], [])
        tools = [self.tool("mcp__x__idle")]
        a = self.first(opt, tools)["tools"]
        opt.memo.note_tools("other", [], ["mcp__x__idle"])         # now used elsewhere
        later = self.start + tool_turn(0, "Read", {"file_path": "/x/a"}, output="x")
        b = parsed(opt.optimize(full_request(later, tools)))["tools"]
        self.assertEqual(strip_cc(a), strip_cc(b))                # same bytes: the cache stays valid

    def test_keep_prefix_flag(self):
        tools = [self.tool("Bash")]
        out = self.first(Optimizer(None, Memo(), prefix_rewrite=False), tools)
        self.assertEqual(strip_cc(out["tools"]), strip_cc(tools))
        self.assertIn("Be brief.  \n", out["system"][0]["text"])  # untouched apart from the note


class NarrationTests(unittest.TestCase):
    TABLE = "\n".join(["| Name            | Value          | Notes                  |",
                       "|-----------------|----------------|------------------------|"]
                      + [f"| item number {i:<4} | {i * 7:<14} | something about it     |" for i in range(40)])

    def convo(self, text):
        return [{"role": "user", "content": [{"type": "text", "text": "go"}]},
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
                {"role": "user", "content": [{"type": "text", "text": "thanks"}]}]

    def test_table_padding_is_removed_and_code_fences_are_kept(self):
        text = "Summary:\n\n" + self.TABLE + "\n\n```\n| keep   |   this |\n```\n"
        opt = Optimizer(None, Memo())
        opt.optimize(full_request(self.convo(text)[:1]))
        out = parsed(opt.optimize(full_request(self.convo(text))))["messages"][1]["content"][0]["text"]
        self.assertLess(len(out), len(text) * 0.8)
        self.assertIn("| keep   |   this |", out)
        self.assertIn("| item number 7 | 49 | something about it |", out)
        self.assertNotIn("|-----------------|", out)

    def test_keep_narration_flag_and_short_text(self):
        text = "Summary:\n\n" + self.TABLE
        opt = Optimizer(None, Memo(), narration=False)
        opt.optimize(full_request(self.convo(text)[:1]))
        out = parsed(opt.optimize(full_request(self.convo(text))))["messages"][1]["content"][0]["text"]
        self.assertEqual(out, text)

    def test_lossy_narration_pruning_is_opt_in(self):
        prose = ("This paragraph explains the architecture in plain words and it goes on for a good while so that "
                 "it is long enough to be considered for pruning by the compressor stage. ") * 30
        text = "Intro line.\n\n" + prose + "\n\nClosing line."
        for flag, shorter in ((False, False), (True, True)):
            opt = Optimizer(FakeCompressor(), Memo(), prune_narration=flag)
            opt.optimize(full_request(self.convo(text)[:1]))
            out = parsed(opt.optimize(full_request(self.convo(text))))["messages"][1]["content"][0]["text"]
            self.assertEqual(len(out) < len(text) * 0.9, shorter)


class InstrumentationTests(unittest.TestCase):
    def test_composition_splits_context_by_category(self):
        h = [{"role": "user", "content": [{"type": "text", "text": "hello " * 50}]}]
        h += tool_turn(0, "Write", {"file_path": "/x/a.py", "content": BIG_FILE}, output="created")
        c = breakdown.composition(json.loads(full_request(h)))
        for key in ("system", "tools", "user_text", "args:Write", "result:Write", "thinking"):
            self.assertGreater(c[key], 0, key)
        self.assertGreater(c["args:Write"], c["result:Write"])

    def test_profile_row_reports_savings_per_lever(self):
        opt = Optimizer(None, Memo())
        start = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        opt.optimize(full_request(start))
        h = start + tool_turn(0, "Write", {"file_path": "/x/a.py", "content": BIG_FILE}, output="created")
        raw = full_request(h)
        row = breakdown.record(raw, opt.optimize(raw), "abc")
        self.assertLess(row["total_after"], row["total_before"])
        self.assertGreater(row["saved_by"]["args:Write"], 0)


class FallbackTests(unittest.TestCase):
    """End to end through the real proxy handler, with a fake API that rejects abridged requests."""

    def test_rejected_rewrite_is_resent_untouched_and_conversation_stays_untouched(self):
        import http.client
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import proxy

        seen = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                seen.append(body)
                code, out = (400, b'{"error":"bad input"}') if b'"content":"[proxy:' in body else (200, b'{"usage":{"input_tokens":1}}')
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        saved = (proxy.UPSTREAM, proxy.OPTIMIZER, proxy.DRY_RUN)
        proxy.UPSTREAM = ("http", "127.0.0.1", up.server_port)
        proxy.OPTIMIZER = Optimizer(None, Memo())
        proxy.DRY_RUN = False
        px = ThreadingHTTPServer(("127.0.0.1", 0), proxy.ProxyHandler)
        for server in (up, px):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            start = [{"role": "user", "content": [{"type": "text", "text": "build it"}]}]

            def post(body):
                conn = http.client.HTTPConnection("127.0.0.1", px.server_port, timeout=10)
                conn.request("POST", "/v1/messages", body, {"Content-Type": "application/json"})
                status = conn.getresponse().status
                conn.close()
                return status

            def turn(n):
                h = list(start)
                for i in range(n):
                    h += tool_turn(i, "Write", {"file_path": f"/x/a{i}.py", "content": BIG_FILE}, output="created")
                return full_request(h)

            self.assertEqual(post(full_request(start)), 200)
            n = len(seen)
            original = turn(1)
            self.assertEqual(post(original), 200)            # client never sees the 400
            self.assertEqual(len(seen) - n, 2)                # rewritten attempt, then the original
            self.assertIn(b'"content":"[proxy:', seen[n])
            self.assertEqual(seen[n + 1], original)           # byte-for-byte what Claude Code sent
            n = len(seen)
            self.assertEqual(post(turn(2)), 200)
            self.assertEqual(len(seen) - n, 1)                # poisoned: straight through, untouched
            self.assertNotIn(b'"content":"[proxy:', seen[n])
        finally:
            proxy.UPSTREAM, proxy.OPTIMIZER, proxy.DRY_RUN = saved
            px.shutdown()
            up.shutdown()


# ------------------------------------------------------------------- money

import economics


class EconomicsTests(unittest.TestCase):
    P = economics.price_for("claude-sonnet-5-5")

    def test_prices(self):
        self.assertEqual((self.P.inp, self.P.out, self.P.read), (2.0, 10.0, 0.10))
        self.assertAlmostEqual(self.P.write1h, 4.0)
        self.assertEqual(economics.price_for("claude-sonnet-5-5-20261001").model, "claude-sonnet-5-5")
        self.assertEqual(economics.price_for("claude-opus-5-5").read, 0.20)
        self.assertEqual(economics.price_for("claude-opus-4-8").read, 0.5)   # 0.1x when no read price is listed
        self.assertIsNone(economics.price_for("huihui_ai/dolphin3"))
        self.assertIsNone(economics.price_for(None))

    def test_usage_cost_prices_each_bucket(self):
        usage = {"cache_read_input_tokens": 1_000_000, "input_tokens": 1_000_000, "output_tokens": 1_000_000,
                 "cache_creation": {"ephemeral_1h_input_tokens": 1_000_000, "ephemeral_5m_input_tokens": 1_000_000}}
        c = economics.usage_cost(usage, self.P)
        self.assertAlmostEqual(c.read, 0.10)
        self.assertAlmostEqual(c.write, 2.5 + 4.0)
        self.assertAlmostEqual(c.fresh, 2.0)
        self.assertAlmostEqual(c.output, 10.0)
        old = economics.usage_cost({"cache_creation_input_tokens": 1_000_000}, self.P)   # no breakdown: 5m
        self.assertAlmostEqual(old.write, 2.5)

    def test_unchanged_request_has_exactly_zero_difference(self):
        usage = {"cache_read_input_tokens": 37_777, "cache_creation": {"ephemeral_1h_input_tokens": 4_321},
                 "input_tokens": 12, "output_tokens": 500}
        pieces = [30_000.0, 3_000.0, 2_000.0, 1_000.0, 4_000.0]
        actual, other = economics.compare(usage, self.P, pieces, list(pieces))
        self.assertAlmostEqual(actual.total, other.total, places=12)

    def test_saving_in_the_cached_prefix_is_priced_at_the_read_rate(self):
        # 100k cached (5 pieces of 20k), 10k new tail; the rewrite removes 10k tokens from the prefix
        usage = {"cache_read_input_tokens": 100_000, "cache_creation": {"ephemeral_1h_input_tokens": 10_000},
                 "input_tokens": 0, "output_tokens": 0}
        orig = [20_000.0] * 5 + [10_000.0]
        new = [20_000.0, 10_000.0, 20_000.0, 20_000.0, 20_000.0, 10_000.0]
        a, o = economics.compare(usage, self.P, orig, new)
        self.assertAlmostEqual(a.total - o.total, 10_000 * 0.10e-6, places=9)
        self.assertAlmostEqual(a.write, o.write)            # the uncached tail is untouched

    def test_saving_in_the_new_tail_is_priced_at_the_write_rate(self):
        usage = {"cache_read_input_tokens": 100_000, "cache_creation": {"ephemeral_1h_input_tokens": 10_000},
                 "input_tokens": 0, "output_tokens": 0}
        orig = [20_000.0] * 5 + [10_000.0]
        new = [20_000.0] * 5 + [4_000.0]                     # tail shrinks by 6k tokens
        a, o = economics.compare(usage, self.P, orig, new)
        self.assertAlmostEqual(a.total - o.total, 6_000 * 4.0e-6, places=9)   # 1h write = 2x input
        self.assertAlmostEqual(a.read, o.read)

    def test_compare_in_the_other_direction(self):
        """Live proxy: billed = rewritten; the original side must cost more."""
        usage = {"cache_read_input_tokens": 50_000, "cache_creation": {"ephemeral_1h_input_tokens": 5_000},
                 "input_tokens": 0, "output_tokens": 300}
        new = [10_000.0] * 5 + [5_000.0]
        orig = [10_000.0] * 5 + [9_000.0]
        billed, without = economics.compare(usage, self.P, new, orig)
        self.assertGreater(without.total, billed.total)
        self.assertAlmostEqual(without.output, billed.output)

    def test_ledger_rows_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "l.jsonl")
            led = economics.Ledger(path)
            usage = {"cache_read_input_tokens": 100_000, "cache_creation": {"ephemeral_1h_input_tokens": 10_000},
                     "input_tokens": 3, "output_tokens": 700}
            class R:  # what Optimizer returns when it rewrote the tail
                orig_pieces = [20_000.0] * 5 + [10_000.0]
                new_pieces = [20_000.0] * 5 + [4_000.0]
            row = led.add("abcdef012345", "claude-sonnet-5-5", usage, R(), dry_run=False)
            self.assertEqual(row["conv"], "abcdef01")
            self.assertGreater(row["saved"], 0)
            self.assertAlmostEqual(row["billed"], row["with_proxy"], places=6)
            self.assertAlmostEqual(row["without_proxy"] - row["with_proxy"], row["saved"], places=6)
            dry = led.add("abcdef012345", "claude-sonnet-5-5", usage, R(), dry_run=True)
            self.assertAlmostEqual(dry["billed"], dry["without_proxy"], places=6)   # nothing rewritten
            self.assertGreater(dry["saved"], 0)
            self.assertIsNone(led.add("x", "some-local-model", usage, None))        # unpriced: skipped
            untouched = led.add("zz", "claude-sonnet-5-5", usage, None)
            self.assertEqual(untouched["saved"], 0)
            total, convs = led.snapshot()
            self.assertEqual(total.requests, 3)
            self.assertEqual(convs["abcdef01"].requests, 2)
            self.assertEqual(len(open(path).read().splitlines()), 3)

    def test_sniffer_reads_final_output_tokens_from_the_stream(self):
        sse = (b'event: message_start\ndata: {"message":{"usage":{"input_tokens":5,"cache_read_input_tokens":9,'
               b'"output_tokens":1}}}\n\n' + b'event: content_block_delta\ndata: {"x":1}\n\n' * 50 +
               b'event: message_delta\ndata: {"usage":{"output_tokens":321}}\n\n')
        s = UsageSniffer(True)
        for i in range(0, len(sse), 64):
            s.feed(sse[i:i + 64])
        usage = s.finish()
        self.assertEqual(usage["output_tokens"], 321)
        self.assertEqual(usage["cache_read_input_tokens"], 9)

    def test_optimizer_result_carries_model_and_piece_sizes(self):
        opt = Optimizer(None, Memo())
        start = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        opt.optimize(full_request(start))
        h = start + tool_turn(0, "Write", {"file_path": "/x/a.py", "content": BIG_FILE}, output="created")
        res = opt.optimize(full_request(h))
        self.assertEqual(res.model, "m")
        self.assertEqual(len(res.orig_pieces), len(h) + 1)
        self.assertGreater(sum(res.orig_pieces), sum(res.new_pieces))

    def test_report_reads_a_ledger(self):
        import contextlib
        import report
        with tempfile.TemporaryDirectory() as tmp:
            led = os.path.join(tmp, "l.jsonl")
            ledger = economics.Ledger(led)
            class R:
                orig_pieces = [10.0, 5000.0, 9000.0]
                new_pieces = [10.0, 5000.0, 3000.0]
            ledger.add("c1c1c1c1", "claude-sonnet-5-5", {"cache_read_input_tokens": 5000,
                       "cache_creation": {"ephemeral_1h_input_tokens": 3000}, "output_tokens": 100}, R())
            import sys
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sys.argv = ["report.py", led]
                report.main()
                sys.argv = ["report.py", led, "--requests", "c1c1"]
                report.main()
            out = buf.getvalue()
            self.assertIn("c1c1c1c1", out)
            self.assertIn("TOTAL", out)


if __name__ == "__main__":
    unittest.main()
