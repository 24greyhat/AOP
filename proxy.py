"""Shrink Claude Code's API requests (cache-safely) and forward them to Anthropic."""

import argparse
import atexit
import base64
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.util
import os
import re
import signal
from threading import Lock, Thread
from urllib.parse import urlsplit

import breakdown
import economics
from shrink import Compressor
from dashboard import Dashboard, Stats
from optimizer import Memo, Optimizer


UPSTREAM = ("https", "api.anthropic.com", 443)
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade",
}
LOG_LOCK = Lock()

SESSION = []
DUMP_PATH = None   # --dump: write the original (untouched) request bodies here
DRY_RUN = False    # --dry-run: measure savings but forward the original bodies
PROFILE_PATH = None  # --profile: one JSON line per request with token composition and savings per lever
OPTIMIZER = None
STATS = Stats()
LEDGER = economics.Ledger()
DASH = Dashboard(STATS)
DASH.ledger = LEDGER
_MODEL = re.compile(rb'"model"\s*:\s*"([^"]+)"')


def is_messages_request(command, path):
    return command == "POST" and path.split("?", 1)[0] == "/v1/messages"


class UsageSniffer:
    """Reads token usage out of a response while it is relayed (never alters it)."""

    LIMIT = 1 << 20

    def __init__(self, sse):
        self.sse = sse
        self.buf = bytearray()
        self.tail = b""
        self.usage = None

    def feed(self, data):
        self.tail = (self.tail + data)[-8192:]
        if self.usage is not None or len(self.buf) > self.LIMIT:
            return
        self.buf += data
        if self.sse:
            start = self.buf.find(b"event: message_start")
            start = self.buf.find(b"data:", start) if start >= 0 else -1
            end = self.buf.find(b"\n", start) if start >= 0 else -1
            if end >= 0:
                try:
                    self.usage = json.loads(self.buf[start + 5:end])["message"]["usage"]
                except (ValueError, KeyError, TypeError):
                    self.usage = {}

    def finish(self):
        if self.usage is None and not self.sse and 0 < len(self.buf) <= self.LIMIT:
            try:
                self.usage = json.loads(self.buf).get("usage")
            except (ValueError, AttributeError):
                pass
        usage = self.usage or None
        if usage and self.sse:   # message_start carries the input side; the last message_delta the output total
            found = re.findall(rb'"output_tokens"\s*:\s*(\d+)', self.tail)
            if found:
                usage = dict(usage, output_tokens=int(found[-1]))
        return usage


def forwarded_headers(headers, exclude):
    connection_tokens = {
        token.strip().lower()
        for value in headers.get_all("Connection", [])
        for token in value.split(",")
    }
    blocked = HOP_BY_HOP | connection_tokens | exclude
    return [(name, value) for name, value in headers.items() if name.lower() not in blocked]


def dump_request(method, target, headers, body):
    """--dump: append the original request body to a JSON file (same format as before)."""
    global SESSION
    try:
        content = body.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(body).decode("ascii")
        encoding = "base64"

    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "target": target,
        "headers": [(key, value) for key, value in headers.items()],
        "body_bytes": len(body),
        "body_encoding": encoding,
        "body": content,
    }
    with LOG_LOCK:
        data = json.dumps(record['body'])
        SESSION.append(data)

        with open(DUMP_PATH, "w") as f:
            f.write(json.dumps(SESSION))


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60

    def log_message(self, format, *args):
        pass  # Request records are written to stdout by log_request.

    def read_exactly(self, size):
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("Incomplete request body")
        return data

    def read_chunk_line(self):
        line = self.rfile.readline(8192)
        if not line.endswith(b"\r\n"):
            raise ValueError("Invalid chunked request")
        return line[:-2]

    def read_body(self):
        lengths = self.headers.get_all("Content-Length", [])
        transfers = self.headers.get_all("Transfer-Encoding", [])
        if transfers:
            if lengths or [value.strip().lower() for value in transfers] != ["chunked"]:
                raise ValueError("Unsupported request framing")
            body = bytearray()
            while True:
                size = self.read_chunk_line().split(b";", 1)[0]
                if not size or any(byte not in b"0123456789abcdefABCDEF" for byte in size):
                    raise ValueError("Invalid chunk size")
                size = int(size, 16)
                if not size:
                    while self.read_chunk_line():
                        pass  # Consume trailers before forwarding the body.
                    return bytes(body)
                body.extend(self.read_exactly(size))
                if self.read_exactly(2) != b"\r\n":
                    raise ValueError("Invalid chunk ending")

        if not lengths:
            return b""
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise ValueError("Invalid Content-Length")
        return self.read_exactly(int(lengths[0]))

    def send_upstream(self, body, sniff):
        """Open a connection to the API, send the request, return (connection, response)."""
        scheme, host, port = UPSTREAM
        connect = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        upstream = connect(host, port, timeout=300)
        try:
            upstream.putrequest(self.command, self.path, skip_accept_encoding=True)
            # Ask for an uncompressed reply on message calls so token usage can be read.
            skip = {"host", "content-length", "expect"} | ({"accept-encoding"} if sniff else set())
            for name, value in forwarded_headers(self.headers, skip):
                upstream.putheader(name, value)
            if sniff:
                upstream.putheader("Accept-Encoding", "identity")
            if body or "Content-Length" in self.headers or "Transfer-Encoding" in self.headers:
                upstream.putheader("Content-Length", str(len(body)))
            upstream.endheaders(body)
            return upstream, upstream.getresponse()
        except BaseException:
            upstream.close()
            raise

    def proxy_request(self):
        self.close_connection = True
        if not self.path.startswith("/"):
            self.send_error(400, "Expected an API path")
            return
        try:
            body = self.read_body()
        except (ValueError, OverflowError):
            self.send_error(400, "Invalid request body")
            return
        except OSError:
            return

        sniff = is_messages_request(self.command, self.path)
        conv = ""
        result = None
        fell_back = False
        original = body
        if sniff:
            if DUMP_PATH:
                dump_request(self.command, self.path, self.headers, body)
            if OPTIMIZER and "Content-Encoding" not in self.headers:
                result = OPTIMIZER.optimize(body)
                conv = result.conv
                STATS.request(result)
                DASH.log(DASH.request_line(result))
                if PROFILE_PATH:
                    Thread(target=lambda: breakdown.append_jsonl(PROFILE_PATH, breakdown.record(original, result, conv)),
                           daemon=True).start()
                if not DRY_RUN:
                    body = result.body
                DASH.refresh()
        rewritten = body is not original and body != original

        try:
            upstream, response = self.send_upstream(body, sniff)
            if rewritten and response.status == 400:
                # Never let a rewrite cost a request: show why, then resend what Claude Code sent.
                detail = response.read(1500).decode("utf-8", "replace")
                response.close()
                upstream.close()
                DASH.log(DASH.amber(f"⚠ API rejected the rewritten request, resending the original: {detail[:160]}"))
                OPTIMIZER.poison(conv)  # this conversation goes through untouched from here on
                fell_back = True
                upstream, response = self.send_upstream(original, sniff)
        except (http.client.HTTPException, OSError, ValueError) as error:
            DASH.log(DASH.red(f"upstream request failed ({type(error).__name__})"))
            try:
                self.send_error(502, "Anthropic API connection failed")
            except OSError:
                pass  # Client already hung up; nobody left to tell.
            return

        sniffer = UsageSniffer("text/event-stream" in (response.getheader("Content-Type") or "")) if sniff else None
        try:
            self.relay_response(response, sniffer)
        except (http.client.HTTPException, OSError):
            # Closing an interrupted response leaves its framing incomplete.
            pass
        finally:
            response.close()
            upstream.close()
            if sniffer:
                usage = sniffer.finish()
                warning = STATS.usage(usage, conv)
                if warning:
                    DASH.log(DASH.amber(f"⚠ cache miss: {warning}"))
                m = _MODEL.search(original[:4096])
                model = (result.model if result else "") or (m.group(1).decode("utf-8", "replace") if m else "")
                row = LEDGER.add(conv, model, usage, None if fell_back else result, DRY_RUN)
                if row:
                    DASH.log(DASH.money_line(row))
                DASH.refresh()

    def relay_response(self, response, sniffer=None):
        has_body = self.command != "HEAD" and response.status not in (204, 304)
        excluded = {"content-length"} if response.chunked else set()
        headers = forwarded_headers(response.headers, excluded)
        chunked = has_body and not any(name.lower() == "content-length" for name, _ in headers)

        self.send_response_only(response.status, response.reason)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Connection", "close")
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        if has_body:
            # read1 forwards available bytes promptly for server-sent events.
            while data := response.read1(64 * 1024):
                if sniffer:
                    sniffer.feed(data)
                if chunked:
                    self.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
                else:
                    self.wfile.write(data)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = proxy_request


def main():
    global UPSTREAM, DUMP_PATH, DRY_RUN, OPTIMIZER, PROFILE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080, help="local port (default: 8080)")
    parser.add_argument("--upstream", default="https://api.anthropic.com",
                        help="API origin to forward to (default: https://api.anthropic.com)")
    parser.add_argument("--model", default="small",
                        help="LLMLingua-2 model: small (bert-base, fast), large (xlm-roberta) or a HF id")
    parser.add_argument("--rate", type=float, default=0.6,
                        help="fraction of tokens LLMLingua keeps in prose it prunes (default: 0.6)")
    parser.add_argument("--device", default="cpu", help="torch device for LLMLingua: cpu, mps, cuda (default: cpu)")
    parser.add_argument("--no-compress", action="store_true", help="skip LLMLingua pruning (lossless cleanup only)")
    parser.add_argument("--no-convert", action="store_true", help="skip markitdown file conversion")
    parser.add_argument("--full-reads", action="store_true", help="always send whole files (default: big source files are narrowed to what matches the task)")
    parser.add_argument("--lossless-prompts", action="store_true", help="clean your messages but never prune pasted prose")
    parser.add_argument("--keep-prompts", action="store_true", help="never touch your own messages (default: lossless cleanup of long ones)")
    parser.add_argument("--keep-args", action="store_true",
                        help="never abridge the assistant's executed Write/Bash/Edit arguments")
    parser.add_argument("--elide-min-tokens", type=int, default=400,
                        help="smallest Write/Edit argument worth abridging (default: 400)")
    parser.add_argument("--bash-min-tokens", type=int, default=250,
                        help="smallest Bash command worth abridging (default: 250)")
    parser.add_argument("--keep-prefix", action="store_true",
                        help="leave the system prompt and tool definitions exactly as Claude Code sends them")
    parser.add_argument("--slim-after", type=int, default=5, metavar="N",
                        help="slim an MCP tool's definition once N conversations offered it and none used it "
                             "(0 = never slim; default: 5)")
    parser.add_argument("--keep-narration", action="store_true", help="never touch the assistant's own text")
    parser.add_argument("--prune-narration", action="store_true",
                        help="also prune long prose in the assistant's older text with LLMLingua (lossy; off by default)")
    parser.add_argument("--no-cold", action="store_true",
                        help="never compact history when a conversation returns after its cache expired")
    parser.add_argument("--cold-through-thinking", action="store_true",
                        help="let cold compaction edit history before signed thinking blocks (may 400; the proxy "
                             "then resends the original request). Default: leave that history alone")
    parser.add_argument("--cache-ttl", type=int, default=None, metavar="SECONDS",
                        help="prompt-cache lifetime to assume (default: read it from the requests, 1 h or 5 min)")
    parser.add_argument("--cold-keep", type=int, default=10, metavar="N",
                        help="newest messages a cold-cache compaction never touches (default: 10)")
    parser.add_argument("--cold-min-tokens", type=int, default=1000,
                        help="smallest old tool output a cold-cache compaction ages (default: 1000)")
    parser.add_argument("--keep-repeat-reads", action="store_true",
                        help="never replace a repeated whole-file Read by a pointer to the identical earlier one")
    parser.add_argument("--ledger", metavar="FILE",
                        help="append one JSON line per request with its real cost and what it would have cost "
                             "without the proxy (read it with report.py)")
    parser.add_argument("--profile", metavar="FILE",
                        help="append one JSON line per request: token composition before/after and savings per lever")
    parser.add_argument("--no-optimize", action="store_true", help="forward requests untouched")
    parser.add_argument("--dry-run", action="store_true", help="measure savings but forward the original requests")
    parser.add_argument("--memo", default=os.path.expanduser("~/.cache/aop-proxy/memo.sqlite"),
                        help="where shrunken blocks are remembered across restarts ('' = memory only)")
    parser.add_argument("--dump", metavar="FILE", help="also write the original request bodies to FILE (JSON)")
    args = parser.parse_args()

    target = urlsplit(args.upstream if "//" in args.upstream else f"https://{args.upstream}")
    UPSTREAM = (target.scheme, target.hostname, target.port or (443 if target.scheme == "https" else 80))
    DUMP_PATH, DRY_RUN, PROFILE_PATH = args.dump, args.dry_run, args.profile
    LEDGER.path = args.ledger
    DASH.dry_run = args.dry_run

    compressor = None
    if not (args.no_optimize or args.no_compress):
        compressor = Compressor(model=args.model, rate=args.rate, device=args.device)
        compressor.start()  # loads in the background; first request waits for it if needed
    if not args.no_optimize:
        OPTIMIZER = Optimizer(compressor, Memo(args.memo or None), convert=not args.no_convert,
                              clean_prompts=not args.keep_prompts,
                              prune_prompts=not args.lossless_prompts,
                              targeted_reads=not args.full_reads,
                              elide_args=not args.keep_args, elide_min_tokens=args.elide_min_tokens,
                              bash_min_tokens=args.bash_min_tokens,
                              prefix_rewrite=not args.keep_prefix, slim_after=args.slim_after or 10**9,
                              narration=not args.keep_narration, prune_narration=args.prune_narration,
                              cold_cache=not args.no_cold, cache_ttl=args.cache_ttl, cold_keep=args.cold_keep,
                              cold_through_thinking=args.cold_through_thinking,
                              cold_min_tokens=args.cold_min_tokens, dedupe_reads=not args.keep_repeat_reads)

    def terminate(signum, frame):
        raise KeyboardInterrupt  # same clean shutdown path as Ctrl+C

    signal.signal(signal.SIGTERM, terminate)

    with ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler) as server:
        DASH.start()
        atexit.register(DASH.stop)
        mode = "pass-through" if args.no_optimize else "dry-run" if DRY_RUN else "optimizing"
        DASH.log(f"{DASH.bold('http://127.0.0.1:' + str(server.server_port))} → {args.upstream}  "
                 f"{DASH.dim('[' + mode + ']')}")
        here = os.path.dirname(os.path.abspath(__file__))
        for package in ("llmlingua", "markitdown"):
            spec = importlib.util.find_spec(package)
            if spec and spec.origin and os.path.dirname(os.path.abspath(spec.origin)) == here:
                DASH.log(DASH.red(f"⚠ {package}.py in this folder shadows the real '{package}' package "
                                  f"— rename or delete it or that feature will not work"))
        if compressor:
            DASH.log(DASH.dim(f"LLMLingua-2 model loading in background: {compressor.model_id}"))
        DASH.log(DASH.dim("Prompts are inspected and rewritten locally; only the result is sent to the API."))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            DASH.stop()


if __name__ == "__main__":
    main()
