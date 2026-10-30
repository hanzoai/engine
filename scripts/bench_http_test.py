#!/usr/bin/env python3
"""bench_http.py against a fake engine that streams in vLLM's shape and in hanzo's shape."""
import http.server
import io
import json
import os
import socket
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_http  # noqa: E402

GAP = 0.05  # seconds between token chunks


def chunk(delta=None, finish=None, usage=None, choices=True):
    body = {"id": "x", "object": "chat.completion.chunk", "model": "fake"}
    body["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []
    if usage:
        body["usage"] = usage
    return body


def vllm_stream():
    """Role chunk, 3 reasoning + 3 content chunks carrying 8 tokens, finish, then usage 2 s later."""
    yield chunk({"role": "assistant", "content": ""}), 0
    for text in ["Let", " me think", " now"]:
        yield chunk({"reasoning": text}), GAP
    for text in ["The", " answer", " is"]:
        yield chunk({"content": text}), GAP
    yield chunk({}, finish="length"), 0
    yield chunk(choices=False, usage={"prompt_tokens": 42, "completion_tokens": 8,
                                       "total_tokens": 50}), 2.0


def hanzo_stream():
    """reasoning_content deltas, usage on the finish chunk."""
    yield chunk({"role": "assistant"}), 0
    for text in ["Let", " me think", " now"]:
        yield chunk({"reasoning_content": text}), GAP
    for text in ["The", " answer", " is"]:
        yield chunk({"content": text}), GAP
    yield chunk({}, finish="length", usage={"prompt_tokens": 42, "completion_tokens": 8,
                                             "total_tokens": 50}), 0


def no_usage_stream():
    yield chunk({"content": "a"}), 0
    yield chunk({"content": "b"}), GAP
    yield chunk({}, finish="stop"), 0


class Fake(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, shape, hold=0.0):
        super().__init__(("127.0.0.1", 0), Handler)
        self.shape, self.hold = shape, hold
        self.prompts, self.lock, self.inflight, self.peak = [], threading.Lock(), 0, 0
        self.spans = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}/v1/chat/completions"


class Handler(http.server.BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        # Stream chunks as written, as a real server does; Nagle would add delayed-ACK stalls.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        server = self.server
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        assert req["stream_options"] == {"include_usage": True}
        with server.lock:
            server.prompts.append(req["messages"][0]["content"])
            server.inflight += 1
            server.peak = max(server.peak, server.inflight)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        time.sleep(server.hold)
        written = []
        for body, delay in server.shape():
            time.sleep(delay)
            self.wfile.write(f"data: {json.dumps(body)}\n\n".encode())
            self.wfile.flush()
            if bench_http.is_token_chunk(body):
                written.append(time.monotonic())
        with server.lock:
            server.spans.append(written[-1] - written[0] if written else 0.0)
        self.wfile.write(b"data: [DONE]\n\n")
        with server.lock:
            server.inflight -= 1


def serve(shape, hold=0.0):
    server = Fake(shape, hold)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run(argv):
    out = io.StringIO()
    with redirect_stdout(out):
        code = bench_http.main(argv)
    return code, [json.loads(line) for line in out.getvalue().splitlines() if line]


class Streams(unittest.TestCase):
    def check_shape(self, shape):
        server = serve(shape)
        try:
            code, rows = run([server.url, "--sizes", "4", "--gen", "8"])
        finally:
            server.shutdown()
        self.assertEqual(code, 0)
        (row,) = rows
        self.assertEqual(row["prompt_tokens"], 42)
        self.assertEqual(row["completion_tokens"], 8)
        self.assertEqual(row["chunks"], 6)
        # The window is the server's first-to-last token chunk; the 2 s usage delay and the empty
        # role and finish chunks are outside it.
        (span,) = server.spans
        self.assertAlmostEqual(row["decode_s"], span, delta=0.03)
        self.assertLess(row["decode_s"], 1.5)
        self.assertAlmostEqual(row["decode_tps"], 7 / row["decode_s"], places=6)
        self.assertAlmostEqual(row["itl_p50_s"], span / 5, delta=0.03)

    def test_vllm_shape(self):
        self.check_shape(vllm_stream)

    def test_hanzo_shape(self):
        self.check_shape(hanzo_stream)

    def test_no_usage_fails(self):
        server = serve(no_usage_stream)
        try:
            code, rows = run([server.url, "--sizes", "4"])
        finally:
            server.shutdown()
        self.assertNotEqual(code, 0)
        self.assertIn("no usage", rows[0]["error"])

    def test_concurrency_overlaps_with_distinct_nonces(self):
        server = serve(hanzo_stream, hold=0.3)
        try:
            code, rows = run([server.url, "--sizes", "4", "--concurrency", "4"])
        finally:
            server.shutdown()
        self.assertEqual(code, 0)
        self.assertEqual(len(server.prompts), 4)
        nonces = {p.split(".")[0] for p in server.prompts}
        self.assertEqual(len(nonces), 4)
        self.assertTrue(all(p.startswith("Session ") for p in server.prompts))
        self.assertGreaterEqual(server.peak, 2)
        aggregate = [r for r in rows if r.get("aggregate")]
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["completion_tokens"], 32)


if __name__ == "__main__":
    unittest.main()
