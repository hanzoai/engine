#!/usr/bin/env python3
"""Prefill and decode rates from a streaming OpenAI-compatible chat endpoint.

The prompt is a uuid nonce (so no prefix cache can serve it), WORDS repeated `size` times, then
"Summarize in one sentence." Sizes 4, 40, 400 and 4000 are about 220, 1,335, 12,497 and 124,094
Qwen3.8 tokens; the prompt length reported is the server's own usage.prompt_tokens.

A token chunk is a chunk whose delta carries content, reasoning, reasoning_content or tool_calls
(vLLM streams thinking as `reasoning`, hanzo as `reasoning_content`). Timing uses token chunks only:

  TTFT   = first token chunk - send
  decode = (usage.completion_tokens - 1) / (last token chunk - first token chunk)

Under speculation one chunk carries several tokens, so tokens come from usage, never from chunk
counts, and a stream without usage fails. Gaps between token chunks are one sample per step, the
way vLLM records inter-token latency under speculation.

Output is one JSON line per request (and one aggregate line per concurrent batch) on stdout.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request
import uuid

WORDS = ("The bandwidth of a decode step is the only number that matters here. "
         "Every token read is a byte moved, and the arithmetic is idle waiting for it. ")
TOKEN_FIELDS = ("content", "reasoning", "reasoning_content", "tool_calls")
SIZES = "4,40,400,4000"


def prompt_for(size):
    return f"Session {uuid.uuid4()}. " + WORDS * size + "\n\nSummarize in one sentence."


def base_of(url):
    """http://host:port from any URL on the server."""
    parts = url.split("/")
    return "/".join(parts[:3])


def first_model(url):
    with urllib.request.urlopen(base_of(url) + "/v1/models", timeout=30) as r:
        data = json.load(r)
    return data["data"][0]["id"]


def is_token_chunk(chunk):
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if any(delta.get(field) for field in TOKEN_FIELDS):
            return True
    return False


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[k]


def stream(url, model, prompt, gen, timestamps=False, on_first=None):
    """One streaming request. Returns the per-request measurement dict. `on_first` runs once,
    when the first token chunk arrives."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    sent = time.monotonic()
    chunks = []
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            now = time.monotonic()
            if chunk.get("usage"):
                usage = chunk["usage"]
            if is_token_chunk(chunk):
                if not chunks and on_first:
                    on_first()
                chunks.append(now)
    if usage is None:
        raise RuntimeError("the stream carried no usage; include_usage is required")
    if not chunks:
        raise RuntimeError("the stream carried no tokens")
    tokens = usage["completion_tokens"]
    first, last = chunks[0], chunks[-1]
    gaps = [b - a for a, b in zip(chunks, chunks[1:])]
    result = {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": tokens,
        "chunks": len(chunks),
        "ttft_s": first - sent,
        "prefill_tps": usage["prompt_tokens"] / (first - sent),
        "decode_s": last - first,
        "decode_tps": (tokens - 1) / (last - first) if tokens > 1 and last > first else None,
        "tokens_per_chunk": tokens / len(chunks),
        "itl_p50_s": percentile(gaps, 0.5),
        "itl_p95_s": percentile(gaps, 0.95),
        "itl_max_s": max(gaps) if gaps else None,
        "_first": first,
        "_last": last,
    }
    if timestamps:
        result["chunk_times_s"] = [t - sent for t in chunks]
    return result


def profile(url, action):
    req = urllib.request.Request(base_of(url) + f"/v1/profile/{action}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        r.read()


def run_batch(args, model, size, concurrency, label, on_first=None):
    results = [None] * concurrency
    errors = [None] * concurrency

    def one(i):
        try:
            results[i] = stream(args.url, model, prompt_for(size), args.gen, args.timestamps,
                                on_first)
        except Exception as e:  # recorded, never hidden
            errors[i] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=one, args=(i,)) for i in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = []
    for i in range(concurrency):
        row = {"label": label, "size": size, "concurrency": concurrency, "stream": i}
        if errors[i]:
            row["error"] = errors[i]
        else:
            row.update({k: v for k, v in results[i].items() if not k.startswith("_")})
        lines.append(row)
    ok = [r for r in results if r]
    if concurrency > 1 and ok:
        span = max(r["_last"] for r in ok) - min(r["_first"] for r in ok)
        decoded = sum(r["completion_tokens"] - 1 for r in ok)
        lines.append({
            "label": label, "size": size, "concurrency": concurrency, "aggregate": True,
            "streams_ok": len(ok),
            "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
            "completion_tokens": sum(r["completion_tokens"] for r in ok),
            "decode_tps": decoded / span if span > 0 else None,
        })
    return lines, ok, errors


# ---------------------------------------------------------------------------------------------
# /metrics check. Families are vLLM's names; hanzo serves the same names under `hanzo:` plus
# mamba_usage_perc for hybrid models. Spec families exist only when a proposer is attached.

CORE = [
    "num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
    "prefix_cache_queries", "prefix_cache_hits", "num_preemptions",
    "prompt_tokens", "generation_tokens", "request_success",
    "request_prompt_tokens", "request_generation_tokens", "iteration_tokens_total",
    "time_to_first_token_seconds", "inter_token_latency_seconds",
    "e2e_request_latency_seconds", "request_queue_time_seconds",
    "request_inference_time_seconds", "request_prefill_time_seconds",
    "request_decode_time_seconds",
]
SPEC = [
    "spec_decode_num_drafts", "spec_decode_num_draft_tokens",
    "spec_decode_num_accepted_tokens", "spec_decode_num_accepted_tokens_per_pos",
]
HYBRID = {"hanzo": ["mamba_usage_perc"], "vllm": []}
SUFFIXES = ("_total", "_created", "_bucket", "_sum", "_count")


def parse_metrics(text):
    """{sample name: [(labels dict, value)]} from Prometheus text."""
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        if "{" in name_labels:
            name, _, rest = name_labels.partition("{")
            labels = {}
            for part in _split_labels(rest.rstrip("}")):
                k, _, v = part.partition("=")
                labels[k] = v.strip('"')
        else:
            name, labels = name_labels, {}
        samples.setdefault(name, []).append((labels, float(value)))
    return samples


def _split_labels(body):
    out, cur, quoted, escaped = [], "", False, False
    for ch in body:
        if escaped:
            cur += ch
            escaped = False
        elif ch == "\\":
            cur += ch
            escaped = True
        elif ch == '"':
            cur += ch
            quoted = not quoted
        elif ch == "," and not quoted:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def family_of(name, prefix):
    base = name[len(prefix) + 1:]
    for suffix in SUFFIXES:
        if base.endswith(suffix) and base[: -len(suffix)] in ALL_FAMILIES:
            return base[: -len(suffix)]
    return base


ALL_FAMILIES = set(CORE) | set(SPEC) | {"mamba_usage_perc"}


def detect_prefix(samples):
    for prefix in ("hanzo", "vllm"):
        if any(n.startswith(prefix + ":") for n in samples):
            return prefix
    raise RuntimeError("no hanzo: or vllm: metrics")


def required_families(prefix, expect):
    required = list(CORE)
    if "spec" in expect:
        required += SPEC
    if "hybrid" in expect:
        required += HYBRID[prefix]
    return required


def check_families(samples, expect):
    prefix = detect_prefix(samples)
    present = {family_of(n, prefix) for n in samples if n.startswith(prefix + ":")}
    missing = [f for f in required_families(prefix, expect) if f not in present]
    return prefix, missing


def total(samples, prefix, family, suffix="_total", labels=None):
    s = 0.0
    for name in (f"{prefix}:{family}{suffix}", f"{prefix}:{family}"):
        for lab, v in samples.get(name, []):
            if labels and any(lab.get(k) != v2 for k, v2 in labels.items()):
                continue
            s += v
        if name in samples:
            break
    return s


def scrape(url):
    with urllib.request.urlopen(base_of(url) + "/metrics", timeout=30) as r:
        return r.read().decode()


class Scraper(threading.Thread):
    """Scrapes /metrics every 0.2 s while requests run and keeps the peak of running."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url, self.stop, self.peak_running, self.texts = url, threading.Event(), 0.0, []

    def run(self):
        while not self.stop.is_set():
            try:
                text = scrape(self.url)
                samples = parse_metrics(text)
                prefix = detect_prefix(samples)
                running = total(samples, prefix, "num_requests_running", suffix="")
                self.peak_running = max(self.peak_running, running)
                self.texts.append(text)
            except Exception:
                pass
            self.stop.wait(0.2)


def check_run(before, after, results, concurrency, peak_running, expect):
    """Counter deltas against what the client saw. Returns a list of failures."""
    failures = []
    prefix, missing = check_families(after, expect)
    if missing:
        failures.append(f"missing families: {', '.join(missing)}")
    n = len(results)

    def delta(family, suffix="_total", labels=None):
        return total(after, prefix, family, suffix, labels) - total(before, prefix, family, suffix, labels)

    got = delta("request_success")
    if got != n:
        failures.append(f"request_success rose {got}, expected {n}")
    gen = sum(r["completion_tokens"] for r in results)
    got = delta("generation_tokens")
    if got != gen:
        failures.append(f"generation_tokens rose {got}, expected {gen}")
    prompt = sum(r["prompt_tokens"] for r in results)
    got = delta("prompt_tokens")
    if got != prompt:
        failures.append(f"prompt_tokens rose {got}, expected {prompt}")
    got = delta("time_to_first_token_seconds", "_count")
    if got != n:
        failures.append(f"time_to_first_token_seconds count rose {got}, expected {n}")
    if concurrency >= 2 and peak_running < 2:
        failures.append(f"num_requests_running peaked at {peak_running} under concurrency {concurrency}")
    if "spec" in expect:
        accepted = delta("spec_decode_num_accepted_tokens")
        per_pos = delta("spec_decode_num_accepted_tokens_per_pos")
        if accepted != per_pos:
            failures.append(f"per-position accepted {per_pos} != accepted {accepted}")
        if delta("spec_decode_num_drafts") <= 0:
            failures.append("spec_decode_num_drafts did not rise")
    return prefix, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url", help="chat completions URL, e.g. http://127.0.0.1:30000/v1/chat/completions")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--sizes", default=SIZES)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--model", help="model id (default: the first id from /v1/models)")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", metavar="FILE", help="also append the JSON lines to FILE")
    ap.add_argument("--timestamps", action="store_true", help="record every token chunk's time")
    ap.add_argument("--check-metrics", action="store_true",
                    help="scrape /metrics before, during and after, and check the families and deltas")
    ap.add_argument("--expect", default="", help="comma list: spec, hybrid")
    ap.add_argument("--save-metrics", metavar="FILE", help="write the final /metrics scrape here")
    ap.add_argument("--profile", action="store_true",
                    help="POST /v1/profile/start after the first token and /v1/profile/stop at the end")
    args = ap.parse_args(argv)
    expect = {e for e in args.expect.split(",") if e}
    model = args.model or first_model(args.url)
    out = open(args.json, "a") if args.json else None
    failed = False

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()

    for size in [int(s) for s in args.sizes.split(",")]:
        before = parse_metrics(scrape(args.url)) if args.check_metrics else None
        scraper = Scraper(args.url) if args.check_metrics else None
        if scraper:
            scraper.start()
        on_first = None
        if args.profile:
            # The window opens at the first token, so it covers decode, not the prefill.
            once = threading.Lock()
            started = []

            def on_first():
                with once:
                    if not started:
                        profile(args.url, "start")
                        started.append(True)
        lines, ok, errors = run_batch(args, model, size, args.concurrency, args.label, on_first)
        if args.profile:
            profile(args.url, "stop")
        if scraper:
            scraper.stop.set()
            scraper.join()
        for row in lines:
            emit(row)
        if any(errors):
            failed = True
        if args.check_metrics:
            text = scrape(args.url)
            after = parse_metrics(text)
            if args.save_metrics:
                with open(args.save_metrics, "w") as f:
                    f.write(text)
            prefix, failures = check_run(before, after, ok, args.concurrency,
                                         scraper.peak_running, expect)
            emit({"label": args.label, "size": size, "metrics": prefix, "ok": not failures,
                  "failures": failures, "peak_running": scraper.peak_running})
            failed |= bool(failures)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
