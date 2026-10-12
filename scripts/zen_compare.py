#!/usr/bin/env python3
"""The same probes against two OpenAI-compatible servers, and the table that sets them side by side.

  zen_compare.py run  URL LABEL OUT.json [--model zen6] [--embed-url URL --embed-model zen-embedding-2]
  zen_compare.py table A.json B.json

`run` asks, greedy and thinking off unless a probe says otherwise:
  decode   code, chat and prose prompts, 384 tokens: decode tok/s as scripts/bench_http.py counts it
           (completion_tokens - 1 over first to last token chunk), and the text's SHA-256
  ttft     a nonce-prefixed prompt of about 1K and 16K tokens, one token out: seconds to the first token
  tools    five tool prompts in the qwen3_coder format: the calls each server parsed (name and arguments)
  effort   the six reasoning_effort names on one prompt: completion tokens, to show the mapping at work
  embed    (with --embed-url) a text, an image, a clip and a sound through /v1/embeddings, ms each
The server's /metrics, when it has speculative counters (vLLM `spec_decode_*`), gives the acceptance
over the run. `table` prints markdown, and for the probes both ran, whether the greedy text is equal.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
import uuid

PROMPTS = {
    "code": "Write a Python function merge_intervals(intervals) that merges overlapping intervals, with type hints and a docstring, then three pytest tests for it.",
    "chat": "I have a job interview tomorrow for a backend engineer role and I am nervous. Give me five practical tips for tonight and for the morning, in a friendly tone.",
    "prose": "Write a 400-word story about a lighthouse keeper who discovers that the light has been answering a signal from the sea.",
}
FILLER = ("The bandwidth of a decode step is the only number that matters here. "
          "Every token read is a byte moved, and the arithmetic is idle waiting for it. ")
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                    "metric": {"type": "boolean"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["query"]}}},
]
TOOL_PROMPTS = [
    "What is the weather in Paris for the next 3 days, in metric units?",
    "Open /etc/hostname and show me the first 5 lines.",
    "Search for the tags rust and gpu: fastest FP4 GEMM on Blackwell.",
    "Check the weather in Oslo and also read the file /tmp/notes.txt.",
    "What is 2 + 2? Answer directly without using any tool.",
]
EFFORTS = ["minimal", "low", "medium", "high", "xhigh", "max"]


def base(url):
    return "/".join(url.split("/")[:3])


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream(url, body):
    """One streaming chat completion: text, tool calls, usage and token-chunk times."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    sent = time.monotonic()
    times, text, calls, usage = [], [], {}, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            token = False
            for choice in chunk.get("choices") or []:
                d = choice.get("delta") or {}
                if d.get("content"):
                    text.append(d["content"]); token = True
                if d.get("reasoning") or d.get("reasoning_content"):
                    token = True
                for tc in d.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    f = tc.get("function") or {}
                    slot["name"] += f.get("name") or ""
                    slot["arguments"] += f.get("arguments") or ""
                    token = True
            if token:
                times.append(time.monotonic())
    if usage is None or not times:
        raise RuntimeError("no usage or no tokens in the stream")
    n = usage["completion_tokens"]
    out = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": n, "ttft_s": times[0] - sent,
           "decode_tps": (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else None,
           "text": "".join(text), "calls": [calls[k] for k in sorted(calls)]}
    return out


def chat(model, prompt, max_tokens, **extra):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}, **extra}


def scrape(url):
    """Speculative counters of a vLLM-style /metrics, or {}."""
    try:
        with urllib.request.urlopen(base(url) + "/metrics", timeout=10) as r:
            lines = r.read().decode().splitlines()
    except Exception:
        return {}
    out = {}
    for line in lines:
        for key in ("spec_decode_num_drafts_total", "spec_decode_num_accepted_tokens_total",
                    "spec_decode_num_draft_tokens_total"):
            if line.startswith("vllm:" + key + "{") or line.startswith(key + "{"):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def run(args):
    chat_url = args.url.rstrip("/") + "/v1/chat/completions"
    result = {"label": args.label, "model": args.model, "decode": {}, "ttft": {}, "tools": [], "effort": {}, "embed": {}}
    before = scrape(args.url)
    for name, prompt in PROMPTS.items():
        runs = [stream(chat_url, chat(args.model, prompt, 384)) for _ in range(args.reps)]
        best = max(runs, key=lambda r: r["decode_tps"] or 0)
        result["decode"][name] = {"decode_tps": best["decode_tps"], "all_tps": [r["decode_tps"] for r in runs],
                                  "tokens": best["completion_tokens"],
                                  "sha256": hashlib.sha256(best["text"].encode()).hexdigest(), "head": best["text"][:160]}
    for name, words in (("1K", 12), ("16K", 190)):
        runs = []
        for _ in range(args.reps):
            prompt = f"Session {uuid.uuid4()}. " + FILLER * words + "\n\nSummarize in one sentence."
            runs.append(stream(chat_url, chat(args.model, prompt, 1)))
        result["ttft"][name] = {"ttft_s": min(r["ttft_s"] for r in runs), "prompt_tokens": runs[0]["prompt_tokens"]}
    for prompt in TOOL_PROMPTS:
        r = stream(chat_url, chat(args.model, prompt, 512, tools=TOOLS))
        calls = []
        for c in r["calls"]:
            try:
                calls.append({"name": c["name"], "arguments": json.loads(c["arguments"] or "{}")})
            except json.JSONDecodeError:
                calls.append({"name": c["name"], "arguments": "INVALID " + c["arguments"][:80]})
        result["tools"].append({"prompt": prompt, "calls": calls})
    for effort in EFFORTS:
        body = chat(args.model, "A farmer has 17 sheep; all but 9 run away. How many are left? Explain.", 1024,
                    reasoning_effort=effort)
        body["chat_template_kwargs"] = {"enable_thinking": True}
        r = stream(chat_url, body)
        result["effort"][effort] = r["completion_tokens"]
    after = scrape(args.url)
    drafts = after.get("spec_decode_num_drafts_total", 0) - before.get("spec_decode_num_drafts_total", 0)
    if drafts > 0:
        accepted = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
        result["acceptance"] = {"drafts": drafts, "accepted_per_draft": accepted / drafts,
                                "tokens_per_step": 1 + accepted / drafts}
    if args.embed_url:
        result["embed"] = embed_probe(args)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result)[:600])


def embed_probe(args):
    url = args.embed_url.rstrip("/") + "/v1/embeddings"
    inputs = {"text": "task: search result | query: where is the cache key built?"}
    out = {}
    for name, item in inputs.items():
        times = []
        for _ in range(args.reps + 2):
            t = time.monotonic()
            post(url, {"model": args.embed_model, "input": item})
            times.append(time.monotonic() - t)
        out[name] = {"ms_p50": sorted(times)[len(times) // 2] * 1000}
    return out


def table(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    rows = [f"| | {A['label']} | {B['label']} |", "|---|---|---|"]
    f = lambda v, p=1: "n/a" if v is None else f"{v:.{p}f}"
    for name in PROMPTS:
        x, y = A["decode"][name], B["decode"][name]
        rows.append(f"| decode tok/s, {name} | {f(x['decode_tps'])} | {f(y['decode_tps'])} |")
        rows.append(f"| greedy text, {name} | {x['sha256'][:12]} | {y['sha256'][:12]}{' (equal)' if x['sha256'] == y['sha256'] else ''} |")
    for name in A["ttft"]:
        rows.append(f"| TTFT {name} ({A['ttft'][name]['prompt_tokens']} / {B['ttft'][name]['prompt_tokens']} tok), s | {f(A['ttft'][name]['ttft_s'], 2)} | {f(B['ttft'][name]['ttft_s'], 2)} |")
    same = sum(1 for p, q in zip(A["tools"], B["tools"]) if p["calls"] == q["calls"])
    rows.append(f"| tool probes with equal calls | {same} of {len(A['tools'])} | |")
    rows.append(f"| completion tokens by effort | {A['effort']} | {B['effort']} |")
    for r in (A, B):
        if "acceptance" in r:
            rows.append(f"| acceptance ({r['label']}) | {f(r['acceptance']['accepted_per_draft'], 2)} accepted per draft | |")
    print("\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url"); r.add_argument("label"); r.add_argument("out")
    r.add_argument("--model", default="zen6"); r.add_argument("--reps", type=int, default=3)
    r.add_argument("--embed-url"); r.add_argument("--embed-model", default="zen-embedding-2")
    t = sub.add_parser("table")
    t.add_argument("a"); t.add_argument("b")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else table(args.a, args.b)
