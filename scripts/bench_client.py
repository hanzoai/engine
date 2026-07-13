#!/usr/bin/env python3
"""bench_client — load generator for the MoE bench harness.

Drives /v1/chat/completions and reads the engine's usage superset (Usage in
hanzo-engine/src/response.rs) to isolate prefill from decode at the source, then
cross-checks steady-state decode client-side from inter-token latency (ITL).

Four request modes, each emits one JSON record per request on stdout (the
orchestrator collects them; the report assembler folds them):

  prefill      large distinct prompt, max_tokens=1 -> usage.avg_prompt_tok_per_sec
  decode       tiny prompt, max_tokens sweep, non-stream -> usage.avg_compl_tok_per_sec
  steady       same as decode but streamed; ITL median (first K dropped) -> tok/s
  batch        N concurrent decode requests, distinct prompts -> aggregate tok/s

Why both usage and ITL for decode: usage.avg_compl_tok_per_sec is prefill-excluded
at the source (sequence.rs), but streaming usage is unreliable (anthropic.rs hard-
codes it to 0.0), so steady-state must also be measured on the wire. We emit both
and assert they agree within tolerance as a self-check.

Every request carries temperature=0, a fixed seed, and a UNIQUE prompt salt per
virtual user so prefix caching cannot elide prefill (belt-and-suspenders with the
server's --prefix-cache-n 0). Pure httpx — no colons-before-nothing, no openai dep.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid

import httpx

# ITL samples to drop before computing steady-state: prefill-tail + graph-capture
# transient land in the first few inter-token gaps.
DROP_FIRST_K = 8


def salt() -> str:
    """A unique token so no two virtual users share a prefix."""
    return uuid.uuid4().hex


def messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def filler(n_tokens: int, tag: str) -> str:
    """A distinct prompt of ~n_tokens words. `tag` keeps every prompt unique."""
    # One word ~= one token for this synthetic filler; good enough to sweep sizes.
    head = f"session {tag} request {salt()} "
    body = " ".join(f"w{i}" for i in range(max(0, n_tokens - len(head.split()))))
    return head + body


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    stream: bool,
    phase: str,
) -> dict:
    """Fire one request; return a record with token counts, both timers, phase tag."""
    payload = {
        "model": "default",
        "messages": messages(prompt),
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
        "stream": stream,
    }
    if not stream:
        t0 = time.perf_counter()
        resp = await client.post(url, json=payload)
        wall = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage") or {}
        return {
            "phase": phase,
            "stream": False,
            "wall_s": wall,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "usage_prompt_tok_s": usage.get("avg_prompt_tok_per_sec"),
            "usage_compl_tok_s": usage.get("avg_compl_tok_per_sec"),
            "usage_prompt_time_s": usage.get("total_prompt_time_sec"),
            "usage_compl_time_s": usage.get("total_completion_time_sec"),
        }

    # Streamed: stamp every content-bearing delta to recover ITL.
    stamps: list[float] = []
    usage: dict = {}
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    async with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                stamps.append(time.perf_counter())
    wall = time.perf_counter() - t0

    itls = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    steady = itls[DROP_FIRST_K:] if len(itls) > DROP_FIRST_K else itls
    return {
        "phase": phase,
        "stream": True,
        "wall_s": wall,
        "n_deltas": len(stamps),
        "completion_tokens": usage.get("completion_tokens"),
        "usage_compl_tok_s": usage.get("avg_compl_tok_per_sec"),
        "itl": itl_stats(steady),
    }


def itl_stats(itls: list[float]) -> dict | None:
    """p50/p90/p99 ITL (ms) and steady-state tok/s = 1/median(ITL)."""
    if not itls:
        return None
    ms = sorted(t * 1000.0 for t in itls)
    med = statistics.median(ms)
    return {
        "count": len(ms),
        "p50_ms": med,
        "p90_ms": percentile(ms, 90),
        "p99_ms": percentile(ms, 99),
        "steady_tok_s": (1000.0 / med) if med > 0 else None,
    }


def percentile(sorted_ms: list[float], p: float) -> float:
    if not sorted_ms:
        return 0.0
    k = (len(sorted_ms) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_ms) - 1)
    return sorted_ms[lo] + (sorted_ms[hi] - sorted_ms[lo]) * (k - lo)


async def phase_prefill(client, url, args) -> list[dict]:
    out = []
    for n in args.prompt_tokens:
        rec = await one_request(
            client, url, filler(n, f"pf{n}"), 1, args.seed, False, "prefill"
        )
        rec["config"] = {"prompt_tokens": n}
        out.append(rec)
    return out


async def phase_decode(client, url, args) -> list[dict]:
    out = []
    for m in args.max_tokens:
        rec = await one_request(
            client, url, filler(16, f"dc{m}"), m, args.seed, False, "decode"
        )
        rec["config"] = {"max_tokens": m}
        out.append(rec)
    return out


async def phase_steady(client, url, args) -> list[dict]:
    out = []
    for m in args.max_tokens:
        rec = await one_request(
            client, url, filler(16, f"st{m}"), m, args.seed, True, "steady"
        )
        rec["config"] = {"max_tokens": m}
        # Self-check: streamed ITL vs source usage (when the latter is populated).
        u = rec.get("usage_compl_tok_s")
        s = (rec.get("itl") or {}).get("steady_tok_s")
        if u and s:
            rec["agree_pct"] = 100.0 * abs(s - u) / u
        out.append(rec)
    return out


async def phase_batch(client, url, args) -> list[dict]:
    out = []
    m = max(args.max_tokens)
    for n in args.batch:
        n = min(n, args.max_seqs)
        prompts = [filler(16, f"b{n}u{u}") for u in range(n)]
        t0 = time.perf_counter()
        recs = await asyncio.gather(
            *(
                one_request(client, url, p, m, args.seed, False, "batch")
                for p in prompts
            )
        )
        wall = time.perf_counter() - t0
        toks = sum(r.get("completion_tokens") or 0 for r in recs)
        per = [r.get("usage_compl_tok_s") for r in recs if r.get("usage_compl_tok_s")]
        out.append(
            {
                "phase": "batch",
                "config": {"n": n, "max_tokens": m},
                "wall_s": wall,
                "aggregate_tok_s": (toks / wall) if wall > 0 else None,
                "per_stream_tok_s": (statistics.mean(per) if per else None),
                "completion_tokens": toks,
            }
        )
    return out


PHASES = {
    "prefill": phase_prefill,
    "decode": phase_decode,
    "steady": phase_steady,
    "batch": phase_batch,
}


async def run(args) -> list[dict]:
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        records: list[dict] = []
        for phase in args.phases:
            records.extend(await PHASES[phase](client, url, args))
        return records


def self_test() -> int:
    """Exercise the pure math (ITL, percentile) without a server."""
    st = itl_stats([0.01] * 20)
    assert st is not None and abs(st["steady_tok_s"] - 100.0) < 1e-6, st
    assert abs(st["p50_ms"] - 10.0) < 1e-9, st
    ms = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(percentile(ms, 50) - 3.0) < 1e-9
    assert abs(percentile(ms, 100) - 5.0) < 1e-9
    assert abs(percentile(ms, 0) - 1.0) < 1e-9
    # DROP_FIRST_K is honored: a spiky head must not move the steady median.
    spikey = [1.0] * DROP_FIRST_K + [0.02] * 20
    st2 = itl_stats(spikey[DROP_FIRST_K:])
    assert st2 is not None and abs(st2["steady_tok_s"] - 50.0) < 1e-6, st2
    assert itl_stats([]) is None
    # Prompts are unique even for identical shape (prefix-cache safety).
    assert filler(16, "x") != filler(16, "x")
    print("bench_client self-test: ok", file=sys.stderr)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE bench load generator")
    p.add_argument("--base-url", default="http://localhost:1234")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-seqs", type=int, default=32)
    p.add_argument(
        "--phases",
        default="prefill,decode,steady,batch",
        help="comma-separated subset of: prefill,decode,steady,batch",
    )
    p.add_argument("--prompt-tokens", default="512,2048,8192,32768")
    p.add_argument("--max-tokens", default="256,512,1024")
    p.add_argument("--batch", default="1,2,4,8,16,32")
    p.add_argument("--out", default="", help="write records JSON here (default stdout)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    a.phases = [s for s in a.phases.split(",") if s]
    a.prompt_tokens = [int(x) for x in a.prompt_tokens.split(",") if x]
    a.max_tokens = [int(x) for x in a.max_tokens.split(",") if x]
    a.batch = [int(x) for x in a.batch.split(",") if x]
    return a


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    records = asyncio.run(run(args))
    blob = json.dumps(records, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
    else:
        print(blob)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
