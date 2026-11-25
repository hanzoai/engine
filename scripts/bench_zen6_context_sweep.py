#!/usr/bin/env python3
"""Sweep intermediate context lengths for Zen6 automatic mode selection.

Runs Dense, LSA, DFlash2, and DFlash2+LSA across intermediate context lengths
(default: 5k, 8k, 12k, 16k, 20k, 24k, 30k) using `hanzo-engine serve`, records
GPU utilization before/after each run to flag external contention, and parses
both `serve_api:` and `serve_trace:` telemetry via `serve_trace_report`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from serve_trace_report import aggregate_records, parse_log_lines_with_anomalies

BASE_PARAGRAPH = (
    "In distributed consensus protocols and high-throughput state machines, deterministic ordering of transactions "
    "under asynchronous network conditions requires careful coordination between validator committees. Each validator "
    "maintains a local mempool of pending operations, verifies cryptographic signatures, and proposes candidate blocks "
    "according to a rotating leader schedule. When network partitions or transient delays occur, view-change protocols "
    "ensure liveness without violating safety invariants, provided fewer than one-third of voting weight is Byzantine. "
    "Meanwhile, execution engines pipeline signature verification, state trie prefetching, and speculative execution "
    "to saturate modern multi-core and GPU-accelerated hardware architectures. "
)


def parse_context_spec(token: str) -> tuple[str, int]:
    s = token.strip().lower()
    if s.endswith("k"):
        n = int(float(s[:-1]) * 1024)
        return s, n
    n = int(s)
    label = f"{n // 1024}k" if n % 1024 == 0 else str(n)
    return label, n


def make_prompt(target_tokens: int, question: str) -> str:
    # BASE_PARAGRAPH tokenizes to ~115-120 tokens in Zen6/Qwen3.
    repeats = max(1, round(target_tokens / 116))
    body = BASE_PARAGRAPH * repeats
    return f"Context document:\n{body}\n\nQuestion: {question}"


def gpu_utilization_pct() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        first = out.splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


def http_req(port: int, method: str, path: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 600) -> tuple[int, Any]:
    url = f"http://127.0.0.1:{port}{path}"
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def wait_ready(proc: subprocess.Popen[bytes], port: int, timeout: int = 240) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited early with code {proc.returncode}")
        try:
            code, data = http_req(port, "GET", "/v1/models", timeout=3)
            if code == 200 and data.get("data"):
                return data["data"][0]["id"]
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError("Server did not become ready in time")


def run_completion(port: int, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            pkt = json.loads(data_str)
            if pkt.get("usage"):
                usage = pkt["usage"]
            choices = pkt.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or delta.get("reasoning_content") or ""
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(c)
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    c_tok = int(usage.get("completion_tokens", len(chunks)))
    p_tok = int(usage.get("prompt_tokens", 0))
    decode_s = max(1e-6, total - ttft)
    return {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "decode_tok_s": round(max(0, c_tok - 1) / decode_s if c_tok > 1 else c_tok / decode_s, 2),
        "output_preview": "".join(chunks)[:160],
    }


def run_case(
    binary: str,
    model_path: str,
    dflash_path: str | None,
    mode: str,
    ctx_label: str,
    ctx_tokens: int,
    max_tokens: int,
    trials: int,
    port: int,
    log_dir: Path,
    admin_token: str,
    lsa_min_seq_len: int,
    dflash_block_size: int | None,
) -> dict[str, Any]:
    use_lsa = mode in ("lsa", "combo")
    use_dflash = mode in ("dflash", "combo")
    if use_dflash and not dflash_path:
        raise ValueError(f"Mode {mode} requested without --dflash")

    pa_ctx = max(16384, ctx_tokens + max_tokens + 2048)
    # Force LSA below tested ctx when lsa mode is explicitly tested at 5K+.
    effective_lsa_min = min(lsa_min_seq_len, max(1024, ctx_tokens // 2)) if use_lsa else lsa_min_seq_len

    log_path = log_dir / f"sweep_{ctx_label}_{mode}.log"
    if log_path.exists():
        log_path.unlink()

    cmd = [
        binary,
        "serve",
        "--format",
        "gguf",
        "-m",
        os.path.dirname(model_path),
        "-f",
        os.path.basename(model_path),
        "--port",
        str(port),
        "--max-seqs",
        "2",
        "--prefix-cache-n",
        "16",
        "--pa-context-len",
        str(pa_ctx),
        "--no-ui",
        "--no-advertise",
    ]
    if use_lsa:
        cmd.extend([
            "--lsa",
            "--lsa-min-seq-len",
            str(effective_lsa_min),
            "--lsa-top-k",
            "512",
            "--lsa-top-p",
            "128",
        ])
    if use_dflash and dflash_path:
        cmd.extend(["--dflash", dflash_path])
        if dflash_block_size:
            cmd.extend(["--dflash-block-size", str(dflash_block_size)])

    env = os.environ.copy()
    env["ENGINE_LSA_ADMIN_TOKEN"] = admin_token
    gpu_before = gpu_utilization_pct()
    trial_results: list[dict[str, Any]] = []

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        try:
            wait_ready(proc, port)
            if use_lsa:
                http_req(
                    port,
                    "POST",
                    "/v1/lsa",
                    {
                        "min_seq_len": effective_lsa_min,
                        "window_size": 512,
                        "sink_size": 4,
                        "top_k": 512,
                        "top_p": 128,
                        "rotary_layout": "prefix",
                        "rotary_dim": 64,
                    },
                    headers={"Authorization": f"Bearer {admin_token}"},
                )
            # Warmup short request (excluded from throughput via min_prompt_tokens).
            run_completion(port, "Reply with OK.", max_tokens=1)
            for idx in range(trials):
                prompt = make_prompt(
                    ctx_tokens,
                    f"Trial {idx + 1}: list the four core mechanisms for safety and high throughput in 4 concise bullets.",
                )
                trial_results.append(run_completion(port, prompt, max_tokens=max_tokens))
            log_f.flush()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()

    gpu_after = gpu_utilization_pct()
    with open(log_path, "r", errors="replace") as f:
        records, pairing = parse_log_lines_with_anomalies(f.readlines(), source=log_path.name)
    trace_summary = aggregate_records(
        records,
        min_completion_tokens=min(8, max_tokens),
        min_prompt_tokens=max(512, ctx_tokens // 2),
        pairing_anomalies=pairing,
    )
    return {
        "context_label": ctx_label,
        "target_tokens": ctx_tokens,
        "mode": mode,
        "gpu_util_before_pct": gpu_before,
        "gpu_util_after_pct": gpu_after,
        "contended": bool((gpu_before or 0) > 10 or (gpu_after or 0) > 10),
        "trials": trial_results,
        "serve_trace": trace_summary,
        "log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", default="/tmp/wt-engine-lsa-dflash/target/release/hanzo-engine")
    parser.add_argument("--model", default="/data/models/zen6-source/4B-gguf/Ternary-Bonsai-2-4B-PQ2_0.gguf")
    parser.add_argument("--dflash", default="/data/models/zen6-source/4B-gguf/Ternary-Bonsai-2-4B-DFlash2-BF16.gguf")
    parser.add_argument("--contexts", default="5k,8k,12k,16k,20k,24k,30k")
    parser.add_argument("--modes", default="dense,lsa,dflash,combo")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--port", type=int, default=18946)
    parser.add_argument("--lsa-min-seq-len", type=int, default=4096)
    parser.add_argument("--dflash-block-size", type=int, default=8)
    parser.add_argument("--log-dir", default="/tmp/zen6_context_sweep")
    parser.add_argument("--out-json", default="/tmp/zen6_context_sweep.json")
    parser.add_argument("--admin-token", default="hanzo-lsa-test-admin-secret-2026")
    parser.add_argument("--allow-contended", action="store_true", help="Run even when GPU utilization is initially non-zero")
    args = parser.parse_args(argv)

    gpu_now = gpu_utilization_pct()
    if gpu_now is not None and gpu_now > 15 and not args.allow_contended:
        print(
            f"Refusing to run uncontended context sweep while GPU utilization is {gpu_now}%. Re-run with --allow-contended if intentional.",
            file=sys.stderr,
        )
        return 1

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    contexts = [parse_context_spec(c) for c in args.contexts.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results: list[dict[str, Any]] = []
    for ctx_label, ctx_tokens in contexts:
        for mode in modes:
            res = run_case(
                binary=args.bin,
                model_path=args.model,
                dflash_path=args.dflash if mode in ("dflash", "combo") else None,
                mode=mode,
                ctx_label=ctx_label,
                ctx_tokens=ctx_tokens,
                max_tokens=args.max_tokens,
                trials=args.trials,
                port=args.port,
                log_dir=log_dir,
                admin_token=args.admin_token,
                lsa_min_seq_len=args.lsa_min_seq_len,
                dflash_block_size=args.dflash_block_size,
            )
            results.append(res)
            with open(args.out_json, "w") as f:
                json.dump({"results": results}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
