#!/usr/bin/env python3
"""Two /v1/embeddings servers on the same inputs: the vectors, the latency, the concurrency, the memory.

  embed_compare.py INPUTS.jsonl URL_A URL_B [--model-a M] [--model-b M] [--pid-a PID] [--pid-b PID] [--out R.json]

INPUTS.jsonl holds one JSON string per line (the model's task prefix already in it). Per input the cosine of the
two servers' vectors (the least and the median); retrieval agreement: for each of the first Q inputs taken as
queries, the top-10 of the rest by each server's vectors, and how many of A's top-1 and top-10 B shares;
latency p50/p95 of one input a request, and of batches of 16; throughput at 1, 4 and 16 concurrent requests;
the resident set of each pid, and the GPU memory `nvidia-smi` attributes to it.
"""
import argparse
import concurrent.futures as cf
import json
import subprocess
import time
import urllib.request

import numpy as np


def embed(url, model, texts):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/embeddings", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)["data"]
    v = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def timed(url, model, texts, rounds):
    ms = []
    for i in range(rounds + 2):
        t = time.perf_counter()
        embed(url, model, texts)
        if i >= 2:
            ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return {"p50_ms": round(ms[len(ms) // 2], 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1)}


def throughput(url, model, texts, workers, seconds=15):
    done, stop = [0], time.monotonic() + seconds

    def work(k):
        n = 0
        while time.monotonic() < stop:
            embed(url, model, [texts[(k + n) % len(texts)]])
            n += 1
        return n

    with cf.ThreadPoolExecutor(workers) as ex:
        counts = list(ex.map(work, range(workers)))
    return round(sum(counts) / seconds, 1)


def memory(pid):
    if not pid:
        return None
    rss = next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")) / 1024
    gpu = None
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        p, m = [x.strip() for x in line.split(",")]
        if p == str(pid):
            gpu = int(m)
    return {"rss_mib": round(rss), "gpu_mib": gpu}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs"); ap.add_argument("url_a"); ap.add_argument("url_b")
    ap.add_argument("--model-a", default="zen-embedding-2"); ap.add_argument("--model-b", default="zen-embedding-2")
    ap.add_argument("--pid-a", type=int); ap.add_argument("--pid-b", type=int)
    ap.add_argument("--queries", type=int, default=20); ap.add_argument("--out")
    args = ap.parse_args()
    texts = [json.loads(l) for l in open(args.inputs)]
    result = {"inputs": len(texts)}
    vecs = {}
    for name, url, model in (("a", args.url_a, args.model_a), ("b", args.url_b, args.model_b)):
        vecs[name] = np.concatenate([embed(url, model, texts[i:i + 16]) for i in range(0, len(texts), 16)])
    cos = (vecs["a"] * vecs["b"]).sum(1)
    result["cosine"] = {"min": round(float(cos.min()), 6), "median": round(float(np.median(cos)), 6), "below_0.999": int((cos < 0.999).sum())}
    q = args.queries
    top = {n: [np.argsort(-(v[q:] @ v[i]))[:10] for i, _ in enumerate(range(q))] for n, v in vecs.items()}
    result["retrieval"] = {"queries": q, "top1_same": sum(int(top["a"][i][0] == top["b"][i][0]) for i in range(q)),
                           "top10_overlap_mean": round(float(np.mean([len(set(top["a"][i]) & set(top["b"][i])) for i in range(q)])), 2)}
    for name, url, model, pid in (("a", args.url_a, args.model_a, args.pid_a), ("b", args.url_b, args.model_b, args.pid_b)):
        short = [t for t in texts if len(t) < 300][:16] or texts[:16]
        result[name] = {"one": timed(url, model, short[:1], 40), "batch16": timed(url, model, short, 15),
                        "per_second": {w: throughput(url, model, texts, w) for w in (1, 4, 16)}, "memory": memory(pid)}
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
