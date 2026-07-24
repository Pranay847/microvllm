#!/usr/bin/env python3
"""Concurrent load generator for microvllm's /generate endpoint.

Its first job is a correctness check, not a benchmark: fire N requests at a
configurable concurrency and assert that every one comes back, with no dropped,
duplicated, or mismatched responses. Against a server started with --mock-echo it
also verifies that each response body echoes that request's own prompt, which is
the end-to-end proof that the queue/worker boundary never crosses wires.

Usage:
    # start a server first, e.g.:
    #   microvllm --mock-echo --port 8080
    #   microvllm --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --quiet
    python benchmarks/load_gen.py --requests 500 --concurrency 32 --check-echo

Standard library only (urllib) so it runs anywhere Python 3 does.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def one_request(base_url, prompt, max_tokens):
    """Send a single /generate request. Returns (ok, status, text, latency_s, error)."""
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        base_url + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
            latency = time.perf_counter() - start
            return (True, resp.status, payload.get("text", ""), latency, None)
    except urllib.error.HTTPError as e:
        return (False, e.code, "", time.perf_counter() - start, f"HTTP {e.code}")
    except Exception as e:  # noqa: BLE001 - report anything that goes wrong
        return (False, 0, "", time.perf_counter() - start, str(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument(
        "--check-echo",
        action="store_true",
        help="assert each response equals its prompt (server must be --mock-echo)",
    )
    args = ap.parse_args()

    # Every prompt is unique, so a mismatched or duplicated response is detectable.
    prompts = [f"req-{i}-token{i * 7 % 1000}" for i in range(args.requests)]

    results = {}
    lock = threading.Lock()
    t0 = time.perf_counter()

    def task(idx):
        ok, status, text, latency, err = one_request(
            args.url, prompts[idx], args.max_tokens
        )
        with lock:
            results[idx] = (ok, status, text, latency, err)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(task, i) for i in range(args.requests)]
        for _ in as_completed(futures):
            pass

    wall = time.perf_counter() - t0

    # --- correctness ---
    missing = [i for i in range(args.requests) if i not in results]
    failures = [(i, r) for i, r in results.items() if not r[0]]
    mismatches = []
    if args.check_echo:
        for i, (ok, _status, text, _lat, _err) in results.items():
            if ok and text != prompts[i]:
                mismatches.append((i, prompts[i], text))

    latencies = [r[3] for r in results.values() if r[0]]
    ok_count = sum(1 for r in results.values() if r[0])

    print(f"requests:     {args.requests} @ concurrency {args.concurrency}")
    print(f"wall time:    {wall:.2f}s  ({args.requests / wall:.1f} req/s)")
    print(f"succeeded:    {ok_count}")
    print(f"failed:       {len(failures)}")
    if latencies:
        latencies.sort()
        p = lambda q: latencies[min(len(latencies) - 1, int(q * len(latencies)))]  # noqa: E731
        print(
            f"latency (s):  p50={statistics.median(latencies):.3f} "
            f"p95={p(0.95):.3f} p99={p(0.99):.3f} max={latencies[-1]:.3f}"
        )

    problems = []
    if missing:
        problems.append(f"{len(missing)} requests produced no result at all")
    if failures:
        codes = {}
        for _i, r in failures:
            codes[r[4]] = codes.get(r[4], 0) + 1
        problems.append(f"{len(failures)} failed: {codes}")
    if mismatches:
        problems.append(f"{len(mismatches)} responses did not match their prompt")
        for i, want, got in mismatches[:5]:
            print(f"  MISMATCH #{i}: sent {want!r} got {got!r}", file=sys.stderr)

    if problems:
        print("\nFAIL:", "; ".join(problems), file=sys.stderr)
        return 1

    checked = " and all echoes matched" if args.check_echo else ""
    print(f"\nOK: all {ok_count} requests returned correctly{checked}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
