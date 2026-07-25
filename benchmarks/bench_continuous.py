#!/usr/bin/env python3
"""Continuous vs. static batching under a mixed-length workload.

This is the project's headline measurement. Uniform-length traffic is static batching's
best case (every sequence finishes on the same step, so no slot idles); real traffic is
mixed, and that is where holding a batch until its longest member finishes costs you.

The workload is 80% short requests (16-32 tokens) and 20% long ones (256-512), which is
roughly the shape of chat traffic. Under static batching a short request unlucky enough to
share a batch with a long one holds its slot for the long one's entire generation while
producing nothing. Continuous batching retires it immediately and refills the slot.

Two numbers matter, and they are different questions:
  * throughput  -- how much total work the engine gets through
  * short-request tail latency -- whether a small request stays fast under mixed load

The second is the one users feel, and is where the gap should be widest.

Methodology matches docs/benchmarks.md: one server per arm started up front, arms
INTERLEAVED not grouped so thermal drift hits both equally, a discarded warm-up, and
median + IQR over several rounds rather than a mean.

Usage:
    python benchmarks/bench_continuous.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""
import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def wait_healthy(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.3)
    return False


def one_request(port, prompt, max_tokens):
    """Returns (ok, completion_tokens, latency_seconds)."""
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            payload = json.loads(resp.read().decode())
            return (True, payload["usage"]["completion_tokens"], time.perf_counter() - start)
    except Exception as e:  # noqa: BLE001
        print(f"  request failed: {e}", file=sys.stderr)
        return (False, 0, time.perf_counter() - start)


def build_workload(n, seed, short_frac=0.8):
    """Fixed mixed-length workload. Identical across arms so the comparison is fair."""
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        if rng.random() < short_frac:
            jobs.append(("short", f"Briefly answer question {i}:", rng.randint(16, 32)))
        else:
            jobs.append(("long", f"Write at length about subject {i}:", rng.randint(256, 512)))
    rng.shuffle(jobs)
    return jobs


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def run_round(port, jobs, concurrency):
    """Run the whole workload; return per-class latencies and aggregate throughput."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(
            pool.map(lambda j: (j[0],) + one_request(port, j[1], j[2]), jobs)
        )
    wall = time.perf_counter() - start

    short_lat = sorted(r[3] for r in results if r[0] == "short" and r[1])
    long_lat = sorted(r[3] for r in results if r[0] == "long" and r[1])
    tokens = sum(r[2] for r in results if r[1])
    failures = sum(1 for r in results if not r[1])
    return {
        "tok_s": tokens / wall if wall > 0 else 0.0,
        "wall": wall,
        "short_p50": statistics.median(short_lat) if short_lat else 0.0,
        "short_p95": pct(short_lat, 0.95),
        "short_p99": pct(short_lat, 0.99),
        "long_p50": statistics.median(long_lat) if long_lat else 0.0,
        "failures": failures,
    }


def summarize(samples, key):
    vals = sorted(s[key] for s in samples)
    return statistics.median(vals), vals[len(vals) // 4], vals[(3 * len(vals)) // 4]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default=None)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base-port", type=int, default=8500)
    args = ap.parse_args()

    binary = args.binary or os.path.expanduser("~/build/microvllm/wsl-release/src/microvllm")
    arms = ["static", "continuous"]
    jobs = build_workload(args.requests, args.seed)
    n_short = sum(1 for j in jobs if j[0] == "short")
    print(
        f"workload: {len(jobs)} requests -- {n_short} short (16-32 tok), "
        f"{len(jobs) - n_short} long (256-512 tok)\n"
    )

    servers = {}
    try:
        for idx, arm in enumerate(arms):
            port = args.base_port + idx
            proc = subprocess.Popen(
                [
                    binary, "--model", args.model,
                    "--threads", str(args.threads),
                    "--port", str(port),
                    "--batch-size", str(args.batch_size),
                    "--scheduler", arm,
                    "--queue", str(max(256, args.requests * 2)),
                    "--quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            servers[arm] = (proc, port)
            print(f"starting {arm} scheduler on port {port}...", flush=True)

        for arm, (_p, port) in servers.items():
            if not wait_healthy(port):
                print(f"{arm} server never became healthy", file=sys.stderr)
                return 1
        print("all servers healthy\n", flush=True)

        print("warm-up round (discarded)...", flush=True)
        warm = build_workload(6, args.seed + 99)
        for _arm, (_p, port) in servers.items():
            run_round(port, warm, args.concurrency)

        samples = {arm: [] for arm in arms}
        for rnd in range(1, args.rounds + 1):
            for arm in arms:  # interleaved
                _p, port = servers[arm]
                r = run_round(port, jobs, args.concurrency)
                samples[arm].append(r)
                print(
                    f"  round {rnd}  {arm:<11} {r['tok_s']:7.2f} tok/s   "
                    f"short p50={r['short_p50']:6.2f}s p99={r['short_p99']:6.2f}s   "
                    f"wall={r['wall']:6.1f}s"
                    + (f"  FAILURES={r['failures']}" if r["failures"] else ""),
                    flush=True,
                )

        print(f"\n{'metric':<22}{'static':>12}{'continuous':>14}{'change':>12}")
        print("-" * 60)
        rows = [
            ("throughput tok/s", "tok_s", True),
            ("short p50 latency s", "short_p50", False),
            ("short p95 latency s", "short_p95", False),
            ("short p99 latency s", "short_p99", False),
            ("long p50 latency s", "long_p50", False),
            ("wall clock s", "wall", False),
        ]
        for label, key, higher_better in rows:
            s_med, _, _ = summarize(samples["static"], key)
            c_med, _, _ = summarize(samples["continuous"], key)
            if s_med == 0:
                change = "n/a"
            elif higher_better:
                change = f"{c_med / s_med:.2f}x"
            else:
                change = f"{(1 - c_med / s_med) * 100:+.0f}%"
            print(f"{label:<22}{s_med:>12.2f}{c_med:>14.2f}{change:>12}")

        print(f"\n{'':22}{'IQR (static)':>22}{'IQR (continuous)':>24}")
        for label, key, _ in rows[:4]:
            _, sq1, sq3 = summarize(samples["static"], key)
            _, cq1, cq3 = summarize(samples["continuous"], key)
            print(f"{label:<22}{sq1:>10.2f}-{sq3:<11.2f}{cq1:>12.2f}-{cq3:<11.2f}")

        print(
            f"\nconfig: batch {args.batch_size}, concurrency {args.concurrency}, "
            f"{args.threads} threads, {args.rounds} interleaved rounds, seed {args.seed}"
        )
        return 0
    finally:
        for _arm, (proc, _port) in servers.items():
            proc.send_signal(signal.SIGINT)
        for _arm, (proc, _port) in servers.items():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
