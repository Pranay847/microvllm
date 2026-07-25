#!/usr/bin/env python3
"""Throughput vs. batch size for microvllm's static batching scheduler.

Methodology (see docs/benchmarks.md -- this laptop is a 15 W part whose numbers are
worthless if measured carelessly):

  * One server per batch size, all started up front. llama.cpp mmaps the weights, so the
    model pages are shared across processes and only the KV cache is duplicated.
  * Arms are INTERLEAVED (round 1: bs=1,4,8,...; round 2: bs=1,4,8,...), never grouped,
    so thermal drift hits every arm equally instead of penalising whichever ran last.
  * A discarded warm-up round precedes the measured rounds.
  * Reported as MEDIAN and IQR across rounds, not mean, because a stray scheduling
    hiccup on a shared laptop skews a mean badly.
  * Thread count is held constant at the Phase 0 operating point for every arm.

The workload is uniform-length on purpose: every request asks for the same number of
tokens, so static batching is measured at its best. Ragged lengths are where static
batching falls apart, and that is exactly the comparison Phase 4 makes.

Usage:
    python benchmarks/bench_batching.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def wait_healthy(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(0.3)
    return False


def generate(port, prompt, max_tokens):
    """One /generate call. Returns completion_tokens, or 0 on failure."""
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())["usage"]["completion_tokens"]
    except Exception as e:  # noqa: BLE001
        print(f"  request failed: {e}", file=sys.stderr)
        return 0


def run_round(port, n_requests, max_tokens, concurrency):
    """Fire n_requests concurrently; return (output_tokens, wall_seconds)."""
    prompts = [f"Write about topic number {i}:" for i in range(n_requests)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        counts = list(pool.map(lambda p: generate(port, p, max_tokens), prompts))
    return sum(counts), time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default=None, help="path to the microvllm executable")
    ap.add_argument("--batch-sizes", default="1,4,8,16,32")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=8200)
    args = ap.parse_args()

    binary = args.binary or os.path.expanduser(
        "~/build/microvllm/wsl-release/src/microvllm"
    )
    sizes = [int(s) for s in args.batch_sizes.split(",")]

    # --- start one server per arm -------------------------------------------------
    servers = {}
    try:
        for idx, bs in enumerate(sizes):
            port = args.base_port + idx
            proc = subprocess.Popen(
                [
                    binary, "--model", args.model,
                    "--threads", str(args.threads),
                    "--port", str(port),
                    "--batch-size", str(bs),
                    "--queue", str(max(256, args.requests * 2)),
                    "--quiet",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            servers[bs] = (proc, port)
            print(f"starting batch-size {bs} on port {port}...", flush=True)

        for bs, (_proc, port) in servers.items():
            if not wait_healthy(port):
                print(f"server for batch-size {bs} never became healthy", file=sys.stderr)
                return 1
        print("all servers healthy\n", flush=True)

        # --- warm-up round (discarded) ------------------------------------------
        print("warm-up round (discarded)...", flush=True)
        for bs, (_proc, port) in servers.items():
            run_round(port, min(8, args.requests), args.max_tokens, args.requests)

        # --- measured rounds, INTERLEAVED ---------------------------------------
        samples = {bs: [] for bs in sizes}
        for rnd in range(1, args.rounds + 1):
            for bs in sizes:  # interleaved: every arm sampled once per round
                _proc, port = servers[bs]
                tokens, wall = run_round(
                    port, args.requests, args.max_tokens, args.requests
                )
                tps = tokens / wall if wall > 0 else 0.0
                samples[bs].append(tps)
                print(
                    f"  round {rnd}  bs={bs:<3} {tokens:>5} tok in {wall:6.2f}s"
                    f"  -> {tps:7.2f} tok/s",
                    flush=True,
                )

        # --- report --------------------------------------------------------------
        print(f"\n{'batch':>6} {'median tok/s':>13} {'IQR':>16} {'speedup':>9}")
        print("-" * 50)
        baseline = statistics.median(samples[sizes[0]])
        rows = []
        for bs in sizes:
            vals = sorted(samples[bs])
            med = statistics.median(vals)
            q1 = vals[len(vals) // 4]
            q3 = vals[(3 * len(vals)) // 4]
            speedup = med / baseline if baseline > 0 else 0.0
            rows.append((bs, med, q1, q3, speedup))
            print(f"{bs:>6} {med:>13.2f} {q1:>7.2f}-{q3:<8.2f} {speedup:>8.2f}x")

        print(
            f"\nconfig: {args.requests} requests x {args.max_tokens} tokens, "
            f"{args.threads} threads, {args.rounds} interleaved rounds"
        )
        return 0
    finally:
        for _bs, (proc, _port) in servers.items():
            proc.send_signal(signal.SIGINT)
        for _bs, (proc, _port) in servers.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
