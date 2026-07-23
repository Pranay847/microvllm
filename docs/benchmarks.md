# Benchmarks

All numbers on the development machine unless stated otherwise.

| | |
|---|---|
| CPU | Intel Core Ultra 7 155U — 2 P-cores + 8 E-cores + 2 LP-E cores (12C/14T), AVX2 / AVX_VNNI / F16C / FMA, **no AVX-512** |
| Memory | 15.4 GB host, 7.8 GB visible to WSL2 |
| OS | Ubuntu 24.04.3 under WSL2 on Windows 11 |
| Compiler | gcc 13.3.0, `-O3 -DNDEBUG` (`wsl-release` preset) |
| Backend | llama.cpp b10103 (`c588c4f`), CPU, `LLAMAFILE=1 OPENMP=1 REPACK=1` |
| Model | Qwen2.5-0.5B-Instruct Q4_K_M — 462.96 MiB, 630.17 M params |

This is a 15 W laptop part with a hybrid core layout that WSL flattens into 14 uniform
threads. **Absolute throughput here is laptop-class and not the point.** Every claim this
project makes is a *relative* comparison between scheduling strategies measured on the same
machine, in the same session, with the same thread count.

---

## Phase 0 — baseline and thread-count selection

Before any batching work, the single-sequence ceiling has to be known, and a thread count has
to be fixed so later comparisons are not confounded by it.

`llama-bench -p 512 -n 128 -r 3`, tokens/sec, mean ± stddev:

| Threads | pp512 (prefill) | tg128 (decode) |
|--------:|----------------:|---------------:|
| 2  | 170.47 ± 5.71   | 39.92 ± 2.46   |
| 4  | 215.03 ± 61.63  | 45.96 ± 6.15   |
| 6  | 157.96 ± 24.78  | 65.12 ± 22.39  |
| **8**  | **191.82 ± 8.30** | **63.39 ± 1.32** |
| 10 | 194.16 ± 44.13  | 69.88 ± 1.04   |
| 12 | 213.01 ± 2.87   | 43.57 ± 27.16  |

### Decision: `--threads 8`

Held constant across every arm of every later benchmark.

`t=10` has the best decode mean (69.88), but its prefill variance is severe (±44.13, 23% RSD).
`t=8` gives up ~9% decode throughput for stability on *both* axes — 2.1% RSD on decode, 4.3% on
prefill. Phase 4 measures time-to-first-token (prefill-sensitive) alongside throughput
(decode-sensitive), so a point that is reproducible on both is worth more than a point that
wins on one.

`t=12` is actively bad (43.57 ± 27.16): oversubscribing 14 hardware threads puts the model in
contention with system threads and the LP-E cores.

### Caveat on these numbers

`-r 3` gives three samples per cell, so the ± figures are themselves unreliable — enough to
rank configurations and reject `t=12`, not enough to publish. The final suite uses ≥5
repetitions reported as **median and IQR**, with interleaved arms (ABABAB) so thermal drift
hits every arm equally. `t=8` will be re-validated then.

### What the shape of this data means

Decode goes from 39.92 tok/s at 2 threads to 69.88 at 10 — **1.75× throughput for 5× the
threads.** That is the memory-bandwidth wall, and it is the single most important number on
this page, because it is *why batching works*:

At batch size 1, generating one token requires streaming all 463 MiB of weights through the
memory system to produce a single token. At batch size 16, the same weight traffic produces
sixteen tokens. Decode is bandwidth-bound, not compute-bound, so the marginal cost of another
sequence in the batch is nearly free until compute or KV-cache capacity becomes the binding
constraint.

Prefill behaves differently — it is compute-bound and scales with thread count (up to ~215
tok/s), which is exactly why the scheduler has to treat prefill and decode as different kinds
of work rather than as generic "steps".

---

## Baseline to beat

| Metric | Value |
|---|---|
| Single-sequence decode | **63.39 tok/s** (`t=8`, tg128) |
| Single-sequence prefill | **191.82 tok/s** (`t=8`, pp512) |

Phase 3 (static batching) and Phase 4 (continuous batching) are measured against this, plus
against `llama-server` as an external reference.

---

## Planned measurements

- **Phase 3** — throughput vs. batch size (1/4/8/16/32), locating the plateau
- **Phase 4** — continuous vs. static under mixed load (80% short 16–32 tokens, 20% long
  256–512): throughput, TTFT and TPOT percentiles, plus Poisson arrivals swept to the latency knee
- **Phase 5** — KV utilization and preemption counts under a constrained `--kv-blocks` budget;
  TTFT with and without prefix sharing
