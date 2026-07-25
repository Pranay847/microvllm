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

## Phase 3 — throughput vs. batch size (static batching)

End-to-end through the HTTP server: 32 concurrent requests, 32 output tokens each, uniform
length. `benchmarks/bench_batching.py`, 5 interleaved rounds after a discarded warm-up, one
server process per arm, `--threads 8` throughout.

| Batch size | Median tok/s | IQR | Speedup |
|---:|---:|---:|---:|
| 1  | 35.36 | 32.69 – 38.62 | 1.00× |
| 4  | 55.25 | 42.38 – 57.90 | 1.56× |
| 8  | 63.61 | 62.52 – 64.27 | 1.80× |
| **16** | **72.84** | 72.05 – 73.09 | **2.06×** |
| 32 | 56.85 | 56.59 – 61.23 | 1.61× |

**Peak is 2.06× at batch size 16, and throughput *regresses* at 32.**

### Reading the curve

The gain is real but sublinear, and it is bounded by the same memory-bandwidth wall the
Phase 0 thread sweep exposed. Batching amortises weight traffic — one pass over 463 MiB of
weights advances every sequence in the batch instead of one — so throughput climbs steeply
from 1 to 8. Past that, per-step overhead that does *not* amortise (attention over each
sequence's own KV cache, sampling per sequence, the scheduler's own bookkeeping) grows
linearly with batch size and starts to dominate.

The regression at 32 is the more interesting half. Two effects compound:

- **KV-cache pressure.** 32 sequences × 4096 context × ~12 KB/token of KV is a far larger
  working set than 16, and it stops fitting the cache hierarchy the way smaller batches do.
  This is precisely the resource Phase 5's block allocator exists to manage.
- **Ragged completion.** Static batching holds the whole batch until its *last* member
  finishes. With 32 sequences the odds that at least one runs long rise, and every finished
  sequence's slot sits idle until then. Batch occupancy decays over the batch's lifetime, and
  the wider the batch the more of it decays.

Note the variance, too: batch 16 is the *tightest* arm (IQR 72.05–73.09, ~1.4% spread) while
batch 1 is the loosest (32.69–38.62, ~17%). Serial execution is dominated by per-request
overhead and scheduling jitter; a well-fed batch is dominated by steady memory traffic, which
is far more reproducible.

### The limitation this measures

Uniform-length requests are static batching's **best case** — every sequence finishes on the
same step, so no slot idles. The workload here was chosen that way deliberately, to isolate
the batching gain from the batching *penalty*.

Real traffic is mixed-length, and that is where static batching breaks down: a 16-token
request batched with a 512-token one occupies its slot for all 512 steps, producing nothing
for 496 of them. Phase 4's continuous batching retires finished sequences and admits waiting
ones every step, so that slot goes back to work immediately. The mixed-length comparison
between the two is the project's headline measurement, and this table is its control arm.

### A crash the benchmark found

The first run of this sweep returned zeros for the entire `bs=32` arm — every request refused.
The server was not slow, it was **dying**: `LlamaEngineConfig::n_seq_max` defaults to 16, and
submitting a batch with more sequences than the context was built for makes llama.cpp abort
the process. Fixed in two places: `main` now sizes the context to the requested batch, and the
scheduler clamps `max_batch_size` to `engine.caps().n_seq_max` regardless, on the principle
that no command-line value should be able to abort the server. Both are pinned by regression
tests.

The 60-test unit suite could not have caught this — the mock engine's sequence capacity was
never exceeded. It took a real model at a real batch size, which is the argument for
benchmarking against the real backend rather than trusting the mock alone.

---

## Planned measurements

- ~~**Phase 3** — throughput vs. batch size (1/4/8/16/32), locating the plateau~~ ✅ above
- **Phase 4** — continuous vs. static under mixed load (80% short 16–32 tokens, 20% long
  256–512): throughput, TTFT and TPOT percentiles, plus Poisson arrivals swept to the latency knee
- **Phase 5** — KV utilization and preemption counts under a constrained `--kv-blocks` budget;
  TTFT with and without prefix sharing
