# Architecture

## The problem

A naive LLM server processes one request at a time. Under concurrent load this wastes almost
all available compute, for two reasons:

1. **Autoregressive generation is sequential.** Each new token needs the attention keys and
   values of every previous token. Recomputing them is O(n²) wasted work, so they are cached —
   and the KV-cache, not compute, becomes the binding constraint.
2. **A single sequence cannot saturate the hardware.** Decoding one token for one sequence is
   memory-bandwidth-bound. Decoding one token for sixteen sequences costs barely more wall-clock
   than doing it for one.

The job of a serving engine is to keep as many sequences in flight as memory allows, and to
never let one request's shape stall another's progress.

## Component map

```
  HTTP threads (cpp-httplib pool)          Engine thread (owns llama_context)
 ┌──────────────────────────────┐         ┌────────────────────────────────────┐
 │ POST /generate  (blocking)   │         │  Scheduler::step()  ── loop        │
 │ POST /generate/stream (SSE)  │         │   1. reap finished seqs            │
 │ GET  /metrics                │         │   2. admit from queue (if blocks)  │
 └───────────┬──────────────────┘         │   3. build llama_batch             │
             │ RequestSpec + ITokenSink   │   4. engine.decode()               │
             v                            │   5. sample per-seq, emit to sink  │
      ┌──────────────┐   pop              │   6. check stops / grow / preempt  │
      │ RequestQueue │ ─────────────────► │   7. record metrics                │
      │ mutex+condvar│                    └──────────┬─────────────────────────┘
      └──────────────┘                               │
                                     ┌───────────────┴───────────────┐
                                     v                               v
                            ┌─────────────────┐           ┌────────────────────┐
                            │ BlockAllocator  │           │   IModelEngine     │
                            │ free list       │           ├────────────────────┤
                            │ per-seq tables  │           │ LlamaModelEngine   │
                            │ refcount + COW  │           │ MockModelEngine    │
                            │ prefix index    │           └────────────────────┘
                            └─────────────────┘
```

## The threading invariant

**Exactly one thread ever touches `llama_context`.** llama.cpp's context is not thread-safe,
so all backend calls happen on the engine thread. HTTP threads only ever touch:

- the `RequestQueue` (mutex + condition variable),
- a per-request sink: an atomic cancellation flag plus an SPSC token ring,
- metrics counters.

That is the entire shared surface. It is small on purpose — it is what `test_concurrency`
exercises under ThreadSanitizer, and a small surface is what makes a clean TSan run mean
something rather than being an accident of low contention.

## Why the model sits behind an interface

`IModelEngine` has two implementations: `LlamaModelEngine` and a deterministic
`MockModelEngine` with configurable per-token latency and stop behaviour. This is the single
most load-bearing decision in the project:

- **The scheduler becomes testable.** Scripted arrival traces can assert the exact sequence of
  admissions, preemptions, and completions. Continuous batching is then *proven* correct rather
  than inferred from a speedup.
- **CI needs no model.** A 469 MB download on every push is untenable; `src/core` has no
  llama.cpp dependency at all.
- **Sanitizer runs stay signal-rich.** Instrumenting ggml's thread pool buries first-party
  findings under backend noise. With the backend excluded, any TSan report is necessarily ours.
- **Scheduling cost is isolated.** Benchmarks against the mock engine measure the scheduler,
  not the model.

## Scheduling: static vs. continuous batching

**Static batching** collects N requests, prefills them together, and decodes in lockstep until
*all* finish. If one request wants 500 tokens and another wants 20, the short one occupies a
slot doing nothing for 480 steps.

**Continuous batching** rebuilds the batch every step. Finished sequences are retired and their
cache freed; newly arrived ones are admitted into the vacated capacity immediately. Each
sequence tracks its own position and state — there is no lockstep. This is the single largest
throughput win in real serving, and the effect is largest under *mixed-length* load, which is
why the headline benchmark uses an 80% short / 20% long workload.

### Prefill vs. decode

Prefill (processing the prompt) is compute-bound and processes many tokens at once. Decode is
memory-bandwidth-bound and processes one token per sequence per step. A long prefill admitted
whole will stall every decode in the batch — head-of-line blocking *inside* the batch.

The scheduler therefore treats prefill as a budgeted resource: **chunked prefill** splits a long
prompt across several steps so decodes keep progressing. This trades that request's
time-to-first-token for everyone else's inter-token latency, which is a policy knob worth
measuring rather than a setting worth guessing.

## KV-cache: the block allocator

Memory is modelled as fixed-size blocks (default 16 tokens). Each sequence owns a `BlockTable`
mapping logical positions to physical block ids — a page table. Blocks are refcounted so that
sequences sharing a prompt prefix can share the underlying blocks, with copy-on-write at the
point they diverge.

Invariants, enforced by unit tests:

- `blocks.size() == ceil(n_tokens / block_size)` for every live sequence
- no block is ever handed out twice
- every block returns to the free list exactly once, at refcount zero

**What the allocator actually owns.** llama.cpp holds the KV bytes. This allocator holds the
policy: who gets admitted, who gets evicted, what is shared. With `kv_unified = true`, one
block corresponds to `block_size` cells of llama.cpp's unified cache and
`total_blocks = n_ctx / block_size`. Every allocator decision is mirrored into the backend via
`llama_memory_seq_rm` (free) and `llama_memory_seq_cp` (share). Prefix sharing is therefore a
real memory saving, not a simulated one — but the allocator is a governor over the backend's
memory, not a replacement for it.

**Preemption.** When a running sequence needs another block and the pool is empty, the
most-recently-admitted sequence is evicted, its blocks freed, and it is requeued to be
recomputed on readmission. LIFO order protects the progress of older, closer-to-done requests
from being thrown away. This mirrors vLLM's recompute preemption mode.

**Why the budget is configurable.** Qwen2.5-0.5B uses roughly 12 KB of KV per token, so this
hardware cannot exhaust the cache by accident. `--kv-blocks` exists so admission control,
preemption, and eviction are observable and benchmarkable by design.

## Benchmark methodology

The target machine is a 15 W laptop CPU with hybrid P/E cores that WSL presents as a flat
14-thread topology. Careless measurement on such a machine produces numbers that do not
survive scrutiny, so:

- thread count is swept once, then **held constant across every arm**
- arms are **interleaved (ABABAB), never grouped (AAABBB)**, so thermal drift hits both equally
- ≥5 repetitions, reported as **median and IQR**, with a discarded warm-up
- `llama-server` is measured as an external reference, not just our own static-batching baseline

Absolute throughput here is laptop-class. **The relative comparison is the claim.**
