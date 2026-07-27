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

## The per-sequence state machine

`SequenceState` holds everything a single request needs to advance on its own: its prompt,
its position, its stop strings, its token budget, its accounting. Both schedulers and the
single-request generator drive the same object, so the stop and accounting rules are defined
and tested exactly once rather than reimplemented per driver.

Two details there matter more than they look:

- **Positions are derived, not tracked.** The prompt occupies `[0, n_prompt)` and the k-th
  generated token is fed back in at `n_prompt + k - 1`. Once sequences advance out of
  lockstep, a mutable position counter is precisely where off-by-one bugs live; deriving it
  makes that class of bug unrepresentable.
- **Output is held back by `max_stop_len - 1` bytes.** A stop string can complete using bytes
  already generated, so emitting eagerly would leak text that must be suppressed. The held-back
  tail is flushed on a clean finish and discarded on a stop-string finish, which is what makes
  the streamed deltas and the buffered result provably identical.

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

The continuous scheduler therefore builds each step's batch in a deliberate order:

1. **Decodes claim one token each, first.** Sequences already generating always make progress;
   an arriving prompt can never stall them.
2. **Prefills share whatever token budget remains**, capped at `prefill_chunk` per sequence, so
   a 2000-token prompt is spread over many steps instead of monopolising one.

That trades the arriving request's time-to-first-token for everyone else's inter-token latency
— a policy knob worth measuring rather than a setting worth guessing, which is why it is a
flag (`--prefill-chunk`) and not a constant.

### Admission control on context length

llama.cpp divides its KV pool across `n_seq_max`, so a request is bounded by `n_ctx_seq`, not
`n_ctx` — raising the batch size silently shrinks every request's usable context. The scheduler
checks `prompt + max_tokens` against `n_ctx_seq` when admitting and rejects an oversized request
immediately with `kContextOverflow` (surfaced as HTTP 400), rather than admitting it and failing
deep in the backend after the prefill compute has already been spent. This is the same class of
decision Phase 5's block allocator generalises: knowing what will fit before starting work.

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
from being thrown away. This mirrors vLLM's recompute preemption mode. Two rules keep it from
degenerating, both learned the hard way (see [benchmarks.md](benchmarks.md)):

- **Never evict the sequence you are making room for.** Otherwise it is readmitted, grows,
  and is evicted again — a livelock indistinguishable from a hang.
- **Admit with headroom.** Reserving only a request's prompt blocks leaves nothing for it to
  grow into, so a newcomer immediately starves an incumbent, which preempts the newcomer, and
  the pair thrash without progressing. Admission therefore requires one spare block per
  active sequence plus one for the newcomer.

**Once the scheduler takes a request, it owns it.** Deferred and preempted requests wait in a
scheduler-local pending list rather than going back on the queue, so a normal shutdown cannot
strand work the server had already accepted.

**Why the budget is configurable.** Qwen2.5-0.5B uses roughly 12 KB of KV per token, so this
hardware cannot exhaust the cache by accident. `--kv-blocks` exists so admission control,
preemption, and eviction are observable and benchmarkable by design.

**Prefix sharing.** Block-aligned prompt prefixes are hashed; a hit increfs the donor's
blocks and mirrors the KV with `llama_memory_seq_cp`, so the new sequence prefills only its
divergent suffix — real work avoided, not bookkeeping.

**Donor retention.** A prefix would otherwise die with the request that produced it, limiting
sharing to sequences that overlap in time. Instead a retiring sequence's prefix is copied into
a reserved *donor slot* (ids allocated above the batch slots) and kept, so later,
non-overlapping requests still hit. Donors cost real resources — a sequence id and pinned
blocks — so they are bounded by `--prefix-donors`, evicted least-recently-used, and
**reclaimed before any live sequence is preempted**: a donor is cache whose loss costs a
recompute later, while a live sequence is work already done.

Two constraints here are easy to get wrong, and both fail silently or violently rather than
obviously:

- **The donor slot must be registered with the engine before copying into it.** A sequence
  exists only once the engine knows about it, so copying into an unregistered slot silently
  does nothing — and the next hit then inherits an empty prefix and produces *wrong output*
  rather than merely missing.
- **The context must be created with `kv_unified = true`.** `llama_memory_seq_cp` supports a
  partial range only on a unified cache; the per-stream path asserts `is_full` and aborts the
  process. Sharing a prefix is by definition a partial copy, so the non-unified default turns
  the first genuine cache hit into a crash.

## Benchmark methodology

The target machine is a 15 W laptop CPU with hybrid P/E cores that WSL presents as a flat
14-thread topology. Careless measurement on such a machine produces numbers that do not
survive scrutiny, so:

- thread count is swept once, then **held constant across every arm**
- arms are **interleaved (ABABAB), never grouped (AAABBB)**, so thermal drift hits both equally
- ≥5 repetitions, reported as **median and IQR**, with a discarded warm-up
- `llama-server` is measured as an external reference, not just our own static-batching baseline

Absolute throughput here is laptop-class. **The relative comparison is the claim.**
