# microvllm

A miniature LLM inference server in C++20. It implements the techniques that make real
serving engines fast under concurrent load — **continuous batching**, **request scheduling**,
and a **block-based KV-cache allocator with admission control** — around llama.cpp as the
tensor backend.

> **Status: Phase 3 complete.** A concurrent HTTP server serves a real model with **static
> batching** — multiple sequences packed into one forward pass, measured at **2.06× the
> single-sequence throughput** at batch size 16
> ([benchmarks](docs/benchmarks.md#phase-3--throughput-vs-batch-size-static-batching)).
> Continuous batching (Phase 4) and the paged KV-cache allocator (Phase 5) are next. See
> [docs/architecture.md](docs/architecture.md) for the design.

## What this is, and what it isn't

This project is the **serving infrastructure**, not the model math. llama.cpp performs the
matrix multiplications; everything around it — the queue, the scheduler, the batching loop,
the cache accounting, the HTTP layer — is the deliverable. That split is the same one vLLM
makes between its scheduler and its kernels.

One boundary is worth stating plainly up front, because it is the first thing a careful
reader will ask about: **llama.cpp owns the KV-cache bytes.** The `BlockAllocator` in this
project owns the *policy* — admission, eviction, preemption, and prefix sharing — and mirrors
every decision into llama.cpp's cache via `llama_memory_seq_rm` and `llama_memory_seq_cp`.
It is a real allocator with real invariants, but it is a governor over someone else's memory,
not a replacement for it.

## Requirements

Development targets **WSL2 / Linux**. This is not a stylistic preference: ThreadSanitizer has
no Windows implementation in either MSVC or clang-cl, and TSan validation of the scheduling
path is a core goal. A Windows/MSVC preset exists as a compile check only.

- CMake ≥ 3.25, Ninja, a C++20 compiler (gcc 13+ or clang 18+)
- ~2 GB disk for build trees, ~500 MB for the model

## Quick start

```bash
bash scripts/setup_wsl.sh      # apt deps + verifies asan/tsan/ubsan actually link
bash scripts/fetch_model.sh    # Qwen2.5-0.5B-Instruct Q4_K_M (~469 MB) into models/
cmake --preset wsl-release
cmake --build --preset wsl-release
ctest --preset wsl-release
```

## Build presets

| Preset | Purpose |
|---|---|
| `wsl-debug` | Day-to-day development |
| `wsl-release` | **All published benchmark numbers come from here** |
| `wsl-asan-ubsan` | AddressSanitizer + UBSan, first-party targets only |
| `wsl-tsan` | ThreadSanitizer, first-party targets only |
| `win-msvc` | Cross-platform compile check (no TSan available) |

Sanitizer presets set `MICROVLLM_BUILD_LLAMA=OFF` on purpose. Instrumenting ggml's thread pool
buries first-party findings under backend noise and makes the build unusably slow; with the
backend excluded, any report from `wsl-tsan` is necessarily about code in this repo.

On Ubuntu 24.04 (including GitHub Actions runners) TSan binaries abort at startup with
`unexpected memory mapping` — the kernel's `vm.mmap_rnd_bits=32` exceeds what TSan's shadow
mapping tolerates. The build handles this by running TSan tests under `setarch -R`, which needs
no privileges. Race detection is unaffected: TSan works from happens-before tracking, not
address layout.

## Layout

```
include/microvllm/   public headers
src/core/            queue, scheduler, block allocator, metrics  (no llama.cpp dependency)
src/engine/          llama.cpp facade + deterministic mock engine
src/http/            HTTP + SSE layer
tests/               unit tests; test_concurrency is the TSan target
benchmarks/          google/benchmark + Python load generator
third_party/         llama.cpp submodule, pinned to b10103
```

`src/core` deliberately does not depend on llama.cpp. That is what lets the entire scheduling
path be unit-tested in milliseconds against a deterministic mock engine, run under
ThreadSanitizer without backend noise, and execute in CI without downloading a model.

## License

MIT. Vendored llama.cpp is MIT, under its own copyright.
