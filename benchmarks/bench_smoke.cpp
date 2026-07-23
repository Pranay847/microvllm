#include "microvllm/types.hpp"

#include <benchmark/benchmark.h>

// Phase 0 placeholder: proves google/benchmark links and runs. Replaced in
// Phase 3 by bench_scheduler.cpp (static vs continuous batching over the mock engine).
static void BM_FinishReasonToString(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(microvllm::to_string(microvllm::FinishReason::kEos));
    }
}
BENCHMARK(BM_FinishReasonToString);
