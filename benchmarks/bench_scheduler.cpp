// Scheduler microbenchmarks against the deterministic mock engine.
//
// These deliberately exclude the model: with a zero-cost "forward pass", what remains is
// pure scheduling cost -- forming batches, building BatchItems, advancing per-sequence
// state, and fulfilling promises. That separates two questions the end-to-end numbers in
// docs/benchmarks.md conflate: "does batching help?" (it does, because decode is
// bandwidth-bound) and "does the scheduler itself cost anything?" (it should not).
#include <benchmark/benchmark.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "microvllm/mock_engine.hpp"
#include "microvllm/scheduler.hpp"
#include "microvllm/types.hpp"

namespace {

using namespace microvllm;  // NOLINT(google-build-using-namespace) - benchmark-local

Request make_req(RequestId id, std::uint32_t max_tokens) {
    Request r;
    r.id              = id;
    r.spec.prompt     = "benchmark prompt";
    r.spec.max_tokens = max_tokens;
    r.cancel          = std::make_shared<std::atomic<bool>>(false);
    return r;
}

// Cost of running a batch of N sequences to completion. With a free forward pass, wall
// time here is scheduler overhead alone; tokens/s should rise close to linearly with the
// batch size, since each decode step services N sequences.
void BM_RunBatch(benchmark::State& state) {
    const auto batch_size = static_cast<std::size_t>(state.range(0));
    constexpr std::uint32_t kTokens = 64;

    MockModelEngine engine(
        MockModelEngine::Config{.response = std::string(kTokens, 'x'), .n_seq_max = 64});
    RequestQueue queue(1024);
    Scheduler    sched(engine, queue, SchedulerConfig{.max_batch_size = batch_size});

    for (auto _ : state) {
        state.PauseTiming();
        std::vector<Request> batch;
        batch.reserve(batch_size);
        for (std::size_t i = 0; i < batch_size; ++i) {
            batch.push_back(make_req(static_cast<RequestId>(i), kTokens));
        }
        state.ResumeTiming();

        sched.run_batch(std::move(batch));
    }

    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(batch_size * kTokens));
    state.counters["seqs"] = static_cast<double>(batch_size);
}
BENCHMARK(BM_RunBatch)->Arg(1)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

// Cost of a single sequence's per-token state machine, isolated from batching: stop
// scanning, holdback flushing, and usage accounting. Sensitive to how many stop strings a
// request carries, since each is searched on every generated token.
void BM_SequenceStepWithStops(benchmark::State& state) {
    const auto n_stops = static_cast<std::size_t>(state.range(0));
    constexpr std::uint32_t kTokens = 128;

    MockModelEngine engine(
        MockModelEngine::Config{.response = std::string(kTokens, 'x'), .n_seq_max = 4});
    RequestQueue queue(16);
    Scheduler    sched(engine, queue, SchedulerConfig{.max_batch_size = 1});

    for (auto _ : state) {
        state.PauseTiming();
        Request r = make_req(1, kTokens);
        for (std::size_t i = 0; i < n_stops; ++i) {
            r.spec.stop.push_back("STOP" + std::to_string(i));  // never matches
        }
        std::vector<Request> batch;
        batch.push_back(std::move(r));
        state.ResumeTiming();

        sched.run_batch(std::move(batch));
    }

    state.SetItemsProcessed(state.iterations() * kTokens);
    state.counters["stops"] = static_cast<double>(n_stops);
}
BENCHMARK(BM_SequenceStepWithStops)->Arg(0)->Arg(1)->Arg(4)->Arg(16);

}  // namespace
