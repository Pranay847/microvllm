#pragma once

#include <cstddef>
#include <vector>

#include "microvllm/model_engine.hpp"
#include "microvllm/request_queue.hpp"

namespace microvllm {

struct SchedulerConfig {
    std::size_t max_batch_size = 8;
};

// Static batching.
//
// The scheduler forms a batch of up to max_batch_size requests, prefills them, then
// decodes them in lockstep: one llama_batch per step containing one token for every
// still-running sequence. That is the whole throughput argument -- decode is
// memory-bandwidth-bound, so streaming the model's weights once to advance sixteen
// sequences costs barely more than advancing one.
//
// "Static" names the limitation this design accepts: the batch is fixed once it starts.
// A sequence that finishes early frees nothing for a waiting request, and the batch
// occupies the engine until its *longest* member is done, so a 16-token request queued
// behind a 512-token one waits for all 512. Phase 4 fixes exactly this by admitting and
// retiring sequences every step; measuring the gap between the two is the point.
//
// Threading: one scheduler thread owns the engine, exactly as EngineWorker did. HTTP
// threads only touch the RequestQueue.
class Scheduler {
public:
    Scheduler(IModelEngine& engine, RequestQueue& queue, SchedulerConfig config = {});

    // Pull batches from the queue and run them until the queue is closed and drained.
    void run();

    // Run one already-formed batch to completion, fulfilling every promise. Exposed for
    // tests and benchmarks that drive batches directly.
    void run_batch(std::vector<Request> batch);

private:
    // Block for one request, then sweep up whatever else is already waiting, without
    // waiting for the batch to fill: a lone request must never be delayed hoping for company.
    [[nodiscard]] std::vector<Request> next_batch();

    IModelEngine&   engine_;
    RequestQueue&   queue_;
    SchedulerConfig config_;
};

}  // namespace microvllm
