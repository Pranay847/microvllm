#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "microvllm/model_engine.hpp"
#include "microvllm/request_queue.hpp"

namespace microvllm {

enum class BatchingMode {
    // Form a batch, run it to completion, then form the next one. A slot freed early
    // stays idle until the whole batch finishes.
    kStatic,
    // Retire finished sequences and admit waiting ones every step, so a freed slot goes
    // straight back to work.
    kContinuous,
};

struct SchedulerConfig {
    std::size_t  max_batch_size = 8;
    BatchingMode mode           = BatchingMode::kContinuous;

    // Prompt tokens one sequence may submit per step. A long prompt admitted whole would
    // stall every decode sharing that step, so prefill is spread across steps instead.
    // Lower values protect other sequences' inter-token latency at the cost of this
    // sequence's time-to-first-token. 0 disables chunking (prefill goes in whole).
    std::size_t prefill_chunk = 128;
};

// Batches many sequences into one forward pass, in either of two modes.
//
// The shared premise: decode is memory-bandwidth-bound, so streaming the model's weights
// once to advance sixteen sequences costs barely more than advancing one.
//
// kStatic forms a batch, runs it to completion, then forms the next. Its weakness is
// structural -- the batch occupies the engine until its *longest* member finishes, so a
// 16-token request sharing a batch with a 512-token one holds a slot for all 512 steps
// while producing nothing for 496 of them.
//
// kContinuous rebuilds the batch every step: finished sequences are retired and their
// slots immediately refilled from the queue. Each sequence keeps its own position and
// state, so they need not be in lockstep. Prefill and decode work coexist in one step --
// decodes are budgeted first so an arriving prompt cannot stall sequences already
// generating, and long prompts are split across steps (prefill_chunk).
//
// Both modes share all the machinery except the loop itself, which is deliberate: the
// benchmark comparing them then controls for everything but the scheduling policy.
//
// Threading: one scheduler thread owns the engine. HTTP threads only touch the RequestQueue.
class Scheduler {
public:
    Scheduler(IModelEngine& engine, RequestQueue& queue, SchedulerConfig config = {});

    // Serve requests from the queue until it is closed and drained.
    void run();

    // Run one already-formed batch to completion (static semantics), fulfilling every
    // promise. Exposed for tests and benchmarks that drive batches directly.
    void run_batch(std::vector<Request> batch);

    [[nodiscard]] const SchedulerConfig& config() const { return config_; }

private:
    struct Job;  // one in-flight request: sink, sequence state, promise

    // Build a Job and register its sequence with the engine.
    static std::unique_ptr<Job> make_job(IModelEngine& engine, Request req, SeqId seq);
    // Fulfill a job's promise from its sink.
    static void deliver(Job& job);

    void run_static();
    void run_continuous();

    // Block for one request, then sweep up whatever else is already waiting, without
    // waiting for the batch to fill: a lone request must never be delayed hoping for company.
    [[nodiscard]] std::vector<Request> next_batch();

    // Finish a job, release its engine slot, and fulfill its promise.
    void retire(Job& job);

    IModelEngine&   engine_;
    RequestQueue&   queue_;
    SchedulerConfig config_;
};

}  // namespace microvllm
