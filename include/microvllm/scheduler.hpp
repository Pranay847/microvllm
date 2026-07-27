#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "microvllm/block_allocator.hpp"
#include "microvllm/latency_histogram.hpp"
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

    // KV-cache budget, in blocks. 0 means "derive from the engine's context" -- the
    // natural size, where the pool exactly covers what the backend can hold. Setting it
    // lower deliberately constrains the cache so admission control, preemption, and
    // eviction are observable on hardware whose cache cannot otherwise be exhausted.
    std::uint32_t kv_blocks  = 0;
    std::uint32_t block_size = 16;  // tokens per block, matching vLLM's default

    // Reserved sequence slots holding prefixes past their request's retirement, so
    // sequential traffic behind a shared prompt can hit. Each costs one llama.cpp sequence
    // id and pins real blocks, so they are bounded and reclaimed LRU -- and reclaimed
    // before any live sequence is preempted, since cache should yield to work in progress.
    // 0 disables retention, leaving sharing to sequences that overlap in time.
    std::uint32_t prefix_donors = 4;

    // Share KV cache between requests with a common prompt prefix. Chat traffic is highly
    // redundant -- the same system prompt or conversation history leads every request --
    // and prefill is the compute-bound half of inference, so recomputing it is pure waste.
    bool prefix_caching = true;

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

    // Observable scheduling state, for /metrics and for tests asserting that admission
    // control and preemption actually fired rather than merely not crashing.
    //
    // stats() is safe to call from any thread: the counters are atomic internally and
    // this returns a plain snapshot. /metrics is served on an HTTP thread while the
    // scheduler thread is updating them, so a non-atomic read here would be a data race.
    struct Stats {
        std::uint64_t admitted    = 0;
        std::uint64_t completed   = 0;
        std::uint64_t preemptions = 0;  // sequences evicted to free cache, then requeued
        std::uint64_t deferred    = 0;  // admissions delayed because the pool was full
        std::uint64_t prefix_hits = 0;  // requests that inherited a cached prompt prefix
        std::uint64_t prefix_tokens_saved = 0;  // prompt tokens not prefilled thanks to hits
        std::uint64_t donors_retained  = 0;     // prefixes kept past their request's exit
        std::uint64_t donor_evictions  = 0;     // donors dropped, by LRU or block pressure
        std::uint32_t donors_held      = 0;     // donor slots currently occupied
        std::uint64_t prompt_tokens     = 0;    // billable input tokens served
        std::uint64_t completion_tokens = 0;    // billable output tokens served
        std::uint32_t kv_blocks_total = 0;
        std::uint32_t kv_blocks_used  = 0;
    };
    [[nodiscard]] Stats stats() const;

    // Latency distributions, for /metrics percentile queries. Safe to read from any
    // thread; the scheduler thread observes into them as requests complete.
    [[nodiscard]] const LatencyHistogram& ttft_histogram() const { return ttft_hist_; }
    [[nodiscard]] const LatencyHistogram& e2e_histogram() const { return e2e_hist_; }

private:
    struct Job;  // one in-flight request: sink, sequence state, promise

    // Build a Job and register its sequence with the engine.
    static std::unique_ptr<Job> make_job(IModelEngine& engine, Request req, SeqId seq);
    // Fulfill a job's promise and record its timing. Non-static because this is the one
    // point every request passes through on completion, which makes it the right place to
    // observe latency and token counts exactly once.
    void deliver(Job& job);

    void run_static();
    void run_continuous();

    // Copy a retiring sequence's prompt prefix into a donor slot so it outlives the
    // request. No-op when donors are disabled or the prefix is shorter than a block.
    void retain_prefix(Job& job);
    // Free everything an evicted donor held (blocks + engine sequence).
    void release_donor(const PrefixCache::Entry& donor);
    // Drop the least-recently-used donor to free blocks. True if anything was reclaimed.
    [[nodiscard]] bool reclaim_donor_blocks();

    // Evict the most-recently-admitted sequence to free cache blocks, requeueing it for
    // recompute. LIFO on purpose: the newest sequence has the least work invested, so
    // throwing it away costs the least and older, closer-to-done requests keep their
    // progress instead of being starved. Returns false if there is nothing to preempt.
    // `except` is never evicted -- without it, making room for a starved sequence can
    // evict that same sequence, which is readmitted, grows, and is evicted again.
    [[nodiscard]] bool preempt_one(std::vector<std::unique_ptr<Job>>& active,
                                   std::vector<SeqId>&               free_slots,
                                   std::vector<Request>&             pending,
                                   const Job*                        except = nullptr);

    // Move a Job's request back out so it can be retried: promise, cancel flag, and sink
    // travel with it, so a deferred or preempted client waits rather than seeing an error.
    static Request reclaim(Job& job);

    // Block for one request, then sweep up whatever else is already waiting, without
    // waiting for the batch to fill: a lone request must never be delayed hoping for company.
    [[nodiscard]] std::vector<Request> next_batch();

    // Finish a job, release its engine slot, and fulfill its promise.
    void retire(Job& job);

    IModelEngine&                   engine_;
    RequestQueue&                   queue_;
    SchedulerConfig                 config_;
    std::unique_ptr<BlockAllocator> blocks_;
    PrefixCache                     prefix_cache_;

    // Written only by the scheduler thread, read by HTTP threads via stats().
    struct AtomicStats {
        std::atomic<std::uint64_t> admitted{0};
        std::atomic<std::uint64_t> completed{0};
        std::atomic<std::uint64_t> preemptions{0};
        std::atomic<std::uint64_t> deferred{0};
        std::atomic<std::uint64_t> prefix_hits{0};
        std::atomic<std::uint64_t> prefix_tokens_saved{0};
        std::atomic<std::uint64_t> donors_retained{0};
        std::atomic<std::uint64_t> donor_evictions{0};
        std::atomic<std::uint32_t> donors_held{0};
        std::atomic<std::uint64_t> prompt_tokens{0};
        std::atomic<std::uint64_t> completion_tokens{0};
        std::atomic<std::uint32_t> kv_blocks_total{0};
        std::atomic<std::uint32_t> kv_blocks_used{0};
    };
    AtomicStats      stats_;
    LatencyHistogram ttft_hist_;
    LatencyHistogram e2e_hist_;
};

}  // namespace microvllm
