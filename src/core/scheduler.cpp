#include "microvllm/scheduler.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/sequence_state.hpp"

namespace microvllm {

// One in-flight request: its sink, its per-sequence state machine, and the promise to
// fulfill. Held by unique_ptr because BatchItem holds spans into SequenceState, which
// must not move while a batch references it.
struct Scheduler::Job {
    Request                        request;
    std::shared_ptr<ITokenSink>    sink;   // buffered or streaming; kept alive here
    std::unique_ptr<SequenceState> state;
    std::unique_ptr<BlockTable>    blocks;  // this sequence's KV-cache page table
};

void Scheduler::deliver(Job& job) {
    // SequenceState is the source of truth for the outcome; the sink is only the delivery
    // channel. Reading everything from the state keeps distinctions like kContextOverflow
    // vs kError intact (a sink's generic on_error would flatten them) and means this
    // works unchanged whether the sink buffered the response or streamed it.
    GenResult result;
    result.text   = std::string(job.state->emitted_text());
    result.reason = job.state->reason();
    result.usage  = job.state->usage();
    if (result.reason == FinishReason::kError ||
        result.reason == FinishReason::kContextOverflow) {
        result.error = job.state->error();
    }
    result.timing = job.request.timing;
    if (!result.timing.finished) {
        result.timing.finished = Clock::now();
    }

    // Record once, here, so every completion path -- normal, cancelled, preempted-then-
    // finished, failed -- is counted exactly once and none can be forgotten.
    stats_.prompt_tokens.fetch_add(result.usage.prompt_tokens, std::memory_order_relaxed);
    stats_.completion_tokens.fetch_add(result.usage.completion_tokens,
                                       std::memory_order_relaxed);
    if (result.timing.first_token) {
        ttft_hist_.observe(result.timing.ttft_ms());
    }
    e2e_hist_.observe(result.timing.total_ms());

    job.request.result.set_value(std::move(result));
}

std::unique_ptr<Scheduler::Job> Scheduler::make_job(IModelEngine& engine, Request req, SeqId seq) {
    auto job     = std::make_unique<Job>();
    job->request = std::move(req);
    // A caller-supplied sink (streaming) is used as-is; otherwise buffer into a
    // CollectingSink and return the whole response through the promise.
    job->sink = job->request.sink
                    ? job->request.sink
                    : std::make_shared<CollectingSink>(job->request.cancel);

    std::vector<Token> prompt = engine.tokenize(job->request.spec.prompt, /*add_special=*/true);
    job->state = std::make_unique<SequenceState>(seq, job->request.spec, *job->sink,
                                                 std::move(prompt));

    // Admission control on context length. llama.cpp divides the KV pool across
    // sequences, so a request is bounded by n_ctx_seq, not n_ctx. Catching it here turns
    // an opaque mid-generation "failed to find a memory slot" -- after the prefill
    // compute has already been spent -- into an immediate, actionable rejection.
    const EngineCaps  caps   = engine.caps();
    const std::size_t needed = job->state->usage().prompt_tokens + job->request.spec.max_tokens;
    if (caps.n_ctx_seq > 0 && needed > caps.n_ctx_seq) {
        job->state->fail("request needs " + std::to_string(needed) +
                             " tokens (prompt + max_tokens) but the per-request context "
                             "holds " + std::to_string(caps.n_ctx_seq) +
                             "; raise --ctx or lower max_tokens",
                         FinishReason::kContextOverflow);
    }

    engine.begin_sequence(seq, job->request.spec.sampling);
    return job;
}

Scheduler::Scheduler(IModelEngine& engine, RequestQueue& queue, SchedulerConfig config)
    : engine_(engine), queue_(queue), config_(config) {
    // The engine's sequence capacity is the authority. Submitting a batch with more
    // sequences than the context was built for makes llama.cpp abort the process, so
    // clamp rather than trust the caller: no --batch-size value may crash the server.
    const auto n_seq_max = static_cast<std::size_t>(engine_.caps().n_seq_max);
    if (n_seq_max > 0 && config_.max_batch_size > n_seq_max) {
        std::fprintf(stderr,
                     "microvllm: batch size %zu exceeds the engine's n_seq_max %zu; "
                     "clamping to %zu\n",
                     config_.max_batch_size, n_seq_max, n_seq_max);
        config_.max_batch_size = n_seq_max;
    }
    // Size the KV budget. Default: exactly what the backend can hold, so the allocator
    // mirrors reality rather than inventing a limit. A smaller --kv-blocks constrains it
    // deliberately, which is how admission control and preemption become observable on
    // hardware whose cache a 0.5B model cannot otherwise exhaust.
    const std::uint32_t block_size = config_.block_size == 0 ? 16 : config_.block_size;
    std::uint32_t       pool       = config_.kv_blocks;
    if (pool == 0) {
        const std::uint32_t cells = engine_.caps().n_ctx > 0 ? engine_.caps().n_ctx : 4096;
        pool = (cells + block_size - 1) / block_size;
    }
    config_.block_size = block_size;
    config_.kv_blocks  = pool;
    blocks_            = std::make_unique<BlockAllocator>(pool, block_size);
    stats_.kv_blocks_total.store(pool, std::memory_order_relaxed);

    if (config_.max_batch_size == 0) {
        config_.max_batch_size = 1;
    }
}

std::vector<Request> Scheduler::next_batch() {
    std::vector<Request> batch;

    auto first = queue_.pop();  // blocks; nullopt once closed and drained
    if (!first) {
        return batch;
    }
    batch.push_back(std::move(*first));

    // Sweep up whatever else already arrived, but never wait for the batch to fill --
    // a lone request must not pay a latency penalty hoping for company.
    while (batch.size() < config_.max_batch_size) {
        auto more = queue_.try_pop();
        if (!more) {
            break;
        }
        batch.push_back(std::move(*more));
    }
    return batch;
}

void Scheduler::run() {
    if (config_.mode == BatchingMode::kContinuous) {
        run_continuous();
    } else {
        run_static();
    }
}

void Scheduler::run_static() {
    while (true) {
        std::vector<Request> batch = next_batch();
        if (batch.empty()) {
            return;  // queue closed and drained
        }
        run_batch(std::move(batch));
    }
}

Scheduler::Stats Scheduler::stats() const {
    Stats s;
    s.admitted            = stats_.admitted.load(std::memory_order_relaxed);
    s.completed           = stats_.completed.load(std::memory_order_relaxed);
    s.preemptions         = stats_.preemptions.load(std::memory_order_relaxed);
    s.deferred            = stats_.deferred.load(std::memory_order_relaxed);
    s.prefix_hits         = stats_.prefix_hits.load(std::memory_order_relaxed);
    s.prefix_tokens_saved = stats_.prefix_tokens_saved.load(std::memory_order_relaxed);
    s.prompt_tokens       = stats_.prompt_tokens.load(std::memory_order_relaxed);
    s.completion_tokens   = stats_.completion_tokens.load(std::memory_order_relaxed);
    s.kv_blocks_total     = stats_.kv_blocks_total.load(std::memory_order_relaxed);
    s.kv_blocks_used      = stats_.kv_blocks_used.load(std::memory_order_relaxed);
    return s;
}

void Scheduler::retire(Job& job) {
    job.state->finish();  // idempotent
    // A cache entry names a live sequence whose KV is about to stop existing, so it must
    // go before the sequence does -- otherwise a later hit would copy from a slot that has
    // been recycled by an unrelated request.
    prefix_cache_.evict_sequence(job.state->seq());
    if (job.blocks) {
        job.blocks->release();  // return this sequence's cache to the pool
    }
    engine_.release_sequence(job.state->seq());
    stats_.completed.fetch_add(1, std::memory_order_relaxed);
    deliver(job);
}

// Move a Job's request back out so it can be retried later. Used when a sequence is
// deferred at admission or preempted mid-flight: the promise, cancel flag, and sink
// travel with it, so the client sees a slower response rather than an error.
Request Scheduler::reclaim(Job& job) {
    Request out;
    out.id     = job.request.id;
    out.spec   = std::move(job.request.spec);
    out.cancel = std::move(job.request.cancel);
    out.sink   = std::move(job.request.sink);
    out.result = std::move(job.request.result);
    // Keep the original enqueue stamp: from the client's perspective the wait began when
    // they sent the request, not when it was re-admitted. The per-attempt stamps are
    // cleared, since a preempted request is recomputed from scratch.
    out.timing          = job.request.timing;
    out.timing.admitted = std::nullopt;
    out.timing.first_token = std::nullopt;
    return out;
}

bool Scheduler::preempt_one(std::vector<std::unique_ptr<Job>>& active,
                            std::vector<SeqId>&                free_slots,
                            std::vector<Request>&              pending,
                            const Job*                         except) {
    // LIFO: evict the most recently admitted sequence. It has the least work invested, so
    // discarding it wastes the least compute, and older sequences closer to finishing keep
    // their progress rather than being starved. This is vLLM's recompute preemption.
    //
    // `except` is never evicted. Without it, making room for a starved sequence can evict
    // that same sequence, which is then readmitted, grows, and is evicted again -- a
    // livelock that looks exactly like a hang.
    for (auto it = active.rbegin(); it != active.rend(); ++it) {
        Job& job = **it;
        if (job.state->finished() || &job == except) {
            continue;
        }

        // The request goes to the FRONT of pending so it is retried before newer
        // arrivals -- it has already waited once and partially run, and sending it to the
        // back would risk starving it behind a stream of new work. It is recomputed from
        // scratch on readmission, which is vLLM's recompute preemption.
        Request requeued = reclaim(job);

        prefix_cache_.evict_sequence(job.state->seq());  // its KV is about to disappear
        job.blocks->release();
        engine_.release_sequence(job.state->seq());
        free_slots.push_back(job.state->seq());

        active.erase(std::next(it).base());
        pending.insert(pending.begin(), std::move(requeued));
        stats_.preemptions.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

// Continuous batching: the batch is rebuilt every step rather than run to completion.
//
//   1. admit  -- fill free slots from the queue (blocking only when nothing is in flight)
//   2. build  -- decodes claim one token each first, prefills chunk into what is left
//   3. decode -- one forward pass over the mixed batch
//   4. accept -- advance each sequence independently; retire and free the slot on finish
//
// The ordering in step 2 is the policy that matters: budgeting decodes first means an
// arriving prompt can never stall sequences that are already generating, and chunking
// means a long prompt is spread over several steps instead of monopolising one.
void Scheduler::run_continuous() {
    // Slot ids are recycled as sequences come and go, so they come from a free list
    // rather than being a position in the active vector.
    std::vector<SeqId> free_slots;
    free_slots.reserve(config_.max_batch_size);
    for (std::size_t i = config_.max_batch_size; i > 0; --i) {
        free_slots.push_back(static_cast<SeqId>(i - 1));
    }

    std::vector<std::unique_ptr<Job>> active;
    std::vector<Request>              pending;  // owned by the scheduler, awaiting capacity
    const std::size_t n_batch = engine_.caps().n_batch;

    while (true) {
        // --- 1. take work from the queue ---------------------------------------------
        // Once a request leaves the queue the scheduler owns it: a deferred or preempted
        // request waits in `pending` rather than being handed back, so a queue close (a
        // normal shutdown) can never strand work that was already accepted.
        while (pending.size() + active.size() < config_.max_batch_size) {
            const bool idle = active.empty() && pending.empty();
            // Block only when nothing is in flight; otherwise waiting here would stall
            // sequences that are mid-generation.
            std::optional<Request> req = idle ? queue_.pop() : queue_.try_pop();
            if (!req) {
                break;
            }
            pending.push_back(std::move(*req));
        }
        if (active.empty() && pending.empty()) {
            return;  // queue closed and drained, nothing left to finish
        }

        // --- 2. admit what fits ------------------------------------------------------
        while (!pending.empty() && !free_slots.empty()) {
            const SeqId slot = free_slots.back();
            auto        job  = make_job(engine_, std::move(pending.front()), slot);

            // Admission control, with headroom. Reserving only the prompt's blocks is not
            // enough: every running sequence needs at least one more block to cross its
            // next boundary, so admitting into a pool with no slack means the newcomer
            // immediately starves someone, that someone preempts the newcomer, and the
            // pair thrash without either making progress. Requiring one spare block per
            // active sequence (plus one for the newcomer) guarantees the batch can always
            // advance a step before pressure returns.
            //
            // The first sequence is exempt: if nothing is running there is no one to
            // starve, and refusing it would mean never running anything.
            job->blocks = std::make_unique<BlockTable>(*blocks_);
            const std::uint32_t want =
                blocks_->blocks_for(job->state->usage().prompt_tokens);
            const std::uint32_t headroom =
                active.empty() ? 0 : static_cast<std::uint32_t>(active.size()) + 1;
            const bool fits = job->state->finished() ||
                              (blocks_->free_blocks() >= want + headroom &&
                               job->blocks->ensure_capacity(job->state->usage().prompt_tokens));
            if (!fits) {
                if (active.empty()) {
                    // Nothing is running, so no amount of waiting will free capacity:
                    // this request simply cannot fit the pool. Say so rather than
                    // deferring it forever.
                    job->state->fail(
                        "KV cache exhausted: prompt needs " +
                            std::to_string(blocks_->blocks_for(job->state->usage().prompt_tokens)) +
                            " blocks but the pool holds " +
                            std::to_string(blocks_->total_blocks()),
                        FinishReason::kContextOverflow);
                    job->state->finish();
                    engine_.release_sequence(slot);
                    pending.erase(pending.begin());
                    deliver(*job);
                    continue;
                }
                // Put it back untouched and let running sequences drain first.
                engine_.release_sequence(slot);
                pending.front() = reclaim(*job);
                stats_.deferred.fetch_add(1, std::memory_order_relaxed);
                break;
            }

            // Prefix sharing: if another live sequence already holds the KV for a leading
            // run of this prompt, inherit it rather than prefilling it again. The copy is
            // mirrored into the backend (llama_memory_seq_cp), so the saving is real work
            // avoided -- not just refcount bookkeeping.
            if (config_.prefix_caching && !job->state->finished()) {
                const auto hit = prefix_cache_.find_longest(job->state->prompt(),
                                                            blocks_->block_size());
                if (hit && hit->seq != job->state->seq()) {
                    engine_.copy_sequence(hit->seq, job->state->seq(),
                                          static_cast<Pos>(hit->n_tokens));
                    job->blocks->adopt_shared_prefix(hit->blocks, hit->n_tokens);
                    job->state->adopt_cached_prefix(hit->n_tokens);
                    stats_.prefix_hits.fetch_add(1, std::memory_order_relaxed);
                    stats_.prefix_tokens_saved.fetch_add(hit->n_tokens, std::memory_order_relaxed);
                }
            }

            pending.erase(pending.begin());
            free_slots.pop_back();
            job->request.timing.admitted = Clock::now();
            stats_.admitted.fetch_add(1, std::memory_order_relaxed);
            active.push_back(std::move(job));
        }
        if (active.empty()) {
            continue;
        }

        // Sequences that could not get the block they need this step; resolved after the
        // accept loop by preempting someone (which may be one of them).
        std::vector<Job*> grew_short;

        try {
            // --- 2. build the mixed batch -------------------------------------------
            std::vector<BatchItem> items;
            std::vector<Job*>      sampling;
            items.reserve(active.size());

            // Decodes first: one token each, so generation always makes progress.
            std::size_t budget = n_batch;
            for (auto& job : active) {
                if (job->state->finished() || job->state->prefilling()) {
                    continue;
                }
                if (budget == 0) {
                    break;
                }
                items.push_back(job->state->decode_item());
                sampling.push_back(job.get());
                --budget;
            }

            // Prefills share whatever token budget remains.
            const std::size_t chunk_cap =
                config_.prefill_chunk == 0 ? n_batch : config_.prefill_chunk;
            for (auto& job : active) {
                if (job->state->finished() || !job->state->prefilling() || budget == 0) {
                    continue;
                }
                const std::size_t take = std::min({budget, chunk_cap,
                                                   job->state->prefill_remaining()});
                if (take == 0) {
                    continue;
                }
                const BatchItem item = job->state->take_prefill_chunk(take);
                budget -= item.tokens.size();
                if (item.sample) {
                    sampling.push_back(job.get());
                }
                items.push_back(item);
            }

            // --- 3. decode ----------------------------------------------------------
            if (!items.empty()) {
                const std::vector<GenStep> steps = engine_.decode(items);

                // --- 4. accept ------------------------------------------------------
                for (std::size_t k = 0; k < sampling.size(); ++k) {
                    Job& job = *sampling[k];
                    // Stamped before accept() so it marks when the engine produced the
                    // token, not when the stop rules finished considering it.
                    if (!job.request.timing.first_token) {
                        job.request.timing.first_token = Clock::now();
                    }
                    if (!job.state->accept(engine_, steps.at(k))) {
                        job.state->finish();
                        continue;
                    }
                    // The sequence grew by a token. Most steps stay inside the block it
                    // already holds; crossing a boundary needs another one.
                    const std::uint32_t needed = job.state->usage().prompt_tokens +
                                                 job.state->usage().completion_tokens;
                    if (!job.blocks->ensure_capacity(needed)) {
                        grew_short.push_back(&job);
                        continue;
                    }

                    // Publish this sequence's prompt prefix once prefill is done, so
                    // later requests sharing it can inherit the KV instead of recomputing
                    // it. Only whole blocks are offered -- the last, partially-filled
                    // block is still being written and cannot be shared.
                    if (config_.prefix_caching && !job.state->prefilling()) {
                        const auto prompt   = job.state->prompt();
                        const auto bs       = blocks_->block_size();
                        const auto n_blocks = static_cast<std::uint32_t>(prompt.size() / bs);
                        if (n_blocks > 0 && job.blocks->n_blocks() >= n_blocks) {
                            const std::uint32_t n_tokens = n_blocks * bs;
                            PrefixCache::Entry  entry;
                            entry.seq      = job.state->seq();
                            entry.n_tokens = n_tokens;
                            entry.blocks.assign(job.blocks->blocks().begin(),
                                                job.blocks->blocks().begin() + n_blocks);
                            prefix_cache_.insert(PrefixCache::hash_prefix(prompt, n_tokens),
                                                 std::move(entry));
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            // A backend failure leaves the shared context untrustworthy, so every
            // in-flight sequence fails rather than silently producing garbage.
            for (auto& job : active) {
                if (!job->state->finished()) {
                    job->state->fail(e.what());
                }
            }
        }

        // --- retire finished sequences and reclaim their slots -----------------------
        for (auto it = active.begin(); it != active.end();) {
            if ((*it)->state->finished()) {
                free_slots.push_back((*it)->state->seq());
                retire(**it);
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        // --- resolve cache exhaustion ------------------------------------------------
        // Retirement above may already have freed enough. If not, evict LIFO until every
        // starved sequence can grow. Preemption is the escape valve that lets the server
        // stay correct under a cache budget instead of failing when it runs out.
        for (Job* starved : grew_short) {
            const auto it = std::find_if(active.begin(), active.end(),
                                         [&](const auto& j) { return j.get() == starved; });
            if (it == active.end()) {
                continue;  // already retired or preempted
            }
            const std::uint32_t needed = starved->state->usage().prompt_tokens +
                                         starved->state->usage().completion_tokens;

            // A sequence larger than the entire pool can never be satisfied, no matter
            // who is evicted. Say so rather than evicting everyone else pointlessly.
            if (blocks_->blocks_for(needed) > blocks_->total_blocks()) {
                starved->state->fail(
                    "KV cache exhausted: sequence needs " +
                        std::to_string(blocks_->blocks_for(needed)) +
                        " blocks but the pool holds only " +
                        std::to_string(blocks_->total_blocks()),
                    FinishReason::kContextOverflow);
                starved->state->finish();
                continue;
            }

            // Otherwise evicting others is guaranteed to make room eventually: with every
            // other sequence gone the pool holds total - starved's own blocks free, which
            // covers the shortfall exactly because needed fits the pool. So this loop
            // terminates -- either it fits, or there is nothing left to evict.
            while (!starved->blocks->ensure_capacity(needed)) {
                if (!preempt_one(active, free_slots, pending, /*except=*/starved)) {
                    break;
                }
            }
        }

        // A preemption or forced failure may have finished sequences; sweep once more.
        for (auto it = active.begin(); it != active.end();) {
            if ((*it)->state->finished()) {
                free_slots.push_back((*it)->state->seq());
                retire(**it);
                it = active.erase(it);
            } else {
                ++it;
            }
        }

        // Publish pool utilization from this thread. BlockAllocator is not thread-safe, so
        // /metrics must read a snapshot rather than reach into the allocator itself.
        stats_.kv_blocks_used.store(blocks_->used_blocks(), std::memory_order_relaxed);
    }
}

void Scheduler::run_batch(std::vector<Request> batch) {
    if (batch.empty()) {
        return;
    }

    // Sequence ids are slot indices within this batch. Static batching holds the whole
    // batch until every member finishes, so a slot is never reused mid-batch.
    std::vector<std::unique_ptr<Job>> jobs;
    jobs.reserve(batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
        auto job    = make_job(engine_, std::move(batch[i]), static_cast<SeqId>(i));
        job->blocks = std::make_unique<BlockTable>(*blocks_);
        // Static batching cannot preempt (the batch is fixed once it starts), so a
        // sequence that will not fit the pool is rejected here rather than failing later.
        if (!job->state->finished() &&
            !job->blocks->ensure_capacity(job->state->usage().prompt_tokens +
                                          job->request.spec.max_tokens)) {
            job->state->fail("KV cache exhausted: not enough blocks for this batch",
                             FinishReason::kContextOverflow);
        } else {
            job->request.timing.admitted = Clock::now();
            stats_.admitted.fetch_add(1, std::memory_order_relaxed);
        }
        jobs.push_back(std::move(job));
    }

    try {
        // --- Prefill ---
        // Each sequence's prompt is submitted and its first token sampled. Prompts are
        // packed into as few decode calls as the engine's n_batch allows.
        const std::size_t n_batch = engine_.caps().n_batch;
        std::size_t       i       = 0;
        while (i < jobs.size()) {
            std::vector<BatchItem> items;
            std::vector<Job*>      submitted;
            std::size_t            tokens = 0;

            while (i < jobs.size()) {
                // A job rejected at admission (e.g. context overflow) is already
                // finished and must never reach the engine.
                if (jobs[i]->state->finished()) {
                    ++i;
                    continue;
                }
                // Always admit at least one prompt, even if it alone exceeds n_batch:
                // the engine reports that as an error rather than silently truncating.
                const std::size_t len = jobs[i]->state->prefill_remaining();
                if (!items.empty() && tokens + len > n_batch) {
                    break;
                }
                tokens += len;
                items.push_back(jobs[i]->state->prefill_item());
                submitted.push_back(jobs[i].get());
                ++i;
            }

            if (items.empty()) {
                continue;  // nothing admissible in this slice
            }

            const std::vector<GenStep> steps = engine_.decode(items);
            for (std::size_t k = 0; k < submitted.size(); ++k) {
                Job& j = *submitted[k];
                if (!j.request.timing.first_token) {
                    j.request.timing.first_token = Clock::now();
                }
                if (!j.state->accept(engine_, steps.at(k))) {
                    j.state->finish();
                    engine_.release_sequence(j.state->seq());
                }
            }
        }

        // --- Decode ---
        // One llama_batch per step carrying a single token for every still-running
        // sequence. This is the batching win: one pass over the model's weights advances
        // every sequence at once. The loop ends only when the last one finishes, which is
        // precisely the head-of-line blocking that Phase 4's continuous batching removes.
        while (true) {
            std::vector<BatchItem> items;
            std::vector<Job*>      active;
            for (auto& j : jobs) {
                if (!j->state->finished()) {
                    items.push_back(j->state->decode_item());
                    active.push_back(j.get());
                }
            }
            if (active.empty()) {
                break;
            }

            const std::vector<GenStep> steps = engine_.decode(items);
            for (std::size_t k = 0; k < active.size(); ++k) {
                Job& j = *active[k];
                if (!j.request.timing.first_token) {
                    j.request.timing.first_token = Clock::now();
                }
                if (!j.state->accept(engine_, steps.at(k))) {
                    j.state->finish();
                    engine_.release_sequence(j.state->seq());
                }
            }
        }
    } catch (const std::exception& e) {
        // A backend failure takes down the whole batch: the shared context's state is
        // untrustworthy. Release every slot and report the error to each waiter rather
        // than leaving any of them blocked forever.
        for (auto& j : jobs) {
            if (!j->state->finished()) {
                j->state->fail(e.what());
            }
            j->state->finish();  // idempotent
            engine_.release_sequence(j->state->seq());
        }
    }

    for (auto& jp : jobs) {
        Job& j = *jp;
        j.state->finish();  // idempotent; covers any sequence not yet notified
        if (j.blocks) {
            j.blocks->release();  // return this batch's cache before the next one forms
        }
        stats_.completed.fetch_add(1, std::memory_order_relaxed);
        deliver(j);
    }
    stats_.kv_blocks_used.store(blocks_->used_blocks(), std::memory_order_relaxed);
}

}  // namespace microvllm
