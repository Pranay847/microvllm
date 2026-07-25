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

void Scheduler::retire(Job& job) {
    job.state->finish();  // idempotent
    engine_.release_sequence(job.state->seq());
    deliver(job);
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
    const std::size_t n_batch = engine_.caps().n_batch;

    while (true) {
        // --- 1. admit ---------------------------------------------------------------
        while (!free_slots.empty()) {
            // Block only when there is nothing in flight; otherwise a wait here would
            // stall sequences that are mid-generation.
            std::optional<Request> req = active.empty() ? queue_.pop() : queue_.try_pop();
            if (!req) {
                if (active.empty()) {
                    return;  // queue closed and drained, nothing left to finish
                }
                break;
            }
            const SeqId slot = free_slots.back();
            free_slots.pop_back();
            active.push_back(make_job(engine_, std::move(*req), slot));
        }
        if (active.empty()) {
            continue;
        }

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
                    if (!job.state->accept(engine_, steps.at(k))) {
                        job.state->finish();
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
        jobs.push_back(make_job(engine_, std::move(batch[i]), static_cast<SeqId>(i)));
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
        deliver(j);
    }
}

}  // namespace microvllm
