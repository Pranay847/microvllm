#include "microvllm/scheduler.hpp"

#include <cstdio>
#include <exception>
#include <memory>
#include <utility>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/sequence_state.hpp"

namespace microvllm {
namespace {

// One in-flight request: its sink, its per-sequence state machine, and the promise to
// fulfill. Held by unique_ptr because BatchItem holds spans into SequenceState, which
// must not move while a batch references it.
struct Job {
    Request                        request;
    std::unique_ptr<CollectingSink> sink;
    std::unique_ptr<SequenceState>  state;
};

void deliver(Job& job) {
    GenResult result;
    result.text   = job.sink->text();
    result.reason = job.sink->reason();
    result.usage  = job.sink->usage();
    if (result.reason == FinishReason::kError) {
        result.error = job.sink->error();
    }
    job.request.result.set_value(std::move(result));
}

}  // namespace

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
    while (true) {
        std::vector<Request> batch = next_batch();
        if (batch.empty()) {
            return;  // queue closed and drained
        }
        run_batch(std::move(batch));
    }
}

void Scheduler::run_batch(std::vector<Request> batch) {
    if (batch.empty()) {
        return;
    }

    std::vector<Job> jobs;
    jobs.reserve(batch.size());

    // Sequence ids are slot indices within this batch. Static batching holds the whole
    // batch until every member finishes, so a slot is never reused mid-batch.
    for (std::size_t i = 0; i < batch.size(); ++i) {
        const auto seq = static_cast<SeqId>(i);
        Job        job;
        job.request = std::move(batch[i]);
        job.sink    = std::make_unique<CollectingSink>(job.request.cancel);
        jobs.push_back(std::move(job));

        Job& j = jobs.back();
        j.state = std::make_unique<SequenceState>(
            seq, j.request.spec, *j.sink,
            engine_.tokenize(j.request.spec.prompt, /*add_special=*/true));
        engine_.begin_sequence(seq, j.request.spec.sampling);
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
                const BatchItem item = jobs[i].state->prefill_item();
                // Always admit at least one prompt, even if it alone exceeds n_batch:
                // the engine reports that as an error rather than silently truncating.
                if (!items.empty() && tokens + item.tokens.size() > n_batch) {
                    break;
                }
                tokens += item.tokens.size();
                items.push_back(item);
                submitted.push_back(&jobs[i]);
                ++i;
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
            for (Job& j : jobs) {
                if (!j.state->finished()) {
                    items.push_back(j.state->decode_item());
                    active.push_back(&j);
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
        for (Job& j : jobs) {
            if (!j.state->finished()) {
                j.state->fail(e.what());
            }
            j.state->finish();  // idempotent
            engine_.release_sequence(j.state->seq());
        }
    }

    for (Job& j : jobs) {
        j.state->finish();  // idempotent; covers any sequence not yet notified
        deliver(j);
    }
}

}  // namespace microvllm
