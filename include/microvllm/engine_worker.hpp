#pragma once

#include "microvllm/model_engine.hpp"
#include "microvllm/request_queue.hpp"

namespace microvllm {

// The single engine thread. It is the only thread that ever touches the model, which
// is what makes the whole system safe: llama_context is not thread-safe, so all decode
// work is funnelled here while HTTP threads only touch the RequestQueue.
//
// run() pops requests one at a time, drives generate_one against the engine, and
// fulfills each request's promise exactly once -- including on backend failure, so a
// waiting HTTP thread never hangs. Phase 2 processes serially on a single KV slot; the
// scheduler replaces this loop with continuous batching in Phase 4.
class EngineWorker {
public:
    EngineWorker(IModelEngine& engine, RequestQueue& queue);

    // Process requests until the queue is closed and drained, then return. Runs on the
    // dedicated engine thread.
    void run();

private:
    void process(Request& req);

    IModelEngine& engine_;
    RequestQueue& queue_;

    // Serial processing means one fixed KV-cache slot is enough.
    static constexpr SeqId kSeq = 0;
};

}  // namespace microvllm
