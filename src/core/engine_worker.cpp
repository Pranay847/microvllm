#include "microvllm/engine_worker.hpp"

#include <exception>
#include <utility>

#include "microvllm/collecting_sink.hpp"
#include "microvllm/generator.hpp"

namespace microvllm {

EngineWorker::EngineWorker(IModelEngine& engine, RequestQueue& queue)
    : engine_(engine), queue_(queue) {}

void EngineWorker::run() {
    while (auto req = queue_.pop()) {
        process(*req);
    }
}

void EngineWorker::process(Request& req) {
    GenResult result;
    try {
        // The sink shares the request's cancel flag, so a client disconnect observed on
        // an HTTP thread stops generation here.
        CollectingSink sink(req.cancel);
        generate_one(engine_, kSeq, req.spec, sink);

        result.text   = sink.text();
        result.reason = sink.reason();
        result.usage  = sink.usage();
        if (sink.reason() == FinishReason::kError) {
            result.error = sink.error();
        }
    } catch (const std::exception& e) {
        // A backend failure may have left this KV slot dirty; clear it so the next
        // request starts clean. Then report the error rather than hanging the caller.
        engine_.release_sequence(kSeq);
        result.reason = FinishReason::kError;
        result.error  = e.what();
    }

    // Always fulfill the promise exactly once, on every path, so no HTTP thread hangs.
    req.result.set_value(std::move(result));
}

}  // namespace microvllm
