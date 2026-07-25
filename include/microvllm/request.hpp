#pragma once

#include <atomic>
#include <future>
#include <memory>
#include <string>

#include "microvllm/token_sink.hpp"
#include "microvllm/types.hpp"

namespace microvllm {

// The outcome of one generation, delivered from the engine thread back to the HTTP
// thread that submitted the request.
struct GenResult {
    std::string  text;
    FinishReason reason = FinishReason::kError;
    Usage        usage;
    std::string  error;  // populated only when reason == kError
};

// A unit of work handed from an HTTP thread to the engine thread through RequestQueue.
//
// Move-only: it owns a std::promise, the write end of the result channel. The HTTP
// thread keeps the paired future and blocks on it. `cancel` is shared with the HTTP
// thread so a client disconnect can trip generation without a data race.
struct Request {
    RequestId                          id = 0;
    RequestSpec                        spec;
    std::shared_ptr<std::atomic<bool>> cancel;  // never null once submitted
    std::promise<GenResult>            result;

    // Where generated text goes as it is produced. Null means "buffer it" -- the
    // scheduler installs a CollectingSink and the whole response arrives via `result`.
    // The streaming endpoint supplies a StreamingSink so the HTTP thread can write SSE
    // events while generation is still running.
    std::shared_ptr<ITokenSink> sink;

    Request() = default;
    Request(Request&&) noexcept            = default;
    Request& operator=(Request&&) noexcept = default;
    Request(const Request&)                = delete;
    Request& operator=(const Request&)     = delete;
};

}  // namespace microvllm
