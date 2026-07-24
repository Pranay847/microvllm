#pragma once

#include <atomic>
#include <string>

#include "microvllm/token_sink.hpp"
#include "microvllm/types.hpp"

namespace microvllm {

// The sink for the blocking /generate endpoint: it accumulates the whole response in
// memory and records how generation ended. Concatenating the generator's on_text
// deltas reproduces the exact response body, so this is the authoritative result --
// no separate "final text" channel is needed.
//
// Phase 1 uses it synchronously on one thread. Its cancellation flag is atomic so that
// in Phase 2 an HTTP thread can trip it on client disconnect while the engine thread reads it.
class CollectingSink final : public ITokenSink {
public:
    void on_text(std::string_view delta) override { text_ += delta; }

    void on_finish(FinishReason reason, const Usage& usage) override {
        reason_ = reason;
        usage_  = usage;
        done_   = true;
    }

    void on_error(std::string_view message) override {
        error_  = message;
        reason_ = FinishReason::kError;
        done_   = true;
    }

    [[nodiscard]] bool cancelled() const override { return cancel_.load(std::memory_order_relaxed); }

    // Ask the in-flight generation to stop; it will finish with kCancelled.
    void cancel() { cancel_.store(true, std::memory_order_relaxed); }

    [[nodiscard]] const std::string& text() const { return text_; }
    [[nodiscard]] FinishReason reason() const { return reason_; }
    [[nodiscard]] const Usage& usage() const { return usage_; }
    [[nodiscard]] bool done() const { return done_; }
    [[nodiscard]] const std::string& error() const { return error_; }

private:
    std::string       text_;
    std::string       error_;
    Usage             usage_{};
    FinishReason      reason_ = FinishReason::kUnset;
    bool              done_   = false;
    std::atomic<bool> cancel_{false};
};

}  // namespace microvllm
