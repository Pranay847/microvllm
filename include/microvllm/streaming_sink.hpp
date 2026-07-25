#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "microvllm/token_sink.hpp"
#include "microvllm/types.hpp"

namespace microvllm {

// The sink behind POST /generate/stream.
//
// This is where the Phase 1 decision to make ITokenSink emit *text deltas* pays off: the
// scheduler already streams output incrementally with stop strings correctly held back, so
// streaming needs no changes to the generator, the scheduler, or the engine -- only a
// different sink.
//
// Unlike CollectingSink, this one crosses a thread boundary: the scheduler thread produces
// deltas and the HTTP thread consumes them to write SSE events. It is therefore a small
// blocking queue with a terminal marker, not just an accumulator.
class StreamingSink final : public ITokenSink {
public:
    struct Event {
        std::string  text;
        bool         terminal = false;
        FinishReason reason   = FinishReason::kUnset;
        Usage        usage{};
        std::string  error;
    };

    explicit StreamingSink(std::shared_ptr<std::atomic<bool>> cancel)
        : cancel_(cancel ? std::move(cancel) : std::make_shared<std::atomic<bool>>(false)) {}

    void on_text(std::string_view delta) override {
        if (delta.empty()) {
            return;
        }
        Event e;
        e.text = std::string(delta);
        push(std::move(e));
    }

    void on_finish(FinishReason reason, const Usage& usage) override {
        Event e;
        e.terminal = true;
        e.reason   = reason;
        e.usage    = usage;
        push(std::move(e));
    }

    void on_error(std::string_view message) override {
        Event e;
        e.terminal = true;
        e.reason   = FinishReason::kError;
        e.error    = std::string(message);
        push(std::move(e));
    }

    [[nodiscard]] bool cancelled() const override {
        return cancel_->load(std::memory_order_relaxed);
    }

    void cancel() { cancel_->store(true, std::memory_order_relaxed); }

    // Block until an event is available and return it. The terminal event is the last one
    // a caller will receive; there is always exactly one.
    [[nodiscard]] Event next() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !events_.empty(); });
        Event e = std::move(events_.front());
        events_.pop_front();
        return e;
    }

private:
    void push(Event e) {
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            events_.push_back(std::move(e));
        }
        cv_.notify_one();
    }

    mutable std::mutex                 mutex_;
    std::condition_variable            cv_;
    std::deque<Event>                  events_;
    std::shared_ptr<std::atomic<bool>> cancel_;
};

}  // namespace microvllm
